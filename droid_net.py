import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from modules.extractor import BasicEncoder
from modules.corr import CorrBlock
from modules.gru import ConvGRU, DispConvGRU
from modules.clipping import GradientClip

from lietorch import SE3
from geom.ba import BA, MoBA

import geom.projective_ops as pops
from geom.graph_utils import graph_to_edge_list, keyframe_indicies

from torch_scatter import scatter_mean


def cvx_upsample(data, mask):
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, [3,3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)

    return up_data

def upsample_disp(disp, mask):
    batch, num, ht, wd = disp.shape
    disp = disp.view(batch*num, ht, wd, 1)
    mask = mask.view(batch*num, -1, ht, wd)
    return cvx_upsample(disp, mask).view(batch, num, 8*ht, 8*wd)


class GraphAgg(nn.Module):
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            GradientClip(),
            nn.Softplus())

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8*8*9, 1, padding=0))

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch*num, ch, ht, wd)

        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))

        net = net.view(batch, num, 128, ht, wd)
        net = scatter_mean(net, ix, dim=1)
        net = net.view(-1, 128, ht, wd)

        net = self.relu(self.conv2(net))

        eta = self.eta(net).view(batch, -1, ht, wd)
        upmask = self.upmask(net).view(batch, -1, 8*8*9, ht, wd)

        return .01 * eta, upmask


class DispGraphAgg(nn.Module):
    def __init__(self):
        super(DispGraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8*8*9, 1, padding=0))

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch*num, ch, ht, wd)

        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))

        net = net.view(batch, num, 128, ht, wd)
        net = scatter_mean(net, ix, dim=1)
        net = net.view(-1, 128, ht, wd)

        net = self.relu(self.conv2(net))
        upmask = self.upmask(net).view(batch, -1, 8*8*9, ht, wd)

        return upmask


class FlowUpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2*3 + 1)**2

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True))

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True))

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip(),
            nn.Sigmoid())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip())

        self.gru = ConvGRU(128, 128+128+64)
        self.agg = GraphAgg()

    def forward(self, net, inp, corr, flow=None, ii=None, jj=None):
        """ M3-SLAM update operator """

        batch, num, ch, ht, wd = net.shape

        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)

        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch*num, -1, ht, wd)
        inp = inp.view(batch*num, -1, ht, wd)        
        corr = corr.view(batch*num, -1, ht, wd)
        flow = flow.view(batch*num, -1, ht, wd)

        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)

        ### update variables ###
        delta = self.delta(net).view(*output_dim)
        weight = self.weight(net).view(*output_dim)

        delta = delta.permute(0,1,3,4,2)[...,:2].contiguous()
        weight = weight.permute(0,1,3,4,2)[...,:2].contiguous()

        net = net.view(*output_dim)

        if ii is not None:
            eta, upmask = self.agg(net, ii.to(net.device))
            return net, delta, weight, eta, upmask

        else:
            return net, delta, weight


class DepthUpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2*3 + 1)**2
        self.size_disp_enc = 7
        self.aggregation = "mean"

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True))

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 3, padding=1),
            GradientClip())

        self.gru = DispConvGRU(h_planes=128, i_planes=128+128+self.size_disp_enc*self.size_disp_enc)
        self.agg = GraphAgg()

    def disp_encoder(self, disp):
        batch, _, ht, wd = disp.shape
        dispkxk = F.unfold(disp, [self.size_disp_enc, self.size_disp_enc], padding=self.size_disp_enc//2)
        dispkxk = dispkxk.view(batch, self.size_disp_enc ** 2, ht, wd)
        disp1x1 = disp.view(batch, 1, ht, wd)

        return dispkxk - disp1x1

    def forward(self, net, inp, corr, disp=None, ii=None, jj=None):
        """ M3SLAM update operator """

        batch, num, source_view_num, ch, ht, wd = corr.shape
        output_dim = (batch, num, -1, ht, wd)

        if depth is None:
            depth = torch.ones(batch, num, 1, ht, wd, device=net.device)

        net = net.view(batch*num, -1, ht, wd)
        inp = inp.view(batch*num, -1, ht, wd)        
        disp = disp.view(batch*num, -1, ht, wd)
        corr = corr.view(batch*num, -1, ht, wd)

        # if self.aggregation is 'mean':
        #     corr = torch.mean(corr, dim=1, keepdim=False)
        # elif self.aggregation is 'max':
        #     corr = torch.max(corr, dim=1, keepdim=False)
        # elif self.aggregation is 'std':
        #     corr = torch.std(corr, dim=1, keepdim=False) 

        corr = self.corr_encoder(corr)
        disp = self.disp_encoder(disp)
        net = self.gru(net, inp, corr, disp)

        ### update variables ###
        delta = self.delta(net).view(*output_dim)
        delta = delta.permute(0,1,3,4,2)[...,:2].contiguous()

        net = net.view(*output_dim)

        if ii is not None:
            upmask = self.agg(net, ii.to(net.device))
            return net, delta, upmask

        else:
            return net, delta


class DroidNet(nn.Module):
    def __init__(self):
        super(DroidNet, self).__init__()
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')
        self.flowupdate = FlowUpdateModule()
        self.depthupdate = DepthUpdateBlock()


    def extract_features(self, images):
        """ run feeature extraction networks """

        # normalize images
        images = images[:, :, [2,1,0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        fmaps = self.fnet(images)
        net = self.cnet(images)
        
        net, inp = net.split([128,128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        return fmaps, net, inp


    def forward(self, Gs, images, disps, intrinsics, graph=None, num_steps=25, fixedp=2):
        """ Estimates SE3 or Sim3 between pair of frames """

        u = keyframe_indicies(graph)
        ii, jj, kk = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        fmaps, net, inp = self.extract_features(images)
        net, inp = net[:,ii], inp[:,ii]
        disps_net = net
        of_net = net
        corr_fn = CorrBlock(fmaps[:,ii], fmaps[:,jj], num_levels=4, radius=3)

        ht, wd = images.shape[-2:]
        coords0 = pops.coords_grid(ht//8, wd//8, device=images.device)
        
        coords1, _ = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
        target = coords1.clone()

        Gs_list, disp_list, residual_list = [], [], []
        # one step pose; one step depth
        disp_list.append(torch.ones_like((disps.shape[0], disps.shape[1], ht, wd))))
        for step in range(num_steps):
            #perform pose refinement
            if step % 2 == 0:
                Gs = Gs.detach()
                disps = disps.detach()
                coords1 = coords1.detach()
                target = target.detach()

                # extract motion features
                corr = corr_fn(coords1)
                resd = target - coords1
                flow = coords1 - coords0

                motion = torch.cat([flow, resd], dim=-1)
                motion = motion.permute(0,1,4,2,3).clamp(-64.0, 64.0)

                of_net, of_delta, weight, eta, upmask = \
                    self.flowupdate(of_net, inp, corr, motion, ii, jj)

                target = coords1 + of_delta

                for i in range(2):
                    Gs = MoBA(target, weight, eta, Gs, disps, intrinsics, ii, jj, fixedp=2)

                Gs_list.append(Gs)
            #perform depth refinement
            elif step % 2 == 1:
                Gs = Gs.detach() 
                disps = disps.detach()
                coords1 = coords1.detach()
                target = target.detach()

                # extract corr
                corr = corr_fn(coords1)

                # average corr in the source views 
                _, ix = torch.unique(ii, return_inverse=True)
                corr = scatter_mean(corr, ix, dim=1)                

                disps_net, disps_delta, upmask = \
                    self.depthupdate(disps_net, inp, corr, disps, ii, jj)

                disps = disps + disps_delta
                up_disps = upsample_disp(disps, upmask)

                disp_list.append(up_disps)

            coords1, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
            residual = (target - coords1)
            residual_list.append(valid_mask * residual)

        return Gs_list, disp_list, residual_list
