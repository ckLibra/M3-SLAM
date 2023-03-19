import torch
import torch.nn as nn


class ConvGRU(nn.Module):
    def __init__(self, h_planes=128, i_planes=128):
        super(ConvGRU, self).__init__()
        self.do_checkpoint = False
        self.convz = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)
        self.convr = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)
        self.convq = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)

        self.w = nn.Conv2d(h_planes, h_planes, 1, padding=0)

        self.convz_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0)
        self.convr_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0)
        self.convq_glo = nn.Conv2d(h_planes, h_planes, 1, padding=0)

    def forward(self, net, *inputs):
        inp = torch.cat(inputs, dim=1)
        net_inp = torch.cat([net, inp], dim=1)

        b, c, h, w = net.shape
        glo = torch.sigmoid(self.w(net)) * net
        glo = glo.view(b, c, h*w).mean(-1).view(b, c, 1, 1)

        z = torch.sigmoid(self.convz(net_inp) + self.convz_glo(glo))
        r = torch.sigmoid(self.convr(net_inp) + self.convr_glo(glo))
        q = torch.tanh(self.convq(torch.cat([r*net, inp], dim=1)) + self.convq_glo(glo))

        net = (1-z) * net + z * q
        return net

class DispConvGRU(nn.Module):
    def __init__(self, kernel_z = 3, kernel_r=3, kernel_q=3, h_planes=None, i_planes=None):
        super(ConvGRU, self).__init__()
        self.do_checkpoint = False
        self.convz = nn.Conv2d(h_planes+i_planes, h_planes, kernel_z, padding=kernel_z//2)
        self.convr = nn.Conv2d(h_planes+i_planes, h_planes, kernel_z, padding=kernel_z//2)
        self.convq = nn.Conv2d(h_planes+i_planes, h_planes, kernel_z, padding=kernel_z//2)

    def forward(self, net, *inputs):
        inp = torch.cat(inputs, dim=1)
        net_inp = torch.cat([net, inp], dim=1)
        z = self.convz(net_inp)
        z = torch.sigmoid(z)
        r = torch.sigmoid(self.convr(net_inp))
        q = torch.tanh(self.convq(torch.cat([r*net, inp], dim=1)))
        net = (1-z) * net + z * q
        return net