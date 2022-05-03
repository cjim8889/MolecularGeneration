from survae.transforms.bijections import Bijection
from survae.utils import sum_except_batch

import torch
from torch.nn import functional as F
from torch import nn

class MaskedCouplingFlow(Bijection):
    def __init__(self, ar_net, mask, input_channel=4):
        super(MaskedCouplingFlow, self).__init__()
        
        self.input_channel = input_channel
        self.ar_net = ar_net
        
        self.register_buffer("mask", mask)
        self.scaling_factor = nn.Parameter(torch.zeros(input_channel))

    def forward(self, x):
        return self._transform(x, forward=True)

    def inverse(self, z):
        return self._transform(z, forward=False)

    def _transform(self, z, forward=True):
        z_masked = z * self.mask
        alpha, beta = self.ar_net(z_masked).chunk(2, dim=1)

        # scaling factor idea inspired by UvA github to stabilise training 
        scaling_factor = self.scaling_factor.exp().view(1, -1, 1, 1)
        alpha = torch.tanh(alpha / scaling_factor) * scaling_factor

        alpha = alpha * (1 - self.mask)
        beta = beta * (1 - self.mask)
        
        if forward:
            z = (z + beta) * torch.exp(alpha) # Exp to ensure invertibility
            log_det = sum_except_batch(alpha)
        else:
            z = (z * torch.exp(-alpha)) - beta
            log_det = -sum_except_batch(alpha)
        
        return z, log_det
