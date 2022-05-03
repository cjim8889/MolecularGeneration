from survae.transforms.bijections import ConditionalBijection
from survae.utils import sum_except_batch

from torch import nn
import torch

class ConditionalARNet(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.LazyConv2d(128, 1, 1, 0),
            nn.LazyBatchNorm2d(),
            nn.GELU(),
            nn.LazyConv2d(128, 1, 1, 0),
            nn.LazyBatchNorm2d(),
            nn.GELU(),
            nn.LazyConv2d(8, 1, 1, 0),
            nn.ReLU(),
        )

    # x: B x K x H x W
    # context: B x C x H x W
    def forward(self, x, context):
        z = torch.cat((x, context), dim=1)
        z = self.net(z)

        return z


class MaskedConditionalCouplingFlow(ConditionalBijection):
    def __init__(self, ar_net, mask):
        super(MaskedConditionalCouplingFlow, self).__init__()
        
        self.ar_net = ar_net
        self.register_buffer("mask", mask)
        self.scaling_factor = nn.Parameter(torch.zeros(4))

    def forward(self, x, context):
        return self._transform(x, context, forward=True)

    def inverse(self, z, context):
        return self._transform(z, context, forward=False)

    def _transform(self, z, context, forward=True):
        z_masked = z * self.mask
        alpha, beta = self.ar_net(z_masked, context).chunk(2, dim=1)

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
