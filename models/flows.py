import torch.nn as nn
import torch
import numpy as np
from einops.layers.torch import Rearrange
from models.dequantization import Dequantization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_adj_mask(c_in, invert=False):
    mask = torch.cat([torch.ones(c_in//2, dtype=torch.float32),
                      torch.zeros(c_in-c_in//2, dtype=torch.float32)])
    mask = mask.view(1, 1, 1, c_in, 1)
    if invert:
        mask = 1 - mask
    return mask

class MaskedCouplingFlow(nn.Module):
    def __init__(self, input_channel, mask, affine=True, device=device):
        super().__init__()

        self.input_channel = input_channel
        self.scaling_factor = nn.Parameter(torch.zeros(input_channel))
        self.register_buffer("mask", mask.to(device))

        self.network = nn.Sequential(
            nn.Conv3d(input_channel, 64, 3, 1, 1),
            nn.LazyBatchNorm3d(),
            nn.GELU(),
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.LazyBatchNorm3d(),
            nn.ReLU(),
            nn.Conv3d(64, input_channel * 2, 3, 1, 1),
            nn.LazyBatchNorm3d(),
            nn.ReLU(),
        )

    def _transform(self, z, forward=True):

        z_masked = z * self.mask
        alpha, beta = self.network(z_masked).chunk(2, dim=1)

        # scaling factor idea inspired by UvA github to stabilise training 
        scaling_factor = self.scaling_factor.exp().view(1, -1, 1, 1, 1)
        alpha = torch.tanh(alpha / scaling_factor) * scaling_factor


        alpha = alpha * (1 - self.mask)
        beta = beta * (1 - self.mask)
        
        if forward:
            z = (z + beta) * torch.exp(alpha) # Exp to ensure invertibility
            # z = z / alpha + beta
            log_det = torch.sum(alpha, dim=[1, 2, 3, 4])
        else:
            # z = (z - beta) * alpha
            z = (z * torch.exp(-alpha)) - beta
            log_det = -torch.sum(alpha, dim=[1, 2, 3, 4])
        
        return z, log_det

    def forward(self, x):
        z, log_det = self._transform(x, forward=True)

        return z, log_det

    def inverse(self, z):
        x, log_det = self._transform(z, forward=False)

        return x, log_det

        
class AdjacencyFlows(nn.Module):
    def __init__(self, input_channel=1, t=5, affine=True):
        super().__init__()

        self.rearrange = Rearrange("B H W D -> B 1 D H W")
        self.invertarrange = Rearrange("B 1 D H W -> B H W D")

        self.flows = nn.ModuleList()


        invert = False

        self.flows.append(Dequantization(quants=2))

        for idx in range(t):
            flow = MaskedCouplingFlow(input_channel, create_adj_mask(29, invert=invert))
            self.flows.append(flow)

            invert = not invert

    def forward(self, x):
        log_det = torch.zeros(x.shape[0], device=device)
        
        z = self.rearrange(x)
        
        for idx in range(len(self.flows)):

            z, log_det_idx = self.flows[idx](z)

            log_det += log_det_idx

        
        return z, log_det, 

    def inverse(self, z):
        log_det = torch.zeros(z.shape[0], device=device)

        for idx in range(len(self.flows) - 1, -1, -1):
            z, log_det_idx = self.flows[idx].inverse(z)
            log_det += log_det_idx

        x = self.invertarrange(z)
        return x, log_det
        