from survae.transforms.bijections import Bijection
from .utils import create_mask
from .coupling import MaskedCouplingFlow

import torch
import numpy as np
from torch import nn

class ARNet(nn.Module):
    
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
    def forward(self, x):
        z = self.net(x)

        return z


class AdjacencyBlockFlow(Bijection):
    def __init__(self, max_nodes=29):
        super(AdjacencyBlockFlow, self).__init__()
        self.step_size = int(np.ceil(max_nodes / 5.))
        self.transforms = nn.ModuleList()

        for idx in range(0, max_nodes, self.step_size):
            ar_net = ARNet()
            tr = MaskedCouplingFlow(ar_net, mask=create_mask([idx, max(idx + self.step_size, max_nodes)], max_nodes))
            self.transforms.append(tr)


    def forward(self, x):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            x, ldj = transform(x)
            log_prob += ldj
        
        return x, log_prob

    def inverse(self, z):
        
        log_prob = torch.zeros(z.shape[0], device=z.device)
        for idx in range(len(self.transforms) - 1, -1, -1):
            z, ldj = self.transforms[idx].inverse(z)
            log_prob += ldj
        
        return z, log_prob
