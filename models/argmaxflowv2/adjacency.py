from survae.transforms.bijections import Bijection
from .utils import create_mask
from .coupling import MaskedCouplingFlow

import torch
import numpy as np
from torch import nn


class ARNet(nn.Module):
    
    def __init__(self, hidden_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.LazyConv2d(hidden_dim, 3, 1, 1),
            nn.LazyBatchNorm2d(),
            nn.GELU(),
            nn.LazyConv2d(hidden_dim, 3, 1, 1),
            nn.LazyBatchNorm2d(),
            nn.GELU(),
            nn.LazyConv2d(2, 1, 1, 0),
            nn.ReLU(),
        )

    # x: B x 1 x H x num_classes
    def forward(self, x):
        z = self.net(x)

        return z


class AdjacencyBlockFlow(Bijection):
    def __init__(self, max_nodes=9, mask_ratio=5., input_channel=1, hidden_dim=128, inverted_mask=False):
        super(AdjacencyBlockFlow, self).__init__()
        self.step_size = int(np.ceil(max_nodes / mask_ratio))
        self.transforms = nn.ModuleList()

        for idx in range(0, max_nodes, self.step_size):
            ar_net = ARNet(hidden_dim=hidden_dim)
            tr = MaskedCouplingFlow(ar_net, input_channel=input_channel, mask=create_mask([idx, max(idx + self.step_size, max_nodes)], max_nodes, invert=inverted_mask))
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
