from survae.transforms.bijections import ConditionalBijection
import torch
from torch import nn

import numpy as np
from .coupling import ConditionalARNet, MaskedConditionalCouplingFlow
from ..utils import create_mask

class ConditionalAdjacencyBlockFlow(ConditionalBijection):
    def __init__(self, max_nodes=29):
        super(ConditionalAdjacencyBlockFlow, self).__init__()
        self.step_size = int(np.ceil(max_nodes / 5.))
        self.transforms = nn.ModuleList()

        for idx in range(0, max_nodes, self.step_size):
            ar_net = ConditionalARNet()
            tr = MaskedConditionalCouplingFlow(ar_net, mask=create_mask([idx, max(idx + self.step_size, max_nodes)], max_nodes))
            self.transforms.append(tr)


    def forward(self, x, context):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            x, ldj = transform(x, context)
            log_prob += ldj
        
        return x, log_prob

    def inverse(self, z, context):
        log_prob = torch.zeros(z.shape[0], device=z.device)
        for idx in range(len(self.transforms) - 1, -1, -1):
            z, ldj = self.transforms[idx].inverse(z, context)
            log_prob += ldj
        
        return z, log_prob
