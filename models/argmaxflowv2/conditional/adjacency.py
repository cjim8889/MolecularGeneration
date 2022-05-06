from survae.transforms.bijections import ConditionalBijection
import torch
from torch import nn

import numpy as np
from .coupling import MaskedConditionalCouplingFlow
from ..utils import create_mask

class ConditionalARNet(nn.Module):
    
    def __init__(self, embedding_dim=7, context_size=1, hidden_dim=64):
        super().__init__()
        
        self.embed = nn.Sequential(
            nn.Linear(embedding_dim, 4),
            nn.ReLU(),
        )
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

    # x: B x 1 x 45 x 4
    # context: B x C x 45 x embedding_dim
    def forward(self, x, context):
        context = self.embed(context) # B x C x 45 x 4

        z = torch.cat((x, context), dim=1)
        z = self.net(z)

        return z



class ConditionalAdjacencyBlockFlow(ConditionalBijection):
    def __init__(self, max_nodes=9, embedding_dim=7, context_size=1, hidden_dim=128, mask_ratio=2., inverted_mask=False):
        super(ConditionalAdjacencyBlockFlow, self).__init__()
        self.step_size = int(np.ceil(max_nodes / mask_ratio))
        self.transforms = nn.ModuleList()


        # context: B x context_size x 45 x embedding_dim
        for idx in range(0, max_nodes, self.step_size):
            ar_net = ConditionalARNet(embedding_dim=embedding_dim, context_size=context_size, hidden_dim=hidden_dim)
            tr = MaskedConditionalCouplingFlow(ar_net, mask=create_mask([idx, max(idx + self.step_size, max_nodes)], max_nodes, invert=inverted_mask))
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

