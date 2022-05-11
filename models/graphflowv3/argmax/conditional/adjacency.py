from survae.transforms.bijections import ConditionalBijection
import torch
from torch import nn

import numpy as np
from .coupling import MaskedConditionalCouplingFlow
from ..utils import create_mask

class ConditionalARNet(nn.Module):
    
    def __init__(self, embedding_dim=7, num_classes=5, context_size=1, hidden_dim=64):
        super().__init__()
        
        self.embed = nn.Sequential(
            nn.Linear(embedding_dim, num_classes),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.LazyConv2d(hidden_dim, 3, 1, 1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(hidden_dim, 1, 1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(2, 1, 1, 0),
            nn.ReLU(),
        )

    # x: B x 1 x 45 x 5
    # context: B x C x 45 x embedding_dim
    def forward(self, x, context):
        context = self.embed(context) # B x C x 45 x 5

        z = torch.cat((x, context), dim=1)
        z = self.net(z)

        return z

def ar_net_init(ConditionalARNet, **kwargs):
    def create():
        return ConditionalARNet(**kwargs)

    return create


class ConditionalAdjacencyBlockFlow(ConditionalBijection):
    def __init__(self, ar_net_init, 
            max_nodes=9,
            mask_ratio=2., 
            inverted_mask=False, 
            mask_init=create_mask, 
            split_dim=1):

        super(ConditionalAdjacencyBlockFlow, self).__init__()
        self.step_size = int(np.ceil(max_nodes / mask_ratio))
        self.transforms = nn.ModuleList()


        # context: B x context_size x 45 x embedding_dim
        for idx in range(0, max_nodes, self.step_size):
            net = ar_net_init()
            tr = MaskedConditionalCouplingFlow(net, mask=mask_init([idx, max(idx + self.step_size, max_nodes)], max_nodes, invert=inverted_mask), split_dim=split_dim)
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

