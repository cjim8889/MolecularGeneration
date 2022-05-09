import torch
import torch.nn as nn
from ..argmaxflowv2 import ContextNet, ConditionalAdjacencyBlockFlow
from survae.transforms.bijections import ConditionalBijection, ActNormBijection2d, Conv1x1
from torch_geometric.nn import DenseGCNConv

from .surjectives import AtomSurjection
from einops.layers.torch import Rearrange
from .utils import create_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ContextNet(nn.Module):
    def __init__(self, context_size=16, num_classes=4, embedding_dim=5, hidden_dim=32):
        super(ContextNet, self).__init__()
        #Assume input B x 45
        self.net = nn.Sequential(
            Rearrange("B H W C -> B C H W"),
            # Padding is flaky and requires further investigation 
            nn.LazyConv2d(hidden_dim, kernel_size=1, stride=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(hidden_dim, kernel_size=1, stride=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(1, kernel_size=1, stride=1),
            nn.ReLU(),
            Rearrange("B 1 H W -> B H W"),
            nn.Linear(9, 9),
            nn.ReLU(),
        )

    def forward(self, x):

        return self.net(x)

class ConditionalARNet(nn.Module):
    
    def __init__(self, embedding_dim=7, num_classes=7, context_size=1, hidden_dim=64):
        super().__init__()
        
        
        self.indices = torch.triu_indices(9, 9, device=device)
        self.graph_nets = nn.ModuleList([
            DenseGCNConv(num_classes + 9, num_classes),
        ])

        self.net = nn.Sequential(
            Rearrange("B H W -> B 1 H W"),
            nn.LazyConv2d(hidden_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.LazyConv2d(hidden_dim * 2, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.LazyConv2d(2, kernel_size=1, stride=1),
        )
    # x: B x 1 x 9 x 7
    # context: context: B x C x 45 x Embedding_dim adj: B x 1 x 45 x 1

    def forward(self, x, context):
        adj_dense = context['b_adj']

        c = context['context']

        z = x.squeeze(1)
        z = torch.cat((z, c), dim=-1)

        for graph in self.graph_nets:
            z = graph(z, adj_dense)

        return self.net(z)


class AtomGraphFlowV2(nn.Module):
    def __init__(self, context_init=None, mask_ratio=9., block_length=6, max_nodes=9, context_size=1, embedding_dim=7, hidden_dim=64, inverted_mask=False) -> None:
        super().__init__()

        if context_init is None:
            context_init = ContextNet(context_size=context_size, num_classes=5, embedding_dim=5, hidden_dim=hidden_dim)

        self.context_init = context_init
        
        self.surjection = AtomSurjection()
        self.transforms = nn.ModuleList()

        self.transforms.append(self.surjection)

        for idx in range(block_length):
            # norm = ActNormBijection2d(num_features=1)

            # self.transforms.append(norm)

            # conv1x1 = Conv1x1(num_channels=1, orthogonal_init=True, slogdet_cpu=True)
            # self.transforms.append(conv1x1)

            cf = ConditionalAdjacencyBlockFlow(
                ar_net=ConditionalARNet,
                max_nodes=max_nodes,
                embedding_dim=embedding_dim,
                num_classes=7,
                context_size=context_size,
                inverted_mask=inverted_mask,
                mask_ratio=mask_ratio,
                mask_init=create_mask,
            )

            self.transforms.append(cf)
        

    def forward(self, x, context):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        if self.context_init is not None:
            context = {
                "context": self.context_init(context['adj']),
                "b_adj": context['b_adj']
            }

        for transform in self.transforms:
            if isinstance(transform, ConditionalBijection):
                x, ldj = transform(x, context)
            else:
                x, ldj = transform(x)

            log_prob += ldj

        
        return x, log_prob

    def inverse(self, z, context):
        log_prob = torch.zeros(z.shape[0], device=z.device)

        if self.context_init is not None:
            context = {
                "context": self.context_init(context['adj']),
                "b_adj": context['b_adj']
            }

        for idx in range(len(self.transforms) - 1, -1, -1):
            if isinstance(self.transforms[idx], ConditionalBijection):
                z, ldj = self.transforms[idx].inverse(z, context)
            else:
                z, ldj = self.transforms[idx].inverse(z)

            log_prob += ldj
        
        return z, log_prob