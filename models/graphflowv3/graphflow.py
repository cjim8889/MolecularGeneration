from pprint import pp
import torch
import torch.nn as nn
from .argmax import ConditionalAdjacencyBlockFlow, ar_net_init
from survae.transforms.bijections import ConditionalBijection
from torch_geometric.nn import DenseGCNConv

from .surjectives import AtomSurjection
from einops.layers.torch import Rearrange
from .utils import create_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AdjContextNet(nn.Module):
    def __init__(self, context_size=16, num_classes=5, embedding_dim=7, hidden_dim=32):
        super(AdjContextNet, self).__init__()
        #Assume input B x 45
        self.net = nn.Sequential(
            nn.Embedding(num_classes, embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            Rearrange("B H E -> B 1 H E"),
            # Padding is flaky and requires further investigation 
            nn.LazyConv2d(hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LazyConv2d(hidden_dim, kernel_size=1, stride=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(context_size, kernel_size=1, stride=1),
            nn.ReLU(), # B x C x 45 x E
        )

    def forward(self, x):

        return self.net(x)

class ConditionalARNet(nn.Module):
    
    def __init__(self, context_embedding_dim=7, x_width=7, max_nodes=9, hidden_dim=64):
        super().__init__()
        
        self.graph_nets = nn.ModuleList([
            DenseGCNConv(x_width, x_width),
            DenseGCNConv(x_width, x_width),
        ])

        self.net = nn.Sequential(
            nn.LazyConv2d(hidden_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.LazyConv2d(hidden_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.LazyConv2d(2, kernel_size=1, stride=1),
        )

        self.context_net = nn.Sequential(
            nn.LazyConv2d(hidden_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.LazyConv2d(1, kernel_size=1, stride=1),
            nn.ReLU(),
            Rearrange("B 1 H E -> B 1 (H E)"),
            nn.Linear(45 * context_embedding_dim, max_nodes * x_width),
            nn.ReLU(),
            Rearrange("B 1 (H E) -> B 1 H E", H=max_nodes, E=x_width),
        )
    # x: B x 1 x 9 x 7
    # context: context: B x C x 45 x Embedding_dim adj: B x 1 x 45 x 1

    def forward(self, x, context):
        adj_dense = context['b_adj']

        c = self.context_net(context['context'])

        z = x.squeeze(1)

        for graph in self.graph_nets:
            z = graph(z, adj_dense)

        return self.net(torch.cat((z.unsqueeze(1), c), dim=1))


class AtomGraphFlowV3(nn.Module):
    def __init__(self, context_init=None, mask_ratio=9., block_length=6, surjection_length=4, max_nodes=9, context_size=1, embedding_dim=7, hidden_dim=64, inverted_mask=False) -> None:
        super().__init__()

        if context_init is None:
            context_init = AdjContextNet(context_size=context_size, hidden_dim=hidden_dim)

        self.context_init = context_init
        
        self.surjection = AtomSurjection(hidden_dim=hidden_dim, block_length=surjection_length)
        self.transforms = nn.ModuleList()

        self.transforms.append(self.surjection)

        ar_net_func = ar_net_init(
            ConditionalARNet,
            context_embedding_dim=7,
            x_width=7, 
            max_nodes=max_nodes, 
            hidden_dim=hidden_dim
        )

        for idx in range(block_length):
            cf = ConditionalAdjacencyBlockFlow(
                ar_net_init=ar_net_func,
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