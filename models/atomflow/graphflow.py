from numpy import indices
import torch
import torch.nn as nn
from ..argmaxflowv2 import ContextNet, ConditionalAdjacencyBlockFlow
from survae.transforms.bijections import ConditionalBijection
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
            nn.Embedding(num_classes, embedding_dim), # B x 45 x embedding_dim
            Rearrange("B W E -> B 1 W E"),
            # Padding is flaky and requires further investigation 
            nn.Conv2d(1, hidden_dim, kernel_size=1, stride=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, context_size, kernel_size=(1, embedding_dim), stride=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(True),
        )

    def forward(self, x):

        return self.net(x)

class ConditionalARNet(nn.Module):
    
    def __init__(self, embedding_dim=7, num_classes=7, context_size=1, hidden_dim=64):
        super().__init__()
        
        
        self.indices = torch.triu_indices(9, 9, device=device)
        self.graph_net = DenseGCNConv(num_classes, num_classes)

        self.context_net = nn.Sequential(
            nn.Flatten(start_dim=2, end_dim=-1),
            nn.Linear(45, 9 * num_classes),
            nn.ReLU(True),
            nn.Unflatten(dim=-1, unflattened_size=(9, 7))
        )

        self.net = nn.Sequential(
            nn.LazyConv2d(hidden_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.LazyConv2d(hidden_dim, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.LazyConv2d(2, kernel_size=1, stride=1),
        )
    # x: B x 1 x 9 x 7
    # context: context: B x C x 45 x Embedding_dim adj: B x 1 x 45 x 1
    def forward(self, x, context):
        adj_dense = torch.zeros((x.shape[0], 9, 9), device=device)
        adj_dense[..., self.indices[0], self.indices[1]] = context['adj'].view(x.shape[0], 45).float()
        adj_dense[adj_dense > 0.] = 1.

        z = x.squeeze(1)
        z = self.graph_net(z, adj_dense).unsqueeze(1)

        c = self.context_net(context['context'])


        z = torch.cat([z, c], dim=1)

        return self.net(z)


class AtomGraphFlow(nn.Module):
    def __init__(self, context_init=None, mask_ratio=9., block_length=6, max_nodes=9, context_size=1, embedding_dim=7, hidden_dim=64, inverted_mask=False) -> None:
        super().__init__()

        if context_init is None:
            context_init = ContextNet(context_size=context_size, num_classes=5, embedding_dim=5, hidden_dim=hidden_dim)

        self.context_init = context_init
        
        self.surjection = AtomSurjection()
        self.transforms = nn.ModuleList()

        self.transforms.append(self.surjection)

        for idx in range(block_length):
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
                "context": self.context_init(context),
                "adj": context
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
                "context": self.context_init(context),
                "adj": context
            }

        for idx in range(len(self.transforms) - 1, -1, -1):
            if isinstance(self.transforms[idx], ConditionalBijection):
                z, ldj = self.transforms[idx].inverse(z, context)
            else:
                z, ldj = self.transforms[idx].inverse(z)

            log_prob += ldj
        
        return z, log_prob