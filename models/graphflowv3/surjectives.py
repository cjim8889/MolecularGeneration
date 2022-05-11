import torch
import torch.nn as nn
from .argmax import ContextNet, ConditionalNormal, ArgmaxSurjection, ConditionalAdjacencyBlockFlow, ar_net_init
from .argmax import ConditionalARNet

from survae.flows import ConditionalInverseFlow
from survae.distributions import ConditionalNormal

from ..dequantization import Dequantization
from survae.transforms.surjections import Surjection
from .utils import create_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AtomSurjection(Surjection):
    stochastic_forward = True

    def __init__(self, num_classes=6, max_nodes=9, hidden_dim=64, mask_ratio=9., block_length=4):
        super().__init__()

        context_size = 16
        embedding_dim = 7
        inverted_mask = False

        context_net = ContextNet(context_size=context_size, num_classes=num_classes, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

        encoder_base = ConditionalNormal(nn.Sequential(
            nn.Linear(embedding_dim, num_classes),
            nn.ReLU(),
            nn.Conv2d(context_size, 2, kernel_size=1, stride=1)
        ), split_dim=1)

        ar_net_func = ar_net_init(
            ConditionalARNet,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            context_size=context_size,
        )

        transform = [ConditionalAdjacencyBlockFlow(
            ar_net_init=ar_net_func,
            max_nodes=max_nodes, 
            inverted_mask=inverted_mask, 
            mask_ratio=mask_ratio, 
            mask_init=create_mask) for _ in range(block_length)]

        conditional_flow = ConditionalInverseFlow(base_dist=encoder_base, context_init=context_net, transforms=transform)

        self.categorical_surjection = ArgmaxSurjection(conditional_flow, num_classes)
        self.ordinal_surjection = Dequantization(alpha=.0001, quants=10, device=device)

    def forward(self, x):

        categorical = x[..., :-1].squeeze(-1).long()
        ordinal = x[..., -1].squeeze(-1).long()

        z_c, ldj_c = self.categorical_surjection(categorical)
        z_o, ldj_o = self.ordinal_surjection(ordinal)

        z_o = z_o.view(z_o.shape[0], 1, z_o.shape[1], 1)

        z = torch.cat([z_c, z_o], dim=-1)

        return z, ldj_c + ldj_o

    def inverse(self, z):
        
        z_c = z[..., :-1]
        z_o = z[..., -1]

        categorical, ldj_c = self.categorical_surjection.inverse(z_c)
        ordinal, ldj_o = self.ordinal_surjection.inverse(z_o)

        return torch.cat([categorical, ordinal], dim=1).permute(0, 2, 1), 0.