from survae.flows import ConditionalInverseFlow
from survae.distributions import ConditionalNormal
from einops.layers.torch import Rearrange

from .surjectives import ArgmaxSurjection
from .adjacency import AdjacencyBlockFlow
from .conditional import ConditionalAdjacencyBlockFlow
from torch import nn
import torch

class ContextNet(nn.Module):
    def __init__(self, context_size=16, num_classes=4, embedding_dim=16):
        super(ContextNet, self).__init__()
        #Assume input B x H x W
        self.net = nn.Sequential(
            nn.Embedding(num_classes, embedding_dim),
            Rearrange("B H W E -> B E H W"),
            nn.Conv2d(embedding_dim, 32, kernel_size=5, stride=1, padding=2),
            nn.LazyBatchNorm2d(),
            nn.ReLU(True),
            nn.Conv2d(32, context_size, kernel_size=3, stride=1, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(True),
        )

    def forward(self, x):

        return self.net(x)

class ArgmaxFlow(nn.Module):
    def __init__(self):
        super().__init__()

        self.transforms = nn.ModuleList()

        context_size = 16
        context_net = ContextNet(context_size=context_size)
        encoder_base = ConditionalNormal(nn.Conv2d(context_size, 8, kernel_size=3, stride=1, padding=1), split_dim=1)
        t = ConditionalAdjacencyBlockFlow()
        c = ConditionalInverseFlow(base_dist=encoder_base, context_init=context_net, transforms=[t])

        surjection = ArgmaxSurjection(c, 4)


        self.transforms.append(surjection)

        for i in range(6):
            flow = AdjacencyBlockFlow()

            self.transforms.append(flow)

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