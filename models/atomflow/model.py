import torch
import torch.nn as nn
from ..argmaxflowv2 import ArgmaxFlow, ContextNet, ConditionalNormal, ArgmaxSurjection, ConditionalAdjacencyBlockFlow
from survae.flows import ConditionalInverseFlow, ConditionalFlow
from survae.distributions import ConditionalNormal
from ..dequantization import Dequantization
from .surjectives import AtomSurjection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class AtomFlow(nn.Module):
    def __init__(self, AdjacencyFlow) -> None:
        super().__init__()

        self.surjection = AtomSurjection()

        self.flows = 

    def forward(self, x):
        
        return self.surjection(x)

    def inverse(self, z):
        
        return self.surjection.inverse(z)