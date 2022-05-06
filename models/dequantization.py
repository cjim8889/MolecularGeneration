from torch.nn import functional as F
import torch.nn as nn
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dequantization(nn.Module):

    def __init__(self, alpha=1e-5, quants=2, device=device):
        """
        Inputs:
            alpha - small constant that is used to scale the original input.
                    Prevents dealing with values very close to 0 and 1 when inverting the sigmoid
            quants - Number of possible discrete values (usually 256 for 8-bit image)
        """
        super().__init__()
        self.alpha = torch.tensor(alpha, device=device)
        self.quants = torch.tensor(quants, device=device)

    def dequant(self, z):
        # Transform discrete values to continuous volumes
        z = z.to(torch.float32)
        z = z + torch.rand_like(z).detach()

        return z

    def inverse(self, z):

        log_det = (-z-2*F.softplus(-z)).sum(dim=[1])
        z = torch.sigmoid(z)

        z = z * self.quants
        log_det += torch.log(self.quants) * np.prod(z.shape[1:])

        z = torch.floor(z).clamp(min=0, max=self.quants-1).to(torch.int64)

        return z, log_det

    def forward(self, x):
        log_det = torch.zeros(x.shape[0], device=x.device)

        # Add Uniform Noise into x
        z = x.to(torch.float)
        z = z + torch.rand_like(z).detach()


        z = z / self.quants
        log_det += -torch.log(self.quants) * np.prod(z.shape[1:])

        z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
        log_det += torch.log(1 - self.alpha) * np.prod(z.shape[1:])


        log_det += (-torch.log(z) - torch.log(1-z)).sum(dim=[1])
        z = torch.logit(z, eps=1e-06)


        return z, log_det
