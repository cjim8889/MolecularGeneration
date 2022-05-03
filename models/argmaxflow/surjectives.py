from survae.transforms.surjections import Surjection
from einops.layers.torch import Rearrange
from survae.transforms import Softplus
from torch.nn import functional as F
import torch

class ArgmaxSurjection(Surjection):
    stochastic_forward = True

    def __init__(self, encoder, num_classes):
        super(ArgmaxSurjection, self).__init__()

        self.encoder = encoder
        self.num_classes = num_classes
        self.softplus = Softplus()
        self.rearrange = Rearrange("B C H W -> B H W C")
        self.invert = Rearrange("B H W C -> B C H W")
    
    def forward(self, x):
      u, log_pu = self.encoder.sample_with_log_prob(context=x)

      u = self.rearrange(u)
      
      index = x.unsqueeze(-1)
      u_max = torch.take_along_dim(u, index, dim=-1)

      u_x = u_max - u

      u_tmp = F.softplus(u_x)
      ldj = F.logsigmoid(u_x)

      v = u_max - u_tmp

      ldj = ldj.scatter_(3, index, 0.)
      v = v.scatter_(3, index, u_max)

      v = self.invert(v)

      log_pz = log_pu - torch.sum(ldj, dim=[1, 2, 3])

      return v, -log_pz

    def inverse(self, z):
      return z.argmax(dim=1), 0.
