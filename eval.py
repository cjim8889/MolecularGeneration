import torch
from models import atomflow


from models.atomflow import AtomFlow
from models.argmaxflowv2 import ArgmaxFlow

from rdkit import Chem
from torch_geometric.nn import DenseGraphConv
from models.atomflow.graphflow import ContextNet
from models.atomflow.graphflow import AtomGraphFlow
from survae.transforms.bijections import Conv1x1, ActNormBijection2d


device = torch.device("cpu")

flow = Conv1x1(
    num_channels=1,
    orthogonal_init=True,
    slogdet_cpu=True
)

norm = ActNormBijection2d(
    num_features=1
)

z, ldj = norm(torch.randn(1, 1, 9, 2))

print(z, z.shape)