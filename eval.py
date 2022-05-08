import torch
from models import atomflow


from models.atomflow import AtomFlow
from models.argmaxflowv2 import ArgmaxFlow

from rdkit import Chem
from torch_geometric.nn import DenseGraphConv
from models.atomflow.graphflow import ContextNet
from models.atomflow.graphflow import AtomGraphFlow



device = torch.device("cpu")

# net = ContextNet(context_size=1, num_classes=5, embedding_dim=5, hidden_dim=32)

# z = net(torch.ones(1, 45).long())

# print(z.shape)

net = AtomGraphFlow(
    mask_ratio=9,
    block_length=1,
    max_nodes=9,
    context_size=1,
    hidden_dim=32
)


z, _ = net(torch.ones(1, 9, 2).long(), torch.ones(1, 45).long())

print(z.shape)
print(sum([p.numel() for p in net.parameters()]))