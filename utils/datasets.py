from torch_geometric.transforms import BaseTransform
import torch.nn as nn
import torch_geometric
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, DenseDataLoader
import torch
from torch_sparse import matmul, SparseTensor, spmm
from torch_scatter import segment_sum_coo, segment_coo, scatter
import numpy as np

# import rdkit


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class ToDense(BaseTransform):
    r"""Converts a sparse adjacency matrix to a dense adjacency matrix with
    shape :obj:`[num_nodes, num_nodes, *]` (functional name: :obj:`to_dense`).

    Args:
        num_nodes (int): The number of nodes. If set to :obj:`None`, the number
            of nodes will get automatically inferred. (default: :obj:`None`)
    """
    def __init__(self, num_nodes=None):
        self.num_nodes = num_nodes

    def __call__(self, data):
        assert data.edge_index is not None

        orig_num_nodes = data.num_nodes

        if self.num_nodes is None:
            num_nodes = orig_num_nodes
        else:
            assert orig_num_nodes <= self.num_nodes
            num_nodes = self.num_nodes

        if data.edge_attr is None:
            edge_attr = torch.ones(data.edge_index.size(1), dtype=torch.float)
        else:
            edge_attr = data.edge_attr

        size = torch.Size([num_nodes, num_nodes] + list(edge_attr.size())[1:])
        adj = torch.sparse_coo_tensor(data.edge_index, edge_attr, size)
        data.adj = adj.to_dense()
        data.edge_index = None
        data.edge_attr = None
        data.z = None # Added to use the QM9 dataset

        data.mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.mask[:orig_num_nodes] = 1

        

        if data.x is not None:
            size = [num_nodes - data.x.size(0)] + list(data.x.size())[1:]
            data.x = torch.cat([data.x, data.x.new_zeros(size)], dim=0)

        if data.pos is not None:
            size = [num_nodes - data.pos.size(0)] + list(data.pos.size())[1:]
            data.pos = torch.cat([data.pos, data.pos.new_zeros(size)], dim=0)

        if data.y is not None and (data.y.size(0) == orig_num_nodes):
            size = [num_nodes - data.y.size(0)] + list(data.y.size())[1:]
            data.y = torch.cat([data.y, data.y.new_zeros(size)], dim=0)

        return data

    def __repr__(self) -> str:
        if self.num_nodes is None:
            return super().__repr__()
        return f'{self.__class__.__name__}(num_nodes={self.num_nodes})'


def get_datasets():
    transform = T.Compose([ToDense(29)])
    dataset = QM9(root="./qm9-datasets", transform=transform)

    transformed_x = dataset.data.x[:, :6]
    dataset.data.x = transformed_x

    train_loader = DenseDataLoader(dataset[:int(len(dataset) * 0.8)], batch_size=128)
    test_loader = DenseDataLoader(dataset[int(len(dataset) * 0.8):], batch_size=128)

    return train_loader, test_loader