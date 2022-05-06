from torch_geometric.transforms import BaseTransform
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from .loader import ModifiedDenseDataLoader
import torch
from .qm9 import ModifiedQM9

# import rdkit


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class ToDenseAdjV2(BaseTransform):
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
        data.orig_adj = adj.to_dense().float()
        
        tmp = torch.ones(data.orig_adj.shape[0], data.orig_adj.shape[1], 1) * 0.5

        data.adj = torch.cat((tmp, data.orig_adj), dim=-1).argmax(dim=-1)
        data.adj = data.adj[torch.tril_indices(9,9).unbind()].long()


        data.edge_index = None
        data.edge_attr = None
        data.z = None # Added to use the QM9 dataset

        data.mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.mask[:orig_num_nodes] = 1

        
        if data.x is not None:
            size = [num_nodes - data.x.size(0)] + list(data.x.size())[1:]
            data.x = torch.cat([data.x, data.x.new_zeros(size)], dim=0)

            data.x = data.x[..., :6]

            categorical = data.x[..., :-1]
            ordinal = data.x[..., -1:]

            tmp = torch.ones(categorical.shape[0], 1) * 0.5
            # print(tmp.shape, categorical.shape)

            categorical = torch.cat((tmp, categorical), dim=-1).argmax(dim=-1).long()

            data.x = torch.cat([categorical.unsqueeze(1), ordinal], dim=-1)

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

class ToDenseAdjV1(BaseTransform):
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
        data.adj = adj.to_dense().float()

        tmp = torch.ones(data.adj.shape[0], data.adj.shape[1], 1) * 0.5

        data.adj = torch.cat((tmp, data.adj[..., :-1]), dim=-1).argmax(dim=-1).long()

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


def get_datasets(type="mqm9", batch_size=128, shuffle=True, num_workers=4):

    if type == "mqm9":
        # Modified QM9 Dataset where all hydrogen atoms are removed
        # Max_num_nodes: 9
        transform = T.Compose([ToDenseAdjV2(num_nodes=9)])
        dataset = ModifiedQM9(root="./mqm9-datasets", pre_transform=transform)

        transformed_x = dataset.data.x[:, :6]
        dataset.data.x = transformed_x

    elif type == "qm9":

        transform = T.Compose([ToDenseAdjV1(29)])
        dataset = QM9(root="./qm9-datasets", pre_transform=transform)

        transformed_x = dataset.data.x[:, :6]
        dataset.data.x = transformed_x

    train_loader = ModifiedDenseDataLoader(dataset[:int(len(dataset) * 0.8)], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = ModifiedDenseDataLoader(dataset[int(len(dataset) * 0.8):], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_datasets(type="mqm9")
    
    data = next(iter(train_loader))

    print(data.adj[0])
    print(torch.all(data.adj[0] == data.adj[0].t()))