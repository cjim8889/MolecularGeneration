import torch
from models.argmaxflowv2.model import ContextNet
from models.argmaxflowv2.conditional import ConditionalAdjacencyBlockFlow
from models.atomflow import create_mask

from models.dequantization import Dequantization

from utils import get_datasets
from models.atomflow import AtomFlow


if __name__ == '__main__':
    
    # transform = T.Compose([ToDenseAdjV2(num_nodes=9)])
    # dataset = ModifiedQM9(root="./mqm9-datasets", pre_transform=transform)


    # for data in dataset:
    #     if data.adj.shape[0] != 45:
    #         print(data)
    #     if data.x.shape[0] != 9:
    #         print(data)

    t = torch.tensor([[8, 6, 6, 6, 6, 7, 6, 8, 6]])

    deq = Dequantization(alpha=0.01, quants=9)

    print(deq(t), t.shape)
    # train_loader, test_loader = get_datasets(type="mqm9", batch_size=128)

    # batch = next(iter(train_loader))

    # g = AtomFlow(block_length=2, max_nodes=9, context_size=16, hidden_dim=64, inverted_mask=False)
    
    # z, ldj = g(batch.x, batch.adj)
    # x, _ = g.inverse(z, batch.adj)

    # print(z.shape, batch.x[0])
   