import torch
from models.argmaxflowv2.model import ContextNet
from models.argmaxflowv2.conditional import ConditionalAdjacencyBlockFlow
from survae.distributions import ConditionalNormal
from survae.flows import ConditionalInverseFlow
from torch import embedding, nn
from models.argmaxflowv2 import create_mask
from models.argmaxflowv2 import ArgmaxSurjection, ArgmaxFlow
from utils.qm9 import ModifiedQM9
import torch_geometric.transforms as T

from utils import ToDenseAdjV2
from utils import get_datasets, create_model_and_optimiser_sche

if __name__ == '__main__':
    
    # transform = T.Compose([ToDenseAdjV2(num_nodes=9)])
    # dataset = ModifiedQM9(root="./mqm9-datasets", pre_transform=transform)


    # for data in dataset:
    #     if data.adj.shape[0] != 45:
    #         print(data)
    #     if data.x.shape[0] != 9:
    #         print(data)
    train_loader, test_loader = get_datasets(type="mqm9", batch_size=128)

    batch = next(iter(train_loader))
    print(batch.adj[:10])

    # zeros = torch.zeros(45, dtype=torch.long)
    
    # for batch in train_loader:
    #     for i in range(batch.adj.shape[0]):
            
    #         adj = batch.adj[i]
    #         orig_adj = batch.orig_adj[i]
    #         x = batch.x[i]
    #         if torch.all(adj == zeros):
    #             tmp = torch.ones(orig_adj.shape[0], orig_adj.shape[1], 1) * 0.5

    #             t_adj = torch.cat((tmp, orig_adj[..., :-1]), dim=-1).argmax(dim=-1)
        

    #             print(batch.adj[i], "\n\n", t_adj,"\n\n",  x,"\n\n", orig_adj,"\n\n")

    #             print(torch.all(t_adj == t_adj.t()))
    # for batch in train_loader:
    
        # print(batch.adj)
    # config = dict(
    #         epochs=10,
    #         batch_size=128,
    #         optimiser="Adam",
    #         learning_rate=1e-03,
    #         weight_decay=1e-06,
    #         momentum=0.9,
    #         dataset="MQM9",
    #         architecture="Flow",
    #         flow="ArgmaxAdjV2",
    #         model=ArgmaxFlow,
    #         t=6,
    #         inverted_mask=False,
    #         upload=False,
    #         upload_interval=2,
    #         hidden_dim=64,
    #         mask_ratio=2.
    #     )


    # model, _, _ = create_model_and_optimiser_sche(config=config)
    # model.eval()


    # z, _ = model(batch.adj[:1])
    # x, _ =model.inverse(z)


    # base = torch.distributions.Normal(loc=0., scale=1.)
    # print(torch.allclose(x, batch.adj[:1]))

    # z = base.sample((1, 1, 45, 4)).long()

    # print(torch.sum(base.log_prob(z), dim=[1, 2, 3]))

    # generated_x, _ = model.inverse(z)

    # print(generated_x)
    # x = torch.zeros(1, 45).long()
    # x[0, 5:10] = 1

    # model = ArgmaxFlow(
    #     context_size=8,
    #     num_classes=4,
    #     embedding_dim=7,
    #     hidden_dim=128,
    #     max_nodes=9,
    #     t=6
    # )

    # z, _ = model(x)
    # print(z.shape)

    # print(sum([p.numel() for p in model.parameters()]))

    # context_size = 1
    # c = ContextNet(context_size=context_size, num_classes=4, embedding_dim=7)

    # # context: B x context_size x 45 x embedding_dim

    # encoder_base = ConditionalNormal(nn.Sequential(
    #     nn.Linear(7, 4),
    #     nn.ReLU(),
    #     nn.Conv2d(context_size, 2, kernel_size=3, stride=1, padding=1)
    # ), split_dim=1)
    
    # t = ConditionalAdjacencyBlockFlow()

    # c = ConditionalInverseFlow(base_dist=encoder_base, context_init=c, transforms=[t])

    # surjection = ArgmaxSurjection(c, 4)

    # z, _ = c.sample_with_log_prob(x)
    # z, _ = surjection(x)
    # out, _ = surjection.inverse(z)
    # print(z.shape)

    # print(out, x)