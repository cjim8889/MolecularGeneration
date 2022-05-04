import torch
from models.argmaxflowv2.model import ContextNet
from models.argmaxflowv2.conditional import ConditionalAdjacencyBlockFlow
from survae.distributions import ConditionalNormal
from survae.flows import ConditionalInverseFlow
from torch import embedding, nn
from models.argmaxflowv2 import create_mask
from models.argmaxflowv2 import ArgmaxSurjection, ArgmaxFlow

if __name__ == '__main__':
    
    x = torch.zeros(1, 45).long()
    x[0, 5:10] = 1

    model = ArgmaxFlow(
        context_size=8,
        num_classes=4,
        embedding_dim=7,
        hidden_dim=128,
        max_nodes=9,
        t=6
    )

    z, _ = model(x)
    print(z.shape)

    print(sum([p.numel() for p in model.parameters()]))

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