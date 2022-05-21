import torch

from torch import nn

from models.graphflowv4.argmax import ConditionalNormal, ArgmaxSurjection, ConditionalAdjacencyBlockFlow, ar_net_init
from survae.flows import ConditionalInverseFlow
from survae.distributions import ConditionalNormal
from models.graphflowv4.surjectives import ContextNet


if __name__ == '__main__':
    context_size =16
    num_classes = 6
    embedding_dim = 7
    hidden_dim = 32


    context_net = ContextNet(context_size=context_size, num_classes=num_classes, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    encoder_base = ConditionalNormal(nn.Sequential(
            nn.LazyLinear(num_classes),
            nn.ReLU(),
            nn.LazyConv2d(2, kernel_size=1, stride=1),
            nn.ReLU(),
        ), split_dim=1)
    conditional_flow = ConditionalInverseFlow(base_dist=encoder_base, context_init=context_net, transforms=[])


    z, _ = conditional_flow.sample_with_log_prob(context=torch.randint(0, num_classes, (1, 9)))
    # z = context_net(torch.randint(0, num_classes, (1, 9)))
    print(z.shape)