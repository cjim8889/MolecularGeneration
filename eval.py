import torch

from models.graphflowv3.argmax import ContextNet, ArgmaxSurjection
from models.graphflowv3 import AtomSurjection
from survae.distributions import ConditionalNormal
import torch.nn as nn

from utils import get_datasets

device = torch.device("cpu")

context_size=16
num_classes=5
embedding_dim=7
hidden_dim=64


if __name__ == "__main__":
    train_loader, test_loadesr = get_datasets(type="mqm9", batch_size=128)

    batch = next(iter(train_loader))
    print(batch.adj.shape)

    sur = AtomSurjection()

    z, ldj = sur(torch.ones(1, 9, 2).long())
    print(z.shape)