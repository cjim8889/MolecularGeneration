import torch.nn as nn
import torch
import argparse
from utils import ArgmaxAdjacencyExp


parser = argparse.ArgumentParser(description="Molecular Generation MSc Project")

parser.add_argument("--type", help="Type of experiments e.g. argmaxadj")
parser.add_argument("--epochs", help="Number of epochs", type=int, default=100)
parser.add_argument("--batch_size", help="Batch size", type=int, default=128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, 0, 0.001)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.type == "argmaxadj":

        config = dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimiser="Adam",
            learning_rate=1e-03,
            scheduler="StepLR",
            scheduler_gamma=0.99,
            scheduler_step=1,
            bpd=False,
            dataset="QM9",
            architecture="Flow",
            weight_init=weight_init
        )

        exp = ArgmaxAdjacencyExp(config)


    exp.train()




