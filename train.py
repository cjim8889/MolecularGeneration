import torch.nn as nn
import torch
import argparse
from utils import ArgmaxAdjacencyExp, ArgmaxAdjacencyV2Exp


parser = argparse.ArgumentParser(description="Molecular Generation MSc Project")

parser.add_argument("--type", help="Type of experiments e.g. argmaxadj")
parser.add_argument("--epochs", help="Number of epochs", type=int, default=100)
parser.add_argument("--batch_size", help="Batch size", type=int, default=128)
parser.add_argument("--block_length", help="Block length t parameter for V2 experiments", type=int, default=12)

parser.add_argument("--optimiser", help="Optimiser", type=str, default="Adam")
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-03)
parser.add_argument("--weight_decay", help="Weight decay", type=float, default=0.0)


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
            optimiser=args.optimiser,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            scheduler="StepLR",
            scheduler_gamma=0.99,
            scheduler_step=1,
            bpd=False,
            dataset="QM9",
            architecture="Flow",
            weight_init=weight_init
        )

        exp = ArgmaxAdjacencyExp(config)

    elif args.type == "argmaxadjv2":

        config = dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimiser=args.optimiser,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            scheduler="StepLR",
            scheduler_gamma=0.98,
            scheduler_step=1,
            bpd=False,
            dataset="MQM9",
            architecture="Flow",
            weight_init=weight_init,
            t=args.block_length
        )

        exp = ArgmaxAdjacencyV2Exp(config)


    exp.train()




