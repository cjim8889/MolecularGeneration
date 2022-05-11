import torch.nn as nn
import torch
import argparse

from utils import ArgmaxAdjacencyExp, ArgmaxAdjacencyV2Exp, AtomExp


parser = argparse.ArgumentParser(description="Molecular Generation MSc Project")

parser.add_argument("--type", help="Type of experiments e.g. argmaxadj")
parser.add_argument("--epochs", help="Number of epochs", type=int, default=100)
parser.add_argument("--batch_size", help="Batch size", type=int, default=128)
parser.add_argument("--block_length", help="Block length t parameter for V2 experiments", type=int, default=12)

parser.add_argument("--optimiser", help="Optimiser", type=str, default="Adam")
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-03)
parser.add_argument("--weight_decay", help="Weight decay", type=float, default=0.0)
parser.add_argument("--momentum", help="Momentum for the SGD optimiser", type=float, default=0.9)

parser.add_argument("--invert_mask", help="Invert masking of atom", type=bool, default=False)
parser.add_argument("--hidden_dim", help="Hidden dimension", type=int, default=64)
parser.add_argument("--mask_ratio", help="Mask ratio", type=float, default=2.)

parser.add_argument("--scheduler", help="Scheduler", type=str, default="StepLR")
parser.add_argument("--scheduler_step", help="Scheduler step", type=int, default=3)
parser.add_argument("--scheduler_gamma", help="Scheduler gamma", type=float, default=0.96)

parser.add_argument("--upload", help="Upload to wandb", type=bool, default=False)
parser.add_argument("--upload_interval", help="Upload to wandb every n epochs", type=int, default=10)

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
            momentum=args.momentum,
            scheduler=args.scheduler,
            scheduler_gamma=args.scheduler_gamma,
            scheduler_step=args.scheduler_step,
            bpd=False,
            dataset="QM9",
            architecture="Flow",
            weight_init=weight_init,
            upload=args.upload,
            upload_interval=args.upload_interval,
        )

        exp = ArgmaxAdjacencyExp(config)

    elif args.type == "argmaxadjv2":

        config = dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimiser=args.optimiser,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            momentum=args.momentum,
            scheduler_gamma=args.scheduler_gamma,
            scheduler_step=args.scheduler_step,
            hidden_dim=args.hidden_dim,
            bpd=False,
            dataset="MQM9",
            architecture="Flow",
            weight_init=weight_init,
            t=args.block_length,
            inverted_mask=args.invert_mask,
            upload=args.upload,
            upload_interval=args.upload_interval,
            mask_ratio=args.mask_ratio,
        )

        exp = ArgmaxAdjacencyV2Exp(config)

    elif args.type == "atom":

        config = dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimiser=args.optimiser,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            momentum=args.momentum,
            scheduler_gamma=args.scheduler_gamma,
            scheduler_step=args.scheduler_step,
            hidden_dim=args.hidden_dim,
            bpd=False,
            dataset="MQM9",
            architecture="Flow",
            weight_init=weight_init,
            t=args.block_length,
            inverted_mask=args.invert_mask,
            upload=args.upload,
            upload_interval=args.upload_interval,
            mask_ratio=args.mask_ratio,
        )

        exp = AtomExp(config)

    elif args.type == "atomgraph":

        config = dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimiser=args.optimiser,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            momentum=args.momentum,
            scheduler_gamma=args.scheduler_gamma,
            scheduler_step=args.scheduler_step,
            hidden_dim=args.hidden_dim,
            bpd=False,
            dataset="MQM9",
            flow="AtomGraph",
            architecture="Flow",
            weight_init=weight_init,
            t=args.block_length,
            inverted_mask=args.invert_mask,
            upload=args.upload,
            upload_interval=args.upload_interval,
            mask_ratio=args.mask_ratio,
        )

        exp = AtomExp(config)

    elif args.type == "atomgraphv2":

        config = dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimiser=args.optimiser,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            momentum=args.momentum,
            scheduler_gamma=args.scheduler_gamma,
            scheduler_step=args.scheduler_step,
            hidden_dim=args.hidden_dim,
            bpd=False,
            dataset="MQM9",
            flow="AtomGraphV2",
            architecture="Flow",
            weight_init=weight_init,
            t=args.block_length,
            inverted_mask=args.invert_mask,
            upload=args.upload,
            upload_interval=args.upload_interval,
            mask_ratio=args.mask_ratio,
        )

        exp = AtomExp(config)

    elif args.type == "atomgraphv3":

        config = dict(
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimiser=args.optimiser,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            scheduler=args.scheduler,
            momentum=args.momentum,
            scheduler_gamma=args.scheduler_gamma,
            scheduler_step=args.scheduler_step,
            hidden_dim=args.hidden_dim,
            bpd=False,
            dataset="MQM9",
            flow="AtomGraphV3",
            architecture="Flow",
            weight_init=weight_init,
            t=args.block_length,
            inverted_mask=args.invert_mask,
            upload=args.upload,
            upload_interval=args.upload_interval,
            mask_ratio=args.mask_ratio,
        )

        exp = AtomExp(config)
        
    exp.train()





