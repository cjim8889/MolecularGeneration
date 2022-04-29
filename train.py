import math
import torch.nn as nn
import torch
import numpy as np
from tqdm.notebook import trange, tqdm
from models.flows import AdjacencyFlows
from utils.datasets import get_datasets
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, 0, 0.001)

def create_model_and_optimiser_sche(config):

    model = config['model'](t=config['t'], affine=config['affine'])
    model = model.to(device)
    
    if "weight_init" in config:
        model.apply(config["weight_init"])

    optimiser = None
    if config['optimiser'] == "Adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    scheduler = None
    if "scheduler" in config:
        if config["scheduler"] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=config['scheduler_step'], gamma=config["scheduler_gamma"])

    return model, optimiser, scheduler

@torch.jit.script
def criterion(log_prob, log_det):
    return - torch.mean(log_prob + log_det)

if __name__ == "__main__":
    config = dict(
        epochs=250,
        t=24,
        flow="MaskedCouplingFlow",
        batch_size=256,
        optimiser="Adam",
        learning_rate=1e-03,
        affine=False,
        dequantization=True,
        scheduler="StepLR",
        scheduler_gamma=0.99,
        scheduler_step=1,
        bpd=False,
        dataset="QM9",
        architecture="Flow",
        model=AdjacencyFlows,
        weight_init=weight_init
    )

    train_loader, test_loader = get_datasets()

    network, optimiser, scheduler = create_model_and_optimiser_sche(config)
    base = torch.distributions.Normal(loc=0., scale=1.)

    network(torch.zeros(1, 29, 29, 4, device=device))
    print(f"Model Parameters: {sum([p.numel() for p in network.parameters()])}")
    
    with wandb.init(project="molecule-flow", config=config):
        step = 0
        for epoch in range(config['epochs']):
            loss_step = 0
            loss_ep_train = 0
            network.train()

            with tqdm(train_loader, unit="batch") as tepoch: 
                for idx, batch_data in enumerate(tepoch):

                    adj = batch_data.adj

                    adj_t = adj.view(adj.shape[0] * 29 * 29, -1)
                    adj_t[torch.logical_not(adj_t.bool()).all(dim=1)] = torch.tensor([0, 0, 0, 1], dtype=torch.float32)

                    input = adj_t.view(*adj.shape)
                    input = input.to(device)

                    optimiser.zero_grad(set_to_none=True)

                    z, log_det = network(input)
                    log_prob = torch.sum(base.log_prob(z), dim=[1, 2, 3, 4])

                    loss = criterion(log_prob, log_det)
                    loss.backward()

                    nn.utils.clip_grad_norm_(network.parameters(), 1)
                    optimiser.step()

                    loss_step += loss
                    loss_ep_train += loss


                    
                    step += 1

                    if idx % 5 == 0:
                        ll = (loss_step / 5.).item()
                        wandb.log({"epoch": epoch, "NLL": ll}, step=step)

                        tepoch.set_description(f"Epoch {epoch}")
                        tepoch.set_postfix(Loss=ll)
                        
                        loss_step = 0
                        
            scheduler.step()
            if epoch % 3 == 0:
                torch.save({
                'epoch': epoch,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, f"model_checkpoint_{epoch}.pt")
            
                wandb.save(f"model_checkpoint_{epoch}.pt")

