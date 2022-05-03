from numpy import intp
from .datasets import get_datasets
from models.argmaxflow.model import ArgmaxFlow

import wandb
import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model_and_optimiser_sche(config):
    
    if config['flow'] == "ArgmaxAdj":
        model = ArgmaxFlow()
        model = model.to(device)
    else:
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
def argmax_criterion(log_prob, log_det):
    return - torch.mean(log_prob + log_det)

class ArgmaxAdjacencyExp:
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        self.config['flow'] = "ArgmaxAdj" 
        self.config['model'] = ArgmaxFlow

        self.batch_size = self.config["batch_size"] if "batch_size" in self.config else 128

        self.train_loader, self.test_loader = get_datasets(self.batch_size)
        self.network, self.optimiser, self.scheduler = create_model_and_optimiser_sche(self.config)
        self.base = torch.distributions.Normal(loc=0., scale=1.)

    def train(self):
        self.network(torch.zeros(1, 29, 29, device=device).long())
        print(f"Model Parameters: {sum([p.numel() for p in self.network.parameters()])}")

        with wandb.init(project="molecule-flow", config=self.config):
            step = 0
            for epoch in range(self.config['epochs']):
                loss_step = 0
                loss_ep_train = 0
                self.network.train()

                for idx, batch_data in enumerate(self.train_loader):

                    adj = batch_data.adj

                    adj_t = adj.view(adj.shape[0] * 29 * 29, -1)
                    adj_t[torch.logical_not(adj_t.bool()).all(dim=1)] = torch.tensor([0, 0, 0, 1], dtype=torch.float32)

                    input = adj_t.view(*adj.shape)
                    input = input.argmax(dim=-1)

                    input = input.to(device)

                    self.optimiser.zero_grad(set_to_none=True)

                    z, log_det = self.network(input)
                    log_prob = torch.sum(self.base.log_prob(z), dim=[1, 2, 3])

                    loss = argmax_criterion(log_prob, log_det)
                    loss.backward()

                    nn.utils.clip_grad_norm_(self.network.parameters(), 1)
                    self.optimiser.step()

                    loss_step += loss
                    loss_ep_train += loss

                    
                    step += 1
                    if idx % 5 == 0:
                        ll = (loss_step / 5.).item()
                        wandb.log({"epoch": epoch, "NLL": ll}, step=step)

                        
                        loss_step = 0

                self.scheduler.step()
                wandb.log({"NLL/Epoch": loss_ep_train / len(self.train_loader)}, step=epoch)
                if epoch % 3 == 0:
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.network.state_dict(),
                    'optimizer_state_dict': self.optimiser.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    }, f"model_checkpoint_{epoch}.pt")
                
                    wandb.save(f"model_checkpoint_{epoch}.pt")

