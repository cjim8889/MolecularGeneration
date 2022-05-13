from .datasets import get_datasets
from models.atomflow import AtomFlow
from models.atomflow.graphflow import AtomGraphFlow
from models.graphflowv2 import AtomGraphFlowV2
from models.graphflowv3 import AtomGraphFlowV3
from .utils import create_model_and_optimiser_sche, argmax_criterion
import torch.nn as nn
import wandb
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class AtomExp:
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config

        if self.config['flow'] == "AtomGraph":
            self.config['model'] = AtomGraphFlow
        elif self.config['flow'] == "AtomGraphV2":
            self.config['model'] = AtomGraphFlowV2
        elif self.config['flow'] == "AtomGraphV3":
            self.config['model'] = AtomGraphFlowV3
        else:
            self.config['flow'] = "AtomFlow" 
            self.config['model'] = AtomFlow
        

        if "hidden_dim" not in self.config:
            self.config['hidden_dim'] = 128

        self.batch_size = self.config["batch_size"] if "batch_size" in self.config else 128

        self.train_loader, self.test_loadesr = get_datasets(type="mqm9", batch_size=self.batch_size)
        self.network, self.optimiser, self.scheduler = create_model_and_optimiser_sche(self.config)
        self.base = torch.distributions.Normal(loc=0., scale=1.)

    def train(self):
        if self.config['flow'] == "AtomGraph" or self.config['flow'] == "AtomGraphV3":
            self.network(torch.zeros(1, 9, 2, device=device).long(), {
                "adj": torch.zeros(1, 45, device=device).long(),
                "b_adj": torch.zeros(1, 9, 9, device=device).long()
            })
        elif self.config['flow'] == "AtomGraphV2":
            self.network(torch.zeros(1, 9, 2, device=device).long(), {
                "adj": torch.zeros(1, 9, 9, 5, device=device).float(),
                "b_adj": torch.zeros(1, 9, 9, device=device).long()
            })
        else:
            self.network(torch.zeros(1, 9, 2, device=device).long(), torch.zeros(1, 45, device=device).long())
        print(f"Model Parameters: {sum([p.numel() for p in self.network.parameters()])}")
        # print(self.network)

        with wandb.init(project="molecule-flow", config=self.config) as run:
            step = 0
            for epoch in range(self.config['epochs']):
                loss_step = 0
                loss_ep_train = 0
                self.network.train()

                for idx, batch_data in enumerate(self.train_loader):

                    b_adj = batch_data.b_adj
                    x = batch_data.x
                    
                    x = x.to(device)
                    b_adj = b_adj.to(device)

                    self.optimiser.zero_grad()

                    context = None

                    if self.config['flow'] == "AtomGraph" or self.config['flow'] == "AtomGraphV3":
                        adj = batch_data.adj.to(device)
                        context = {
                            "adj": adj,
                            "b_adj": b_adj
                        }
                    elif self.config['flow'] == "AtomGraphV2":
                        adj = batch_data.orig_adj.to(device)
                        context = {
                            "adj": adj.float(),
                            "b_adj": b_adj
                        }
                    else:
                        context = batch_data.adj.to(device)

                    z, log_det = self.network(x, context)
                    log_prob = torch.sum(self.base.log_prob(z), dim=[1, 2, 3])
                    loss = argmax_criterion(log_prob, log_det)
                    loss.backward()

                    # nn.utils.clip_gsrad_norm_(self.network.parameters(), 1)
                    self.optimiser.step()

                    loss_step += loss
                    loss_ep_train += loss

                    
                    step += 1
                    if idx % 5 == 0:
                        ll = (loss_step / 5.).item()
                        # print(f"Epoch: {epoch}, Step: {idx}, Loss: {ll}")
                        wandb.log({"epoch": epoch, "NLL": ll}, step=step)

                        
                        loss_step = 0

                self.scheduler.step()
                wandb.log("Learning Rate/Epoch", self.scheduler.get_last_lr()[0])
                wandb.log({"NLL/Epoch": (loss_ep_train / len(self.train_loader)).item()}, step=epoch)
                if self.config['upload']:
                    if epoch % self.config['upload_interval'] == 0:
                        torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.network.state_dict(),
                        'optimizer_state_dict': self.optimiser.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        }, f"model_checkpoint_{run.id}_{epoch}.pt")
                    
                        wandb.save(f"model_checkpoint_{run.id}_{epoch}.pt")

