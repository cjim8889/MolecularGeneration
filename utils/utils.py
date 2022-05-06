from torch import nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.jit.script
def argmax_criterion(log_prob, log_det):
    return - torch.mean(log_prob + log_det)


def create_model_and_optimiser_sche(config):
    
    if config['flow'] == "ArgmaxAdj":
        model = config['model']()
        model = model.to(device)
    elif config['flow'] == "ArgmaxAdjV2":
        model = config['model'](t=config['t'], hidden_dim=config['hidden_dim'], inverted_mask=config['inverted_mask'])
        model = model.to(device)
    elif config['flow'] == "Coupling":
        model = config['model'](t=config['t'], affine=config['affine'])
        model = model.to(device)
    else:
        raise ValueError(f"Unknown flow type: {config['flow']}")
    
    if "weight_init" in config:
        model.apply(config["weight_init"])

    optimiser = None
    if config['optimiser'] == "Adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimiser"] == "AdamW":
        optimiser = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimiser"] == "SGD":
        optimiser = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])
    else:
        raise ValueError(f"Unknown optimiser: {config['optimiser']}")

    scheduler = None
    if "scheduler" in config:
        if config["scheduler"] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=config['scheduler_step'], gamma=config["scheduler_gamma"])


    if "load_from" in config:
        states = torch.load(config["load_from"], map_location=device)

        model.load_state_dict(states["model_state_dict"])
        optimiser.load_state_dict(states["optimiser_state_dict"])

        if "scheduler" in config:
            scheduler.load_state_dict(states["scheduler_state_dict"])

    return model, optimiser, scheduler
