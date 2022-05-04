import torch

def create_mask(range, max_nodes, invert=False):

    r = torch.arange(9, -1, step=-1)

    mask = torch.zeros(r.sum(), dtype=torch.float32)

    mask[r[0:range[0]].sum(): r[0: range[1]].sum()] = 1.
    mask = mask.view(1, 1, r.sum(), 1)

    if invert:
        mask = 1 - mask

    return mask