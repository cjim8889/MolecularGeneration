import torch

def create_mask(range, max_nodes, invert=False):
    mask = torch.zeros(max_nodes, dtype=torch.float32)
    mask[range[0]:range[1]] = 1.
    mask = mask.view(1, 1, max_nodes, 1)

    if invert:
        mask = 1 - mask

    return mask