import torch

def setwise_distance(a, b=None):
    if b is None:
        b = a
    return torch.pow((a.unsqueeze(dim=1) - b.unsqueeze(dim=0)), 2.0).sum(dim=-1)