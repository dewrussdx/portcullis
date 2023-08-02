import torch

DEVICE = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')


def manual_seed(seed: int):
    return torch.manual_seed(seed)
