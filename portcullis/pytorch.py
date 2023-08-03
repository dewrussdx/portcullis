import torch

# The torch device
DEVICE: torch.device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')  # get_device('auto')

print('PyTorch device is:', DEVICE)


def manual_seed(seed: int):
    return torch.manual_seed(seed)
