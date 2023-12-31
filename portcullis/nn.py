"""
Neural Network Base Class
pyTorch Implementation
"""
import torch
import torch.nn as nn
import os
from portcullis.pytorch import DEVICE


class NN(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    # Save model state
    def save(self, path: str = None, verbose: bool = True):
        path = path or f'./models/{self.name}.torch'
        if verbose:
            print('Saving model to', path)
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        torch.save(self.state_dict(), path)

    # Load model state
    def load(self, path: str = None, verbose: bool = True):
        path = path or f'./models/{self.name}.torch'
        if os.path.exists(path):
            if verbose:
                print('Loading model from', path)
            self.load_state_dict(torch.load(path, map_location=DEVICE))
            self.eval()

    @staticmethod
    def polyak_copy(src_nn: nn.Module, dst_nn: nn.Module, tau: float) -> None:
        """Polyak soft update of the target network's weights.
        θ′ ← τ θ + (1 −τ )θ′
        """
        for src_p, dst_p in zip(src_nn.parameters(), dst_nn.parameters()):
            dst_p.data.copy_(tau * src_p.data + (1 - tau) * dst_p.data)
    # Alias
    soft_update = polyak_copy

    @staticmethod
    def copy_states(src_nn: nn.Module, dst_nn: nn.Module) -> None:
        """Synchronize states across networks.
        """
        dst_nn.load_state_dict(src_nn.state_dict())
    # Alias
    hard_update = copy_states
