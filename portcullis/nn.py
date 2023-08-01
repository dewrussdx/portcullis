"""
Neural Network Base Class
pyTorch Implementation
"""
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.optim as optim

DEVICE = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')


class NN(nn.Module):
    DEFAULT_PATH = './models/model.pytorch'

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    # Save model state
    def save(self, path: str = None, verbose: bool = True):
        path = path or NN.DEFAULT_PATH
        if verbose:
            print('Saving model to', path)
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        torch.save(self.state_dict(), path)

    # Load model state
    def load(self, path: str = None, verbose: bool = True):
        path = path or NN.DEFAULT_PATH
        if os.path.exists(path):
            if verbose:
                print('Loading model from', path)
            self.load_state_dict(torch.load(path, map_location=DEVICE))
            self.eval()

    @staticmethod
    def soft_update(src_nn: nn.Module, dst_nn: nn.Module, tau: float) -> None:
        """Soft update of the target network's weights.
        θ′ ← τ θ + (1 −τ )θ′
        """
        src_d = src_nn.state_dict()
        dst_d = dst_nn.state_dict()
        for key in src_d:
            dst_d[key] = src_d[key]*tau + dst_d[key]*(1-tau)
        dst_nn.load_state_dict(dst_d)

    @staticmethod
    def sync_states(src_nn: nn.Module, dst_nn: nn.Module) -> None:
        """Synchronize states across networks.
        """
        dst_nn.load_state_dict(src_nn.state_dict())


"""
Deep Q/Quality Neural Network
pyTorch Implementation
"""


class DQNN(NN):
    # Initialize NN with input, hidden and output layers
    def __init__(self, input_size: int, hdims: (int, int), output_size: int,
                 lr: float = 1e-4, seed: int = None, name: str = None) -> None:
        super().__init__(name)
        self.input_size = input_size
        self.hdims = hdims
        self.output_size = output_size
        self.lr = lr
        self.seed = seed
        self.name = name
        self.seed = torch.manual_seed(seed) if seed else None
        self.fc1 = nn.Linear(self.input_size, self.hdims[0])
        self.fc2 = nn.Linear(self.hdims[0], self.hdims[1])
        self.fc3 = nn.Linear(self.hdims[1], self.output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.to(DEVICE)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
