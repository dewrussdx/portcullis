"""
Neural Network Base Class
pyTorch Implementation
"""
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.optim as optim
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
                 lr: float = 1e-4, name: str = None) -> None:
        super().__init__(name)
        self.input_size = input_size
        self.hdims = hdims
        self.output_size = output_size
        self.lr = lr
        self.name = name
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


class ActorNN(NN):
    def __init__(self, state_dim: int, action_dim: int, max_action: int, name: str=None):
        super().__init__(name)

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class CriticNN(NN):
    def __init__(self, state_dim: int, action_dim: int, name: str=None):
        super().__init__(name)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1