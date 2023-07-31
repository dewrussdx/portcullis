import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from portcullis.nn import NN, DQNN, DEVICE
from portcullis.env import Env, Action, State
from portcullis.mem import Mem, Frag
from collections import namedtuple
import os


class Agent():
    DEFAULT_PATH = './agents/agent.pytorch'

    def __init__(self, name):
        self.name = name

    # Save agent state
    def _save(self, data: any, path: str = None, verbose: bool = True):
        path = path or Agent.DEFAULT_PATH
        if verbose:
            print('Saving agent to', path, '...')
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        torch.save(data, path)

    # Load agent state
    def _load(self, path: str = None, verbose: bool = True):
        path = path or Agent.DEFAULT_PATH
        if os.path.exists(path):
            if verbose:
                print('Loading agent from', path, '...')
            return torch.load(path)
        return None


class DQNNAgent(Agent):
    """Deep Quality Neural Network. This implementation uses separate value and target NN for stability.
    """

    def __init__(self,
                 env: Env,
                 nn_p: DQNN,
                 nn_t: DQNN,
                 mem: Mem,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 eps: float = 1.0,
                 eps_min: float = 5e-2,
                 eps_decay: float = 0.999,
                 tau: float = 5e-3
                 ):
        super().__init__('DQNNAgent')
        self.env = env
        self.nn_p = nn_p  # Policy NN
        self.nn_t = nn_t  # Target NN
        self.mem = mem
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.tau = tau
        self.nn_t.load_state_dict(self.nn_p.state_dict())  # Sync networks
        self.optimizer = optim.AdamW(
            self.nn_p.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()

    def adj_eps(self) -> float:
        """Adjust exploration rate (epsilon-greedy).
        """
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return self.eps

    def learn(self, mem_samples: int) -> None:
        # Sample memory
        if len(self.mem) < mem_samples:
            return
        frags = self.mem.sample(mem_samples)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Frag
        # to Frag of batch-arrays (AoS => SoA.)
        batch = Frag(*zip(*frags))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.nn_p(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(mem_samples, device=DEVICE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.nn_t(
                non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = self.criterion(state_action_values,
                              expected_state_action_values.unsqueeze(1))

        # Optimize the model and apply in-place gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.nn_p.parameters(), 100)
        self.optimizer.step()
        # Target network soft update
        NN.soft_update(self.nn_p, self.nn_t, self.tau)

    def act(self, state) -> Action:
        """Returns action for given state as per current policy.
        """
        if np.random.rand() > self.eps:
            state = torch.tensor(state, dtype=torch.float32,
                                 device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.nn_p(state).max(1)[1].view(1, 1)
        # Explore environment with a random action
        return torch.tensor([[self.env.random_action()]], device=DEVICE, dtype=torch.long)

    def remember(self, state: State, action: Action, reward: float, next_state: State, done: bool) -> None:

        # Convert to torch tensors
        # Note: Action 'action' already converted through act()
        state = torch.tensor(state, dtype=torch.float32,
                             device=DEVICE).unsqueeze(0)
        reward = torch.tensor([reward], device=DEVICE)
        next_state = torch.tensor(
            next_state, dtype=torch.float32, device=DEVICE).unsqueeze(0) if not done else None
        # Memorize
        self.mem.push(state, action, reward, next_state)

    def load(self, path: str = None, verbose: bool = True) -> None:
        checkpoint = super()._load(path, verbose=verbose)
        if checkpoint is None:
            return
        self.nn_p.load_state_dict(checkpoint['nn_p'])
        self.nn_t.load_state_dict(checkpoint['nn_t'])
        self.optimizer.load_state_dict(checkpoint['opt'])
        self.nn_p.eval()
        self.nn_t.eval()
        self.gamma = checkpoint['gamma']
        self.eps = checkpoint['eps']
        self.eps_min = checkpoint['eps_min']
        self.eps_decay = checkpoint['eps_decay']
        self.tau = checkpoint['tau']

    def save(self, path: str = None, verbose: bool = True) -> None:
        checkpoint = {
            'nn_p': self.nn_p.state_dict(),
            'nn_t': self.nn_t.state_dict(),
            'opt': self.optimizer.state_dict(),
            'gamma': self.gamma,
            'eps': self.eps,
            'eps_min': self.eps_min,
            'eps_decay': self.eps_decay,
            'tau': self.tau,
        }
        super()._save(checkpoint, path, verbose=verbose)
