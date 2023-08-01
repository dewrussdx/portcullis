import numpy as np
import torch
from portcullis.nn import NN, DQNN, DEVICE
from portcullis.env import Env, Action, State
from portcullis.mem import Mem
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
            return torch.load(path, map_location=DEVICE)
        return None


class DQNNAgent(Agent):
    """Deep Quality Neural Network. This implementation uses separate value and target 
    networks for stability.
    """

    def __init__(self,
                 env: Env,
                 mem: Mem = Mem(65336),
                 hdims: (int, int) = (256, 128),
                 lr: float = 0.001,
                 gamma: float = 0.99,
                 eps: float = 1.0,
                 eps_min: float = 0.01,
                 eps_decay: float = 0.99,
                 tau: float = 0.005,
                 training: bool = True,
                 name: str = 'DQNNAgent'
                 ):
        super().__init__(name)
        self.env = env
        self.mem = mem
        self.hdims = hdims
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.tau = tau
        self.training = training
        self.criterion = torch.nn.MSELoss()
        self.nn_p = DQNN(self.env.num_features(), self.hdims,
                         self.env.num_actions(), self.lr, name='DQNN_Policy')
        self.nn_t = DQNN(self.env.num_features(), self.hdims,
                         self.env.num_actions(), self.lr, name='DQNN_Policy')
        self._init_nn()

    def _init_nn(self):
        # Always disable learning mode for target network
        self.nn_t.eval()
        # For policy network check the learn_mode flag
        if self.training:
            self.nn_p.train()
        else:
            self.nn_p.eval()
        NN.sync_states(self.nn_p, self.nn_t)

    def adj_eps(self) -> float:
        """Adjust exploration rate (epsilon-greedy).
        """
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return self.eps

    def learn(self, mem_samples: int) -> None:
        # Sample memory
        if len(self.mem) < mem_samples:
            return
        # Convert from AoS to SoA and move to device memory
        # TODO: Optimize
        frags = self.mem.sample(mem_samples)
        states = torch.from_numpy(
            np.vstack([e.state for e in frags])).float().to(DEVICE)
        actions = torch.from_numpy(
            np.vstack([e.action for e in frags])).long().to(DEVICE)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in frags])).float().to(DEVICE)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in frags])).float().to(DEVICE)
        dones = torch.from_numpy(
            np.vstack([e.done for e in frags])).long().to(DEVICE)

        # get q values of the actions that were taken, i.e calculate qpred;
        # actions vector has to be explicitly reshaped to nx1-vector
        q_pred = self.nn_p.forward(states).gather(1, actions.view(-1, 1))

        # calculate target q-values, such that yj = rj + q(s', a'), but if current state is a terminal state, then yj = rj
        # because max returns data structure with values and indices
        q_target = self.nn_t.forward(next_states).max(
            dim=1).values.unsqueeze(1)

        # setting Q(s',a') to 0 when the current state is a terminal state
        q_target[dones] = 0.0
        y_j = rewards + (self.gamma * q_target)

        # calculate the loss as the mean-squared error of yj and qpred
        self.nn_p.optimizer.zero_grad()
        loss = self.criterion(y_j, q_pred)
        loss.backward()
        self.nn_p.optimizer.step()

        # Target network soft update
        NN.soft_update(self.nn_p, self.nn_t, self.tau)

    def act(self, state) -> Action:
        """Returns action for given state as per current policy.
        """
        if np.random.rand() <= self.eps:  # amount of exploration reduces with the epsilon value
            return self.env.random_action()

        if not torch.is_tensor(state):
            state = torch.tensor(
                np.array(state), dtype=torch.float32, device=DEVICE)

        # pick the action with maximum Q-value as per the policy Q-network
        with torch.no_grad():
            action = self.nn_p.forward(state)
            # since actions are discrete, return index that has highest Q
            return torch.argmax(action).item()

    def remember(self, state: State, action: Action, reward: float, next_state: State, done: bool) -> None:
        # Convert to torch tensors
        # state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        # action = torch.tensor(action, dtype=torch.long, device=DEVICE)
        # next_state = torch.tensor(
        #    next_state, dtype=torch.float32, device=DEVICE)
        # reward = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
        # done = torch.tensor(done, dtype=torch.bool, device=DEVICE)
        self.mem.push(state, action, reward, next_state, int(done))

    def load(self, path: str = None, verbose: bool = True) -> None:
        checkpoint = super()._load(path, verbose=verbose)
        if checkpoint is None:
            return
        self.nn_p.load_state_dict(checkpoint['nn_p'])
        self.gamma = checkpoint['gamma']
        self.eps = checkpoint['eps']
        self.eps_min = checkpoint['eps_min']
        self.eps_decay = checkpoint['eps_decay']
        self.tau = checkpoint['tau']
        self.training = checkpoint['training']
        self.mem.clear()
        self._init_nn()

    def save(self, path: str = None, verbose: bool = True) -> None:
        checkpoint = {
            'nn_p': self.nn_p.state_dict(),
            'gamma': self.gamma,
            'eps': self.eps,
            'eps_min': self.eps_min,
            'eps_decay': self.eps_decay,
            'tau': self.tau,
            'training': self.training,
        }
        super()._save(checkpoint, path, verbose=verbose)
