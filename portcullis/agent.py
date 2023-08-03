import numpy as np
import torch
from portcullis.nn import NN, DQNN
from portcullis.env import Env, Action, State
from portcullis.mem import Mem, Frag
from portcullis.pytorch import DEVICE
import os


class Agent():
    def __init__(self, env: Env, mem: Mem, hdims: tuple[int, int], lr: float,
                 gamma: float, eps: float, eps_min: float, eps_decay: float,
                 tau: float, training: bool, name: str,
                 ):
        """DRL Agent Base Class.
        """
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
        self.name = name
        self.epochs = 0
        self.scores = []
        self.high_score = 0.0
        self.gym_env = (type(self.env.action_space).__name__ == 'Discrete')
        if self.gym_env:
            self.num_actions = self.env.action_space.n
            self.num_features = self.env.observation_space.shape[0]
        else:
            self.num_actions = self.env.num_actions()
            self.num_features = self.env.num_features()

    # Save agent state
    def save(self, checkpoint: dict, path: str, verbose: bool, seed: int) -> None:
        path = path or f'./models/{self.name}.torch'
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        checkpoint.update({
            'seed': seed,
            'hdims': self.hdims,
            'lr': self.lr,
            'gamma': self.gamma,
            'eps': self.eps,
            'eps_min': self.eps_min,
            'eps_decay': self.eps_decay,
            'tau': self.tau,
            'name': self.name,
            'epochs': self.epochs,
            'scores': self.scores,
            'high_score': self.high_score,
        })
        if verbose:
            print(f'Saving agent {path}: Epochs={self.epochs}, Score={self.high_score}')
        torch.save(checkpoint, path)

    # Load agent state
    def load(self, path: str, verbose: bool, training: bool) -> dict:
        path = path or f'./models/{self.name}.torch'
        if not os.path.exists(path):
            return None
        checkpoint = torch.load(path, map_location=DEVICE)
        self.training = training
        seed = checkpoint['seed']
        self.hdims = checkpoint['hdims']
        self.lr = checkpoint['lr']
        self.gamma = checkpoint['gamma']
        self.eps = checkpoint['eps']
        self.eps_min = checkpoint['eps_min']
        self.eps_decay = checkpoint['eps_decay']
        self.tau = checkpoint['tau']
        self.name = checkpoint['name']
        self.epochs = checkpoint['epochs']
        self.scores = checkpoint['scores']
        self.high_score = checkpoint['high_score']
        self.env.reset(seed=seed)
        self.mem.clear()
        if verbose:
            print(f'Loaded agent {path}: Epochs={self.epochs}, Score={self.high_score}')
        return checkpoint

    def is_highscore(self, score: float) -> bool:
        """Records how this agent performs. Returns True if score is a new high score.
        """
        self.scores.append(score)
        if score > self.high_score:
            self.high_score = score
            return True
        return False

    def adj_eps(self) -> float:
        """Adjust exploration rate (epsilon-greedy).
        """
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return self.eps


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
        super().__init__(env, mem, hdims, lr, gamma, eps,
                         eps_min, eps_decay, tau, training, name)
        self.nn_p = DQNN(self.num_features, self.hdims, self.num_actions,
                         self.lr, name='DQNN_Policy')
        self.nn_t = DQNN(self.num_features, self.hdims, self.num_actions,
                         self.lr, name='DQNN_Target')
        self.nn_p.optimizer = torch.optim.AdamW(
            self.nn_p.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = torch.nn.SmoothL1Loss()
        self._configure_nn()

    def _configure_nn(self):
        # Always disable training mode for target network
        self.nn_t.eval()
        # For policy network check the training flag
        if self.training:
            print('Training policy network')
            self.nn_p.train()
        else:
            print('Evaluating policy network')
            self.nn_p.eval()

    def learn(self, mem_samples: int, soft_update: bool = True) -> None:
        # Book keeping
        self.epochs += 1
        assert self.training

        # Sample memory
        if len(self.mem) < mem_samples:
            return
        frags = self.mem.sample(mem_samples)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
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

        # Optimize the model
        self.nn_p.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.nn_p.parameters(), 100)
        self.nn_p.optimizer.step()

        # Target network soft update (blending of states)
        if soft_update:
            NN.soft_update(self.nn_p, self.nn_t, self.tau)
        else:  # hard update (sync all states)
            NN.sync_states(self.nn_p, self.nn_t)

    def act(self, state) -> Action:
        """Returns action for given state as per current policy.
        """
        if np.random.rand() <= self.eps and self.training:  # amount of exploration reduces with the epsilon value
            # Return random action from action space
            return self.env.action_space.sample() if self.gym_env else self.env.random_action()

        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32,
                                 device=DEVICE).unsqueeze(0)

        # pick the action with maximum Q-value as per the policy Q-network
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = self.nn_p(state).max(1)[1].view(1, 1)
            return action.item()

    def remember(self, state: State, action: Action, reward: float, next_state: State, done: bool) -> None:
        """Remember a transition fragment and store in replay memory.
        """
        assert self.training
        state = torch.tensor(state, dtype=torch.float32,
                             device=DEVICE).unsqueeze(0)
        action = torch.tensor([[action]], dtype=torch.long, device=DEVICE)
        reward = torch.tensor([reward], dtype=torch.float32, device=DEVICE)
        next_state = torch.tensor(
            next_state, dtype=torch.float32, device=DEVICE).unsqueeze(0) if not done else None
        self.mem.push(state, action, reward, next_state)

    def load(self, path: str = None, verbose: bool = True, training: bool = True) -> None:
        """Load agent state from persistent storage.
        """
        checkpoint = super().load(path, verbose=verbose, training=training)
        if checkpoint:
            self.nn_p.load_state_dict(checkpoint['nn_p'])
            self.nn_p.optimizer.load_state_dict(checkpoint['opt_p'])
            self.nn_t.load_state_dict(checkpoint['nn_p'])
            self.nn_t.optimizer.load_state_dict(checkpoint['opt_t'])
            self.criterion = checkpoint['criterion']        
            self._configure_nn()

    def save(self, path: str = None, verbose: bool = True, seed: int = None) -> None:
        """Save agent state to persistent storage.
        """
        checkpoint = {
            'nn_p': self.nn_p.state_dict(),
            'opt_p': self.nn_p.optimizer.state_dict(),
            'nn_t': self.nn_t.state_dict(),
            'opt_t': self.nn_t.optimizer.state_dict(),
            'criterion': self.criterion,
        }
        super().save(checkpoint, path, verbose=verbose, seed=seed)
