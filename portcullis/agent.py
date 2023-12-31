import os
import copy
import math
import numpy as np
import torch
from portcullis.nn import NN
from portcullis.env import Env, Action, State
from portcullis.replay import ReplayBuffer
from portcullis.pytorch import DEVICE


class Agent():
    def __init__(self, env: Env, mem: ReplayBuffer, hdims: tuple[int, int], lr: float,
                 gamma: float, eps: float, eps_min: float, eps_decay: float,
                 tau: float, name: str,
                 ):
        """Reinforcement Learning Agent Base Class.
        """
        self.env = env
        self.is_gym, self.env_type, self.action_dim, self.state_dim, self.action_max = Env.get_env_spec(
            self.env)
        self.mem = mem
        self.hdims = hdims
        self.lr = lr
        self.gamma = gamma
        # TODO: Some of these params only apply to specific agents
        self.eps = eps  # For reporting
        self.eps_init = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.eps_slope = (self.eps_min - self.eps_init) / self.eps_decay
        self.tau = tau
        self.name = name
        self.training = None
        self.epochs = 0
        self.scores = []
        self.high_score = -1e10
        self.state_shape = (-1, self.state_dim)

    # Save agent state
    def save(self, checkpoint: dict, path: str, verbose: bool, seed: int) -> None:
        """Save agent state. Base class implementation. 
        """
        path = path or f'./models/{self.name}.torch'
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        checkpoint.update({
            'name': self.name,
            'seed': seed,
            'hdims': self.hdims,
            'lr': self.lr,
            'gamma': self.gamma,
            'tau': self.tau,
            'epochs': self.epochs,
            'scores': self.scores,
            'high_score': self.high_score,
        })
        if verbose:
            print(
                f'Saving agent {path}: Epochs={self.epochs}, Score={self.high_score}')
        torch.save(checkpoint, path)

    # Load agent state
    def load(self, path: str, verbose: bool) -> dict:
        """Load agent state. Base class implementation. 
        """
        path = path or f'./models/{self.name}.torch'
        if not os.path.exists(path):
            return None
        checkpoint = torch.load(path, map_location=DEVICE)
        self.name = checkpoint['name']
        seed = checkpoint['seed']
        self.hdims = checkpoint['hdims']
        self.lr = checkpoint['lr']
        self.gamma = checkpoint['gamma']
        self.tau = checkpoint['tau']
        self.epochs = checkpoint['epochs']
        self.scores = checkpoint['scores']
        self.high_score = checkpoint['high_score']
        self.env.reset(seed=seed)
        if verbose:
            print(
                f'Loaded agent {path}: Epochs={self.epochs}, Score={self.high_score}')
        return checkpoint

    def is_highscore(self, score: float) -> bool:
        """Records how this agent performs. Returns True if score is a new high score.
        """
        self.scores.append(score)
        if score > self.high_score:
            self.high_score = score
            return True
        return False

    def dec_eps(self) -> float:
        """Decrement epsilon for e-greedy exploration rate limiting.
        """
        self.eps = self.eps_min + \
            (self.eps_init - self.eps_min) * \
            math.exp(-1. * self.epochs / self.eps_decay)
        return self.eps

    def set_mode(self, training: bool) -> bool:
        """Set training (or evaluation/inference) mode. Return True if mode has changed.
        """
        if training != self.training:
            self.training = training
            return True
        return False


class DQN(Agent):
    class Net(NN):
        """DQN Neural Network Implementation.
        """

        def __init__(self, state_dim: int, hdims: (int, int), action_dim: int,
                     lr: float = 1e-4, name: str = None) -> None:
            super().__init__(name)
            self.state_dim = state_dim
            self.hdims = hdims
            self.action_dim = action_dim
            self.lr = lr
            self.name = name
            self.fc1 = torch.nn.Linear(self.state_dim, self.hdims[0])
            self.fc2 = torch.nn.Linear(self.hdims[0], self.hdims[1])
            self.fc3 = torch.nn.Linear(self.hdims[1], self.action_dim)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            self.to(DEVICE)

        def forward(self, state):
            """Computes forward step of the NN, estimating f(x)
            """
            # Called with either one element to determine next action, or a batch
            # during optimization. Returns tensor([[left0exp,right0exp]...]).
            x = torch.nn.functional.relu(self.fc1(state))
            x = torch.nn.functional.relu(self.fc2(x))
            return self.fc3(x)

    """Deep Quality Neural Network (DQN). This implementation uses separate value and target 
    networks for stability.
    """

    def __init__(self,
                 env: Env,
                 mem: ReplayBuffer = None,
                 hdims: (int, int) = (512, 256),
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 eps: float = 1.0,
                 eps_min: float = 0.01,
                 eps_decay: float = 25e4,
                 alpha=0.6,
                 min_priority=1e-2,
                 tau: float = 0.005,
                 training: bool = True,
                 name: str = 'DQN',
                 ):
        super().__init__(env, mem, hdims, lr, gamma, eps,
                         eps_min, eps_decay, tau, name)
        self.alpha = alpha
        self.min_priority = min_priority
        self.nn_p = DQN.Net(self.state_dim, self.hdims, self.action_dim,
                            self.lr, name='DQN_Policy')
        self.nn_t = DQN.Net(self.state_dim, self.hdims, self.action_dim,
                            self.lr, name='DQN_Target')
        # self.nn_p.optimizer = torch.optim.AdamW(
        #    self.nn_p.parameters(), lr=self.lr, amsgrad=True)
        self.set_mode(training)

    def set_mode(self, training: bool) -> None:
        if super().set_mode(training):
            # Always disable training mode for target network
            self.nn_t.eval()
            # For policy network check the training flag
            if self.training:
                print('Training policy network')
                self.nn_p.train()
            else:
                print('Evaluating policy network')
                self.nn_p.eval()

    def remember(self, state: State, action: Action, next_state: State, reward: float, done: bool) -> None:
        """Remember this transition in memory.
        """
        self.mem.add(state=state, action=action,
                     next_state=next_state, reward=reward, done=done)

    def train(self, soft_update=True, hard_update=False):
        """ Train networks with a priority replay buffer.
        """
        self.epochs += 1

        # Sample replay buffer
        state, action, next_state, reward, done, ind, _ = self.mem.sample()

        # Compute the target Q value
        with torch.no_grad():
            next_action = self.nn_p(next_state).argmax(1, keepdim=True)
            target_Q = (
                reward + done * self.gamma *
                self.nn_t(next_state).gather(1, next_action).reshape(-1, 1)
            )

        # Get current Q estimate
        current_Q = self.nn_p(state).gather(1, action)

        td_loss = (current_Q - target_Q).abs()
        Q_loss = self.lap_huber(td_loss)

        # Optimize the Q network
        self.nn_p.optimizer.zero_grad()
        Q_loss.backward()
        self.nn_p.optimizer.step()

        # Update target network by polyak or full copy
        if soft_update:
            NN.soft_update(self.nn_p, self.nn_t, self.tau)
        if hard_update:
            NN.hard_update(self.nn_p, self.nn_t)

        priority = td_loss.clamp(min=self.min_priority).pow(
            self.alpha).cpu().data.numpy().flatten()
        self.mem.update_priority(ind, priority)

    def lap_huber(self, x: float) -> float:
        """Compute huber loss for loss-adjusted prioritized experience replay (LAP).
        """
        return torch.where(x < self.min_priority, 0.5 * x.pow(2), self.min_priority * x).mean()

    def act(self, state: State) -> Action:
        """Returns action for given state as per current policy.
        """
        if self.training:
            # Amount of exploration reduces with the epsilon value
            if np.random.rand() <= self.dec_eps():
                # Return random action from action space
                choice = self.env.action_space.sample() if self.is_gym else self.env.random_action()
                return Action(choice)

        # Pick the action with maximum Q-value as per the policy Q-network
        with torch.no_grad():
            state = torch.FloatTensor(np.array(state)).reshape(
                self.state_shape).to(DEVICE)
            return Action(self.nn_p(state).argmax(1))

    def load(self, path: str = None, verbose: bool = True) -> None:
        """Load agent state from persistent storage.
        """
        checkpoint = super().load(path, verbose=verbose)
        if checkpoint:
            self.nn_p.load_state_dict(checkpoint['nn_p'])
            self.nn_p.optimizer.load_state_dict(checkpoint['opt_p'])
            self.nn_t.load_state_dict(checkpoint['nn_p'])
            self.nn_t.optimizer.load_state_dict(checkpoint['opt_t'])
            self.eps = checkpoint['eps']
            self.eps_init = checkpoint['eps_init']
            self.eps_min = checkpoint['eps_min']
            self.eps_decay = checkpoint['eps_decay']

    def save(self, path: str = None, verbose: bool = True, seed: int = None) -> None:
        """Save agent state to persistent storage.
        """
        checkpoint = {
            'nn_p': self.nn_p.state_dict(),
            'opt_p': self.nn_p.optimizer.state_dict(),
            'nn_t': self.nn_t.state_dict(),
            'opt_t': self.nn_t.optimizer.state_dict(),
            'eps': self.eps,
            'eps_init': self.eps_init,
            'eps_min': self.eps_min,
            'eps_decay': self.eps_decay,
        }
        super().save(checkpoint, path, verbose=verbose, seed=seed)


# TD3 for continous action spaces
# Paper: https://arxiv.org/abs/1509.02971
# TODO: Replace with TD7

class TD3(Agent):

    class Actor(NN):
        def __init__(self, state_dim: int, action_dim: int, max_action: int, name: str = None):
            super().__init__(name)

            self.l1 = torch.nn.Linear(state_dim, 256)
            self.l2 = torch.nn.Linear(256, 256)
            self.l3 = torch.nn.Linear(256, action_dim)

            self.max_action = max_action

        def forward(self, state):
            a = torch.nn.functional.relu(self.l1(state))
            a = torch.nn.functional.relu(self.l2(a))
            return self.max_action * torch.tanh(self.l3(a))

    class Critic(NN):
        def __init__(self, state_dim: int, action_dim: int, name: str = None):
            super().__init__(name)

            # Q1 architecture
            self.l1 = torch.nn.Linear(state_dim + action_dim, 256)
            self.l2 = torch.nn.Linear(256, 256)
            self.l3 = torch.nn.Linear(256, 1)

            # Q2 architecture
            self.l4 = torch.nn.Linear(state_dim + action_dim, 256)
            self.l5 = torch.nn.Linear(256, 256)
            self.l6 = torch.nn.Linear(256, 1)

        def forward(self, state, action):
            sa = torch.cat([state, action], 1)

            q1 = torch.nn.functional.relu(self.l1(sa))
            q1 = torch.nn.functional.relu(self.l2(q1))
            q1 = self.l3(q1)

            q2 = torch.nn.functional.relu(self.l4(sa))
            q2 = torch.nn.functional.relu(self.l5(q2))
            q2 = self.l6(q2)
            return q1, q2

        def Q1(self, state, action):
            sa = torch.cat([state, action], 1)

            q1 = torch.nn.functional.relu(self.l1(sa))
            q1 = torch.nn.functional.relu(self.l2(q1))
            q1 = self.l3(q1)
            return q1

    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

        self.actor = TD3.Actor(
            state_dim, action_dim, max_action).to(DEVICE)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=3e-4)

        self.critic = TD3.Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(
            batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = torch.nn.functional.mse_loss(current_Q1, target_Q) + \
            torch.nn.functional.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
