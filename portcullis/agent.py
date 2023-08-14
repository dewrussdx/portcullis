import torch.nn.functional as F
import torch.nn as nn
from typing import Callable
from dataclasses import dataclass
import numpy as np
import torch
from portcullis.nn import NN
from portcullis.env import Env, Action, State
from portcullis.replay import Mem, Frag
from portcullis.pytorch import DEVICE
import os
import copy
import math


class Agent():
    def __init__(self, env: Env, mem: Mem, hdims: tuple[int, int], lr: float,
                 gamma: float, eps: float, eps_min: float, eps_decay: float,
                 tau: float, name: str,
                 ):
        """DRL Agent Base Class.
        """
        self.env = env
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
        self.is_gym, self.env_type, self.num_actions, self.num_features, self.max_actions = Env.get_env_spec(
            self.env)

    # Save agent state
    def save(self, checkpoint: dict, path: str, verbose: bool, seed: int) -> None:
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
        self.mem.clear()
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
        """Decrement exploration rate (epsilon-greedy).
        """
        self.eps = max(self.eps_slope * self.epochs +
                       self.eps_init, self.eps_min)
        return self.eps

    def dec_eps2(self) -> float:
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
        # Initialize NN with input, hidden and output layers
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

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, state):
            x = torch.nn.functional.relu(self.fc1(state))
            x = torch.nn.functional.relu(self.fc2(x))
            return self.fc3(x)

    """Deep Quality Neural Network. This implementation uses separate value and target 
    networks for stability.
    """

    def __init__(self,
                 env: Env,
                 mem: Mem = Mem(65336),
                 hdims: (int, int) = (512, 256),
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 eps: float = 1.0,
                 eps_min: float = 0.01,
                 eps_decay: float = 25e4,
                 tau: float = 0.005,
                 training: bool = True,
                 name: str = 'DQN'
                 ):
        super().__init__(env, mem, hdims, lr, gamma, eps,
                         eps_min, eps_decay, tau, name)
        self.nn_p = DQN.Net(self.num_features, self.hdims, self.num_actions,
                            self.lr, name='DQN_Policy')
        self.nn_t = DQN.Net(self.num_features, self.hdims, self.num_actions,
                            self.lr, name='DQN_Target')
        self.nn_p.optimizer = torch.optim.AdamW(
            self.nn_p.parameters(), lr=self.lr, amsgrad=True)
        self.criterion = torch.nn.SmoothL1Loss()
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

    def train(self, batch_size: int = 128, soft_update=True) -> None:
        # Sample memory
        if len(self.mem) < batch_size:
            return
        # Book keeping
        self.epochs += 1
        frags = self.mem.sample(batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Frag(*zip(*frags))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=DEVICE, dtype=torch.bool)
        arr = [s for s in batch.next_state if s is not None]
        non_final_next_states = torch.cat(arr) if len(
            arr) > 0 else torch.zeros(state_batch.shape, device=DEVICE)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of asctions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.nn_p(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size, device=DEVICE)
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

        if soft_update:
            # Target network soft update (blending of states)
            NN.soft_update(self.nn_p, self.nn_t, self.tau)
        else:
            NN.hard_update(self.nn_p, self.nn_t)

    def act(self, state) -> Action:
        """Returns action for given state as per current policy.
        """
        if self.training:
            if np.random.rand() <= self.dec_eps2():  # amount of exploration reduces with the epsilon value
                # Return random action from action space
                return self.env.action_space.sample() if self.is_gym else self.env.random_action()

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

    def remember(self, state: State, action: Action, next_state: State, reward: float, done: bool) -> None:
        """Remember a transition fragment and store in replay memory.
        """
        state = torch.tensor(state, dtype=torch.float32,
                             device=DEVICE).unsqueeze(0)
        action = torch.tensor([[action]], dtype=torch.long, device=DEVICE)
        next_state = torch.tensor(
            next_state, dtype=torch.float32, device=DEVICE).unsqueeze(0) if not done else None
        reward = torch.tensor([reward], dtype=torch.float32, device=DEVICE)
        self.mem.push(state, action, reward, next_state)

    def load(self, path: str = None, verbose: bool = True) -> None:
        """Load agent state from persistent storage.
        """
        checkpoint = super().load(path, verbose=verbose)
        if checkpoint:
            self.nn_p.load_state_dict(checkpoint['nn_p'])
            self.nn_p.optimizer.load_state_dict(checkpoint['opt_p'])
            self.nn_t.load_state_dict(checkpoint['nn_p'])
            self.nn_t.optimizer.load_state_dict(checkpoint['opt_t'])
            self.criterion = checkpoint['criterion']
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
            'criterion': self.criterion,
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


class DDQN(object):
    # Used for Box2D / Toy problems
    class FC(NN):
        def __init__(self, state_dim, num_actions, name: str = None):
            super().__init__(name)
            self.l1 = nn.Linear(state_dim, 256)
            self.l2 = nn.Linear(256, 256)
            self.l3 = nn.Linear(256, num_actions)

        def forward(self, state):
            q = F.relu(self.l1(state))
            q = F.relu(self.l2(q))
            return self.l3(q)

    def __init__(
        self,
        num_actions,
        state_dim,
        discount=0.99,
        optimizer="Adam",
        optimizer_parameters={},
        polyak_target_update=False,
        target_update_frequency=8e3,
        tau=0.005,
        initial_eps=1,
        end_eps=0.001,
        eps_decay_period=25e4,
        eval_eps=0.001,
    ):
        # Determine network type
        self.Q = DDQN.FC(state_dim, num_actions).to(DEVICE)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(
            self.Q.parameters(), **optimizer_parameters)

        self.discount = discount

        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        # Decay for eps
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        # Evaluation hyper-parameters
        self.state_shape = (-1, state_dim)
        self.eval_eps = eval_eps
        self.num_actions = num_actions

        # Number of training iterations
        self.iterations = 0

    def select_action(self, state, eval=False):
        eps = self.eval_eps if eval \
            else max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(0, 1) > eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).reshape(
                    self.state_shape).to(DEVICE)
                return int(self.Q(state).argmax(1))
        else:
            return np.random.randint(self.num_actions)

    def train(self, replay_buffer):
        # Sample replay buffer
        state, action, next_state, reward, done = replay_buffer.sample()

        # Compute the target Q value
        with torch.no_grad():
            next_action = self.Q(next_state).argmax(1, keepdim=True)
            target_Q = (
                reward + done * self.discount *
                self.Q_target(next_state).gather(1, next_action).reshape(-1, 1)
            )

        # Get current Q estimate
        current_Q = self.Q(state).gather(1, action)

        # Compute Q loss
        Q_loss = F.smooth_l1_loss(current_Q, target_Q)

        # Optimize the Q network
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, filename):
        torch.save(self.iterations, filename + "iterations")
        torch.save(self.Q.state_dict(), f"{filename}Q_{self.iterations}")
        torch.save(self.Q_optimizer.state_dict(), filename + "optimizer")

    def load(self, filename):
        self.iterations = torch.load(filename + "iterations")
        self.Q.load_state_dict(torch.load(f"{filename}Q_{self.iterations}"))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(filename + "optimizer"))


class LAP_DDQN(object):
    # Used for Box2D / Toy problems
    class FC(NN):
        def __init__(self, state_dim, num_actions, name: str = None):
            super().__init__(name)
            self.l1 = torch.nn.Linear(state_dim, 256)
            self.l2 = torch.nn.Linear(256, 256)
            self.l3 = torch.nn.Linear(256, num_actions)

        def forward(self, state):
            q = torch.nn.functional.relu(self.l1(state))
            q = torch.nn.functional.relu(self.l2(q))
            return self.l3(q)

    def __init__(
        self,
        num_actions,
        state_dim,
        discount=0.99,
        optimizer="Adam",
        optimizer_parameters={},
        polyak_target_update=False,
        target_update_frequency=8e3,
        tau=0.005,
        initial_eps=1,
        end_eps=0.001,
        eps_decay_period=25e4,
        eval_eps=0.001,
        alpha=0.6,
        min_priority=1e-2
    ):
        # Determine network type
        self.Q = LAP_DDQN.FC(state_dim, num_actions).to(DEVICE)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(
            self.Q.parameters(), **optimizer_parameters)

        self.discount = discount

        # LAP hyper-parameters
        self.alpha = alpha
        self.min_priority = min_priority

        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau

        # Decay for eps
        self.initial_eps = initial_eps
        self.end_eps = end_eps
        self.slope = (self.end_eps - self.initial_eps) / eps_decay_period

        # Evaluation hyper-parameters
        self.state_shape = (-1, state_dim)
        self.eval_eps = eval_eps
        self.num_actions = num_actions

        # Number of training iterations
        self.iterations = 0
        self.eps = 0

    def select_action(self, state, eval=False):
        self.eps = self.eval_eps if eval \
            else max(self.slope * self.iterations + self.initial_eps, self.end_eps)

        # Select action according to policy with probability (1-eps)
        # otherwise, select random action
        if np.random.uniform(0, 1) > self.eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).reshape(
                    self.state_shape).to(DEVICE)
                return int(self.Q(state).argmax(1))
        else:
            return np.random.randint(self.num_actions)

    def train(self, replay_buffer):
        # Sample replay buffer
        state, action, next_state, reward, done, ind, _weights = replay_buffer.sample()

        # Compute the target Q value
        with torch.no_grad():
            next_action = self.Q(next_state).argmax(1, keepdim=True)
            target_Q = (
                reward + done * self.discount *
                self.Q_target(next_state).gather(1, next_action).reshape(-1, 1)
            )

        # Get current Q estimate
        current_Q = self.Q(state).gather(1, action)

        td_loss = (current_Q - target_Q).abs()
        Q_loss = self.huber(td_loss)

        # Optimize the Q network
        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()

        priority = td_loss.clamp(min=self.min_priority).pow(
            self.alpha).cpu().data.numpy().flatten()
        replay_buffer.update_priority(ind, priority)

    def huber(self, x):
        return torch.where(x < self.min_priority, 0.5 * x.pow(2), self.min_priority * x).mean()

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())

    def save(self, filename):
        torch.save(self.iterations, filename + "iterations")
        torch.save(self.Q.state_dict(), f"{filename}Q_{self.iterations}")
        torch.save(self.Q_optimizer.state_dict(), filename + "optimizer")

    def load(self, filename):
        self.iterations = torch.load(filename + "iterations")
        self.Q.load_state_dict(torch.load(f"{filename}Q_{self.iterations}"))
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer.load_state_dict(torch.load(filename + "optimizer"))
