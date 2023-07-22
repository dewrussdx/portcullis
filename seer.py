import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import itertools
import json
import ast
import os


# BUY  if (not invested and fastSMA >= slowSMA)
# SELL if (invested and fastSMA < slowSMA)
# IDLE if (invested and fastSMA >= slowSMA or
#         (not invested and fastSMA < slowSMA)
BUY = 0
SELL = 1
IDLE = 2


df0 = pd.read_csv('sp500_closefull.csv', index_col=0, parse_dates=True)
df0.dropna(axis=0, how='all', inplace=True)
df0.dropna(axis=1, how='any', inplace=True)

df_returns = np.log(df0).diff()
print(df_returns)

N = 1000
train_data = df_returns.iloc[:-N]
test_data = df_returns.iloc[-N:]

features = ['AAPL', 'MSFT', 'AMZN']


class Env(object):
    def __init__(self, df, features):
        self.df = df
        self.features = features
        self.maxsteps = len(df)
        self.index = 0
        self.action_space = [BUY, SELL, IDLE]
        self.invested = False
        self.states = self.df[features].to_numpy()
        self.rewards = self.df['SPY'].to_numpy()
        self.total_buy_and_hold = 0

    def reset(self):
        self.index = 0
        self.total_buy_and_hold = 0
        self.invested = False
        return self.states[self.index]

    def step(self, action):
        self.index += 1
        assert self.index < self.maxsteps
        if action == BUY:
            self.invested = True
        elif action == SELL:
            self.invested = False
        reward = self.rewards[self.index] if self.invested else 0
        next_state = self.states[self.index]
        self.total_buy_and_hold += self.rewards[self.index]
        done = self.index == (self.maxsteps - 1)
        return next_state, reward, done


class StateMapper(object):
    def __init__(self, env, n_bins=6, n_samples=10000):
        # first collect sample states from the environment
        states = []
        done = False
        s = env.reset()
        self.D = len(s)  # number of elements we need
        states.append(s)
        for _ in range(n_samples):
            a = np.random.choice(env.action_space)
            s2, _, done = env.step(a)
            states.append(s2)
            if done:
                s = env.reset()
                states.append(s)
        # convert to np array for easier access
        states = np.array(states)
        # create the bins for each dimension
        self.bins = []
        for d in range(self.D):
            column = np.sort(states[:, d])
            # find the boundaries for each bin
            current_bin = []
            for k in range(n_bins):
                boundary = column[int(n_samples/n_bins * (k+0.5))]
                current_bin.append(boundary)
            self.bins.append(current_bin)

    def transform(self, state):
        x = np.zeros(self.D)
        for d in range(self.D):
            x[d] = int(np.digitize(state[d], self.bins[d]))
        return tuple(x)

    def all_possible_states(self):
        list_of_bins = []
        for d in range(self.D):
            list_of_bins.append(list(range(len(self.bins[d])+1)))
        # print (list_of_bins)
        return itertools.product(*list_of_bins)


class Agent(object):
    def __init__(self, action_size, state_mapper, gamma=8e-1, epsilon=1e-1, learning_rate=1e-1):
        self.action_size = action_size
        self.state_mapper = state_mapper
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.explore = True

        # Initialize Q-table randomly
        self.Q = {}
        for s in self.state_mapper.all_possible_states():
            s = tuple(s)
            for a in range(self.action_size):
                self.Q[(s, a)] = np.random.randn()

    def set_explore(self, explore: bool):
        self.explore = explore

    def act(self, state):
        # Explore action space randomly (epsilon)
        if self.explore and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        # Or pick the best action for the given state (greedy)
        s = self.state_mapper.transform(state)
        act_values = [self.Q[(s, a)]
                      for a in range(self.action_size)]
        return np.argmax(act_values)

    def train(self, state, action, next_state, reward, done):
        s = self.state_mapper.transform(state)
        s2 = self.state_mapper.transform(next_state)
        if done:
            target = reward
        else:
            act_values = [self.Q[(s2, a)] for a in range(self.action_size)]
            target = reward + self.gamma * np.amax(act_values)

        # Run one training step
        self.Q[(s, action)] += self.learning_rate * \
            (target-self.Q[(s, action)])


def play_one_episode(agent, env, training):
    state = env.reset()
    done = False
    total_reward = 0
    agent.set_explore(training)
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        if training:
            agent.train(state, action, next_state, reward, done)
        state = next_state
    return total_reward


def load(path: str, state_mapper: StateMapper) -> Agent:
    print(f'Loading model {path}...')
    with open(path, 'r') as handle:
        model = json.loads(handle.read())
    agent = model['agent']
    inst = Agent(agent['action_size'], state_mapper,
                 agent['gamma'], agent['epsilon'], agent['learning_rate'])
    inst.Q = {ast.literal_eval(
        k): v for k, v in json.loads(agent['Q']).items()}
    return inst


def save(agent: Agent, path: str) -> None:
    print(f'Saving model {path}...')
    model = {'agent': {
        # Q: convert each tuple key to a string before saving as json object
        'Q': json.dumps({str(k): v for k, v in agent.Q.items()}),
        'action_size': agent.action_size,
        'gamma': agent.gamma,
        'epsilon': agent.epsilon,
        'learning_rate': agent.learning_rate,
        'explore': agent.explore,
    },
    }
    with open(path, 'w') as handle:
        handle.write(json.dumps(model))

################################################################


train_env = Env(train_data, features)
test_env = Env(test_data, features)

action_size = len(train_env.action_space)
state_mapper = StateMapper(train_env)

# --------------------------------------------------------------------
path = 'model_1.json'

# Check if model file exists
if not os.path.isfile(path):
    # Bootstrap and save model
    agent = Agent(action_size, state_mapper, gamma=0.9995,
                  epsilon=0.9995, learning_rate=5e-3)
    save(agent, path)
# Load model from file
agent = load(path, state_mapper)

# --------------------------------------------------------------------
num_episodes = 1000
train_rewards = np.empty(num_episodes)
test_rewards = np.empty(num_episodes)
epsilon_decay_rate = 0.99995
for e in range(num_episodes):
    # Train on the train set
    print('Epsilon: '+str(agent.epsilon))
    r = play_one_episode(agent, train_env, training=True)
    train_rewards[e] = r
    # Test on the test set
    tr = play_one_episode(agent, test_env, training=False)
    test_rewards[e] = tr
    print(f'eps: {e+1}/{num_episodes}, train: {r:.5f}, test:{tr:.5f}')
    # Decay epsilon up to a minimum value
    agent.epsilon = max(1e-3, agent.epsilon*epsilon_decay_rate)

save(agent, path)
