from collections import deque
import random
import os  # for creating directories
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import itertools
import json
import ast
import os
import argparse



class EstSPYEnv(object):
    BUY = 0
    SELL = 1
    IDLE = 2
    def __init__(self, df, features):
        self.df = df
        self.features = features
        self.maxsteps = len(df)
        self.index = 0
        self.action_space = [EstSPYEnv.BUY, EstSPYEnv.SELL, EstSPYEnv.IDLE]
        self.invested = False
        self.states = self.df[features].to_numpy()
        self.rewards = self.df['SPY'].to_numpy()
        self.total_buy_and_hold = 0

    def render(self):
        pass

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


class DQNAgent:
    def __init__(self, name, state_size, action_size, gamma=0.95, epsilon=1.0, learning_rate=0.001, epsilon_decay_rate=0.995, epsilon_min=0.01):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        # double-ended queue; acts like list, but elements can be added/removed from either end
        self.memory = deque(maxlen=2000)
        self.gamma = gamma  # decay or discount rate: enables agent to take into account future actions in addition to the immediate ones, but discounted at this rate
        # exploration rate: how much to act randomly; more initially than later due to epsilon decay
        self.epsilon = epsilon
        # decrease number of random explorations as the agent's performance (hopefully) improves over time
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_min = epsilon_min  # minimum amount of random exploration permitted
        # rate at which NN adjusts models parameters via SGD to reduce cost
        self.learning_rate = learning_rate
        self.model = self._build_model()  # private method

    def _build_model(self):
        # neural net to approximate Q-value function:
        model = Sequential()
        # 1st hidden layer; states as input
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        #model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        #model.add(Dense(24, activation='relu'))  # 2nd hidden layer
        # 2 actions, so 2 output neurons: 0 and 1 (L/R)
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # list of previous experiences, enabling re-training later
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, explore=True):
        if explore and np.random.rand() <= self.epsilon:  # if acting randomly, take random action
            return random.randrange(self.action_size)
        # if not acting randomly, predict reward value based on current state
        act_values = self.model.predict(state, verbose=0)
        # pick the action that will give the highest reward (i.e., go left or right?)
        return np.argmax(act_values[0])

    def replay(self, samples):  # method that trains NN with experiences sampled from memory
        # sample a minibatch from memory
        batch = random.sample(self.memory, samples)
        for state, action, reward, next_state, done in batch:  # extract data for each batch sample
            # if done (boolean whether game ended or not, i.e., whether final state or not), then target = reward
            target = reward
            if not done:  # if not done, then predict future discounted reward
                target = (reward + self.gamma *  # (target) = reward + (discount rate gamma) *
                          np.amax(self.model.predict(next_state, verbose=0)[0]))  # (maximum target Q based on future action a')
            # approximately map current state to future discounted reward
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            # single epoch of training with x=state, y=target_f; fit decreases loss btwn target_f and y_hat
            self.model.fit(state, target_f, epochs=1, verbose=0)
        self.epsilon = max(self.epsilon_min, self.epsilon *
                           self.epsilon_decay_rate)

    def load(self, name):
        print('Loading weights from '+str(name))
        self.model.load_weights(name)

    def save(self, name):
        print('Saving weights to '+str(name))
        self.model.save_weights(name)


def play_one_episode(agent, env, training_samples=0):
    state = env.reset()
    state_size = len(env.features)
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0

    memories = 0
    while not done:
        action = agent.act(state, explore=training_samples > 0)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        total_reward += reward
        if training_samples > 0:
            agent.remember(state, action, reward, next_state, done)
            memories += 1
        state = next_state

    print('- Episode created '+str(memories)+' memories')
    if training_samples > 0 and training_samples <= len(agent.memory):
        print('- Learning by replaying ' +
              str(training_samples)+' random memory samples...')
        agent.replay(training_samples)

    return total_reward


# Load and preparse environment data
# TODO: Should be moved into the environment itself
df = pd.read_csv('sp500_closefull.csv', index_col=0, parse_dates=True)
df.dropna(axis=0, how='all', inplace=True)
df.dropna(axis=1, how='any', inplace=True)
df_returns = np.log(df).diff()
N = int(len(df_returns)*0.3)
train_data = df_returns.iloc[1:-N]
test_data = df_returns.iloc[-N:-1]
# print(train_data, test_data)
features = ['AAPL', 'MSFT', 'AMZN']

train_env = EstSPYEnv(train_data, features)
test_env = EstSPYEnv(test_data, features)

state_size = len(train_env.features)
action_size = len(train_env.action_space)

agent = DQNAgent('dale_v1', state_size, action_size, gamma=0.9, epsilon=1, learning_rate=0.01,
                 epsilon_decay_rate=0.99, epsilon_min=0.01)  # initialise agent

output_dir = './models/'
n_episodes = 1000
e_start = 0
# 200:
# agent.epsilon = 0.36512
# agent.load('weights_'+str(agent.name)+'_'+str(e_start)+'.hdf5')

for e in range(e_start+1, n_episodes):
    score = play_one_episode(agent, train_env, training_samples=32)
    if e % 25 == 0:
        for layer in agent.model.layers: print(layer.get_config(), layer.get_weights())
        agent.save(output_dir + 'weights_' +
                   str(agent.name) + '_' + str(e) + '.hdf5')
    print(agent.name+' => episode: {}/{}, e: {:.5}, score: {:.5}'  # print the episode's score and agent's epsilon
          .format(e, n_episodes, agent.epsilon, float(score)))
