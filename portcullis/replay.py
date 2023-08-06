from collections import deque, namedtuple
import numpy as np
import random
import torch
from portcullis.pytorch import DEVICE
from itertools import islice

Frag = namedtuple(
    'Fragment', ('state', 'action', 'reward', 'next_state'))


class Mem():

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.fragments = deque([], maxlen=self.capacity)

    def sample(self, size: int) -> list[Frag]:
        return random.sample(self.fragments, k=size)

    def push(self, *args) -> None:
        self.fragments.append(Frag(*args))

    def clear(self) -> None:
        self.fragments.clear()

    def __len__(self) -> int:
        return self.size()

    def size(self) -> int:
        return len(self.fragments)


class FifoBuffer():
    def __init__(self):
        self.fragments = list()

    def sample(self, size: int) -> list[Frag]:
        items = self.fragments[:size]
        del self.fragments[:size]
        return items

    def push(self, *args) -> None:
        self.fragments.append(Frag(*args))

    def clear(self) -> None:
        self.fragments.clear()

    def __len__(self) -> int:
        return self.size()

    def size(self) -> int:
        return len(self.fragments)


class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(DEVICE),
            torch.FloatTensor(self.action[ind]).to(DEVICE),
            torch.FloatTensor(self.next_state[ind]).to(DEVICE),
            torch.FloatTensor(self.reward[ind]).to(DEVICE),
            torch.FloatTensor(self.not_done[ind]).to(DEVICE),
        )

# Priotized Replay Buffer


class PriotizedReplayBuffer():
    def __init__(self, state_dim, prioritized, batch_size, buffer_size):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)

        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, 1))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

        self.prioritized = prioritized

        if self.prioritized:
            self.tree = SumTree(self.max_size)
            self.max_priority = 1.0
            self.beta = 0.4

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - float(done)

        if self.prioritized:
            self.tree.set(self.ptr, self.max_priority)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        ind = self.tree.sample(self.batch_size) if self.prioritized \
            else np.random.randint(0, self.size, size=self.batch_size)

        batch = (
            torch.FloatTensor(self.state[ind]).to(DEVICE),
            torch.LongTensor(self.action[ind]).to(DEVICE),
            torch.FloatTensor(self.next_state[ind]).to(DEVICE),
            torch.FloatTensor(self.reward[ind]).to(DEVICE),
            torch.FloatTensor(self.not_done[ind]).to(DEVICE)
        )

        if self.prioritized:
            weights = np.array(self.tree.nodes[-1][ind]) ** -self.beta
            weights /= weights.max()
            # Hardcoded: 0.4 + 2e-7 * 3e6 = 1.0. Only used by PER.
            self.beta = min(self.beta + 2e-7, 1)
            batch += (ind, torch.FloatTensor(weights).to(DEVICE).reshape(-1, 1))

        return batch

    def update_priority(self, ind, priority):
        self.max_priority = max(priority.max(), self.max_priority)
        self.tree.batch_set(ind, priority)


class SumTree(object):
    def __init__(self, max_size):
        self.nodes = []
        # Tree construction
        # Double the number of nodes at each level
        level_size = 1
        for _ in range(int(np.ceil(np.log2(max_size))) + 1):
            nodes = np.zeros(level_size)
            self.nodes.append(nodes)
            level_size *= 2

    # Batch binary search through sum tree
    # Sample a priority between 0 and the max priority
    # and then search the tree for the corresponding index

    def sample(self, batch_size):
        query_value = np.random.uniform(0, self.nodes[0][0], size=batch_size)
        node_index = np.zeros(batch_size, dtype=int)

        for nodes in self.nodes[1:]:
            node_index *= 2
            left_sum = nodes[node_index]

            is_greater = np.greater(query_value, left_sum)
            # If query_value > left_sum -> go right (+1), else go left (+0)
            node_index += is_greater
            # If we go right, we only need to consider the values in the right tree
            # so we subtract the sum of values in the left tree
            query_value -= left_sum * is_greater

        return node_index

    def set(self, node_index, new_priority):
        priority_diff = new_priority - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2

    def batch_set(self, node_index, new_priority):
        # Confirm we don't increment a node twice
        node_index, unique_index = np.unique(node_index, return_index=True)
        priority_diff = new_priority[unique_index] - self.nodes[-1][node_index]

        for nodes in self.nodes[::-1]:
            np.add.at(nodes, node_index, priority_diff)
            node_index //= 2
