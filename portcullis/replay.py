import numpy as np
import torch
from portcullis.pytorch import DEVICE
from portcullis.env import State, Action


class ReplayBuffer():
    """Experience Replay Buffer. 
    This buffer can be configured (is_prioritized) as LAP (Loss-Adjusted priotized) xp buffer.
    """

    def __init__(self, state_dim: int, capacity: int = 10_000, batch_size: int = 64, is_prioritized: bool = True):
        """Initialize experience replay buffer.
        """
        self.capacity = capacity
        self.batch_size = batch_size

        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.capacity, state_dim))
        self.action = np.zeros((self.capacity, 1))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.capacity, 1))
        self.not_done = np.zeros((self.capacity, 1))

        self.is_prioritized = is_prioritized
        if self.is_prioritized:
            self.tree = SumTree(self.capacity)
            self.max_priority = 1.0
            self.beta = 0.4

        print(f'Instantiated ReplayBuffer:')
        print(f'- Capacity: {self.capacity}')
        print(f'- Batch Size: {self.batch_size}')
        print(f'- Prioritized: {self.is_prioritized}')

    def add(self, state: State, action: Action, next_state: State, reward: float, done: bool) -> None:
        """Add a transition to the replay buffer.
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - float(done)

        if self.is_prioritized:
            self.tree.set(self.ptr, self.max_priority)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        """Return length of replay buffer.
        """
        return self.size

    def sample(self):
        """Sample memory uniformly (non-prioritized), or weighted by priority.
        """
        ind = self.tree.sample(self.batch_size) if self.is_prioritized \
            else np.random.randint(0, self.size, size=self.batch_size)

        batch = (
            torch.FloatTensor(self.state[ind]).to(DEVICE),
            torch.LongTensor(self.action[ind]).to(DEVICE),
            torch.FloatTensor(self.next_state[ind]).to(DEVICE),
            torch.FloatTensor(self.reward[ind]).to(DEVICE),
            torch.FloatTensor(self.not_done[ind]).to(DEVICE)
        )

        if self.is_prioritized:
            weights = np.array(self.tree.nodes[-1][ind]) ** -self.beta
            weights /= weights.max()
            # Hardcoded: 0.4 + 2e-7 * 3e6 = 1.0. Only used by PER.
            self.beta = min(self.beta + 2e-7, 1)
            batch += (ind, torch.FloatTensor(weights).to(DEVICE).reshape(-1, 1))

        return batch

    def usage(self) -> float:
        """Return buffer usage in percent.
        """
        return self.size * 100. / self.capacity

    def update_priority(self, ind, priority):
        """Update priority for entry at specificed index.
        """
        self.max_priority = max(priority.max(), self.max_priority)
        self.tree.batch_set(ind, priority)


class SumTree(object):
    def __init__(self, capacity: int):
        self.nodes = []
        # Tree construction
        # Double the number of nodes at each level
        level_size = 1
        for _ in range(int(np.ceil(np.log2(capacity))) + 1):
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
