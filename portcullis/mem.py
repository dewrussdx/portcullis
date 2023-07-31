from collections import deque, namedtuple
import numpy as np
import random

Frag = namedtuple(
    'Fragment', ('state', 'action', 'reward', 'next_state'))


class Mem():

    def __init__(self, capacity: int):
        self.fragments = deque([], maxlen=capacity)

    def sample(self, size: int) -> Frag:
        return random.sample(self.fragments, size)

    def push(self, *args) -> None:
        self.fragments.append(Frag(*args))

    def __len__(self) -> int:
        return len(self.fragments)
