from collections import deque, namedtuple
import numpy as np
import random

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
