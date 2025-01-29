import random
from collections import deque


class Memory(object):
    """
        A class that implements a replay buffer for storing agent experience transitions.

        Attributes:
            capacity (int): The maximum number of experiences the memory buffer can hold. memory (deque): A
                            deque that holds the stored experiences, ensuring that the memory size doesn't
                            exceed the capacity.

        Methods:
            remember(sample, priority=None): Adds a new experience to the memory buffer. Optionally stores it
                                             with a priority.
            sample(n): Samples a batch of `n` experiences from the memory buffer.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def remember(self, sample, priority=None):
        if priority is not None:
            self.memory.append((sample, priority))
        else:
            self.memory.append(sample)

    def sample(self, n):
        n = min(n, len(self.memory))
        sample_batch = random.sample(self.memory, n)

        return sample_batch
