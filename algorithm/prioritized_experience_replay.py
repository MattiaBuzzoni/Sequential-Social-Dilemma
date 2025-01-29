import random

from algorithm.sum_tree import SumTree


class Memory(object):
    """
    A class that implements a Prioritized Experience Replay (PER) buffer for storing agent experience transitions,
    utilizing a SumTree data structure for efficient priority sampling and updates.

    Attributes:
        e (float): A small constant added to the error to ensure non-zero priorities.
        capacity (int): The maximum number of experiences the memory buffer can hold.
        memory (SumTree): A SumTree that stores the experiences along with their priority values.
        pr_scale (float): A scaling factor applied to the priority calculation to control the distribution
                          of priorities.
        max_pr (float): The maximum priority value in the memory, updated when a new experience is added.

    Methods:
        get_priority(error): Computes the priority of an experience based on its error using the formula:
                             (error + e) ** pr_scale.
        remember(sample, error): Adds a new experience to the memory buffer with a priority computed from the error.
        sample(n): Samples a batch of `n` experiences from the memory buffer, with priority-based sampling using
                   the SumTree.
        update(batch_indices, errors): Updates the priorities of the experiences in the memory buffer after
                                       they are processed.
    """
    e = 0.05

    def __init__(self, capacity, pr_scale):
        self.capacity = capacity
        self.memory = SumTree(self.capacity)
        self.pr_scale = pr_scale
        self.max_pr = 0

    def get_priority(self, error):
        return (error + self.e) ** self.pr_scale

    def remember(self, sample, error):
        p = self.get_priority(error)

        self_max = max(self.max_pr, p)
        self.memory.add(self_max, sample)

    def sample(self, n):
        sample_batch = []
        sample_batch_indices = []
        sample_batch_priorities = []
        num_segments = self.memory.total() / n

        for i in range(n):
            left = num_segments * i
            right = num_segments * (i + 1)

            s = random.uniform(left, right)
            idx, pr, data = self.memory.get(s)
            sample_batch.append((idx, data))
            sample_batch_indices.append(idx)
            sample_batch_priorities.append(pr)

        return [sample_batch, sample_batch_indices, sample_batch_priorities]

    def update(self, batch_indices, errors):
        for i in range(len(batch_indices)):
            p = self.get_priority(errors[i])
            self.memory.update(batch_indices[i], p)
