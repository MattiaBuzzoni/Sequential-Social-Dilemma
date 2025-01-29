import numpy


class SumTree(object):
    """
        A class that implements a SumTree data structure for efficient priority sampling and updating.
        The SumTree is used in Prioritized Experience Replay (PER) to store and manage the priorities of experiences
        in a way that allows efficient sampling and updates.

        Attributes:
            write (int): The index where the next experience will be written in the memory.
            capacity (int): The maximum number of experiences the memory can store.
            tree (numpy.ndarray): A 1D array representing the SumTree, where the leaf nodes store the priority values
                                   and the internal nodes store the sums of the priorities of their child nodes.
            data (numpy.ndarray): A 1D array storing the actual experiences.

        Methods:
            _propagate(idx, change): Propagates the priority change up the tree to maintain the correct sums at
                                     each internal node.
            _retrieve(idx, s): Recursively retrieves the index of the experience whose priority range contains
                               the value `s`.
            total(): Returns the total sum of priorities in the tree (the root node value).
            add(p, data): Adds a new experience with a priority `p` and stores the experience `data` at the appropriate
                          location in the tree.
            update(idx, p): Updates the priority of an existing experience and propagates the change up the tree.
            get(s): Retrieves an experience based on a priority sampling value `s`, returning the index, priority,
                    and the experience data.
        """

    def __init__(self, capacity):
        self.write = 0
        self.capacity = capacity
        self.tree = numpy.zeros(2*capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # def get_real_idx(self, data_idx):
    #
    #     tempIdx = data_idx - self.write
    #     if tempIdx >= 0:
    #         return tempIdx
    #     else:
    #         return tempIdx + self.capacity

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        # realIdx = self.get_real_idx(dataIdx)

        return idx, self.tree[idx], self.data[dataIdx]
