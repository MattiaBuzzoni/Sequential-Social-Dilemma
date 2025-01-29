import random

import numpy as np

from algorithm.model import DQN
from algorithm.prioritized_experience_replay import Memory as PER
from algorithm.uniform_experience_replay import Memory as UER

MAX_EPSILON = 1
MIN_EPSILON = 0.1

MIN_BETA = 0.4
MAX_BETA = 1.0


class Agent(object):
    epsilon = MAX_EPSILON
    beta = MIN_BETA

    def __init__(self, state_size, action_size, bee_index, net_name, arguments):
        self.state_size = state_size
        self.discount_state = arguments['discount_state']
        if self.discount_state:
            self.state_size += 1   # Add the stock dimension to the state size
        self.action_size = action_size
        self.bee_index = bee_index
        self.learning_rate = arguments['learning_rate']
        self.gamma = 0.99
        self.power = 1
        self.net = DQN(self.state_size, self.action_size, net_name, arguments)
        self.memory_model = arguments['memory']

        if self.memory_model == 'UER':
            self.memory = UER(arguments['memory_capacity'])

        elif self.memory_model == 'PER':
            self.memory = PER(arguments['memory_capacity'], arguments['prioritization_scale'])

        else:
            print('Invalid memory algorithm!')

        self.target_type = arguments['target_type']
        self.update_target_frequency = arguments['target_frequency']
        self.max_exploration_step = arguments['maximum_exploration']
        self.batch_size = arguments['batch_size']
        self.step = 0
        self.test = arguments['test']
        if self.test:
            self.epsilon = 0  # Set epsilon to zero for evaluation test

    def greedy_actor(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.net.predict_one_sample(state))

    def find_targets_per(self, batch):
        batch_len = len(batch)

        # Flatten states to ensure uniform shape
        states = np.array([np.ravel(o[0]).astype(np.float32) for o in batch])
        states_ = np.array([np.ravel(o[3]).astype(np.float32) for o in batch])

        p = self.net.predict(states)
        p_ = self.net.predict(states_)
        pTarget_ = self.net.predict(states_, target=True)

        x = np.zeros((batch_len, self.state_size), dtype=np.float32)
        y = np.zeros((batch_len, self.action_size), dtype=np.float32)
        errors = np.zeros(batch_len)

        for i in range(batch_len):
            o = batch[i][1]
            s = o[0]
            a = o[1]
            r = o[2]
            # s_ = o[3]
            done = o[4]

            t = p[i]
            old_value = t[a]
            if done:
                t[a] = r
            else:
                if self.target_type == 'DDQN':
                    t[a] = r + self.gamma * pTarget_[i][np.argmax(p_[i])]
                elif self.target_type == 'DQN':
                    t[a] = r + self.gamma * np.amax(pTarget_[i])
                else:
                    print('Invalid type for target network!')

            x[i] = s
            y[i] = t
            errors[i] = np.abs(t[a] - old_value)

            return [x, y, errors]

    def find_targets_uer(self, batch):
        batch_len = len(batch)

        # Flatten states to ensure uniform shape
        states = np.array([np.ravel(o[0]).astype(np.float32) for o in batch])
        states_ = np.array([np.ravel(o[3]).astype(np.float32) for o in batch])

        p = self.net.predict(states)
        p_ = self.net.predict(states_)
        pTarget_ = self.net.predict(states_, target=True)

        x = np.zeros((batch_len, self.state_size), dtype=np.float32)
        y = np.zeros((batch_len, self.action_size), dtype=np.float32)
        errors = np.zeros(batch_len)

        for i in range(batch_len):
            o = batch[i]
            s = o[0]
            a = o[1]
            r = o[2]
            # n = o[3]
            done = o[4]

            t = p[i]
            old_value = t[a]
            if done:
                t[a] = r
            else:
                if self.target_type == 'DDQN':
                    t[a] = r + self.gamma * pTarget_[i][np.argmax(p_[i])]
                elif self.target_type == 'DQN':
                    t[a] = r + self.gamma * np.amax(pTarget_[i])
                else:
                    print('Invalid type for target network!')

            x[i] = s
            y[i] = t
            errors[i] = np.abs(t[a] - old_value)

        return [x, y]

    def observe(self, sample):
        if self.memory_model == 'UER':
            self.memory.remember(sample, None)

        elif self.memory_model == 'PER':
            _, _, errors = self.find_targets_per([[0, sample]])
            self.memory.remember(sample, errors[0])

        else:
            print('Invalid memory algorithm!')

    def decay_epsilon(self):
        """ Slowly decrease Epsilon based on our experience
            Returns the result of:
            MAX_BETA + (MIN_BETA - MAX_BETA) * (1 - `t`/t_max) ** power
        """

        self.step += 1

        if self.test:
            self.epsilon = MIN_EPSILON
            self.beta = MAX_BETA
        else:
            if self.step < self.max_exploration_step:
                self.epsilon = (MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) *
                                ((self.max_exploration_step - self.step) / self.max_exploration_step) ** self.power)
                self.beta = (MAX_BETA + (MIN_BETA - MAX_BETA) *
                             (self.max_exploration_step - self.step) / self.max_exploration_step)
            else:
                self.epsilon = MIN_EPSILON

    def replay(self):

        if self.memory_model == 'UER':
            batch = self.memory.sample(self.batch_size)
            x, y = self.find_targets_uer(batch)
            self.net.train(x, y)

        elif self.memory_model == 'PER':
            [batch, batch_indices, batch_priorities] = self.memory.sample(self.batch_size)
            x, y, errors = self.find_targets_per(batch)

            normalized_batch_priorities = [float(i) / sum(batch_priorities) for i in batch_priorities]
            importance_sampling_weights = [(self.batch_size * i) ** (-1 * self.beta)
                                           for i in normalized_batch_priorities]
            normalized_importance_sampling_weights = [float(i) / max(importance_sampling_weights)
                                                      for i in importance_sampling_weights]
            sample_weights = [errors[i] * normalized_importance_sampling_weights[i] for i in range(len(errors))]

            self.net.train(x, y, np.array(sample_weights))

            self.memory.update(batch_indices, errors)
        else:
            print('Invalid memory algorithm!')

    def update_target_model(self):
        if self.step % self.update_target_frequency == 0:
            self.net.update_target_model()
