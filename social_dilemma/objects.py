import numpy as np
from pycolab import things as pythings
from pycolab.prefab_parts import sprites
from scipy.ndimage import convolve

# Parameters controlling the sigmoid function that governs resource dynamics in the environment
SIGMOID_K = 0.6  # Controls the steepness of the sigmoid curve
SIGMOID_C = 1.5  # Controls the horizontal shift of the sigmoid curve (


class PlayerSprite(sprites.MazeWalker):
    def __init__(self, corner, position, character, agent_chars):
        super(PlayerSprite, self).__init__(
            corner, position, character, impassable=['='] + list(agent_chars.replace(character, '')),
            confined_to_board=True)
        self.times_tagged = 0
        self.agent_chars = agent_chars
        self.orientation = np.random.choice(4)
        self.init_pos = position
        self.timeout = 0

    @property
    def visible(self):
        return self._visible

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if actions is not None:
            a = actions[self.agent_chars.index(self.character)]
        else:
            return
        if self._visible:
            if things['.'].curtain[self.position[0], self.position[1]]:
                self.times_tagged += 1
                if self.times_tagged == 2:
                    self.timeout = 25
                    self._visible = False
                    self._teleport(self.init_pos)
                return
            else:
                if a == 0:  # go upward?
                    if self.orientation == 0:
                        self._north(board, the_plot)
                    elif self.orientation == 1:
                        self._east(board, the_plot)
                    elif self.orientation == 2:
                        self._south(board, the_plot)
                    elif self.orientation == 3:
                        self._west(board, the_plot)
                elif a == 1:  # go downward?
                    if self.orientation == 0:
                        self._south(board, the_plot)
                    elif self.orientation == 1:
                        self._west(board, the_plot)
                    elif self.orientation == 2:
                        self._north(board, the_plot)
                    elif self.orientation == 3:
                        self._east(board, the_plot)
                elif a == 2:  # go leftward?
                    if self.orientation == 0:
                        self._west(board, the_plot)
                    elif self.orientation == 1:
                        self._north(board, the_plot)
                    elif self.orientation == 2:
                        self._east(board, the_plot)
                    elif self.orientation == 3:
                        self._south(board, the_plot)
                elif a == 3:  # go rightward?
                    if self.orientation == 0:
                        self._east(board, the_plot)
                    elif self.orientation == 1:
                        self._south(board, the_plot)
                    elif self.orientation == 2:
                        self._west(board, the_plot)
                    elif self.orientation == 3:
                        self._north(board, the_plot)
                elif a == 4:  # turn right?
                    if self.orientation == 3:
                        self.orientation = 0
                    else:
                        self.orientation = self.orientation + 1
                elif a == 5:  # turn left?
                    if self.orientation == 0:
                        self.orientation = 3
                    else:
                        self.orientation = self.orientation - 1
                elif a == 6:  # do nothing?
                    self._stay(board, the_plot)
        else:
            if self.timeout == 0:
                # self._stay(board, the_plot) # disappear only
                self._visible = True
                self.times_tagged = 0
            else:
                self.timeout -= 1


class SightDrape(pythings.Drape):
    """Drape representing the agent's line of sight."""

    def __init__(self, curtain, character, agent_chars, num_pad_pixels):
        super().__init__(curtain, character)
        self.agent_chars = agent_chars
        self.num_pad_pixels = num_pad_pixels
        # Calculate the effective height and width of the grid, excluding padding
        self.h = curtain.shape[0] - (num_pad_pixels * 2 + 2)
        self.w = curtain.shape[1] - (num_pad_pixels * 2 + 2)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # Reset the curtain (line of sight) to False
        np.logical_and(self.curtain, False, self.curtain)
        # Get all agents and process each one
        ags = [things[c] for c in self.agent_chars]
        for agent in ags:
            if agent.visible:
                pos = agent.position
                # Set the line of sight in the direction of the agent's orientation
                if agent.orientation == 0:  # Up
                    self.curtain[pos[0] - 1, pos[1]] = True
                elif agent.orientation == 1:  # Right
                    self.curtain[pos[0], pos[1] + 1] = True
                elif agent.orientation == 2:  # Down
                    self.curtain[pos[0] + 1, pos[1]] = True
                elif agent.orientation == 3:  # Left
                    self.curtain[pos[0], pos[1] - 1] = True
                # Ensure the line of sight does not go through walls ('=' layer)
                self.curtain[:, :] = np.logical_and(self.curtain, np.logical_not(layers['=']))
            else:
                # If the agent is invisible, set the curtain to all False
                self.curtain[:, :] = np.zeros_like(self.curtain)


class ShotDrape(pythings.Drape):
    """Drape representing the agent's tagging ray."""

    def __init__(self, curtain, character, agent_chars, num_pad_pixels):
        super().__init__(curtain, character)
        self.agent_chars = agent_chars
        self.num_pad_pixels = num_pad_pixels
        # Calculate the effective height and width of the grid, excluding padding
        self.h = curtain.shape[0] - (num_pad_pixels * 2 + 2)
        self.w = curtain.shape[1] - (num_pad_pixels * 2 + 2)
        # Height of the tagging ray's reach
        self.scopeHeight = num_pad_pixels + 7

    def update(self, actions, board, layers, backdrop, things, the_plot):
        beam_width = 0
        beam_height = self.scopeHeight
        np.logical_and(self.curtain, False, self.curtain)
        if actions is not None:
            for i, a in enumerate(actions):
                if a == 7:
                    agent = things[self.agent_chars[i]]
                    if agent.visible:
                        pos = agent.position

                        if agent.orientation == 0:
                            if np.any(layers['='][pos[0] - beam_height:pos[0],
                                                  pos[1] - beam_width:pos[1] + beam_width + 1]):
                                collision_idx = np.argwhere(layers['='][pos[0] - beam_height:pos[0],
                                                                        pos[1] - beam_width:pos[1] + beam_width + 1])
                                beam_height = beam_height - (np.max(collision_idx) + 1)
                            self.curtain[pos[0] - beam_height:pos[0],
                                         pos[1] - beam_width:pos[1] + beam_width + 1] = True
                        elif agent.orientation == 1:
                            if np.any(layers['='][pos[0] - beam_width:pos[0] + beam_width + 1,
                                                  pos[1] + 1:pos[1] + beam_height + 1]):
                                collision_idx = np.argwhere(layers['='][pos[0] - beam_width:pos[0] + beam_width + 1,
                                                                        pos[1] + 1:pos[1] + beam_height + 1])
                                beam_height = (np.min(collision_idx[:, 0]))
                            self.curtain[pos[0] - beam_width:pos[0] + beam_width + 1,
                                         pos[1] + 1:pos[1] + beam_height + 1] = True
                        elif agent.orientation == 2:
                            if np.any(layers['='][pos[0] + 1:pos[0] + beam_height + 1,
                                                  pos[1] - beam_width:pos[1] + beam_width + 1]):
                                collision_idx = np.argwhere(layers['='][pos[0] + 1:pos[0] + beam_height + 1,
                                                                        pos[1] - beam_width:pos[1] + beam_width + 1])
                                beam_height = (np.min(collision_idx[:, 0]))
                            self.curtain[pos[0] + 1:pos[0] + beam_height + 1,
                                         pos[1] - beam_width:pos[1] + beam_width + 1] = True
                        elif agent.orientation == 3:
                            if np.any(layers['='][pos[0] - beam_width:pos[0] + beam_width + 1,
                                                  pos[1] - beam_height:pos[1]]):
                                collision_idx = np.argwhere(layers['='][pos[0] - beam_width:pos[0] + beam_width + 1,
                                                                        pos[1] - beam_height:pos[1]])
                                beam_height = beam_height - (np.max(collision_idx) + 1)
                            self.curtain[pos[0] - beam_width:pos[0] + beam_width + 1,
                                         pos[1] - beam_height:pos[1]] = True
                        # self.curtain[:, :] = np.logical_and(self.curtain, np.logical_not(layers['=']))
        else:
            return


class AppleDrape(pythings.Drape):
    """Coins Drape"""

    def __init__(self, curtain, character, agent_chars, num_pad_pixels, discount_state, max_stock, threshold):
        super().__init__(curtain, character)
        self.agent_chars = agent_chars
        self.num_pad_pixels = num_pad_pixels
        self.discount_state = discount_state
        self.apples = np.copy(curtain)
        # Initialize each agent stock
        self.apple_stock = {char: 0 for char in agent_chars}
        self.max_stock = max_stock
        self.apple_deposited = {char: False for char in agent_chars}
        self.threshold = threshold
        self.drop_radius = 2

    def update(self, actions, board, layers, backdrop, things, the_plot):
        # Reset apple_deposited for all agents to False at the beginning of each update
        self.apple_deposited = {char: False for char in self.agent_chars}

        agents_map = np.ones(self.curtain.shape, dtype=bool)
        rewards = []

        if actions is not None:
            for i, a in enumerate(actions):

                rew = self.curtain[things[self.agent_chars[i]].position[0], things[self.agent_chars[i]].position[1]]

                if self.discount_state:
                    if self.apple_stock[self.agent_chars[i]] >= self.max_stock:
                        self.apple_stock[self.agent_chars[i]] += 0  # Does not add an apple to the stock
                        self.curtain[things[self.agent_chars[i]].position[0],
                                     things[self.agent_chars[i]].position[1]] = False  # Remove the apple from the env
                        rew = 0  # Stock is full, reward is 0
                    else:
                        # If 'True' starvation mode is activated
                        if rew:
                            self.apple_stock[self.agent_chars[i]] += 1  # Add an apple to the stock
                            self.curtain[things[self.agent_chars[i]].position[0],
                                         things[self.agent_chars[i]].position[1]] = False
                            # Remove the apple from the env
                        if not rew and not self.apple_stock[self.agent_chars[i]]:
                            rew = -1
                else:
                    if rew:
                        self.apple_stock[self.agent_chars[i]] += 1  # Add an apple to the stock
                        self.curtain[things[self.agent_chars[i]].position[0],
                                     things[self.agent_chars[i]].position[1]] = False

                if self.discount_state:
                    agent = things[self.agent_chars[i]]
                    pos = agent.position  # Agent's current position

                    if agent.visible:

                        if self.apple_stock[self.agent_chars[i]] > 1 and a == 7:
                            # tag_penalty = min(round(0.5 * (self.apple_stock[self.agent_chars[i]]) /
                            #                        (self.max_stock - 1), 1), 0.5)
                            self.apple_stock[self.agent_chars[i]] = 1
                            # rew -= tag_penalty

                        if a == 8:
                            # Check if the stock of apples is greater than the threshold
                            if self.apple_stock[self.agent_chars[i]] > self.threshold:
                                # Define the boundaries of the square section centered around the agent
                                x_min = pos[0] - (self.num_pad_pixels // 2)
                                x_max = pos[0] + (self.num_pad_pixels // 2) + 1
                                y_min = pos[1] - (self.num_pad_pixels // 2)
                                y_max = pos[1] + (self.num_pad_pixels // 2) + 1

                                # Check if there's at least one wall in the square section around the agent
                                wall_ahead = np.any(layers['='][x_min:x_max, y_min:y_max])

                                if wall_ahead:
                                    # Find all free positions within the square area
                                    free_positions = [
                                        (x, y)
                                        for x in range(x_min, x_max)
                                        for y in range(y_min, y_max)
                                        if not layers['='][x, y] and not layers['@'][x, y]
                                        # No wall and no apple at (x, y)
                                    ]

                                    # If there are any free positions, deposit the apple randomly in one of them
                                    if free_positions:
                                        drop_pos = free_positions[
                                            np.random.choice(len(free_positions))]
                                        self.apple_stock[self.agent_chars[i]] -= 1
                                        self.apple_deposited[self.agent_chars[i]] = True
                                        self.curtain[drop_pos] = True
                                else:
                                    # If no wall is ahead, only check for positions with no apples
                                    free_positions = [
                                        (x, y)
                                        for x in range(x_min, x_max)
                                        for y in range(y_min, y_max)
                                        if not layers['@'][x, y]  # No apple at (x, y)
                                    ]
                                    # If there are any free positions, deposit the apple randomly in one of them
                                    if free_positions:
                                        drop_pos = free_positions[np.random.choice(len(free_positions))]
                                        self.apple_stock[self.agent_chars[i]] -= 1
                                        self.apple_deposited[self.agent_chars[i]] = True
                                        self.curtain[drop_pos] = True

                rewards.append(rew * 1)
                agents_map[things[self.agent_chars[i]].position[0], things[self.agent_chars[i]].position[1]] = False
        the_plot.add_reward(rewards)

        # Matrix of local stock of apples
        kernel = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        L = convolve(self.curtain[self.num_pad_pixels + 1:-self.num_pad_pixels - 1,
                     self.num_pad_pixels + 1:-self.num_pad_pixels - 1] * 1, kernel, mode='constant')
        sigmoid = lambda x, k=-SIGMOID_K, c=SIGMOID_C: 1 / (1 + np.exp(-k * (x - c)))
        probs = sigmoid(L, k=-SIGMOID_K, c=SIGMOID_C)
        apple_idx = np.argwhere(np.logical_and(np.logical_and(self.apples, np.logical_not(self.curtain)), agents_map))

        if apple_idx.size > 0:
            i, j = apple_idx[np.random.choice(apple_idx.shape[0])]
            if not self.curtain[i, j]:
                self.curtain[i, j] = np.random.choice([True, False],
                                                      p=[probs[i - self.num_pad_pixels - 1, j -
                                                               self.num_pad_pixels - 1],
                                                         1 - probs[i - self.num_pad_pixels - 1,
                                                                   j - self.num_pad_pixels - 1]])
