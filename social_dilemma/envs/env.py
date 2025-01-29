import os
import random
import string

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from pycolab import ascii_art
from pathlib import Path

from social_dilemma.maps import *
from social_dilemma.objects import PlayerSprite, AppleDrape, SightDrape, ShotDrape
from social_dilemma.utils import build_map, ObservationToArrayWithRGB

DEFAULT_COLOURS = {
    # Colours for agents. R value is a unique identifier
    '1': (245, 0, 0),      # Red
    '2': (0, 0, 245),      # Pure blue
    '3': (245, 0, 245),    # Magenta
    '4': (2, 81, 154),     # Sky blue
    '5': (245, 151, 0),    # Orange
    '6': (100, 245, 245),  # Cyan
    '7': (99, 99, 255),    # Lavender
    '8': (250, 204, 245),  # Pink
    '9': (245, 245, 16),   # Yellow

    '=': (180, 180, 180),  # Steel Impassable wall
    ' ': (0, 0, 0),        # Black background
    '@': (0, 255, 0),      # Green Apples
    '.': (235, 235, 0),    # Yellow beam
    '-': (100, 120, 120)   # Grey scope
}


class HarvestEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, num_agents,
                 visual_radius,
                 max_stock,
                 threshold,
                 map_sketch=SMALL_MAP,
                 full_state=False,
                 discount_state=False):
        super(HarvestEnv, self).__init__()
        # Setup discount mode
        self.discount_state = discount_state
        self.max_stock = max_stock
        self.threshold = threshold
        # Setup spaces
        if discount_state:
            self.action_space = spaces.Discrete(9)
        else:
            self.action_space = spaces.Discrete(8)
        obHeight = obWidth = visual_radius * 2 + 1
        # Setup game
        self.num_agents = num_agents
        self.sight_radius = visual_radius
        self.agent_chars = agent_chars = ''.join(random.sample(string.ascii_letters + string.digits, self.num_agents))
        self.apple_stock = {char: 0 for char in self.agent_chars}
        self.apple_deposited = {char: False for char in self.agent_chars}
        self.starving = {char: False for char in self.agent_chars}
        self.map_height = len(map_sketch)
        self.map_width = len(map_sketch[0])
        self.full_state = full_state
        if full_state:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.map_height + 2, self.map_width + 2, 3),
                                                dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(obHeight, obWidth, 3), dtype=np.uint8)
        self.num_pad_pixels = num_pad_pixels = visual_radius - 1
        self.game_field = build_map(map_sketch, num_pad_pixels=num_pad_pixels, agent_chars=agent_chars)
        self.state = None
        # Pycolab related setup:
        self._game = self.build_game()

        self.colour_map = {}
        for i, char in enumerate(agent_chars):
            self.colour_map[char] = DEFAULT_COLOURS[str((i % len(DEFAULT_COLOURS)) + 1)]

        self.colour_map.update({
            '=': (180, 180, 180),  # Steel Impassable wall
            ' ': (0, 0, 0),  # Black background
            '@': (0, 255, 0),  # Green Apples
            '.': (235, 235, 0),  # Yellow beam
            '-': (75, 75, 75)  # Grey scope
        })

        self.ob_to_image = ObservationToArrayWithRGB(colour_mapping=self.colour_map)
        self.image_counter = 0
        self.apple_positions = self.get_apple_positions()

    def build_game(self):
        agents_order = list(self.agent_chars)
        random.shuffle(agents_order)
        return ascii_art.ascii_art_to_game(
            self.game_field,
            what_lies_beneath=' ',
            sprites=dict(
                [(a, ascii_art.Partial(PlayerSprite, self.agent_chars)) for a in self.agent_chars]),
            drapes={'@': ascii_art.Partial(AppleDrape, self.agent_chars, self.num_pad_pixels,
                                           self.discount_state, self.max_stock, self.threshold),
                    '-': ascii_art.Partial(SightDrape, self.agent_chars, self.num_pad_pixels),
                    '.': ascii_art.Partial(ShotDrape, self.agent_chars, self.num_pad_pixels)},
            # update_schedule=['.'] + agents_order + ['-'] + ['@'],
            update_schedule=['.'] + agents_order + ['-'] + ['@'],
            z_order=['-'] + ['@'] + agents_order + ['.']
        )

    def step(self, n_actions):
        info = {}
        self.state, rewards, _ = self._game.play(n_actions)
        self.apple_stock = self._game.things['@'].apple_stock
        self.apple_deposited =self._game.things['@'].apple_deposited
        observations, apple_stock, done = self.get_observation()

        done = [done] * self.num_agents
        info['apple_stock'] = self.apple_stock
        info['starvation'] = self.is_starving()
        info['apple_deposited'] = self.apple_deposited
        info['apples_collected'] = self.check_apples_collected()

        return observations, apple_stock, rewards, done, info

    def reset(self, **kwargs):
        # Reset the state of the environment to an initial state
        self._game = self.build_game()
        self.state, _, _ = self._game.its_showtime()
        self.apple_stock = {char: 0 for char in self.agent_chars}
        self.apple_deposited = {char: False for char in self.agent_chars}
        observations, apple_stock, _ = self.get_observation()
        if self.discount_state:
            return observations, apple_stock
        else:
            return observations

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        board = self.ob_to_image(self.state)['RGB'].transpose([1, 2, 0])
        board = board[self.num_pad_pixels:self.num_pad_pixels + self.map_height + 2,
                      self.num_pad_pixels:self.num_pad_pixels + self.map_width + 2, :]
        plt.figure(1)
        plt.imshow(board)
        plt.axis("off")
        plt.show(block=False)

        # Images folder path
        path = (Path(__file__).resolve().parents[2] / 'images')
        path.mkdir(parents=True, exist_ok=True)
        # Image names
        image_path = os.path.join(path, "img%04d.png" % self.image_counter)

        plt.savefig(image_path)
        plt.pause(.05)
        plt.clf()
        self.image_counter += 1

    def get_observation(self):
        done = not (np.logical_or.reduce(self.state.layers['@'], axis=None))
        ags = [self._game.things[c] for c in self.agent_chars]
        apple_stock =[]
        obs = []
        board = self.ob_to_image(self.state)['RGB'].transpose([1, 2, 0])

        for a, char in zip(ags, self.agent_chars):
            if char in self.apple_stock:
                apple_stock.append(self.apple_stock[char])
            else:
                apple_stock.append(0)

            if a.visible or a.timeout == 25:
                if self.full_state:
                    ob = np.copy(board)
                    if a.visible:
                        ob[a.position[0], a.position[1], :] = [0, 0, 255]
                    ob = ob[self.num_pad_pixels:self.num_pad_pixels + self.map_height + 2,
                            self.num_pad_pixels:self.num_pad_pixels + self.map_width + 2, :]
                else:
                    ob = np.copy(board[
                                 a.position[0] - self.sight_radius:a.position[0] + self.sight_radius + 1,
                                 a.position[1] - self.sight_radius:a.position[1] + self.sight_radius + 1, :])
                    if a.visible:
                        ob[self.sight_radius, self.sight_radius, :] = [0, 0, 255]
                ob = ob / 255.0
            else:
                # If the agent is time-outed, get empty or constant observation
                if self.full_state:
                    ob = np.zeros_like(board[self.num_pad_pixels:self.num_pad_pixels + self.map_height + 2,
                                       self.num_pad_pixels:self.num_pad_pixels + self.map_width + 2, :])
                else:
                    ob = np.zeros((2 * self.sight_radius + 1, 2 * self.sight_radius + 1, 3))
            obs.append(ob)
        return obs, apple_stock, done

    def get_apple_positions(self):
        if self.state is None or self.state.board is None:
            return []

        current_apple_positions = []
        for y in range(self.state.board.shape[0]):
            for x in range(self.state.board.shape[1]):
                if chr(self.state.board[y, x]) == '@':
                    current_apple_positions.append((y, x))
        return current_apple_positions

    def check_apples_collected(self):
        """Check if any agent has collected an apple."""
        agents_collected_apples = {char: False for char in self.agent_chars}
        current_apple_positions = self.get_apple_positions()
        for agent_char in self.agent_chars:
            agent = self._game.things[agent_char]
            agent_pos = agent.position
            if agent_pos not in current_apple_positions and agent_pos in self.apple_positions:
                agents_collected_apples[agent_char] = True
        self.apple_positions = current_apple_positions  # Update apple positions
        return agents_collected_apples

    def apple_discount(self, discount_amount):
        # Check if the discount mode is activated
        if self.discount_state and any(self.apple_stock.values()):
            # Iterated through the agents and discount if there are apples in their stocks
            for agent_char in self.agent_chars:
                if agent_char in self.apple_stock:
                    if self.apple_stock[agent_char] >= discount_amount:
                        self.apple_stock[agent_char] -= discount_amount
                    else:
                        self.apple_stock[agent_char] = 0

    def is_starving(self):
        # Check if the agent is in starvation mode
        starving_status = {char: False for char in self.agent_chars}
        for agent_char in self.agent_chars:
            if self.discount_state and self.apple_stock[agent_char] == 0:
                starving_status[agent_char] = True
            else:
                starving_status[agent_char] = False
        return starving_status
