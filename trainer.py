import logging
import argparse
import os
import socket
import random
import time
import shutil
import gym

import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt

from algorithm.agent import Agent
from config.default_args import add_default_args
from social_dilemma.envs.env import HarvestEnv, DEFAULT_COLOURS
from utils import report_training_status, get_name_weights
from scipy.ndimage import gaussian_filter1d

SAVE_PATH = os.path.abspath(os.path.dirname(__file__) + '/savings/')


class FigureGenerator:
    def __init__(self, episode_number, max_ts, figures_dir='figures'):
        """
        Initializes the FigureGenerator with number of episodes and steps.

        Parameters:
        episode_number (int): Total number of episodes.
        max_steps (int): Maximum number of time steps per episode.
        """
        self.episode_number = episode_number
        self.max_ts = max_ts

        self.figures_dir = figures_dir
        self._create_figures_dir()

    def _create_figures_dir(self):
        """Create the figures directory if it doesn't exist."""
        try:
            os.makedirs(self.figures_dir, exist_ok=True)
            print(f"Figures directory created at: '{self.figures_dir}'")
        except OSError as error:
            print(f"Error creating figures directory: {error}")

    @staticmethod
    def _get_agent_color(agent_id):
        """
        Retrieve the color associated with the given agent ID.

        Parameters:
        agent_id (int): The ID of the agent (0-indexed).

        Returns:
        tuple: A tuple representing the RGB color of the agent, normalized to [0, 1].
        """
        id_str = str(agent_id + 1)  # Convert agent ID to 1-indexed string
        color = DEFAULT_COLOURS.get(id_str, (0, 0, 0))  # Default to black if not found
        return tuple(c / 255 for c in color)  # Normalize RGB values to [0, 1]

    def plot_cumulative_rewards(self, reward_array, agents, episode):
        """
        Plots cumulative rewards for each agent over time steps for a given episode.

        Parameters:
        reward_array (np.ndarray): The rewards array.
        agents (list): List of agents.
        episode (int): Current episode number.
        """
        plt.figure(figsize=(10, 6))

        for i, agent in enumerate(agents):
            cumulative_rewards = np.cumsum(reward_array[i, :, episode])
            plt.plot(range(1, self.max_ts + 1), cumulative_rewards,
                     label=f'Agent {i + 1}',
                     color=self._get_agent_color(i))

        plt.title(f'Reward per agent - Episode {episode}')
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig('figures/reward.png')
        plt.close()

    def plot_apples_collected(self, apples_collected_array, agents, episode):
        """
        Plots cumulative apples collected for each agent over time steps for a given episode.

        Parameters:
        apples_collected_array (np.ndarray): The apples collected array.
        agents (list): List of agents.
        episode (int): Current episode number.
        """
        plt.figure(figsize=(10, 6))

        for i, agent in enumerate(agents):
            apples_collected = np.cumsum(apples_collected_array[i, :, episode])
            plt.plot(range(1, self.max_ts + 1), apples_collected,
                     label=f'Agent {i + 1}',
                     color=self._get_agent_color(i))

        plt.title(f'Apple gathering per agent (cumulative) - Episode {episode}')
        plt.xlabel('Steps')
        plt.ylabel('Cumulative apples gathering')
        plt.legend()
        plt.savefig('figures/apples_collected.png')
        plt.close()

    def plot_apple_stock(self, apples_stock_array, agents, episode):
        """
        Plots the apples stock for each agent over time steps for a given episode.

        Parameters:
        apples_stock_array (np.ndarray): The agent tagging array.
        agents (list): List of agents.
        episode (int): Current episode number.
        """
        plt.figure(figsize=(10, 6))

        for i, agent in enumerate(agents):
            apple_stock = apples_stock_array[i, :, episode]
            plt.plot(range(1, self.max_ts + 1), apple_stock,
                     label=f'Agent {i + 1}',
                     color=self._get_agent_color(i))

        plt.title(f'Apple stock per Agent - Episode {episode}')
        plt.xlabel('Steps')
        plt.ylabel('Apple stock')
        plt.legend()
        plt.savefig('figures/apple_stock.png')
        plt.close()

    def plot_apple_deposit(self, apple_deposited_array, agents, episode):
        """
        Plots the number of apples deposited for each agent over time steps for a given episode.

        Parameters:
        apples_stock_array (np.ndarray): The agent tagging array.
        agents (list): List of agents.
        episode (int): Current episode number.
        """
        plt.figure(figsize=(10, 6))

        for i, agent in enumerate(agents):
            apple_deposited = np.cumsum(apple_deposited_array[i, :, episode])
            plt.plot(range(1, self.max_ts + 1), apple_deposited,
                     label=f'Agent {i + 1}',
                     color=self._get_agent_color(i))

        plt.title(f'Apple contribution per agent (cumulative) - Episode {episode}')
        plt.xlabel('Steps')
        plt.ylabel('Apple deposited')
        plt.legend()
        plt.savefig('figures/apple_deposit.png')
        plt.close()

    def plot_agent_starvation(self, agent_starvation_array, agents, episode, window_size=100):
        """
        Plots the number of apples deposited for each agent over time steps for a given episode.

        Parameters:
        apples_stock_array (np.ndarray): The agent tagging array.
        agents (list): List of agents.
        episode (int): Current episode number.
        """
        plt.figure(figsize=(10, 6))

        for i, agent in enumerate(agents):
            moving_average = np.convolve(agent_starvation_array[i, :, episode], np.ones(window_size)/window_size,
                                         mode='same')
            plt.plot(range(1, self.max_ts + 1), moving_average,
                     label=f'Agent {i + 1}',
                     color=self._get_agent_color(i))

        plt.title(f' Agent starvation ({window_size}-steps window) - Episode {episode}')
        plt.xlabel('Steps')
        plt.ylabel('Agent starvation')
        plt.legend()
        plt.savefig('figures/agent_starvation.png')
        plt.close()

    def plot_total_reward(self, reward_array, agents, from_episode, to_episode):
        """
        Plots the total cumulative rewards for each agent across episodes,
        starting from the specified episode.

        Parameters:
        reward_array (np.ndarray): The rewards array with shape [num_agents, num_steps, num_episodes].
        episode (int): Total number of episodes to plot.
        start_episode (int): The episode to start plotting from (default: 0).
        """
        plt.figure(figsize=(10, 6))  # Create a figure with the specified size

        # Adjust the range of episodes
        x_range = np.arange(from_episode, to_episode)  # X-axis values for the selected range

        for i, agent in enumerate(agents):
            # Compute the cumulative rewards for the selected range of episodes
            total_rewards = np.sum(reward_array[i, :, from_episode:to_episode], axis=0)
            total_rewards = total_rewards.flatten()

            # Apply a Gaussian filter to smooth the rewards
            smoothed_rewards = gaussian_filter1d(total_rewards, sigma=10)

            # Interpolate the smoothed data
            x_interp = np.linspace(from_episode, to_episode - 1, 1000)
            spline = np.interp(x_interp, x_range, smoothed_rewards)

            # Plot the original reward curve with some opacity
            plt.plot(x_range, total_rewards,
                     alpha=0.25,
                     color=self._get_agent_color(i))

            # Plot the smoothed curve over the original one
            plt.plot(x_interp, spline, label=f'Smoothed Reward Agent {i + 1}',
                     color=self._get_agent_color(i), linestyle='-')

        # Add title, labels, and legend
        plt.title('Smoothed Reward per Agent')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig('figures/total_reward.png')
        plt.close()


class Trainer:
    def __init__(self, arguments, verbose=0):
        """
        Initializes the Trainer class with environment and training parameters.

        :param arguments: A dictionary containing various parameters required for training.
        :param verbose: Verbosity level for logging output.
        """
        # Environment parameters
        self.num_agents = arguments['agents_number']  # Number of agents in the environment
        self.visual_radius = arguments['visual_radius']  # Visual range of the agents
        self.threshold = arguments['threshold']  # Threshold for some condition in the environment
        self.max_stock = arguments['max_stock']  # Maximum number of apples in each agent's stock
        self.discount_state = arguments['discount_state']  # Boolean flag for discount state mode
        self.discount_amount = arguments['discount_amount']  # Amount to discount the reward

        # Initialize the environment with the specified parameters
        self.env = HarvestEnv(num_agents=self.num_agents, visual_radius=self.visual_radius, max_stock=self.max_stock,
                              threshold=self.threshold, discount_state=self.discount_state)

        # Training parameters
        self.episodes_number = arguments['episode_number']  # Total number of episodes for training
        self.max_ts = arguments['max_timestep']  # Maximum number of time step per episode
        self.filling_steps = arguments['first_step_memory']  # Steps to fill the replay memory before training
        self.steps_b_updates = arguments['replay_steps']  # Steps between updates to the agent's model
        self.max_random_moves = arguments['max_random_moves']  # Maximum random actions to take before training
        self.render = arguments['render']  # Flag to render the environment
        self.recorder = arguments['recorder']  # Flag to record videos of the training
        self.generate_figures = arguments['generate_figures']  # Flag to generate figures during training
        self.test = arguments['test']  # Flag to indicate if training is in test mode
        self.verbose = verbose

        if not self.test:
            # Initialize the figures manager for saving the plots of the training model
            self.figure_manager = FigureGenerator(self.episodes_number, self.max_ts)

        # Initialize arrays for storing training data
        self.reward_array = np.zeros((self.num_agents, self.max_ts, self.episodes_number))
        self.action_array = np.zeros((self.num_agents, self.max_ts, self.episodes_number))
        self.apples_collected_array = np.zeros((self.num_agents, self.max_ts, self.episodes_number))
        self.apple_stock_array = np.zeros((self.num_agents, self.max_ts, self.episodes_number))
        if self.discount_state:
            self.agent_starvation_array = np.zeros((self.num_agents, self.max_ts, self.episodes_number))
            self.apple_deposited_array = np.zeros((self.num_agents, self.max_ts, self.episodes_number))

    def run(self, agents):
        """
        Runs the training loop for the specified number of episodes.

        :param agents: A list of agent instances participating in the training.
        """
        if not self.test:
            # Print initial status
            print("== Status ==")
            print(f"Node IP: {socket.gethostbyname(socket.gethostname())}")
            print("Start training ...")
            if self.discount_state:
                print("Discount state mode is enabled!")

        episode_num = 1  # Start from the beginning
        total_step = 0
        max_score = -10000  # Initialize the maximum score

        # Header for the results table
        table_headers = ["Iteration", "Num Episodes", "Episode Length Mean", "Time (s)",
                         "Sample Throughput", "Reward", "Epsilon Greedy"]
        results = []

        # Training loop
        for episode_num in range(episode_num, self.episodes_number + 1):
            start_time = time.time()  # Start timer for the episode

            # Reset the environment for the new episode
            if self.discount_state:
                state, apple_stock = self.env.reset()
            else:
                state = self.env.reset()

            if self.render:
                self.env.render()  # Render the environment if required

            # Make a random number of initial moves
            random_moves = random.randint(0, self.max_random_moves)
            for _ in range(random_moves):
                actions = [4 for _ in range(len(agents))]  # Perform random actions
                state, _, _, _, _ = self.env.step(actions)
                if self.render:
                    self.env.render()

            state = [np.array(state).ravel() for state in state]  # Flatten the state
            if self.discount_state:
                state = [np.concatenate([s, np.array([stk/self.max_stock])]) for s, stk in zip(state, apple_stock)]

            # done = False
            reward_all = [0] * self.num_agents  # Initialize rewards for all agents
            time_step = 0

            # Main loop for each timestep in the episode
            while time_step < self.max_ts:
                self.env.apple_discount(discount_amount=self.discount_amount)  # Apply discount to rewards

                # Get actions from each agent
                actions = [agent.greedy_actor(state[i]) for i, agent in enumerate(agents)]
                next_state, apple_stock, reward, done, info = self.env.step(actions)

                next_state = [np.array(next_state).ravel() for next_state in next_state]  # Flatten the next state
                if self.discount_state:
                    next_state = [np.concatenate([s, np.array([stk/self.max_stock])])
                                  for s, stk in zip(next_state, apple_stock)]

                if not self.test:  # If not in test mode
                    for i, agent in enumerate(agents):
                        # Observe the transition
                        agent.observe((state[i], actions[i], reward[i], next_state[i], done[i]))

                        # Perform agent updates based on the conditions
                        if total_step >= self.filling_steps:
                            agent.decay_epsilon()  # Decay epsilon for exploration-exploitation trade-off
                            if time_step % self.steps_b_updates == 0:
                                agent.replay()  # Replay experiences for learning
                            agent.update_target_model()  # Update the target model

                    # Update agent data
                    agent_keys = list(self.env.agent_chars)

                    for i, agent in enumerate(agents):
                        self.reward_array[i, time_step, episode_num] = reward[i]
                        self.action_array[i, time_step, episode_num] = actions[i]

                        # Check if the agent index is within the bounds of available keys
                        if i < len(agent_keys):
                            self.apples_collected_array[i, time_step, episode_num] = int(
                                info['apples_collected'].get(agent_keys[i], False))  # Convert to 1/0
                            self.apple_stock_array[i, time_step, episode_num] = int(
                                info['apple_stock'].get(agent_keys[i], False))  # Convert to 1/0
                            if hasattr(self, 'apple_deposited_array'):
                                self.apple_deposited_array[i, time_step, episode_num] = int(
                                    info['apple_deposited'].get(agent_keys[i], False))  # Convert to 1/0
                            if hasattr(self, 'agent_starvation_array'):
                                self.agent_starvation_array[i, time_step, episode_num] = int(
                                    info['starvation'].get(agent_keys[i], False))  # Convert to 1/0

                        else:
                            self.apples_collected_array[i, time_step, episode_num] = 0
                            # Default value if out of bounds
                            self.apple_stock_array[i, time_step, episode_num] = 0
                            # Default value if out of bounds

                state = next_state  # Move to the next state
                reward_all = np.round(np.add(reward_all, reward), 1)  # Sum rewards across agents
                total_step += 1  # Increment the total step count
                time_step += 1  # Increment the timestep count

                if self.render:
                    self.env.render()  # Render the environment if required

            end_time = time.time()  # End timer for the episode
            elapsed_time = end_time - start_time
            total_elapsed_time = elapsed_time

            steps = self.max_ts
            sample_throughput = steps / total_elapsed_time  # Calculate sample throughput

            # Record video from rendered environment
            if self.recorder:
                videos_path = (os.path.abspath(os.path.dirname(__file__)) + '/videos')
                images_path = (os.path.abspath(os.path.dirname(__file__)) + '/images')
                print(images_path)
                if not os.path.exists(videos_path):
                    os.makedirs(videos_path)
                os.system(
                    f"ffmpeg -r 4 -i \"{images_path}/img%04d.png\" "
                    "-b:v 40000 -minrate 40000 -maxrate 4000k -bufsize 1835k -c:v mjpeg -qscale:v 0 "
                    + videos_path + "/{a1}_{a2}.avi"
                    .format(a1=self.num_agents, a2=episode_num))

            # Manage savings for checkpoints and weights
            if not self.test:
                if total_step > self.filling_steps:
                    should_save_model = False

                    # Save checkpoint and weights due to a new maximum
                    if any(reward > max_score for reward in reward_all):
                        max_score = max(reward_all)
                        should_save_model = True
                        logging.info(f"Model weights saved due to a new maximum score: {max_score}")

                    # Save checkpoint and weights after n episodes
                    if episode_num % 50 == 0:
                        should_save_model = True
                        logging.info(f"Model weights saved at episode {episode_num}.")

                    if should_save_model:
                        for agent in agents:
                            agent.net.save_weights()

            # Conditional logging of training progress based on verbosity level
            if self.verbose == 0:
                return  # If verbosity is set to 0, do nothing

            elif self.verbose == 1:
                # Print rewards for each agent for the current episode
                print("Episode #{}: {}".format(episode_num, "; ".join("Agent {}: reward: {}".format(i + 1, reward)
                                                                      for i, reward in enumerate(reward_all))))

            elif self.verbose == 2:
                # Collect metrics for summary
                result_row = [
                    int(total_step / self.max_ts),  # Iteration
                    episode_num,  # Episode number
                    self.max_ts,  # Mean Episode Length
                    round(elapsed_time, 2),  # Time per Episode
                    round(sample_throughput, 2),  # Sample Throughput
                    ', '.join(map(str, reward_all)),  # Reward per Agent
                    ', '.join(f"{round(agent.epsilon, 4)}" for agent in agents)  # Epsilon Greed for each agent
                ]
                results.append(result_row)
                # Print metrics summary
                if episode_num % 10 == 0:  # Print every 10 episodes
                    print(tabulate(results[-10:], headers=table_headers, tablefmt='grid'))

                # Report training status at regular intervals after filling_steps are met
                if total_step >= self.filling_steps:
                    if episode_num % 100 == 0 and episode_num > 0:
                        report_training_status(self)

            # Plot various figures related to the agents' performance
            if episode_num % 100 == 0:
                self.figure_manager.plot_cumulative_rewards(self.reward_array, agents, episode_num)
                self.figure_manager.plot_apples_collected(self.apples_collected_array, agents, episode_num)
                self.figure_manager.plot_apple_stock(self.apple_stock_array, agents, episode_num)
                self.figure_manager.plot_total_reward(self.reward_array, agents,
                                                      from_episode=10, to_episode=episode_num)
                if self.discount_state:
                    self.figure_manager.plot_apple_deposit(self.apple_deposited_array, agents, episode_num)
                    self.figure_manager.plot_agent_starvation(self.agent_starvation_array, agents, episode_num)

        # Attempt to remove the 'images' directory and its contents
        try:
            shutil.rmtree(os.path.abspath(os.path.dirname(__file__)) + '/images')
            print(f"Directory 'images' and its contents have been successfully removed.")
        except OSError as error:
            # Print error message if the directory cannot be removed
            print(f"Error removing 'images': {error}")

        print("End training!")  # Indicate the end of the training process


if __name__ == "__main__":
    # Configure base Logger
    logdir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'logs')
    os.makedirs(logdir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=os.path.join(logdir, 'app.log'))

    # Argparse setup
    parser = argparse.ArgumentParser()
    add_default_args(parser)
    args = vars(parser.parse_args())

    # Initialize Trainer
    train = Trainer(args, verbose=2)

    # Prepare agent parameters
    state_size = np.prod(train.env.observation_space.shape)
    if isinstance(train.env.action_space, gym.spaces.Discrete):
        action_space = train.env.action_space.n
    elif isinstance(train.env.action_space, gym.spaces.Box):
        action_space = train.env.action_space.shape[0]
    else:
        raise ValueError("Unknown action space type")

    weights_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    # Initialize agents
    all_agents = [
        Agent(state_size, action_space, b_idx, get_name_weights(b_idx), args)
        for b_idx in range(args['agents_number'])
    ]

    # Error handling for video recording
    if args['recorder'] and not args['render']:
        raise ValueError(
            "You have no --render set, but are trying to record rollout videos (via options --recorder)! "
            "Either set --render or do not use --recorder.")

    try:
        train.run(all_agents)

    except KeyboardInterrupt:
        logging.shutdown()
        try:
            shutil.rmtree(logdir)
        except PermissionError as e:
            print(f"PermissionError: {e}. The log directory is currently in use.")
        print('\nTraining exited early.')

        if SAVE_PATH is None:
            try:
                SAVE_PATH = input(
                    'Would you like to save the trained model? If so, type in a save path, '
                    'otherwise, interrupt with ctrl+c. ')
            except KeyboardInterrupt:
                print('\nExiting...')

        if SAVE_PATH is not None and not train.test:
            print('Saving...')
            os.makedirs(SAVE_PATH, exist_ok=True)

            discount_array = {}
            if hasattr(train, 'apple_deposited_array'):
                discount_array['apple_deposited_array'] = train.apple_deposited_array
            if hasattr(train, 'agent_starvation_array'):
                discount_array['agent_starvation_array'] = train.agent_starvation_array

            np.savez(os.path.join(SAVE_PATH, 'arrays.npz'),
                     reward_array=train.reward_array,
                     action_array=train.action_array,
                     apples_collected_array=train.apples_collected_array,
                     apple_stock_array=train.apple_stock_array, **discount_array)
            print('Saved.')
