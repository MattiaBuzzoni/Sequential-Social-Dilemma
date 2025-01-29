import shutil
import gym
import numpy as np
from pathlib import Path


num_agents = 3

env = gym.make('social_dilemma:HarvestEnv-v1', num_agents=num_agents, visual_radius=15,
               threshold=0, max_stock=0, discount_state=False)

env.reset()
try:
    for t in range(1000):
        n_actions = np.random.randint(low=0, high=env.action_space.n, size=(num_agents,)).tolist()
        n_observations, _, n_rewards, n_done, n_info = env.step(n_actions)
        env.render()
except KeyboardInterrupt:
    shutil.rmtree(Path(__file__).resolve().parents[0] / 'images')
    print("Example Interrupted!")