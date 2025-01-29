from gym.envs.registration import register

register(
    id='HarvestEnv-v1',
    entry_point='social_dilemma.envs:HarvestEnv',
)
