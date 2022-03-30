from gym.envs.registration import register

__version__ = 'develop'
register(
    id='Wordle-v0',
    entry_point='gym_wordle.envs:WordleEnv',
)