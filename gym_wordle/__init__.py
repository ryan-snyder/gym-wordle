from gym.envs.registration import register

__version__ = '0.1.7'
register(
    id='Wordle-v0',
    entry_point='gym_wordle.envs:WordleEnv',
)
register(
    id='WordleEasy-v0',
    entry_point='gym_wordle.envs:WordleEnvEasy'
)
