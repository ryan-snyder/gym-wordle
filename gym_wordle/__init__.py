from gym.envs.registration import register

register(
    id='Wordle-v0',
    entry_point='gym_wordle.envs:WordleEnv',
)