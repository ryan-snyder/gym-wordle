import gym

import gym_wordle

env = gym.make('Wordle-v0')
env.reset()
print('Playing one game of wordle, with random action')
for _ in range(6):
    env.render()
    env.step(env.action_space.sample())
env.close()