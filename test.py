import gym
import os
os.environ['SDL_VIDEODRIVER']='dummy'

import gym_wordle

env = gym.make('WordleEasy-v0', logging=True)
env.reset()
print('Playing one game of wordle, with random action')
for _ in range(6):
    env.render()
    env.step(env.action_space.sample())
env.close()