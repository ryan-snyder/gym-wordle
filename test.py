import gym
import os
os.environ['SDL_VIDEODRIVER']='dummy'

import gym_wordle

env = gym.make('WordleEasy-v0', logging=True)
env.reset()
print('Playing one game of wordle, with random action')
while not env.is_game_over:
    env.render()
    env.step(env.action_space.sample())
    print(env.action_masks())
env.close()