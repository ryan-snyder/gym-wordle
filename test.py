import gym
import os
import numpy as np
os.environ['SDL_VIDEODRIVER']='dummy'

import gym_wordle

env = gym.make('WordleEasy-v0', logging=True)
env.reset()
print('Playing one game of wordle, with random action')
while not env.is_game_over:
    #env.render()
    obs, _, _, _ = env.step(env.action_space.sample())
    print(np.shape(obs))
    env.action_masks()
    print(env.compute_reward(obs['achieved_goal'], obs['desired_goal'], ''))
env.close()