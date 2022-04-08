import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
from gym_wordle.wordle import Wordle
import pygame
import numpy as np
import os
import gym_wordle.envs.state as state

class WordleEnvEasy(gym.Env):
    metadata = {'render.modes': ['human']}
    SCREEN_DIM = 500
    GREEN = "#6aaa64"
    YELLOW = "#c9b458"
    GREY = "#787c7e"
    OUTLINE = "#d3d6da"
    FILLED_OUTLINE = "#878a8c"

    def __init__(self, logging=False, action_type='Discrete', words=100, random=False):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.logging = logging
        self.random = random
        self.screen = None
        self.isopen = False
        self.GUESSES = 6
        self.LETTERS = 5
        if random:
            print('using random solution set and random solution, of size: ', words)
            self.answers = pd.read_csv('{}/wordle-answers-alphabetical.txt'.format(current_dir), header=None, names=['words']).sample(n=words, ignore_index=True)
            self.WORD = self.answers['words'].sample(n=1).tolist()[0].upper()
        else:
            print('using same solution set and random solution, of size: ', words)
            self.answers = pd.read_csv('{}/wordle-answers-alphabetical.txt'.format(current_dir), header=None, names=['words']).sort_values('words').head(words)
            self.WORD = self.answers['words'].sample(n=1).tolist()[0].upper()
        self.WORDLE = Wordle(self.WORD, self.GUESSES, self.LETTERS)
        self.alpha = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        self.colors = ['B', 'Y', 'G']
        self.is_game_over = False
        self.guessed_words = []
        self.blank_letters = []
        self.rewards = []
        file_names = ['{}/wordle-answers-alphabetical.txt'.format(current_dir)]
        #self.word_bank = pd.concat((pd.read_csv(f, header=None, names=['words']) for f in file_names), ignore_index=True).sort_values('words')
        self.word_bank = self.answers
        # our action space is the total amount of possible words to guess
        if action_type == 'Discrete':
            self.action_space = spaces.Discrete(words)
        else:
        # TODO: change this to provide x guesses, and choose the one which scores the highest?
        # or is that too forgiving
            self.action_space = spaces.Box(low=0, high=2315, shape=(1,), dtype='int32')
        #our observation space is the current wordle board in form of (letter, color) with 5x6 (5 letters, 6 guesses)
        #modified to work with gym/baselines
        #same thing basically, only 0-26 is '' to z and 27-29 is B, Y, G
        # first 6 rows are guesses and last 6 rows are colors
        # changed shape to be 3 dimensions so that we can apply conv2d layers to it
        # at some point we should try to normalize the obs space
        # since right now its on a 0-29 scale instead of a 0-1.
        #self.observation_space = spaces.Box(low=0, high=29, shape=(1,12,5), dtype='int32')
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=0, high=6, shape=(417,), dtype='int32'),
            'achieved_goal': spaces.Box(low=27, high=29, shape=(5,), dtype='int32'),
            'desired_goal': spaces.Box(low=29, high=29, shape=(5,), dtype='int32')
        })
        self.current_episode = -1
        self.episode_memory: List[Any] = []
        self.state: state.WordleState = None
        self.state_updater = state.update

    def step(self, action):
        if self.is_game_over:
            return RuntimeError('Episode is already done')
        self._take_action(action)
        reward = 0
        observation = self._get_observation()
        if self.word_bank['words'].to_list()[self.current_guess] == self.WORD.lower():
            self.is_game_over = True
        elif self.WORDLE.g_count == self.GUESSES:
            self.is_game_over = True
        reward += self._get_reward()
        self.rewards.append(reward)
        if self.logging:
            print(self.WORD)
            print(self.guessed_words)
            print(self.rewards)
        return observation, reward, self.is_game_over, {}

    def reset(self):
        self.current_episode = -1
        self.episode_memory.append([])
        self.is_game_over = False
        if self.random:
            self.WORD = self.answers.loc[:,'words'].sample(n=1).tolist()[0].upper()
        else:
            self.WORD = self.answers['words'].sample(n=1).tolist()[0].upper()
        self.WORDLE = Wordle(self.WORD, self.GUESSES, self.LETTERS)
        self.guessed_words = []
        self.blank_letters = []
        self.rewards = []
        if self.logging:
            print(self.WORDLE.word)
        self.close()
        self.state = state.new(self.GUESSES)
        return self._get_observation()

    def render(self, mode='human'):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.SCREEN_DIM, self.SCREEN_DIM))
        font = pygame.font.Font('freesansbold.ttf', 30)
        for col in range(0, 5):
            for row in range(0, 6):
                pygame.draw.rect(self.screen, self.OUTLINE, [col * 100 + 12, row * 100 + 12, 75, 75], 3, 5)
                color = self.GREEN if self.WORDLE.colours[row][col] == 'G' else self.YELLOW if self.WORDLE.colours[row][col] == 'Y' else self.GREY
                piece_text = font.render(self.WORDLE.board[row][col], True, color)
                self.screen.blit(piece_text, (col * 100 + 30, row * 100 + 25))
        #pygame.draw.rect(screen, self.GREEN, [5, turn * 100 + 5, WIDTH - 10, 90], 3, 5)
        if mode == "human":
            pygame.event.pump()
            pygame.display.flip()             
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def _take_action(self, action):
        # turn action into guess
        guess = self.word_bank['words'].to_list()[action]
        self.episode_memory[self.current_episode].append(guess)
        self.guessed_words.append(guess)
        self.current_guess = action
        self.WORDLE.update_board(guess)
        res = self.WORDLE.colours[self.WORDLE.g_count-1]
        for i,l in enumerate(guess):
            if res[i] == 'B' and l not in self.blank_letters:
                self.blank_letters.append(l)
        if self.WORDLE.word.lower() == guess:
            print('~~~~~~AGENT GOT IT RIGHT~~~~~~')
            print(self.guessed_words)
    def _get_reward(self):
        new_reward = 0
        if self.word_bank['words'].to_list()[self.current_guess] == self.WORD.lower():
            if self.WORDLE.g_count > 1:
                new_reward = 10
            else: new_reward = -20
        elif self.WORDLE.g_count == self.GUESSES:
            new_reward = -10
        return new_reward
    def action_masks(self):
        action_mask = [w in self.guessed_words for w in self.word_bank['words'].tolist()]
        #action_mask = [self.guessed_words[key]['action'] for key in self.guessed_words.keys()]
        return action_mask
    # TODO: adjust get reward and compute reward to take into account the desired goal
    # But i think this is fine for right now, since our _get_reward does take into account our desired goal
    def compute_reward(self, achieved_goal, desired_goal, info):
        rewards = np.zeros((len(achieved_goal)))
        for i,result in enumerate(achieved_goal):
            for goal in desired_goal:
                adjusted = 10
                for g in range(len(goal)):
                    adjusted -= goal[g] - result[g]
                    rewards[i] = adjusted
        return np.mean(rewards)

    def _get_observation(self):
        board = np.array(self.WORDLE.board) #2d array of 5x6
        colors = np.array(self.WORDLE.colours) #2d array of 5x6
        results = np.vstack((board, colors)) #stacks boards and colors by rows resulting in a 2d array of 5x12
        convertletterstonum = lambda letter: [self.alpha.index(l) + 1 if l in self.alpha else 0 for l in letter]
        convertcolortonum = lambda color: [self.colors.index(c)+27 for c in color]
        guesses = np.array([convertletterstonum(l) if i <=5 else convertcolortonum(l) for i, l in enumerate(results)])
        colors = np.array(convertcolortonum(colors[self.WORDLE.g_count-1]))
        #guesses3d = np.expand_dims(guesses, axis=0)
        if self.state_updater != None and self.WORDLE.g_count != 0:
            return { 
                'observation': self.state_updater(state=self.state, word=self.guessed_words[self.WORDLE.g_count-1], goal_word=self.WORD.lower()), 
                'achieved_goal': colors, 'desired_goal': [29,29,29,29,29]
            }
        elif self.state_updater != None: 
            return {
                'observation': state.get_nvec(6),
                'achieved_goal': colors, 'desired_goal': [29,29,29,29,29]
            }
        return { 'observation': guesses, 'achieved_goal': colors, 'desired_goal': [29,29,29,29,29]}
