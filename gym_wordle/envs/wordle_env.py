import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
from gym_wordle.wordle import Wordle
import os

class WordleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        print(current_dir, flush=True)
        self.answers = pd.read_csv('{}/wordle-answers-alphabetical.txt'.format(current_dir), header=None, names=['words'])
        self.GUESSES = 6
        self.LETTERS = 5
        self.WORD = self.answers['words'].sample(n=1).tolist()[0].upper()
        self.WORDLE = Wordle(self.WORD, self.GUESSES, self.LETTERS)
        self.alpha = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        self.colors = ['B', 'Y', 'G']
        self.is_game_over = False

        file_names = ['{}/wordle-allowed-guesses.txt'.format(current_dir), '{}/wordle-answers-alphabetical.txt'.format(current_dir)]
        print(file_names)
        self.word_bank = pd.concat((pd.read_csv(f, header=None, names=['words']) for f in file_names), ignore_index=True).sort_values('words')['words'].tolist()
        # our action space is the total amount of possible words to guess
        self.action_space = spaces.Discrete(12972)
        #our observation space is the current wordle board in form of (letter, color) with 5x6 (5 letters, 6 guesses)
        #modified to work with gym/baselines
        #same thing basically, only 0-26 is '' to z and 27-29 is B, Y, G
        # first 6 rows are guesses and last 6 rows are colors
        self.observation_space = spaces.Box(low=0, high=29, shape=(5,12), dtype='int32')
        self.current_episode = -1
        self.episode_memory: List[Any] = []

    def step(self, action):
        if self.is_game_over:
            return RuntimeError('Episode is already done')
        self._take_action(action)
        reward = self._get_reward()
        observation = self._get_observation()
        return observation, reward, self.is_game_over, {}

    def reset(self):
        # TODO
        self.current_episode = -1
        self.episode_memory.append([])
        self.is_game_over = False
        self.WORD = self.answers['words'].sample(n=1).tolist()[0].upper()
        print('Current word is ', self.WORD)
        self.WORDLE = Wordle(self.WORD, self.GUESSES, self.LETTERS)
        return self._get_observation()

    def render(self, mode='human'):
        return None
    def close(self):
        # TODO
        ...

    def _take_action(self, action):
        # turn action into guess
        guess = self.word_bank[action]
        self.episode_memory[self.current_episode].append(guess)

        self.WORDLE.update_board(guess)
        self.is_game_over = self.WORDLE.word == guess or self.WORDLE.g_count == self.GUESSES

    def _get_reward(self):
        result, tries = self.WORDLE.game_result()
        reward = 0
        reward += 6 - tries if tries <=6 else 0
        for c in self.WORDLE.colours[self.WORDLE.g_count-1]:
            if c == self.colors[2]:
                reward += 3
            elif c == self.colors[1]:
                reward += 1
        return reward

    def _get_observation(self):
        board = self.WORDLE.board
        results = self.WORDLE.colours
        print(board)
        print(results)
        convertlettertonum = lambda l: self.alpha.index(l) + 1 if l in self.alpha else 0
        guesses = [ [convertlettertonum(l) for l in r] for r in board]
        convertcolortonum = lambda color: [self.colors.index(c)+27 for c in color]
        colors = [convertcolortonum(r) for r in results]
        return guesses.extend(colors)