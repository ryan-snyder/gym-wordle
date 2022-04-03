import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
from gym_wordle.wordle import Wordle
import pygame
import numpy as np
import os

class WordleEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    SCREEN_DIM = 500
    GREEN = "#6aaa64"
    YELLOW = "#c9b458"
    GREY = "#787c7e"
    OUTLINE = "#d3d6da"
    FILLED_OUTLINE = "#878a8c"

    def __init__(self, logging=False):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.logging = logging
        self.answers = pd.read_csv('{}/wordle-answers-alphabetical.txt'.format(current_dir), header=None, names=['words'])
        self.screen = None
        self.isopen = False
        self.GUESSES = 6
        self.LETTERS = 5
        self.WORD = self.answers['words'].sample(n=1).tolist()[0].upper()
        self.WORDLE = Wordle(self.WORD, self.GUESSES, self.LETTERS)
        self.alpha = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
        self.colors = ['B', 'Y', 'G']
        self.is_game_over = False
        self.guessed_words = []
        self.blank_letters = []

        file_names = ['{}/wordle-allowed-guesses.txt'.format(current_dir), '{}/wordle-answers-alphabetical.txt'.format(current_dir)]
        self.word_bank = pd.concat((pd.read_csv(f, header=None, names=['words']) for f in file_names), ignore_index=True).sort_values('words')['words'].tolist()
        # our action space is the total amount of possible words to guess
        self.action_space = spaces.Discrete(12972)
        #our observation space is the current wordle board in form of (letter, color) with 5x6 (5 letters, 6 guesses)
        #modified to work with gym/baselines
        #same thing basically, only 0-26 is '' to z and 27-29 is B, Y, G
        # first 6 rows are guesses and last 6 rows are colors
        # changed shape to be 3 dimensions so that we can apply conv2d layers to it
        # at some point we should try to normalize the obs space
        # since right now its on a 0-29 scale instead of a 0-1.
        self.observation_space = spaces.Box(low=0, high=29, shape=(1,12,5), dtype='int32')
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
        self.current_episode = -1
        self.episode_memory.append([])
        self.is_game_over = False
        self.WORD = self.answers['words'].sample(n=1).tolist()[0].upper()
        self.WORDLE = Wordle(self.WORD, self.GUESSES, self.LETTERS)
        self.guessed_words = []
        self.blank_letters = []
        if self.logging:
            print(self.WORDLE.word)
        self.close()
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
        guess = self.word_bank[action]
        self.episode_memory[self.current_episode].append(guess)
        self.guessed_words.append(guess)
        if self.logging:
            print(guess)
        self.WORDLE.update_board(guess)
        res = self.WORDLE.colours[self.WORDLE.g_count-1]
        self.blank_letters.extend([ l for i,l in enumerate(guess) if res[i] == 'B' and l not in self.blank_letters])
        self.is_game_over = self.WORDLE.word == guess or self.WORDLE.g_count == self.GUESSES

    def _get_reward(self):
        result, tries = self.WORDLE.game_result()
        rewards = np.zeros(5)
        #heavily penealize guessing the same word multiple times
        #If a word isn't the right guess, we shouldn't guess it again
        #could do the same thing for letters, as if a letter is blank(grey)
        # then the only reason to use a word with a letter in it
        # is to check other letter posistions
        #so it shouldn't be a heavy penalty but it should be a penalty
        for i,c in enumerate(self.WORDLE.colours[self.WORDLE.g_count-1]):
            if c == self.colors[2]:
                rewards[i] = 2
            elif c == self.colors[1]:
                rewards[i] = 1
        #check guesses up to and including our current guess
        if self.logging:
            print(self.WORD)
            print(rewards)
        reward = np.mean(rewards)
        for g in range(self.WORDLE.g_count):
            word = self.WORDLE.board[g]
            current = ''.join(word)
            if current in self.guessed_words:
                return 0
            for l in word: 
                if l in self.blank_letters:
                    reward -= 0.3
        return reward

    def _get_observation(self):
        board = np.array(self.WORDLE.board) #2d array of 5x6
        colors = np.array(self.WORDLE.colours) #2d array of 5x6
        results = np.vstack((board, colors)) #stacks boards and colors by rows resulting in a 2d array of 5x12
        convertletterstonum = lambda letter: [self.alpha.index(l) + 1 if l in self.alpha else 0 for l in letter]
        convertcolortonum = lambda color: [self.colors.index(c)+27 for c in color]
        guesses = np.array([convertletterstonum(l) if i <=5 else convertcolortonum(l) for i, l in enumerate(results)])
        guesses3d = np.expand_dims(guesses, axis=0)
        if self.logging:
            print(np.shape(guesses))
            print(np.shape(guesses3d))
        return guesses3d
