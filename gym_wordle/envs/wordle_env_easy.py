import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
from gym_wordle.wordle import Wordle
import pygame
import numpy as np
import os

class WordleEnvEasy(gym.Env):
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
        self.rewards = []
        self.vowels = ['A','E','I','O','U']
        file_names = ['{}/wordle-answers-alphabetical.txt'.format(current_dir)]
        self.word_bank = pd.concat((pd.read_csv(f, header=None, names=['words']) for f in file_names), ignore_index=True).sort_values('words')
        self.word_bank.loc[:,'v-count'] = self.word_bank.loc[:,'words'].str.lower().str.count(r'[aeiou]') #Count amount of vowels in words
        # our action space is the total amount of possible words to guess
        self.w_bank = self.word_bank
        #self.action_space = spaces.Discrete(2315)
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
            'guesses': spaces.Box(low=0, high=26, shape=(6,5), dtype='int32'),
            'colors': spaces.Box(low=0, high=2, shape=(6,5), dtype='int32')
        })
        self.current_episode = -1
        self.episode_memory: List[Any] = []

    def step(self, action):
        if self.is_game_over:
            return RuntimeError('Episode is already done')
        self.word_score()
        self._take_action(action)
        reward = self._get_reward()
        self.rewards.append(reward)
        observation = self._get_observation()
        if self.word_bank['words'].to_list()[self.current_guess[0]] == self.WORD.lower():
            print(self.rewards)
            print(np.mean(np.array(self.rewards)))
        return observation, reward, self.is_game_over, {}

    def reset(self):
        self.current_episode = -1
        self.episode_memory.append([])
        self.is_game_over = False
        self.WORD = self.answers.loc[:,'words'].sample(n=1).tolist()[0].upper()
        self.WORDLE = Wordle(self.WORD, self.GUESSES, self.LETTERS)
        self.guessed_words = []
        self.blank_letters = []
        self.rewards = []
        self.g_letters = []
        self.y_letters = {}
        self.w_bank = self.word_bank
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
        print(action)
        guess = self.word_bank['words'].to_list()[action[0]]
        self.episode_memory[self.current_episode].append(guess)
        self.guessed_words.append(guess)
        self.current_guess = action
        if self.logging:
            print(guess)
        self.WORDLE.update_board(guess)
        res = self.WORDLE.colours[self.WORDLE.g_count-1]
        self.blank_letters.extend([ l for i,l in enumerate(guess) if res[i] == 'B' and l not in self.blank_letters])
        if self.WORDLE.word.lower() == guess:
            print('~~~~~~AGENT GOT IT RIGHT~~~~~~')
        self.is_game_over = self.WORDLE.word.lower() == guess or self.WORDLE.g_count == self.GUESSES
    def calc_letter_probs(self):
        for x in range(self.WORDLE.letters):
            counts = self.w_bank.loc[:, ('words')].str[x].value_counts(normalize=True).to_dict()
            self.w_bank.loc[:, (f'p-{x}')] = self.w_bank.loc[:, ('words')].str[x].map(counts)
    def parse_board(self):
        self.g_letters = []
        self.y_letters = {}
        if self.WORDLE.g_count > 0:
            g_hold = []
            for x, c in enumerate(self.WORDLE.colours[self.WORDLE.g_count - 1]):
                letter = self.WORDLE.board[self.WORDLE.g_count - 1][x]
                if c == 'Y':
                    if letter not in self.y_letters:
                        self.y_letters[letter] = [x]
                    else:
                        if x not in self.y_letters[letter]:
                            self.y_letters[letter].append(x)
                elif c == 'G':
                    self.prediction[x] = letter
                else:
                    if letter in self.prediction:
                        if letter not in self.y_letters:
                            self.y_letters[letter] = [x]
                        else:
                            self.y_letters[letter].append(x)
                    elif letter not in self.g_letters:
                        self.g_letters.append(letter)
            self.g_letters = [l for l in self.g_letters if l not in self.y_letters and l not in self.prediction]
    def word_score(self):
        self.calc_letter_probs()
        self.prediction = ['' for _ in range(self.WORDLE.letters)]
        self.parse_board()
        if len(self.g_letters) > 0:
            self.w_bank = self.w_bank.loc[~self.w_bank['words'].str.contains('|'.join(self.g_letters).lower())]
            self.g_letters = []
        if len(self.y_letters) > 0:
            y_str = '^' + ''.join(fr'(?=.*{l})' for l in self.y_letters)
            self.w_bank = self.w_bank.loc[self.w_bank['words'].str.contains(y_str.lower())]
            for s, p in self.y_letters.items():
                for i in p:
                    self.w_bank = self.w_bank.loc[self.w_bank['words'].str[i]!=s.lower()]
            self.y_letters = {}
        for i, s in enumerate(self.prediction):
            if s != '':
                self.w_bank = self.w_bank.loc[self.w_bank['words'].str[i]==s.lower()]
        self.w_bank.loc[:, ('w-score')] = 0
        if len(self.w_bank) > 5:
            self.calc_letter_probs() #Recalculate letter position probability
        for x in range(self.WORDLE.letters):
            if self.prediction[x] == '':
                self.w_bank.loc[:, ('w-score')] += self.w_bank[f'p-{x}']
        
        if True not in [True for s in self.prediction if s in self.vowels]:
            self.w_bank.loc[:, ('w-score')] += self.w_bank.loc[:, ('v-count')] / self.WORDLE.letters
    def _get_reward(self):
        if self.WORDLE.g_count > 1:
            self.word_score()
        guess = self.word_bank['words'].to_list()[self.current_guess[0]]
        new_reward = np.nan_to_num(self.w_bank.loc[self.current_guess[0], 'w-score']) if guess in self.w_bank.values else 0
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
        reward = np.mean(rewards)
        for g in range(self.WORDLE.g_count):
            word = self.WORDLE.board[g]
            current = ''.join(word)
            if current in self.guessed_words:
                return 0
            for l in word: 
                if l in self.blank_letters:
                    reward -= 0.5
        if self.logging:
            print(self.WORD)
            print(rewards)
            print(new_reward)
        new_reward += 30 - (tries*5) if guess == self.WORD.lower() else 0
        return new_reward

    def _get_observation(self):
        board = np.array(self.WORDLE.board) #2d array of 5x6
        colors = np.array(self.WORDLE.colours) #2d array of 5x6
        results = np.vstack((board, colors)) #stacks boards and colors by rows resulting in a 2d array of 5x12
        convertletterstonum = lambda letter: [self.alpha.index(l) + 1 if l in self.alpha else 0 for l in letter]
        convertcolortonum = lambda color: [self.colors.index(c) for c in color]
        guesses = np.array([convertletterstonum(l) for l in board])
        colors = np.array([convertcolortonum(c) for c in colors])
        #guesses3d = np.expand_dims(guesses, axis=0)
        return { 'guesses': guesses, 'colors': colors }
