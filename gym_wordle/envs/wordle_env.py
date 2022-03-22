import gym
from gym import error, spaces, utils
from gym.utils import seeding
import panda as pd
from gym_wordle.wordle import Wordle

class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    answers = pd.read_csv('wordle-answers-alphabetical.txt', header=None, names=['words'])

    def __init__(self):
        # TODO
        self.GUESSES = 6
        self.LETTERS = 5
        self.WORD = answers['words'].sample(n=1).tolist()[0].upper()
        self.WORDLE = Worlde(self.WORD, self.GUESSES, self.LETTERS)
        self.is_game_over = False

        file_names = ['wordle-allowed-guesses.txt', 'wordle-answers-alphabetical.txt']
        self.word_bank = pd.concat(
            (pd.read_csv(f, header=None, names=['words']) for f in file_names), ignore_index=True).sort_values('words').tolist()
        # our action space is the total amount of possible words to guess
        self.action_space = spaces.Discrete(12972)
        #our observation space is the current wordle board in form of (letter, color) with 5x6 (5 letters, 6 guesses)
        self.observation_space = spaces.Box(0, 90, (5,6))


        ...
    def step(self, action):
        # TODO
        ...
    def reset(self):
        # TODO
        ...
    def render(self, mode='human'):
        # TODO
        ...
    def close(self):
        # TODO
        ...