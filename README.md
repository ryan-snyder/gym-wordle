# gym-wordle
A wordle environment for openai/gym

# Installation
Install [openai-gym](https://gym.openai.com/docs/)

then install this package with
`pip install -e .`

# Usage
```
import gym 
import gym_wordle
env = gym.make('Wordle-v0')
```
See the [docs](https://gym.openai.com/docs/) for more info

# Environment details

This environment simulates a game of wordle using a wordle python clone from https://github.com/bellerb/wordle_solver

The action space is a discrete space of 12972 numbers which corresponds to a word from a list of all allowed wordle guesses and answers
The observation space is a dict with the guesses and colors for the current game.
Guesses is an array of `shape(5,6) #5 letters and 6 rows` where each element is a number from 0-26 where 0 is `''` and 26 is `z`
Colors is an array of the same shape, only each element is a number from 0-2 where 0 is a blank (or black or grey) square, 1 is a yellow square, and 2 is a green square

The reward calculation is as follows: 

The agent gets 1-6 points depending on how fast the agent guesses the word. For example, getting the word on the first guesses rewards 6 points, getting the word on the second guess rewards 5 points, etc

The agent also is rewarded for colors in the current row (so the current guess). 
Right now, the agent is rewarded 3 points for each green tile, and 1 point for each yellow tile. 
No points are giving for grey tiles
