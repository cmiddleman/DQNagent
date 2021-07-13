import numpy as np 
import matplotlib.pyplot as plt 
import gym
from gym import spaces
from itertools import product


X = 1
O = -1
EMPTY = 0
BOARD_LENGTH = 3

#Status codes
DRAW = 0
IN_PROGRESS = -2

#Reward map
rewards = {
  'win' : 1,
  'lose' : -1,
  'draw' : .1,
  'in_progress' : 0
}

def get_available_actions_mask(board):
  """Finds available actions given a board state.

  Args:
      board (2D numpy array): the board state.

  Returns:
      1D numpy array: mask of available actions as flattened numpy boolean array.
  """

  assert board.shape == (BOARD_LENGTH, BOARD_LENGTH), 'invalid board was passed in to method'
  
  return board.flatten() == EMPTY

def check_game_status(board):
  """Checks the status/winner of the passed in board

  Args:
      board (2D numpy array): the board to check status of (must be of shape (@BOARD_LENGTH, @BOARD_LENGTH))

  Returns:
      int: status code to relay board status info.
  """

  assert board.shape == (BOARD_LENGTH, BOARD_LENGTH), 'invalid board was passed in to status method'

  #check if either X or O won
  for mark in [X,O]:
    #check if winner along rows
    if (board == mark).all(axis = 1).any():
      return mark
    
    #check if winner along columns
    if (board == mark).all(axis=0).any():
      return mark

    #check if winner along diagonals
    if (board[np.arange(BOARD_LENGTH), np.arange(BOARD_LENGTH)] == mark).all():
      return mark
    if (board[np.arange(BOARD_LENGTH), np.arange(BOARD_LENGTH)[::-1]] == mark).all():
      return mark
  
  #check if draw
  if (board != EMPTY).all():
    return DRAW

  #otherwise game still in progress
  return IN_PROGRESS

def action_to_grid(action):
  """converts an action key to grid form.

  Args:
      action (int): the integer code of the action in row then column box order

  Returns:
      row, col: the row and column corresponding to the action
  """
  return action // BOARD_LENGTH, action % BOARD_LENGTH

mark_dict = {
  X : 'X',
  O : 'O',
  EMPTY: ' '
}

class TicTacToe(gym.Env):

    def __init__(self, first = X):
     
      self.observation_space = spaces.Discrete(BOARD_LENGTH**2)
      self.action_space = spaces.Discrete(BOARD_LENGTH**2)
      self.reset(first)
    

    def set_player(self, player):
      assert player in [X,O], 'Attempted to set player to invalid value.'
      board, _ = self.state
      self.state = (board, player)

    def get_obs(self):
      board, player = self.state
      return board.copy(), player

    def step(self, action):
      """Makes environment take a step according to the action passed in.

      Args:
          action (int): integer coded action for which square the current player should go (in this case 0-8)

      Returns:
          observation, reward, done, info: the state after the action has been made, the corresponding reward, whether the state is terminal, debugging info.
      """
      reward = 0
      info = None

      #uncouple the state
      board, player = self.state
      action_row, action_col = action_to_grid(action)

      #check if square is empty, if not return the same state (with no reward, should only happen for human agents)
      if board[action_row, action_col] != EMPTY:
        print('uh oh, illegal move, try again!')
        return self.get_obs(), 0, self.done, info

      board[action_row, action_col] = player
      self.state = (board, -player)

      #get game status
      status = check_game_status(board)

      #check if game is over
      if status != IN_PROGRESS:
        self.done = True
        print('game over', status, 'wins')
        reward = status
        #TODO make sure reward is functioning properly

      return self.get_obs(), reward, self.done, info

    def render(self):
      board, player = self.state
      print("it is {}'s turn".format(mark_dict[player]))
      # for idx, row in enumerate(board):
      #   print(mark_dict[row[0]], '|', mark_dict[row[1]], '|', mark_dict[row[2]])
      #   if idx < BOARD_LENGTH - 1:
      #     print('-----------')
      for row, col in product(np.arange(BOARD_LENGTH), np.arange(BOARD_LENGTH)):
        plt.text(1/8 + col*3/8, 7/8 - row*3/8,mark_dict[board[row, col]] , size = 72, ha = 'center', va = 'center')
      plt.show()

    def reset(self, first = X):
      self.done = False
      self.state = (np.zeros((BOARD_LENGTH,BOARD_LENGTH)), first)

class SuperTicTacToe(gym.Env):
  pass
     






    
