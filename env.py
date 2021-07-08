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
  """converts an action code to grid form.

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
      board = np.zeros((BOARD_LENGTH, BOARD_LENGTH))
      self.state = (board, first)

    def set_player(self, player):
      assert player in [X,O], 'Attempted to set player to invalid value.'
      board, _ = self.state
      self.state = (board, player)

    def get_obs(self):
      board, player = self.state
      return board.copy(), player

    def step(self, action):
      reward = 0
      info = None

      #uncouple the state
      board, player = self.state
      action_row, action_col = action_to_grid(action)

      #check if square is empty, if not return the current state which prompts them to go again
      if board[action_row, action_col] != EMPTY:
        print('uh oh, illegal move, try again!')
        return self.get_obs(), 0, False, None

      board[action_row, action_col] = player
      self.state = (board, -player)

      #get game status
      status = check_game_status(board)

      #check if game is over
      if status != IN_PROGRESS:
        print('game over', status, 'wins')
        return self.get_obs(), reward, True, info

      return self.get_obs(), reward, False, info

    def render(self):
      board, player = self.state
      print("it is {}'s turn".format(mark_dict[player]))
      # for idx, row in enumerate(board):
      #   print(mark_dict[row[0]], '|', mark_dict[row[1]], '|', mark_dict[row[2]])
      #   if idx < BOARD_LENGTH - 1:
      #     print('-----------')
      for row, col in product(np.arange(BOARD_LENGTH), np.arange(BOARD_LENGTH)):
        plt.text(1/8 + col*3/8, 7/8 - row*3/8,mark_dict[board[row, col]] , size = 72, ha = 'center', va = 'center')

    def reset(self, first = X):
      self.state = (np.zeros((3,3)), first)


     






    
