import math
import numpy as np
import cv2

ACTIONS = ['NOOP', 'FIRE','UP','RIGHT','LEFT','DOWN','UPRIGHT','UPLEFT','DOWNRIGHT','DOWNLEFT','UPFIRE','RIGHTFIRE','LEFTFIRE','DOWNFIRE','UPRIGHTFIRE','UPLEFTFIRE','DOWNRIGHTFIRE','DOWNLEFTFIRE']

# this list is mostly needed to list the games in the same order everywhere
GAMES = ['spaceinvaders', 'qbert', 'mspacman', 'pinball','revenge']

# pretty titles for plots/tables
TITLES = {'spaceinvaders': 'Space Invaders',
          'qbert': 'Q*bert',
          'mspacman':'Ms. Pacman',
          'pinball':'Video Pinball',
          'revenge':'Montezumas\'s Revenge'
         }

def preprocess(frame, resize_shape=(84,84)):
    frame = cv2.resize(frame, resize_shape, interpolation=cv2.INTER_AREA)
    return frame[:, :, None]

def get_action_name(action_code):
  assert 0 <= action_code < len(ACTIONS), "%d is not the valid action index." % action_code
  return ACTIONS[action_code]

def get_action_code(action_name):
    assert action_name in ACTIONS, "%s is not the valid action name." % action_name
    return ACTIONS.index(action_name)

def env2game(name):
    ENVS = {'SpaceInvaders-v3': 'spaceinvaders', 
             'MsPacman-v3':'mspacman', 
             'VideoPinball-v3':'pinball',
             'MontezumaRevenge-v3':'revenge',
             'Qbert-v3':'qbert'
            }
    return ENVS[name]

def game2env(name):
    GAMES = {'spaceinvaders':'SpaceInvaders-v3', 
             'mspacman':'MsPacman-v3', 
             'pinball':'VideoPinball-v3',
             'revenge':'MontezumaRevenge-v3',
             'qbert':'Qbert-v3'
            }
    return GAMES[name]
