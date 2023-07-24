import gym
import numpy as np
import sys

from stable_baselines import logger

sys.path.append('/home/code/personal/othello/othellobackend/OthelloAB.py') 

from OthelloAB import findPossibleMoves, newBoardState

class Player():
    def __init__(self, id, token):
        self.id = id
        self.token = token
        

class Token():
    def __init__(self, symbol, number):
        self.number = number
        self.symbol = symbol

class OthelloEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False) -> None:
        super(OthelloEnv, self).__init__()
        self.name = 'othello'
        self.manual = manual

        self.grid_length = 8
        self.n_players = 2
        self.num_squares = self.grid_length ** 2
        self.grid_shape = (self.grid_length, self.grid_length)
        self.action_space = gym.spaces.Discrete(self.num_squares)
        self.observation_space = gym.spaces.Box(-1, 1, self.grid_shape+(2,))
        self.verbose = verbose

    @property
    def observation(self):
        if self.players[self.current_player_num].token.number == 1:
            position = np.array([x.number for x in self.board]).reshape(self.grid_shape)
        else:
            position = np.array([-x.number for x in self.board]).reshape(self.grid_shape)

        la_grid = np.array(self.legal_actions).reshape(self.grid_shape)
        out = np.stack([position,la_grid], axis = -1)
        return out
    
    @property
    def legal_actions(self):
        legal_actions = []
        moves = self.legal_moves()
        for i in range(self.grid_length ** 2):
            if i in moves:
                legal_actions.append(1)
            else:
                legal_actions.append(0)
        return np.array(legal_actions)
    
    def reset(self):
        self.board = [Token('.', 0)] * self.num_squares
        self.players = [Player('1', Token('x', 1)), Player('2', Token('o', -1))]
        self.current_player_num = 0
        self.turns_taken = 0
        self.done = False
        logger.debug(f'\n\n---- NEW GAME ----')
        
        return self.observation
    
    def legal_moves(self, player, board):
        return set(findPossibleMoves(''.join([x.symbol for x in self.board]), player.token.symbol))
    
    def score_reward(self, board, player):
        score = [0, 0]
        for token in board:
            if token.symbol == 'x':
                score[0] += 1
            elif token.symbol == 'o':
                score[1] += 1
        if player.id == 1:
            if score[0] > score[1]:
                return 1
            elif score[0] < score[1]:
                return -1
            else:
                return 0
        elif player.id == 2:
            if score[0] > score[1]:
                return -1
            elif score[0] < score[1]:
                return 1
            else:
                return 0

    def step(self, action):

        reward = [0, 0]

        board = self.board

        if action not in self.legal_moves(self.players[self.current_player_num], board):
            done = True
            reward = [1, 1]
            reward[self.current_player_num] = -1
        else:
            board = newBoardState(''.join([x.symbol for x in board]), self.players[self.current_player_num].token.symbol, action)
            self.turns_taken += 1
            if len(self.legal_moves(self.players[(self.current_player_num + 1) % 2], board)) == 0:
                if len(self.legal_moves(self.players[self.current_player_num], board)) == 0:
                    done = True
                    r = self.score_reward(board, self.players[self.current_player_num])
                    reward = [-r, -r]
                    reward[self.current_player_num] = r
                    self.done = done

                    return self.observation, reward, done, {}
                else:
                    done = False
                    reward = [0, 0]
                    return self.observation, reward, done, {}
        self.done = done

        if not done:
            self.current_player_num = (self.current_player_num + 1) % 2

        return self.observation, reward, done, {}
    
    def render(self, mode='human', close=False, verbose = True):
        logger.debug('')
        if close:
            return
        if self.done:
            logger.debug(f'GAME OVER')
        else:
            logger.debug(f"It is Player {self.current_player.id}'s turn to move")
            
        board_string = ''.join([x.symbol for x in self.board])
        for i in range(self.grid_length):
            logger.debug(board_string[i * self.grid_length: (i + 1) * self.grid_length])

        if self.verbose:
            logger.debug(f'\nObservation: \n{self.observation}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')

