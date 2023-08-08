import gym
import numpy as np
import sys

from stable_baselines import logger


from othello.envs.othelloHelper import findPossibleMoves, newBoardState


class Player():
    def __init__(self, id, token):
        self.id = id
        self.token = token
        

class Token():
    def __init__(self, symbol, number):
        self.number = number
        self.symbol = symbol

    def __repr__(self):
        return self.symbol

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
        self.observation_space = gym.spaces.Box(-1, 1, self.grid_shape+(3,))
        self.verbose = verbose

    @property
    def observation(self):
        moves = self.legal_moves(self.players[self.current_player_num], self.board)
        if self.players[self.current_player_num].token.number == 1:
            position_1 = np.array([1 if x.number == 1 else 0  for x in self.board]).reshape(self.grid_shape)
            position_2 = np.array([1 if x.number == -1 else 0 for x in self.board]).reshape(self.grid_shape)
            position_3 = np.array([0 if i in moves else 1 for i,x in enumerate(self.board)]).reshape(self.grid_shape)
            # position = np.array([x.number for x in self.board]).reshape(self.grid_shape)
            logger.debug(f'My Pieces: {position_1}')
            logger.debug(f'Opponent Pieces: {position_2}')
            logger.debug(f'Legal Moves: {position_3}')
        else:
            position_1 = np.array([1 if x.number == -1 else 0 for x in self.board]).reshape(self.grid_shape)
            position_2 = np.array([1 if x.number == 1 else 0 for x in self.board]).reshape(self.grid_shape)
            position_3 = np.array([0 if i in moves else 1 for i,x in enumerate(self.board)]).reshape(self.grid_shape)
            # position = np.array([-x.number for x in self.board]).reshape(self.grid_shape)
            logger.debug(f'My Pieces: {position_1}')
            logger.debug(f'Opponent Pieces: {position_2}')
            logger.debug(f'Legal Moves: {position_3}')

        # la_grid = np.array(self.legal_actions).reshape(self.grid_shape)
        out = np.stack([position_1, position_2, position_3], axis = -1) 
        return out
    
    @property
    def legal_actions(self):
        legal_actions = []
        moves = self.legal_moves(self.players[self.current_player_num], self.board)
        for i in range(self.grid_length ** 2):
            if i in moves:
                legal_actions.append(1)
            else:
                legal_actions.append(0)
        return np.array(legal_actions)
    
    def reset(self):
        self.board = [Token('.', 0)] * self.num_squares
        self.board[27] = Token('o', -1)
        self.board[28] = Token('x', 1)
        self.board[35] = Token('x', 1)
        self.board[36] = Token('o', -1)
        self.players = [Player('1', Token('x', 1)), Player('2', Token('o', -1))]
        self.current_player_num = 0
        self.turns_taken = 0
        self.done = False
        logger.debug(f'\n\n---- NEW GAME ----')
        
        return self.observation
    
    def legal_moves(self, player, board):
        return set(findPossibleMoves(''.join([x.symbol for x in board]), player.token.symbol))
    
    def score_reward(self, board, player):
        score = [0, 0]
        for token in board:
            if token.symbol == 'x':
                score[0] += 1
            elif token.symbol == 'o':
                score[1] += 1
        if player.id == '1':
            if score[0] > score[1]:
                return 1
            elif score[0] < score[1]:
                return -1
            else:
                return 0
        elif player.id == '2':
            if score[0] > score[1]:
                return -1
            elif score[0] < score[1]:
                return 1
            else:
                return 0

    def step(self, action):

        reward = [0, 0]

        board = self.board
        done = None
        # logger.debug(f'\n\n---- NEW TURN ----')
        # logger.debug(f'Player {self.players[self.current_player_num].id} ({self.players[self.current_player_num].token.symbol})')
        # logger.debug(f'Action: {action}')
        # logger.debug(f'Board: {board}')
        if action not in self.legal_moves(self.players[self.current_player_num], board):
            done = True
            reward = [1, 1]
            reward[self.current_player_num] = -1
        else:
            board = newBoardState(''.join([x.symbol for x in board]), self.players[self.current_player_num].token.symbol, action)
            board = [Token('x', 1) if x == 'x' else Token('o', -1) if x == 'o' else Token('.', 0) for x in board]
            self.board = board
            self.turns_taken += 1
            if len(self.legal_moves(self.players[(self.current_player_num + 1) % 2], board)) == 0:
                if len(self.legal_moves(self.players[self.current_player_num], board)) == 0:
                    done = True
                    r = self.score_reward(board, self.players[self.current_player_num])
                    reward = [-r, -r]
                    reward[self.current_player_num] = r
                else:
                    done = False
                    reward = [0, 0]
            done = False
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
            logger.debug(f"It is Player {self.players[self.current_player_num].token}'s turn to move")
            
        board_string = ''.join([x.symbol for x in self.board])
        for i in range(self.grid_length):
            logger.debug(board_string[i * self.grid_length: (i + 1) * self.grid_length])

        if self.verbose:
            logger.debug(f'\nObservation: \n{self.observation}')
        
        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')

