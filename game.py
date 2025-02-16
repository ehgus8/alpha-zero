from abc import ABC, abstractmethod
import numpy as np

class Game(ABC):

    @staticmethod
    @abstractmethod
    def display_board(board: np.array):
        pass

    @staticmethod
    @abstractmethod
    def get_action_idx(action: tuple[int, int]):
        """
        Returns:
            action index (int): index of action flattened to 1D
        """
        pass

    @staticmethod
    @abstractmethod
    def get_valid_moves(board):
        pass

    @staticmethod
    @abstractmethod
    def make_move(board: np.array, current_player: int, action):
        pass

    @staticmethod
    @abstractmethod
    def undo_move(board: np.array, current_player: int, action):
        pass

    @staticmethod
    @abstractmethod
    def check_winner(board: np.array, player, action):
        pass



        