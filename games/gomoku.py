import numpy as np
import time
from .game import Game
from collections import deque
from ai import Node, MCTS
import utils


class Gomoku(Game):
    rows, cols = 7, 7
    action_dim = rows * cols
    state_dim = rows * cols
    feature_dim = 2
    logger = utils.get_game_logger('Gomoku')

    def __init__(self):
        self.board = np.zeros((Gomoku.feature_dim, Gomoku.rows, Gomoku.cols), dtype=np.float32)

    @staticmethod
    def display_board(board):
        display = np.full((Gomoku.rows, Gomoku.cols), ' ')
        display[board[0] == 1] = 'O'
        display[board[1] == 1] = 'X'
        # for row in display:
            # print(row)
        print("\n  ", end='')
        for c in range(Gomoku.cols):
            print(c, end=' ')
        print()
        for i, row in enumerate(display):
            print(i % 10, ' '.join(row))    
        print()

    @staticmethod
    def get_canonical_board(board: np.array, current_player):
        if current_player == 0:
            return board
        copied_board = np.empty_like(board)
        copied_board[0], copied_board[1] = board[1], board[0]
        return copied_board

    @staticmethod
    def get_action_idx(action: tuple[int, int]):
        """
        Returns:
            action index (int): index of action flattened to 1D
        """
        return action[0] * Gomoku.rows + action[1]

    @staticmethod
    def make_move(board, current_player, action):
        row, col = action
        if board[0, row, col] == 0 and board[1, row, col] == 0:
            board[current_player, row, col] = 1
            return 1 - current_player
        else:
            print("Invalid move. Try again.")
            return current_player

    @staticmethod
    def undo_move(board, current_player, action):
        row, col = action
        board[1 - current_player, row, col] = 0

    @staticmethod
    def check_winner(board, player, action):
        row, col = action
        def dfs(dr, dc):
            q = deque([(row, col)])
            visited = set()
            seq_count = 1
            while q:
                r, c = q.pop()
                visited.add((r, c))
                
                if 0 <= r + dr < Gomoku.rows and 0 <= c + dc < Gomoku.cols and (r + dr, c + dc) not in visited and board[player, r + dr, c + dc] == 1:
                    q.append((r + dr, c + dc))
                    seq_count += 1
                if 0 <= r - dr < Gomoku.rows and 0 <= c - dc < Gomoku.cols and (r - dr, c - dc) not in visited and board[player, r - dr, c - dc] == 1:
                    q.append((r - dr, c - dc))
                    seq_count += 1
                if seq_count >= 5:
                    return True
                
            return False
        
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            if dfs(dr, dc):
                return player
            
        return -1 # No winner
    
    @staticmethod
    def get_valid_moves(board):
        valid_moves = set()
        for r in range(Gomoku.rows):
            for c in range(Gomoku.cols):
                if board[0, r, c] == 1 or board[1, r, c] == 1:
                    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                        if 0 <= r + dr < Gomoku.rows and 0 <= c + dc < Gomoku.cols and board[0, r + dr, c + dc] == 0 and board[1, r + dr, c + dc] == 0:
                            valid_moves.add((r + dr, c + dc))
                        if 0 <= r - dr < Gomoku.rows and 0 <= c - dc < Gomoku.cols and board[0, r - dr, c - dc] == 0 and board[1, r - dr, c - dc] == 0:
                            valid_moves.add((r - dr, c - dc))
        if len(valid_moves) == 0:
            valid_moves.add((Gomoku.rows//2, Gomoku.cols//2))
        return list(valid_moves)
        # return [(r, c) for r in range(Gomoku.rows) for c in range(Gomoku.cols) if board[0, r, c] == 0 and board[1, r, c] == 0]

    @staticmethod
    def mcts(model, board: np.array, root, mcts_iterations, dirichlet = True):
        """
        Call Monte Carlo Tree Search.
        Select -> Expand -> Simulate -> Backup
        """
        MCTS.mcts(model, board, root, Gomoku, mcts_iterations, dirichlet)

    def get_input(board):
        row, col = map(int, input("Enter row and column: ").split())
        if board[0, row, col] == 1 or board[1, row, col] == 1:
            return None
        return (row, col)

    def self_play(self, model, mcts_iter, display = False):

        current_player = 0
        move_count = 0
        boards = []
        actions = [(-1, -1)]
        policy_distributions = []
        qs = []
        start_time = time.time()
        # prev_action = None
        while True:
            root = Node(None, None, current_player, move_count)

            Gomoku.mcts(model, self.board, root, mcts_iter)

            policy_distribution = utils.get_probablity_distribution_of_children(root, Gomoku)
            policy_distributions.append(policy_distribution)
            boards.append(self.board.copy())
            qs.append(root.value / root.visit)

            if model:
                chosen_child = root.sample_child(Gomoku) if move_count < 8 else root.max_visit_child()
            else:
                chosen_child = root.max_visit_child()
            
            current_player = Gomoku.make_move(self.board, current_player, chosen_child.prevAction)
            move_count += 1
            actions.append(chosen_child.prevAction)
            # prev_action = chosen_child.prevAction
            winner = Gomoku.check_winner(self.board, root.currentPlayer, chosen_child.prevAction)
            if winner != -1:
                if display:
                    Gomoku.display_board(self.board)
                    print('time:',time.time() - start_time,'s')
                break
            elif move_count == Gomoku.state_dim:
                if display:
                    Gomoku.display_board(self.board)
                winner = -1
                break

        return boards, actions, policy_distributions, qs, winner