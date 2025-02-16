import numpy as np
from collections import deque
from node import Node
import time
from game import Game
from mcts import MCTS
import utils
class Connect4(Game):
    rows, cols = 6, 7
    action_dim = cols
    state_dim = rows * cols

    logger = utils.get_game_logger('connect4')
    def __init__(self):
        self.board = np.zeros((3, 6, 7), dtype=np.float32)

    @staticmethod
    def display_board(board):
        # 시각화를 위해 ' ' / 'O' / 'X' 로 표시
        display = np.full((6, 7), ' ')
        display[board[0] == 1] = 'O'
        display[board[1] == 1] = 'X'
        print("\n  0 1 2 3 4 5 6")
        for i, row in enumerate(display):
            print(i, ' '.join(row))
        print()
        # print(board[2])

    @staticmethod
    def get_action_idx(move: tuple[int, int]):
        return move[1] # col
    
    @staticmethod
    def get_drop_row(board, col):
        """
        해당 col에 말을 둘 때, 실제 말이 떨어질 row를 반환
        (가장 아래쪽 empty칸(=0)을 찾아서 반환)
        """
        # board[0, row, col], board[1, row, col] 이 모두 0인 가장 아래쪽 row 찾기
        for row in range(5, -1, -1):
            if board[0, row, col] == 0 and board[1, row, col] == 0:
                return row
        return None  # 만약 col이 가득 차 있다면 None
    
    @staticmethod
    def get_valid_moves(board):
        """
        현재 board에서 둘 수 있는 모든 열(column) 리스트를 반환
        (가득 차지 않은 열만 valid)
        """
        valid_moves = []
        for col in range(Connect4.action_dim):
            # top row가 비어있으면(col, row=0)이 아니라, bottom부터 확인
            if board[0, 0, col] == 0 and board[1, 0, col] == 0:
                row = Connect4.get_drop_row(board, col)
                if row is not None:
                    valid_moves.append((row,col))

        return valid_moves
    
    @staticmethod
    def make_move(board, current_player, action):
        row, col = action
        if row is None:
            print("Invalid move: column is full.")
            return current_player  # 변경 없이 그대로 반환
        board[current_player, row, col] = 1
        board[2, :, :] = 1 - current_player
        return 1 - current_player


    @staticmethod
    def undo_move(board, current_player, action):
        row, col = action
        board[1 - current_player, row, col] = 0
        board[2, :, :] = 1 - current_player

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
                
                if 0 <= r + dr < Connect4.rows and 0 <= c + dc < Connect4.cols and (r + dr, c + dc) not in visited and board[player, r + dr, c + dc] == 1:
                    q.append((r + dr, c + dc))
                    seq_count += 1
                if 0 <= r - dr < Connect4.rows and 0 <= c - dc < Connect4.cols and (r - dr, c - dc) not in visited and board[player, r - dr, c - dc] == 1:
                    q.append((r - dr, c - dc))
                    seq_count += 1
                if seq_count >= 4:
                    return True
                
            return False
        # 방향 벡터
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            if dfs(dr, dc):
                return player
        return -1

    
    @staticmethod
    def mcts(model, board: np.array, root, mcts_iterations):
        """
        Call Monte Carlo Tree Search.
        Select -> Expand -> Simulate -> Backup
        """
        MCTS.mcts(model, board, root, Connect4, mcts_iterations)

    def play_against_mcts(self, model, mcts_iterations):
        """
        사람 vs MCTS
        """
        current_player = 0
        move_count = 0

        while True:
            # 1) Human turn
            Connect4.display_board(self.board)
            col = int(input("Enter column (0~6): "))
            row = Connect4.get_drop_row(self.board, col)
            if row is None:
                print("Invalid move. Try again.")
                continue

            current_player = Connect4.make_move(self.board, current_player, (row, col))
            move_count += 1

            winner = Connect4.check_winner(self.board, 1 - current_player, (row, col))
            if winner != -1:
                Connect4.display_board(self.board)
                print("Player", winner, "wins!")
                break
            elif move_count == 42:
                Connect4.display_board(self.board)
                print("It's a draw!")
                break

            # 2) MCTS agent turn
            root = Node(None, None, current_player, move_count)
            start_time = time.time()
            Connect4.mcts(model, self.board, root, mcts_iterations)  # model이 None이면 vanilla MCTS
            model_name = 'network' if model else 'vanilla'
            Connect4.logger.debug(f'model: {model_name} mcts_iteration: {mcts_iterations}, time: {time.time() - start_time}s')
            chosen_child = root.max_visit_child() 
            for child in root.children:
                print(child.to_string(Connect4))
            current_player = Connect4.make_move(self.board, current_player, chosen_child.prevAction)
            move_count += 1

            winner = Connect4.check_winner(self.board, root.currentPlayer, chosen_child.prevAction)
            if winner != -1:
                Connect4.display_board(self.board)
                print("Player", winner, "wins!")
                break
            elif move_count == 42:
                Connect4.display_board(self.board)
                print("It's a draw!")
                break

    def self_play(self, model, mcts_iter = 50, display = False):

        current_player = 0
        move_count = 0
        boards = []
        policy_distributions = []
        start_time = time.time()
        while True:
            root = Node(None, None, current_player, move_count)

            Connect4.mcts(model, self.board, root, mcts_iter)

            policy_distribution = utils.get_probablity_distribution_of_children(root, Connect4)
            policy_distributions.append(policy_distribution)
            boards.append(self.board.copy())

            chosen_child = root.sample_child(Connect4) if model else root.max_visit_child()
            

            current_player = Connect4.make_move(self.board, current_player, chosen_child.prevAction)
            move_count += 1

            winner = Connect4.check_winner(self.board, root.currentPlayer, chosen_child.prevAction)
            if winner != -1:
                if display:
                    Connect4.display_board(self.board)
                    print('time:',time.time() - start_time,'s')
                # Connect4.logger.debug(f'self_play, time: {time.time() - start_time}s')
                break
            elif move_count == 42:
                if display:
                    Connect4.display_board(self.board)
                winner = -1
                break

        return boards, policy_distributions, winner
