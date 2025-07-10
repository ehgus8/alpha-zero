import numpy as np
import torch
import utils

class MCTS:
    # 전역 캐시: 보드 상태의 바이트 표현을 key로, (policy_softmax, value)를 저장합니다.
    cache = {}
    matched = 0
    mcts_count = 0
    @staticmethod
    def mcts(model, board, root, Game, mcts_iterations, dirichlet=True):
        """
        Perform Monte Carlo Tree Search.
        Select -> Expand -> Simulate -> Backup
        """
        for _ in range(mcts_iterations):
            MCTS.mcts_count += 1
            node = root
            trace = [root]
            while node.children:
                node = node.select('network' if model else 'normal')
                trace.append(node)
                Game.make_move(board, 1 - node.currentPlayer, node.prevAction)
            
            if node.parent:
                winner = Game.check_winner(board, 1 - node.currentPlayer, node.prevAction)
                if winner != -1:
                    node.backup(trace, 1, board, Game)
                    continue
                elif node.move_count == Game.state_dim:
                    node.backup(trace, 0, board, Game)
                    continue
            
            valid_moves = Game.get_valid_moves(board)
            if model:
                # 캐시된 결과가 있는지 체크: board를 bytes로 변환하여 key로 사용
                canonical_board = Game.get_canonical_board(board, node.currentPlayer)
                board_key = canonical_board.tobytes()

                if board_key in MCTS.cache:
                    policy_softmax, value = MCTS.cache[board_key]
                    if (not node.parent) and dirichlet:
                        policy_softmax = utils.add_dirichlet_noise(policy_softmax)
                    MCTS.matched += 1
                else:
                    policy_logits, value = model(torch.from_numpy(canonical_board).unsqueeze(0))
                        
                    value = value.detach()
                    policy_logits = policy_logits.squeeze(0).detach().numpy()
                    policy_softmax = np.exp(policy_logits) / np.sum(np.exp(policy_logits))
                    # caching
                    MCTS.cache[board_key] = (policy_softmax, value)
                    if (not node.parent) and dirichlet:
                        policy_softmax = utils.add_dirichlet_noise(policy_softmax)

                node.expand(valid_moves, policy_softmax, Game)
                result = -value.item()
            else:
                node.expand(valid_moves, None, Game)
                result = MCTS.simulate(Game, board, node)
            node.backup(trace, result, board, Game)

    @staticmethod
    def simulate(Game, board, node):
        """
        Simulation (rollout) step of MCTS.
        """
        sim_board = board.copy()
        current_player = node.currentPlayer
        move_count = node.move_count
        winner = -1
        while winner == -1 and move_count < Game.state_dim:
            valid_moves = Game.get_valid_moves(sim_board)
            action = valid_moves[np.random.randint(len(valid_moves))]
            current_player = Game.make_move(sim_board, current_player, action)
            move_count += 1
            winner = Game.check_winner(sim_board, 1 - current_player, action)
        if winner != -1:
            return 1 if winner == (1 - node.currentPlayer) else -1
        return 0


# import game
# import numpy as np
# import torch
# import utils


# class MCTS:
#     @staticmethod
#     def mcts(model, board, root, Game: game.Game, mcts_iterations, dirichlet = True):
#         """
#         Perform Monte Carlo Tree Search.
#         Select -> Expand -> Simulate -> Backup

#         Args:
#             board (np.ndarray): The current board state.
#             current_player (int): The current player.
#             model (nn.Module): The neural network model.
#         """
#         for _ in range(mcts_iterations):
#             node = root
#             trace = [root]
#             while node.children:
#                 node = node.select('network' if model else 'normal')
#                 trace.append(node)

#                 Game.make_move(board, 1 - node.currentPlayer, node.prevAction)
            
#             if node.parent:
#                 winner = Game.check_winner(board, 1 - node.currentPlayer, node.prevAction)
#                 if winner != -1:
#                     # Game.display_board(board)
#                     node.backup(trace, 1, board, Game)
#                     # print(winner, node.currentPlayer, node.prevAction, node.visit, node.value, 'not draw')
#                     continue
#                 elif node.move_count == Game.state_dim:
#                     # Game.display_board(board)
#                     # print(winner, node.currentPlayer, result, 'draw')
#                     node.backup(trace, 0, board, Game)
#                     continue
#             # if node.prevAction:
#                 # board[2, node.prevAction[0], node.prevAction[1]] = 1
#             valid_moves = Game.get_valid_moves(board)
#             if model:
#                 # forward
#                 board_for_model = np.empty_like(board)
#                 if node.currentPlayer == 1: # white
#                 #     # board_for_model[0], board_for_model[1], board_for_model[2], board_for_model[3] = board[1], board[0], board[2], board[3]
#                     board_for_model[0], board_for_model[1], board_for_model[2] = board[1], board[0], board[2]
#                     policy_logits, value = model(
#                         torch.from_numpy(board_for_model).unsqueeze(0)
#                         )
#                 else:
#                     # board_for_model[0], board_for_model[1], board_for_model[2], board_for_model[3] = board[0], board[1], board[2], board[3]
#                     policy_logits, value = model(
#                         torch.from_numpy(board).unsqueeze(0)
#                         )
                    
#                 policy_logits = policy_logits.squeeze(0).numpy()
#                 policy_softmax = np.exp(policy_logits) / np.sum(np.exp(policy_logits))
#                 if (not node.parent) and dirichlet:
#                     policy_softmax = utils.add_dirichlet_noise(policy_softmax)
#                 # Expand
#                 node.expand(valid_moves, policy_softmax, Game)
#                 result = -value.item()
#             else:
#                 node.expand(valid_moves, None, Game)
#                 result = MCTS.simulate(Game, board, node)
#             # if node.prevAction:                
#                 # board[2, node.prevAction[0], node.prevAction[1]] = 0
#             node.backup(trace, result, board, Game)

#     @staticmethod
#     def simulate(Game: game.Game, board, node):
#         """
#         Simulation (rollout) step of MCTS.
#         """
#         sim_board = board.copy()
#         current_player = node.currentPlayer
#         move_count = node.move_count
#         winner = -1
#         while winner == -1 and move_count < Game.state_dim:
#             valid_moves = Game.get_valid_moves(sim_board)

#             action = valid_moves[np.random.randint(len(valid_moves))]

#             current_player = Game.make_move(sim_board, current_player, action)
#             move_count += 1

#             winner = Game.check_winner(sim_board, 1 - current_player, action)

#         if winner != -1:
#             return 1 if winner == (1 - node.currentPlayer) else -1
#         return 0