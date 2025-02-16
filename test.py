import os
from nn import Net
from utils import load_model
import torch
from node import Node
def compete(Game, model1, model2, model1_mcts_iter = 50, model2_mcts_iter = 50, sampling = False, display = False):
        """
        Play a game between two agents.

        Args:
            sampling (bool): Whether to sample the child nodes or select the child node with the maximum visit count.
        Returns: (winner, final_board)
            winner = 0 or 1, -1(draw)
        """
        current_player = 0
        move_count = 0
        game = Game()
        while True:
            if display:
                Game.display_board(game.board)
                print()

            root = Node(None, None, current_player, move_count)
            
            if current_player == 0:
                Game.mcts(model1, game.board, root, model1_mcts_iter)
            else:
                Game.mcts(model2, game.board, root, model2_mcts_iter)

            if sampling:
                chosen_child = root.sample_child(Game)
            else:
                chosen_child = root.max_visit_child()

            current_player = Game.make_move(game.board, current_player, chosen_child.prevAction)
            move_count += 1

            winner = Game.check_winner(game.board, root.currentPlayer, chosen_child.prevAction)
            if winner != -1:
                if display:
                    Game.display_board(game.board)
                    print("Player", winner, "wins!")
                return winner, game.board
            elif move_count == Game.state_dim:
                if display:
                    Game.display_board(game.board)
                    print("It's a draw!")
                return -1, game.board

def compare(Game, best_model, contender_model, best_model_mcts_iter: int, contender_model_mcts_iter: int, iterations: int, sampling=True):
    win_count = [0, 0, 0] # best model's, contender model's, draw
    if best_model is not None:
        best_model.eval()
    if contender_model is not None:
        contender_model.eval()
    with torch.no_grad():
        for i in range(iterations):
            winner, board = compete(Game, best_model if i < iterations//2 else contender_model, 
                                        contender_model if i < iterations//2 else best_model, 
                                        best_model_mcts_iter if i < iterations//2 else contender_model_mcts_iter, 
                                        contender_model_mcts_iter if i < iterations//2 else best_model_mcts_iter,
                                        sampling=sampling, display=False)
            if winner == 0:
                win_count[0 if i < iterations//2 else 1] += 1
            elif winner == 1:
                win_count[1 if i < iterations//2 else 0] += 1
            else:
                win_count[0] += 0.5
                win_count[1] += 0.5
                win_count[2] += 1

    print('win count:', win_count, 'best model win rate:', win_count[0] / iterations, 'contender model win rate:', win_count[1] / iterations, 'draw rate:', win_count[2] / iterations)
    return win_count[1] / iterations