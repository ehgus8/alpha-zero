import os
from nn import Net
from utils import load_model
import torch
from node import Node
import time


def play_against_agent(Game, model, mcts_iterations, human_turn):
        if model is not None:
            model.eval()
        current_player = 0
        move_count = 0
        game = Game()
        while True:
            Game.display_board(game.board)

            if current_player == human_turn:
                action = Game.get_input(game.board)
                if action is None:
                    print("Invalid move. Try again.")
                    continue

                current_player = Game.make_move(game.board, current_player, action)
                move_count += 1

                winner = Game.check_winner(game.board, 1 - current_player, action)
                if winner != -1:
                    Game.display_board(game.board)
                    print("Player", winner, "wins!")
                    break
                elif move_count == Game.state_dim:
                    Game.display_board(game.board)
                    print("It's a draw!")
                    break
            else:
                root = Node(None, None, current_player, move_count)
                start_time = time.time()
                Game.mcts(model, game.board, root, mcts_iterations) 
                model_name = 'network' if model else 'vanilla'
                Game.logger.debug(f'model: {model_name} mcts_iteration: {mcts_iterations}, time: {time.time() - start_time}s')
                chosen_child = root.max_visit_child() 
                for child in root.children:
                    print(child.to_string(Game))
                current_player = Game.make_move(game.board, current_player, chosen_child.prevAction)
                move_count += 1

                winner = Game.check_winner(game.board, root.currentPlayer, chosen_child.prevAction)
                if winner != -1:
                    Game.display_board(game.board)
                    print("Player", winner, "wins!")
                    break
                elif move_count == Game.state_dim:
                    Game.display_board(game.board)
                    print("It's a draw!")
                    break

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
                chosen_child = root.sample_child(Game) if move_count < 10 else root.max_visit_child()
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

def compare(Game, best_model, contender_model, best_model_mcts_iter: int, contender_model_mcts_iter: int, iterations: int, sampling: bool, early_stopping: bool):
    win_count = [0, 0, 0] # best model's, contender model's, draw
    if best_model is not None:
        best_model.eval()
    if contender_model is not None:
        contender_model.eval()
    with torch.no_grad():
        for i in range(iterations):
            start_time = time.time()
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
            Game.display_board(board)
            Game.logger.debug(f'compare iter({i+1}/{iterations}) win count:{win_count}, best model win rate:{win_count[0] / (i+1)}, contender model win rate:{win_count[1] / (i+1)}, draw rate:{win_count[2] / (i+1)}, time:{time.time()-start_time}s')

            if early_stopping:
                if win_count[1] >= int(iterations*0.55):
                    Game.logger.debug(f'compare early stopped, model accepted')
                    return 1
                remained_iter = iterations - (i+1)
                if win_count[1] + remained_iter < int(iterations*0.55):
                    Game.logger.debug(f'compare early stopped, model rejected')
                    return 0
    print('win count:', win_count, 'best model win rate:', win_count[0] / iterations, 'contender model win rate:', win_count[1] / iterations, 'draw rate:', win_count[2] / iterations)
    return win_count[1] / iterations