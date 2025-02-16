import os
from nn import Net
from utils import load_model
import torch

def compare(Game, best_model, contender_model, best_model_mcts_iter: int, contender_model_mcts_iter: int, iterations: int, sampling=True):
    win_count = [0, 0, 0] # best model's, contender model's, draw
    if best_model is not None:
        best_model.eval()
    if contender_model is not None:
        contender_model.eval()
    with torch.no_grad():
        for i in range(iterations):
            game = Game()
            winner, board = game.compete(best_model if i < iterations//2 else contender_model, 
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

if __name__ == "__main__":
    model = Net()
    model_path = os.path.join(os.path.dirname(__file__), 'models/model_v5.pth')
    model = load_model(model, model_path)
    model.eval()

    vanlia_mcts_iter = 400
    model_mcts_iter = 100
    compare(None, model,vanlia_mcts_iter, model_mcts_iter, 40)