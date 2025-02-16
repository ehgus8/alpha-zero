import os
from tictactoe import TicTacToe
from connect4 import Connect4
from nn import Net
from replay_buffer import ReplayBuffer
import train
import test
import utils
import torch

def start_train_loop(Game, older_model, newer_model, 
                     buffer, self_play_iterations, train_iterations, batch_size, display=False):
    buffers_dir = os.path.join(os.path.dirname(__file__), 'replay_buffers')
    update_count = 0
    mcts_iter = 25
    for i in range(1, 1001):
        print()
        print('iteration:', i)
        print('update_count:', update_count)

        train.collect_data(Game, older_model, buffer, iterations=self_play_iterations, mcts_iter=mcts_iter, display=display)
        train.train(newer_model, batch_size, buffer=buffer, train_iterations=train_iterations, device='cpu')
        newer_model.to('cpu')
        newer_model_win_rate = test.compare(Game,older_model, newer_model, mcts_iter, mcts_iter, iterations=60, sampling=True)
        # newer_model_win_rate = test.compare(None, newer_model, min(mcst_iter*(update_count + 1), 800), mcst_iter, iterations=60, sampling=False)
        if newer_model_win_rate > 0.55:
            older_model.load_state_dict(newer_model.state_dict())
            print('update older_model from newer_model')
            update_count+=1

            # model save
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            os.makedirs(models_dir, exist_ok=True)
            torch.save(newer_model.state_dict(), os.path.join(models_dir, f'model_v{update_count}.pth'))

            # buffer save
            os.makedirs(buffers_dir, exist_ok=True)
            buffer.save_pickle(os.path.join(buffers_dir, f'replay_buffer_v{update_count}_i_{i}.pkl'))
            
            # model compare with vanilla mcts
            print(f'vanila mcts (iter: {min(mcts_iter*update_count, 800)}) vs newer model result:')
            test.compare(Connect4, None, newer_model, min(mcts_iter*update_count, 800), mcts_iter, iterations=60, sampling=False)

            
        else:
            newer_model.load_state_dict(older_model.state_dict())
        print('replay buffer size:', buffer.size())
        if i % 20 == 0:
            buffer.save_pickle(os.path.join(buffers_dir, f'replay_buffer_v{update_count}_i_{i}_regular.pkl'))

def select_game():
    print('1. TicTacToe')
    print('2. Connect4')
    game_num = int(input())
    if game_num == 1:
        Game = TicTacToe
    elif game_num == 2:
        Game = Connect4
    return Game

if __name__ == "__main__":
    while True:
        print('select the mode')
        print('1. train')
        print('2. test')
        print('3. play against agent')
        mode = int(input())
        if mode == 1:
            
            Game = select_game()
            """
            Call start train loop
            """
            buffer_size = 120000
            buffer = ReplayBuffer(buffer_size)
            older_model = Net(Game.state_dim, Game.action_dim)
            newer_model = Net(Game.state_dim, Game.action_dim)
            print(older_model)
            newer_model.load_state_dict(older_model.state_dict())

            start_train_loop(Game, older_model, newer_model, buffer, 
                             self_play_iterations=50, batch_size=256, train_iterations=50, 
                             display=True)
        elif mode == 2:
            """
            Test
            """
            num_transformer_layers = 2
            model = Net(num_transformer_layers)
            model = utils.load_model(model, 'model_v23.pth')
            best_model_mcts_iter = 600
            contender_model_mcts_iter = 25
            test.compare(None, model, best_model_mcts_iter, contender_model_mcts_iter, 100, sampling=False)
        elif mode == 3:
            """
            Play against agent
            """
            Game = select_game()
            print('select turn first(0) second(1):')
            human_turn = int(input())
            test.play_against_agent(Game, None, 500, human_turn)
        else:
            print('wrong mode')