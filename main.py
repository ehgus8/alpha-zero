import os
from tictactoe import TicTacToe
from connect4 import Connect4
from gomoku import Gomoku
from nn import Net
from replay_buffer import ReplayBuffer
import train
import test
import utils
import torch
import logging

# logging.basicConfig()

def start_train_loop(Game, older_model, newer_model, model_version,
                     buffer, buffer_size, collect_data_iterations, batch_size, mcts_iter, display=False):
    buffers_dir = os.path.join(os.path.dirname(__file__), 'replay_buffers')
    update_count = model_version
    original_collect_data_iterations = collect_data_iterations

    model_reject_count = 0
    for i in range(1, 1001):
        # train_iterations = int((buffer.size() // batch_size) * (model_version + i) ** 0.5)
        if model_reject_count >= 2:
            collect_data_iterations = original_collect_data_iterations * 2
        else:
            collect_data_iterations = original_collect_data_iterations
        print()
        print('iteration:', i)
        print('update_count:', update_count)

        train.collect_data(Game, older_model, buffer, iterations=collect_data_iterations, mcts_iter=mcts_iter, display=display)
        print(f'buffer status:{buffer.size()}/{buffer_size}')
        train_iterations = int((buffer.size() // batch_size)) * 10
        print('train_iterations:',train_iterations)
        # buffer save
        if i % 1 == 0:
            os.makedirs(buffers_dir, exist_ok=True)
            buffer.save_pickle(os.path.join(buffers_dir, f'replay_buffer_v{update_count}_i_{i}.pkl'))

        Game.logger.debug(f'after collect_data buffer status:{buffer.size()}/{buffer_size}')

        loss, policy_loss, value_loss, l2_reg = train.train(newer_model, batch_size, buffer=buffer, train_iterations=train_iterations, lr=0.001, device='cuda')
        Game.logger.debug(f'loss: {loss}\n\t policy_loss:{policy_loss} value_loss: {value_loss}\n\t l2_loss: {l2_reg}')

        newer_model.to('cpu')
        newer_model_win_rate = test.compare(Game,older_model, newer_model, mcts_iter, mcts_iter, iterations=70, sampling=False, early_stopping=True)
        if newer_model_win_rate > 0.55:
            older_model.load_state_dict(newer_model.state_dict())
            print('update older_model from newer_model')
            update_count+=1
            Game.logger.debug(f'start_train_loop update older_model from newer_model, so now model_v{update_count}')

            # model save
            # models_dir = os.path.join(os.path.dirname(__file__), 'models')
            # os.makedirs(models_dir, exist_ok=True)
            # torch.save(newer_model.state_dict(), os.path.join(models_dir, f'model_v{update_count}.pth'))
            utils.save_model(newer_model, update_count)

            
            
            # model compare with vanilla mcts
            if update_count % 2 == 0:
                print(f'vanila mcts (iter: {min(mcts_iter*update_count, 100)}) vs newer model result:')
                test.compare(Game, None, newer_model, min(mcts_iter*update_count, 100), mcts_iter, iterations=2, sampling=False, early_stopping=False)

            model_reject_count = 0
        else:
            newer_model.load_state_dict(older_model.state_dict())
            model_reject_count += 1
            
        print('replay buffer size:', buffer.size())

def select_game():
    print('1. TicTacToe')
    print('2. Connect4')
    print('3. Gomoku')
    game_num = int(input())
    if game_num == 1:
        Game = TicTacToe
    elif game_num == 2:
        Game = Connect4
    elif game_num == 3:
        Game = Gomoku

    return Game

def expand_transformer_layers(Game, old_model, new_num_layers):
    # 새 모델 생성
    new_model = Net(
        state_dim=Game.state_dim, 
        action_dim=Game.action_dim,
        num_transformer_layers=new_num_layers
    )
    
    # 이전 파라미터 불러오기
    old_state_dict = old_model.state_dict()
    new_state_dict = new_model.state_dict()
    
    # 새 state_dict에 기존 가중치를 복사
    pretrained_dict = {k: v for k, v in old_state_dict.items() if k in new_state_dict}
    
    # 일단 기존 파라미터를 로드
    new_model.load_state_dict(pretrained_dict, strict=False)

    print(f"Loaded previous parameters. Newly added layers initialized randomly.")
    
    return new_model


def train_mode(Game, model_version: int, load_model: bool):
    """
    Call start train loop
    """
    buffer_size = 120000
    buffer = ReplayBuffer(buffer_size)
    buffer.load_pickle('replay_buffer_v2_i_4')
    
    num_transformer_layers = 1
    new_num_layers = 1
    older_model = Net(Game.state_dim, Game.action_dim, num_transformer_layers)
    if load_model:
        older_model = utils.load_model(older_model, f'model_v{model_version}.pth')
    # older_model = expand_transformer_layers(Game, older_model, new_num_layers = new_num_layers)
    
    # newer_model
    newer_model = Net(Game.state_dim, Game.action_dim, new_num_layers)
    newer_model.load_state_dict(older_model.state_dict())
    for name, param in newer_model.named_parameters():
        print(f"{name}: {param.numel()} parameters")

    print(f'buffer status:{buffer.size()}/{buffer_size}')
    start_train_loop(Game, older_model, newer_model, model_version,
                        buffer, buffer_size,
                        collect_data_iterations=200, batch_size=256, mcts_iter=100, 
                        display=True)

if __name__ == "__main__":
    
    # for background execute
    # train_mode(Gomoku, 2,  True)
    while True:
        print('select the mode')
        print('1. train')
        print('2. test')
        print('3. play against agent')
        print('4. only training')
        mode = int(input())
        if mode == 1:
            Game = select_game()
            """
            Call start train loop
            """
            train_mode(Game, 0, False)
        elif mode == 2:
            """
            Test
            """
            Game = Gomoku
            num_transformer_layers = 1
            best_model = Net(Game.state_dim, Game.action_dim, 1)
            best_model = utils.load_model(best_model, 'model_v7.pth')
            contender_model = Net(Game.state_dim, Game.action_dim, 1)
            contender_model = utils.load_model(contender_model, 'model_v2.pth')
            best_model_mcts_iter = 100
            contender_model_mcts_iter = 100
            test.compare(Game, best_model, contender_model, best_model_mcts_iter, contender_model_mcts_iter, 10, sampling=False,early_stopping=False)
        elif mode == 3:
            """
            Play against agent
            """
            Game = select_game()
            print('select turn first(0) second(1):')
            human_turn = int(input())
            num_transformer_layers = 1
            model_version = 14
            older_model = Net(Game.state_dim, Game.action_dim, num_transformer_layers)
            older_model = utils.load_model(older_model, f'model_v{model_version}.pth')
            test.play_against_agent(Game, older_model, 100, human_turn)
        elif mode == 4:
            Game = Gomoku
            buffer_size = 140000
            buffer = ReplayBuffer(buffer_size)
            buffer.load_pickle('replay_buffer_v14_i_32')
            batch_size = 256
            num_transformer_layers = 1
            new_num_layers = 4
            model_version = 14
            best_model = Net(Game.state_dim, Game.action_dim, num_transformer_layers)
            best_model = utils.load_model(best_model, f'model_v{model_version}.pth')
            best_model = expand_transformer_layers(Game, best_model, new_num_layers = new_num_layers)
            train_iterations = int((buffer.size() // batch_size)) * 10
            for name, param in best_model.named_parameters():
                print(f"{name}: {param.numel()} parameters")
            loss, policy_loss, value_loss, l2_reg = train.train(best_model, batch_size, 
                                                                buffer=buffer, 
                                                                train_iterations=train_iterations, 
                                                                lr=0.0004, device='cuda')
            utils.save_model(best_model, 123)
        else:
            print('wrong mode')