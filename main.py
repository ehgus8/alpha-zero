import os
from games import TicTacToe, Connect4, Gomoku
from replay_buffer import ReplayBuffer
import utils
import torch

# 분리된 모듈 임포트
from train_loop import start_train_loop, train_mode
from evaluate import test_mode, play_against_agent_mode


def select_game():
    print('1. TicTacToe')
    print('2. Connect4')
    print('3. Gomoku')
    try:
        game_num = int(input())
        if game_num == 1:
            return TicTacToe
        elif game_num == 2:
            return Connect4
        elif game_num == 3:
            return Gomoku
        else:
            print("잘못된 번호입니다.")
            return select_game()
    except ValueError:
        print("숫자를 입력하세요.")
        return select_game()

if __name__ == "__main__":
    while True:
        print('select the mode')
        print('1. train')
        print('2. test')
        print('3. play against agent')
        print('4. only training')
        try:
            mode = int(input())
        except ValueError:
            print('숫자를 입력하세요.')
            continue
        if mode == 1:
            Game = select_game()
            
            load_choice = ''
            while load_choice.lower() not in ['y', 'n']:
                load_choice = input("최신 모델을 불러와서 학습을 계속하시겠습니까? (y/n): ")
            
            load_model = True if load_choice.lower() == 'y' else False
            train_mode(Game, load_model)
        elif mode == 2:
            test_mode()
        elif mode == 3:
            play_against_agent_mode()
        elif mode == 4:
            print('only training 모드는 추후 분리된 train_loop.py에서 지원 예정입니다.')
        else:
            print('wrong mode')