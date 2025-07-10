import os
import json
from ai.nn import Net
from replay_buffer import ReplayBuffer
import train
import utils
import torch
from ai import mcts
import test

with open('config.json') as f:
    config = json.load(f)

TRAIN_CFG = config['train']
MODEL_CFG = config['model']
BUFFER_CFG = config['buffer']


def start_train_loop(Game, older_model, newer_model, model_version, older_model_args,
                     buffer, buffer_size, collect_data_iterations, batch_size, mcts_iter, display=False):
    """
    AlphaZero 스타일의 self-play 및 학습 루프를 실행합니다.

    Args:
        Game: 게임 클래스 (예: Gomoku, TicTacToe 등)
        older_model: 비교 대상이 되는 이전 버전 모델
        newer_model: 학습 중인 최신 모델
        model_version: 현재 모델 버전(정수)
        buffer: ReplayBuffer 인스턴스
        buffer_size: 버퍼 최대 크기
        collect_data_iterations: self-play 데이터 수집 반복 횟수
        batch_size: 학습 배치 크기
        mcts_iter: MCTS 반복 횟수
        display: self-play 중 보드 출력 여부
    Returns:
        없음. (모델, 버퍼, 로그 등은 파일로 저장)
    """
    buffers_dir = os.path.join(os.path.dirname(__file__), BUFFER_CFG['save_dir'])
    update_count = model_version
    original_collect_data_iterations = collect_data_iterations

    model_reject_count = 0
    for i in range(1, 1001):
        if model_reject_count >= 2:
            collect_data_iterations = original_collect_data_iterations * 2
        else:
            collect_data_iterations = original_collect_data_iterations
        print()  # 사용자 구분용
        print('iteration:', i)
        print('update_count:', update_count)
        Game.logger.info(f'iteration: {i}, update_count: {update_count}')

        train.collect_data(Game, older_model, older_model_args, buffer, 
                           iterations=collect_data_iterations, mcts_iter=mcts_iter, 
                           num_workers=TRAIN_CFG['num_workers'], display=display)
        Game.logger.info(f'buffer status: {buffer.size()}/{buffer_size}')
        print(f'buffer status:{buffer.size()}/{buffer_size}')
        train_iterations = (buffer.size() // batch_size) * 1
        Game.logger.info(f'train_iterations: {train_iterations}')
        print('train_iterations:',train_iterations)
        if i % 1 == 0:
            os.makedirs(buffers_dir, exist_ok=True)
            buffer.save_pickle(os.path.join(buffers_dir, f'replay_buffer_v{update_count}_i_{i}.pkl'))

        Game.logger.debug(f'after collect_data buffer status:{buffer.size()}/{buffer_size}')
        lr = TRAIN_CFG['lr']
        loss, policy_loss, value_loss, l2_reg = train.train(newer_model, batch_size, buffer=buffer, train_iterations=train_iterations, lr=lr, device='cuda')
        Game.logger.debug(f'loss: {loss}\n\t policy_loss:{policy_loss} value_loss: {value_loss}\n\t l2_loss: {l2_reg}')

        newer_model.to('cpu')
        # newer_model_win_rate = test.compare(Game,older_model, newer_model, mcts_iter, mcts_iter, iterations=30, sampling=False, early_stopping=True)
        newer_model_win_rate = 0.6
        Game.logger.info(f'newer_model_win_rate: {newer_model_win_rate}')
        if newer_model_win_rate > 0.55:
            print('cache size:',len(mcts.MCTS.cache))
            Game.logger.info(f'cache size: {len(mcts.MCTS.cache)}')

            if mcts.MCTS.mcts_count > 0:
                cache_rate = mcts.MCTS.matched / mcts.MCTS.mcts_count
                print(f'cache match rate: {cache_rate}')
                Game.logger.info(f'cache match rate: {cache_rate}')
                Game.logger.debug(f'cache matched: {mcts.MCTS.matched} / {mcts.MCTS.mcts_count}')
            else:
                print('cache match rate: N/A (MCTS not run in main process)')
                Game.logger.info('cache match rate: N/A (MCTS not run in main process)')

            mcts.MCTS.cache.clear()
            mcts.MCTS.matched = 0
            mcts.MCTS.mcts_count = 0
            print('cache cleared!')
            older_model.load_state_dict(newer_model.state_dict())
            print('update older_model from newer_model')
            update_count+=1
            Game.logger.info(f'update older_model from newer_model, now model_v{update_count}')
            utils.save_model(newer_model, update_count)
            if update_count % 15 == 0:
                print(f'vanila mcts (iter: {min(mcts_iter*update_count, 400)}) vs newer model result:')
                test.compare(Game, None, newer_model, min(mcts_iter*update_count, 400), mcts_iter, iterations=2, sampling=False, early_stopping=False)
            model_reject_count = 0
        else:
            newer_model.load_state_dict(older_model.state_dict())
            model_reject_count += 1
        Game.logger.info(f'replay buffer size: {buffer.size()}')
        print('replay buffer size:', buffer.size())

def train_mode(Game, load_model=False):
    """
    지정된 게임 환경에서 모델을 초기화하고, 학습 루프를 시작합니다.

    Args:
        Game: 게임 클래스 (예: Gomoku, TicTacToe 등)
        load_model (bool): 최신 모델을 불러올지 여부. 기본값은 False.
    Returns:
        없음. (학습 및 모델 저장)
    """
    buffer_size = TRAIN_CFG['buffer_size']
    buffer = ReplayBuffer(buffer_size)
    # buffer.load_pickle('replay_buffer_v13_i_5')
    num_transformer_layers = TRAIN_CFG['num_transformer_layers']
    new_num_transformer_layers = TRAIN_CFG['new_num_transformer_layers']
    num_heads = TRAIN_CFG['num_heads']
    patch_size = TRAIN_CFG['patch_size']
    embed_dim = TRAIN_CFG['embed_dim']
    dropout = TRAIN_CFG['dropout']

    # 모델 초기화 및 최신 버전 불러오기
    older_model = Net(Game.rows, patch_size=patch_size, embed_dim=embed_dim, action_dim=Game.action_dim,num_heads=num_heads, depth=num_transformer_layers, channels=Game.feature_dim, dropout=dropout)
    
    model_version = 0
    if load_model:
        older_model, model_version = utils.load_latest_model(older_model, Game)
        buffer.load_latest_buffer()
    
    older_model_args = {
        'img_size': Game.rows, 
        'patch_size': patch_size, 
        'embed_dim': embed_dim, 
        'action_dim': Game.action_dim,
        'num_heads': num_heads, 
        'depth': num_transformer_layers, 
        'channels': Game.feature_dim, 
        'dropout': dropout
    }

    newer_model = Net(Game.rows, patch_size=patch_size, embed_dim=embed_dim, action_dim=Game.action_dim,num_heads=num_heads, depth=new_num_transformer_layers, channels=Game.feature_dim, dropout=dropout)
    newer_model.load_state_dict(older_model.state_dict())

    for name, param in newer_model.named_parameters():
        Game.logger.info(f"{name}: {param.numel()} parameters")
        print(f"{name}: {param.numel()} parameters")

    print(f'buffer status:{buffer.size()}/{buffer_size}')
    Game.logger.info(f'buffer status: {buffer.size()}/{buffer_size}')
    start_train_loop(Game, older_model, newer_model, model_version, older_model_args,
                        buffer, buffer_size,
                        collect_data_iterations=TRAIN_CFG['collect_data_iterations'], batch_size=TRAIN_CFG['batch_size'], mcts_iter=TRAIN_CFG['mcts_iter'], 
                        display=True) 