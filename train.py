import torch
from ai.nn import Net
from replay_buffer import ReplayBuffer
import test
import time
import numpy as np
import multiprocessing
import os

def rotate_data(board, policy, k):
    """
    보드와 정책 분포를 k*90도 회전시킵니다.
    Args:
        board (np.ndarray): (C, H, W) 형태의 보드
        policy (np.ndarray): 1D 정책 분포
        k (int): 90도 회전 횟수
    Returns:
        (rotated_board, rotated_policy_1d): 회전된 보드와 정책 분포
    """
    channel, rows, cols = board.shape
    rotated_board = np.rot90(board,k=k, axes=(1, 2)).copy()
    policy_2d = policy.reshape((rows, cols)).copy()
    policy_2d = np.rot90(policy_2d, k=k)
    rotated_policy_1d = policy_2d.flatten()

    return rotated_board, rotated_policy_1d

def flip_data(board, policy, mode):
    """
    보드와 정책 분포를 좌우/상하로 뒤집습니다.
    Args:
        board (np.ndarray): (C, H, W) 형태의 보드
        policy (np.ndarray): 1D 정책 분포
        mode (str): 'lr'(좌우) 또는 'tb'(상하)
    Returns:
        (flipped_board, flipped_policy): 뒤집힌 보드와 정책 분포
    """
    channel, rows, cols = board.shape
    policy_2d = policy.reshape((rows, cols)).copy()
    if mode == 'lr':
        flipped_board = np.flip(board, axis=2).copy() # left right flip
        flipped_policy = np.flip(policy_2d, axis=1).copy()
    else:
        flipped_board = np.flip(board, axis=1).copy() # tob bottom flip
        flipped_policy = np.flip(policy_2d, axis=0).copy()

    return flipped_board, flipped_policy.flatten()
    

def save_data_to_buffer(Game, buffer: ReplayBuffer, data):
    """
    self-play 결과 데이터를 버퍼에 저장하고, 데이터 증강(회전/뒤집기)도 수행합니다.
    Args:
        Game: 게임 클래스
        buffer: ReplayBuffer 인스턴스
        data: (boards, actions, policies, qs, winner, reward) 튜플
    Returns:
        없음
    """
    boards, actions, policies, qs, winner, reward = data
    epsilon = 0.5
    currentPlayer = 0
    for i in range(len(boards)):
            # reward_target = [(reward*(1-epsilon) + qs[i]*epsilon)] if currentPlayer == winner else [((-reward)*(1-epsilon) + qs[i]*epsilon)]
            reward_target = [reward] if currentPlayer == winner else [-reward]
            canonical_board = Game.get_canonical_board(boards[i], currentPlayer)
            currentPlayer = 1 - currentPlayer
            if i == 0 or i == 1: # i = 0 : empty board, i = 1 : only one stone in center if gomoku. so doesn't need agumentation
                buffer.add(canonical_board, policies[i], reward_target)
                continue
            for r in range(0,4): # 0 90 180 270
                board_rot, policy_rot = rotate_data(canonical_board, policies[i], k=r)
                buffer.add(board_rot, policy_rot, reward_target)
                if r == 0 or r == 1: # flip only when 0, 90 rotated
                    board_lr, policy_lr = flip_data(board_rot, policy_rot, 'lr')
                    board_tb, policy_tb = flip_data(board_rot, policy_rot, 'tb')
                    buffer.add(board_lr, policy_lr, reward_target)
                    buffer.add(board_tb, policy_tb, reward_target)
                
        
# 전역 변수로 각 워커의 모델을 저장합니다.
_worker_model = None

def _init_worker(model_class, model_state, model_args):
    """각 워커 프로세스를 초기화하는 함수입니다."""
    global _worker_model
    _worker_model = model_class(**model_args)
    _worker_model.load_state_dict(model_state)
    _worker_model.eval()

def _run_one_game(args):
    """워커가 실행하는 단일 셀프 플레이 게임입니다."""
    Game, mcts_iter, display = args
    game = Game()
    with torch.no_grad():
        boards, actions, policy_distributions, qs, winner = game.self_play(_worker_model, mcts_iter, display)
    
    if winner == -1:
        reward = 0
        game_result_idx = 2  # Draw
    else:
        reward = 1
        game_result_idx = 0 if winner == 0 else 1  # Black or White win
        
    return boards, actions, policy_distributions, qs, winner, reward, game_result_idx


def collect_data(Game, model: Net, model_args: dict, buffer: ReplayBuffer, iterations: int, mcts_iter: int, num_workers: int, display=False):
    """
    MCTS 기반 self-play를 병렬로 실행하여 데이터를 생성하고 버퍼에 저장합니다.
    """
    model.eval()
    model_state = model.state_dict()

    # CPU에서 실행되도록 모델 상태를 확실히 합니다.
    for k, v in model_state.items():
        model_state[k] = v.cpu()

    start_time = time.time()
    
    if num_workers == 0:
        num_workers = os.cpu_count()
    
    args_list = [(Game, mcts_iter, display) for _ in range(iterations)]
    
    # 멀티프로세싱 풀을 사용하여 병렬로 게임을 실행합니다.
    with multiprocessing.Pool(processes=num_workers, initializer=_init_worker, initargs=(Net, model_state, model_args)) as pool:
        results = pool.map(_run_one_game, args_list)
    
    total_time = time.time() - start_time
    
    game_stats = [0, 0, 0]  # [Black wins, White wins, Draws]
    for i, result_data in enumerate(results):
        *game_data, game_result_idx = result_data
        boards, actions, policies, qs, winner, reward = game_data
        
        save_data_to_buffer(Game, buffer, (boards, actions, policies, qs, winner, reward))
        game_stats[game_result_idx] += 1
        
        Game.logger.debug(f'Game {i+1}/{iterations} finished. Winner: {winner}. Buffer size: {buffer.size()}')

    Game.logger.info(f'Finished {iterations} games. Results: {game_stats}')
    Game.logger.debug(f'collect_data iter:{iterations} total time: {total_time}s average time per game: {total_time/iterations}s\n' +
                      f'game results: {game_stats}')


def train(model: Net, batch_size: int, buffer: ReplayBuffer, train_iterations, lr, device):
    """
    버퍼에서 샘플링한 데이터로 모델을 학습합니다.
    Args:
        model: 신경망 모델
        batch_size: 배치 크기
        buffer: ReplayBuffer 인스턴스
        train_iterations: 학습 반복 횟수
        lr: 학습률
        device: 학습 디바이스 (예: 'cuda', 'cpu')
    Returns:
        (loss, policy_loss, value_loss, l2_reg): 최종 손실값들
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # 0.0005 or 0.00025
    start_time = time.time()
    for i in range(train_iterations):
        states, policy_distributions, rewards = buffer.sample(batch_size)
        states = states.to(device)
        policy_distributions = policy_distributions.to(device)
        rewards = rewards.to(device)
        # 예시: policy_loss와 value_loss 외에 L2 정규화 항을 직접 추가하는 경우
        l2_lambda = 1e-4
        l2_reg = 0.0

        for name, param in model.named_parameters():
            # bias와 LayerNorm 파라미터는 weight decay 적용하지 않음
            if 'bias' in name or 'LayerNorm' in name:
                continue
            l2_reg += torch.sum(param ** 2)

        policy_logits, values = model(states)
        log_probs = torch.nn.functional.log_softmax(policy_logits, dim=1)
        policy_loss = torch.mean(torch.sum(-policy_distributions * log_probs, dim=1))
        value_loss = torch.nn.functional.mse_loss(values, rewards)
        loss = policy_loss + value_loss + l2_lambda * l2_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 50 == 0 :
            end_time = time.time()
            print('train_iter:', i+1,'loss:', loss.item(), 'policy_loss:', policy_loss.item(), 'value_loss:', value_loss.item(),'l2_loss:',l2_reg.item(), 'time:', end_time - start_time)
            start_time = end_time
    print('loss:', loss.item(), 'policy_loss:', policy_loss.item(), 'value_loss:', value_loss.item(),'l2_loss:',l2_reg.item())
    return loss.item(), policy_loss.item(), value_loss.item(), l2_reg.item()


