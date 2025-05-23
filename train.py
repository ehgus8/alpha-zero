import torch
from ai.nn import Net
from replay_buffer import ReplayBuffer
import test
import time
import numpy as np

def rotate_data(board, policy, k):
    channel, rows, cols = board.shape
    rotated_board = np.rot90(board,k=k, axes=(1, 2)).copy()
    policy_2d = policy.reshape((rows, cols)).copy()
    policy_2d = np.rot90(policy_2d, k=k)
    rotated_policy_1d = policy_2d.flatten()

    return rotated_board, rotated_policy_1d

def flip_data(board, policy, mode):
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
                
        


def collect_data(Game, model: Net, buffer: ReplayBuffer, iterations: int, mcts_iter: int, display=False):
    """
    Generate self-play data using MCTS.
    Arge:
        model (nn.Module): The neural network model.
    """
    model.eval()
    total_time = 0
    game_results = [0,0,0] # first, second player, draw
    with torch.no_grad():
        for iter in range(iterations):
            game = Game()
            start_time = time.time()
            boards, actions, policy_distributions, qs, winner = game.self_play(model, mcts_iter, display)
            total_time += (time.time() - start_time)
            if winner == -1: # draw
                reward = 0
                game_results[2] += 1
            else:
                reward = 1
                game_results[0 if winner == 0 else 1] += 1
            save_data_to_buffer(Game, buffer, (boards, actions, policy_distributions, qs, winner, reward))
                
            Game.logger.debug(f'collect_data iter({iter+1}/{iterations}) time: {time.time()-start_time}s, game results: {game_results}')
            
            if (iter+1) % 20 == 0:
                print("iter:",iter+1,"Player", winner, "wins!", 'game_results:', game_results)
            print(f'buffer status:{buffer.size()}')
    Game.logger.debug(f'collect_data iter:{iterations} total time: {total_time}s average time per game: {total_time/iterations}s\n' +
                      f'game results: {game_results}')

def train(model: Net, batch_size: int, buffer: ReplayBuffer, train_iterations, lr, device):
    """
    Train the model using MCTS.
    Arge:
        model (nn.Module): The neural network model.
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


