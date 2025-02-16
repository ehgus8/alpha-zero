import torch
from nn import Net
from replay_buffer import ReplayBuffer
import test
import time

def self_play(Game, model: Net, buffer: ReplayBuffer, iterations: int, mcst_iter: int, display=False):
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
            boards, policy_distributions, winner = game.self_play(model, mcst_iter, display)
            total_time += (time.time() - start_time)
            if winner == -1: # draw
                reward = 0
                game_results[2] += 1
            else:
                reward = 1
                game_results[0 if winner == 0 else 1] += 1
            for i in range(len(boards)):
                buffer.add(boards[i], policy_distributions[i], [reward] if boards[i][2,0,0] == winner else [-reward])
            if (iter+1) % 20 == 0:
                print("iter:",iter+1,"Player", winner, "wins!", 'game_results:', game_results)
    Game.logger.debug(f'self_play iter:{iterations} total time: {total_time}s average time per game: {total_time/iterations}s\n' +
                      f'game results: {game_results}')

def train(model: Net, batch_size: int, buffer: ReplayBuffer, train_iterations, device):
    """
    Train the model using MCTS.
    Arge:
        model (nn.Module): The neural network model.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
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

        policy_logits, values = model(states.to(device))
        log_probs = torch.nn.functional.log_softmax(policy_logits, dim=1)
        policy_loss = torch.mean(torch.sum(-policy_distributions * log_probs, dim=1))
        value_loss = torch.nn.functional.mse_loss(values, rewards)
        loss = policy_loss + value_loss + l2_lambda * l2_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 20 == 0 :
            end_time = time.time()
            print('train_iter:', i+1,'loss:', loss.item(), 'policy_loss:', policy_loss.item(), 'value_loss:', value_loss.item(),'l2_loss:',l2_reg.item(), 'time:', end_time - start_time)
            start_time = end_time
    print('loss:', loss.item(), 'policy_loss:', policy_loss.item(), 'value_loss:', value_loss.item(),'l2_loss:',l2_reg.item())


