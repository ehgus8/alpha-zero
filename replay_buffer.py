from collections import deque
import numpy as np
import torch
import pickle
import os

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state: np.ndarray, policy_distribution: np.ndarray, reward: float):
        self.buffer.append((state, policy_distribution, reward))
    
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states, policy_distributions, rewards = zip(*batch)
        states = np.array(states, dtype=np.float32)
        policy_distributions = np.array(policy_distributions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        
        return (torch.tensor(states), 
                torch.tensor(policy_distributions), 
                torch.tensor(rewards))
    
    def head(self, size):
        for i in range(size):
            print(self.buffer[i])
    def tail(self, size):
        for i in range(1, size+1):
            print(self.buffer[-i])

    def size(self):
        return len(self.buffer)
    
    def save_pickle(self, filename):
        folder = os.path.dirname(filename)
        if os.path.exists(folder):
            # 폴더 내에서 'replay_buffer'로 시작하는 파일만 삭제합니다.
            for file in os.listdir(folder):
                if file.startswith('replay_buffer'):
                    file_path = os.path.join(folder, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
        else:
            os.makedirs(folder)

        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.buffer, f)
            print(f"Replay Buffer saved to {filename} successfully.")
        except Exception as e:
            print("Failed to save Replay Buffer.", e)


    def load_pickle(self, filename):
        buffer_path = os.path.join(os.path.dirname(__file__), f'replay_buffers/{filename}.pkl')
        try:
            with open(buffer_path, 'rb') as f:
                loaded_buffer = pickle.load(f)
            self.buffer = deque(loaded_buffer, maxlen=self.buffer_size)
            print(f"Replay Buffer loaded from {filename} successfully.")
        except:
            print("Failed to load Replay Buffer.")