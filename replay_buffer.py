from collections import deque
import numpy as np
import torch
import pickle
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
    
    def size(self):
        return len(self.buffer)
    
    def save_pickle(self, filename):
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.buffer, f)
            print(f"Replay Buffer saved to {filename} successfully.")
        except:
            print("Failed to save Replay Buffer.")


    def load_pickle(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.buffer = pickle.load(f)
            print(f"Replay Buffer loaded from {filename} successfully.")
        except:
            print("Failed to load Replay Buffer.")