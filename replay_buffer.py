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
    
    def save_pickle(self, filename, clean_old=True):
        folder = os.path.dirname(filename)
        if clean_old:
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if file.startswith('replay_buffer'):
                        file_path = os.path.join(folder, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
            else:
                os.makedirs(folder)
        else:
            if not os.path.exists(folder):
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

    def load_latest_buffer(self):
        """
        'replay_buffers' 디렉토리에서 가장 최신 버전의 버퍼를 불러옵니다.
        파일이 없으면 아무것도 하지 않습니다.
        """
        buffers_dir = os.path.join(os.path.dirname(__file__), 'replay_buffers')
        latest_version = -1
        latest_iter = -1
        latest_buffer_path = None
        if os.path.exists(buffers_dir):
            for filename in os.listdir(buffers_dir):
                if filename.startswith('replay_buffer_v') and filename.endswith('.pkl'):
                    try:
                        # 예: replay_buffer_v43_i_26.pkl
                        parts = filename.split('_')
                        v_idx = int(parts[2][1:])  # v43
                        i_idx = int(parts[4].split('.')[0])  # i_26
                        # 버전이 더 높거나, 버전이 같으면 iteration이 더 높은 것
                        if (v_idx > latest_version) or (v_idx == latest_version and i_idx > latest_iter):
                            latest_version = v_idx
                            latest_iter = i_idx
                            latest_buffer_path = os.path.join(buffers_dir, filename)
                    except (IndexError, ValueError):
                        continue
        if latest_buffer_path:
            try:
                with open(latest_buffer_path, 'rb') as f:
                    loaded_buffer = pickle.load(f)
                self.buffer = deque(loaded_buffer, maxlen=self.buffer_size)
                print(f"Replay Buffer loaded from {latest_buffer_path} successfully.")
            except Exception as e:
                print(f"Failed to load Replay Buffer: {e}")
        else:
            print("No saved replay buffer found. Starting with empty buffer.")