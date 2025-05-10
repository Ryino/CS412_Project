import numpy as np
import random
import torch
import torch.nn as nn


class ReplayBuffer:
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.seed = seed
        random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch = self.buffer
        else:
            batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = zip(*batch)
        return state, action, reward, next_state

    def add_batch(self, batch):
        for i in batch:
            self.push(i[0], i[1], i[2], i[3])

    def sample_batch(self, batch_size, length, dim):
        # Get the actual batch size (might be smaller if buffer is not full)
        actual_batch_size = min(batch_size, len(self.buffer))
        
        # Sample from buffer
        states, actions, rewards, next_states = self.sample(actual_batch_size)
        
        # Stack the tensors along the batch dimension
        try:
            s_batch = torch.stack(states, dim=0)
            a_batch = torch.stack(actions, dim=0)
            r_batch = torch.stack([r.unsqueeze(0) for r in rewards], dim=0)
            s2_batch = torch.stack(next_states, dim=0)
        except RuntimeError as e:
            print(f"Error stacking tensors: {e}")
            print(f"States shape: {[s.shape for s in states]}")
            print(f"Actions shape: {[a.shape for a in actions]}")
            print(f"Rewards shape: {[r.shape for r in rewards]}")
            print(f"Next states shape: {[s.shape for s in next_states]}")
            raise

        # Ensure all tensors are on the correct device
        s_batch = s_batch.to(self.device)
        a_batch = a_batch.to(self.device)
        r_batch = r_batch.to(self.device)
        s2_batch = s2_batch.to(self.device)

        return s_batch, a_batch, r_batch, s2_batch

    def __len__(self):
        return len(self.buffer)

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False