import torch
import numpy as np
import random
from collections import deque

class EnhancedDQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr

        self.q_network = self.build_network()
        self.target_network = self.build_network()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_freq = 100
        self.training_step = 0

    def build_network(self):
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.action_dim)
        )

    def act(self, state, training=True):
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_all_experiences(self):
        """
        Returns all experiences currently in the replay buffer.
        Useful for centralized training.
        """
        return list(self.memory)

    def clear_replay_buffer(self):
        """Clears the replay buffer after experiences have been sent for centralized training."""
        self.memory.clear()

    def train_on_batch(self, batch_experiences):
        """
        Train on a combined batch of experiences from multiple agents.
        batch_experiences: list of (state, action, reward, next_state, done)
        """
        if len(batch_experiences) < self.batch_size:
            return

        batch = random.sample(batch_experiences, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (~dones)

        loss = torch.nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()

    def replay(self):
        """
        Optional: If you want to keep single-agent training mode,
        you can keep this method for decentralized learning fallback.
        """
        self.train_on_batch(list(self.memory))

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, path):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
