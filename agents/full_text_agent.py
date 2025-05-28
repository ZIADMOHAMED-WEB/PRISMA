# agents/full_text_agent.py

import os
from pathlib import Path
from agents.shared_enhanced_dqn import EnhancedDQNAgent

class FullTextAgent:
    def __init__(self, state_dim=384, action_dim=2, model_dir="models"):
        self.agent = EnhancedDQNAgent(state_dim, action_dim)
        self.model_path = Path(model_dir) / "full_text_agent.pth"
        self.load_model()

    def act(self, observation, training=True):
        state = observation['global'] if training else observation['local']
        return self.agent.act(state, training)


    def remember(self, state, action, reward, next_state, done):
        self.agent.remember(state, action, reward, next_state, done)

    def train(self):
        self.agent.replay()

    def save_model(self):
        self.agent.save_model(str(self.model_path))

    def load_model(self):
        if self.model_path.exists():
            self.agent.load_model(str(self.model_path))
