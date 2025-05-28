import os
from pathlib import Path
from agents.shared_enhanced_dqn import EnhancedDQNAgent

class TitleAbstractFilterAgent:
    def __init__(self, state_dim=384, action_dim=3, model_dir="models"):
        self.agent = EnhancedDQNAgent(state_dim, action_dim)
        self.model_path = Path(model_dir) / "title_abstract_filter_agent.pth"
        self.load_model()

    def act(self, state, training=True):
        return self.agent.act(state, training)

    def remember(self, state, action, reward, next_state, done):
        self.agent.remember(state, action, reward, next_state, done)

    def get_all_experiences(self):
        """
        Returns all experiences from this agent's replay buffer for centralized training.
        """
        return self.agent.get_all_experiences()

    def clear_replay_buffer(self):
        """
        Clears the replay buffer after sending experiences for centralized training.
        """
        self.agent.clear_replay_buffer()

    def train_on_batch(self, batch_experiences):
        """
        Train the agent centrally on combined experiences from multiple agents.
        """
        self.agent.train_on_batch(batch_experiences)

    def save_model(self):
        self.agent.save_model(str(self.model_path))

    def load_model(self):
        if self.model_path.exists():
            self.agent.load_model(str(self.model_path))
