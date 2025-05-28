import os
from pathlib import Path
from agents.shared_enhanced_dqn import EnhancedDQNAgent

class SearchAgentCTDE:
    def __init__(self, state_dim=386, action_dim=5, model_dir="models"):
        self.agent = EnhancedDQNAgent(state_dim, action_dim)
        self.model_path = Path(model_dir) / "search_agent.pth"
        self.load_model()

    def act(self, state, training=True):
        # Decentralized Execution: act independently per agent
        return self.agent.act(state, training)

    def remember(self, state, action, reward, next_state, done):
        # Store experience in local replay buffer (can be per agent)
        self.agent.remember(state, action, reward, next_state, done)

    def collect_experiences(self):
        """
        Returns all stored experiences from this agent's replay buffer for centralized training.
        The format should be compatible for batch training in the centralized learner.
        """
        return self.agent.get_all_experiences()

    def clear_experiences(self):
        """Clear this agent's replay buffer after experiences have been collected."""
        self.agent.clear_replay_buffer()

    def centralized_train(self, batch_experiences):
        """
        Centralized Training step using aggregated batch experiences from all agents.
        batch_experiences is a list/dataset containing experiences from multiple agents.
        """
        self.agent.train_on_batch(batch_experiences)

    def save_model(self):
        self.agent.save_model(str(self.model_path))

    def load_model(self):
        if self.model_path.exists():
            try:
                self.agent.load_model(str(self.model_path))
            except RuntimeError as e:
                print(f"⚠ Warning: Failed to load model due to {e}. Initializing new model.")
                # Model will remain initialized with random weights
