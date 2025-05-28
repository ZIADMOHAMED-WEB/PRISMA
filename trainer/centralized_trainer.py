# trainers/centralized_trainer.py

from typing import List

class CentralizedTrainer:
    def __init__(self, agents: List):
        """
        Initialize with a list of agents (each should implement
        get_all_experiences(), clear_replay_buffer(), and train_on_batch()).
        """
        self.agents = agents

    def centralized_training_step(self):
        """
        Aggregate experiences from all agents, clear their buffers,
        and train each agent on the combined experiences.
        """
        combined_experiences = []

        # Gather experiences from all agents
        for agent in self.agents:
            combined_experiences.extend(agent.get_all_experiences())
            agent.clear_replay_buffer()

        if len(combined_experiences) == 0:
            print("⚠ Warning: No experiences collected for centralized training.")
            return

        # Train each agent on the combined batch
        for agent in self.agents:
            agent.train_on_batch(combined_experiences)

    def save_all_models(self):
        """
        Save models for all agents.
        """
        for agent in self.agents:
            agent.save_model()

    def load_all_models(self):
        """
        Load models for all agents.
        """
        for agent in self.agents:
            agent.load_model()
