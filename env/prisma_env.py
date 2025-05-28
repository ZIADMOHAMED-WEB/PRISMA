# env/prisma_env.py

from pettingzoo import AECEnv
from gymnasium import spaces
import numpy as np

class PRISMAEnv(AECEnv):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        self.agents = ["search", "title_abstract", "full_text", "prisma_checker"]
        self.possible_agents = self.agents[:]
        self.agent_idx = 0

        self.observation_spaces = {
            agent: spaces.Dict({
                "text": spaces.Box(low=0, high=1, shape=(768,), dtype=np.float32)
            }) for agent in self.agents
        }

        self.action_spaces = {
            "search": spaces.Discrete(5),             # e.g. modify query, change source
            "title_abstract": spaces.Discrete(3),      # include/exclude/maybe
            "full_text": spaces.Discrete(2),           # include/exclude
            "prisma_checker": spaces.Discrete(1),      # no-op or accept
        }

    def reset(self, seed=None, options=None):
        self.agent_idx = 0
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.observations = {
            agent: {"text": np.random.rand(768).astype(np.float32)}
            for agent in self.agents
        }
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self.agents[self.agent_idx]

    def step(self, action):
        agent = self.agent_selection

        # Dummy reward logic (you'll replace this later)
        self.rewards[agent] = float(np.random.rand())
        self.dones[agent] = True
        self.agent_idx += 1

        if self.agent_idx >= len(self.agents):
            self.agents = []
        else:
            self.agent_selection = self.agents[self.agent_idx]

    def observe(self, agent):
        return self.observations[agent]
