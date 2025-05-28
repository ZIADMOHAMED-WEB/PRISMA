import numpy as np
from sentence_transformers import SentenceTransformer
from collections import deque, defaultdict
from typing import List, Dict, Optional


class EnhancedRewardSystem:
    """
    Advanced reward system with PRISMA checklist integration and human feedback,
    updated for CTDE with per-agent feedback tracking and batch reward computation.
    """
    def __init__(self):
        self.relevance_threshold = 0.7
        self.diversity_bonus = 0.1
        self.human_feedback_weight = 0.3
        self.prisma_weight = 0.4
        
        # Feedback history separated by agent_id
        self.feedback_history = defaultdict(lambda: deque(maxlen=1000))
        
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        
        self.checklist_items = [
            'search_strategy_documented', 'inclusion_criteria_clear', 'exclusion_criteria_clear',
            'study_selection_process', 'data_extraction_systematic', 'quality_assessment_performed',
            'results_synthesized', 'limitations_discussed', 'study_bias_assessment', 'certainty_assessment'
        ]

    def compute_search_reward(self, papers: List, query_embedding: np.ndarray, 
                             prisma_data: Dict, human_feedback: Optional[Dict] = None,
                             agent_id: Optional[str] = None) -> float:
        if not papers:
            return -1.0

        relevance_scores = []
        for paper in papers:
            paper_embedding = self.embed_text(paper.summary)
            similarity = np.dot(query_embedding, paper_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(paper_embedding)
            )
            relevance_scores.append(similarity)

        avg_relevance = np.mean(relevance_scores)
        diversity_score = self.calculate_diversity(papers)
        prisma_score = np.mean([prisma_data.get(item, 0.0) for item in ['search_strategy_documented', 'information_sources']])

        feedback_score = 0.0
        if human_feedback and agent_id is not None:
            feedback_score = self.integrate_human_feedback(agent_id, human_feedback)

        reward = (avg_relevance * 0.5 +
                  self.diversity_bonus * diversity_score +
                  self.prisma_weight * prisma_score +
                  self.human_feedback_weight * feedback_score)

        return np.clip(reward, -1.0, 1.0)

    def compute_filter_reward(self, paper_data: Dict, decision: int, 
                             prisma_data: Dict, ground_truth: Optional[int] = None) -> float:
        base_reward = 0.0
        abstract_lower = paper_data.get('abstract', '').lower()
        has_methodology = any(word in abstract_lower
                              for word in ['method', 'approach', 'algorithm', 'framework'])
        has_results = any(word in abstract_lower
                          for word in ['result', 'performance', 'evaluation', 'experiment'])
        citation_count = paper_data.get('citation_count', 0)

        if decision == 1 or decision == 2:  # Include or Maybe
            base_reward = 0.5
            if has_methodology and has_results:
                base_reward += 0.3
            if citation_count > 10:
                base_reward += 0.2
        elif decision == 0:  # Exclude
            base_reward = 0.1
            if not has_methodology or not has_results:
                base_reward += 0.2

        if ground_truth is not None:
            if decision == ground_truth:
                base_reward += 0.5
            else:
                base_reward -= 0.3

        prisma_score = np.mean([prisma_data.get(item, 0.0) for item in ['inclusion_criteria_clear', 'exclusion_criteria_clear', 'study_selection_process']])
        reward = base_reward * 0.6 + self.prisma_weight * prisma_score

        return np.clip(reward, -1.0, 1.0)

    def compute_prisma_reward(self, review_data: Dict) -> float:
        compliance_score = np.mean([review_data.get(item, 0.0) for item in self.checklist_items])
        if compliance_score > 0.8:
            compliance_score += 0.2
        return np.clip(compliance_score, 0.0, 1.0)

    def calculate_diversity(self, papers: List) -> float:
        if len(papers) < 2:
            return 0.0
        embeddings = [self.embed_text(p.summary) for p in papers]
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)
        avg_similarity = np.mean(similarities)
        diversity = 1.0 - avg_similarity
        return max(0.0, diversity)

    def integrate_human_feedback(self, agent_id: str, feedback: Dict) -> float:
        self.feedback_history[agent_id].append(feedback)
        recent_feedback = list(self.feedback_history[agent_id])[-10:]
        if not recent_feedback:
            return 0.0
        relevance_scores = [f.get('relevance', 0.5) for f in recent_feedback]
        quality_scores = [f.get('quality', 0.5) for f in recent_feedback]
        weighted_score = 0.6 * np.mean(relevance_scores) + 0.4 * np.mean(quality_scores)
        return (weighted_score - 0.5) * 2

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True)

    def compute_search_rewards_batch(self,
                                    list_of_papers_lists: List[List],
                                    list_of_query_embeddings: List[np.ndarray],
                                    list_of_prisma_data: List[Dict],
                                    list_of_human_feedback: List[Optional[Dict]],
                                    list_of_agent_ids: List[str]) -> List[float]:
        """
        Batch compute search rewards for multiple agents.
        """
        rewards = []
        for papers, q_emb, prisma, fb, agent_id in zip(
            list_of_papers_lists, list_of_query_embeddings, list_of_prisma_data, list_of_human_feedback, list_of_agent_ids
        ):
            reward = self.compute_search_reward(papers, q_emb, prisma, fb, agent_id)
            rewards.append(reward)
        return rewards
