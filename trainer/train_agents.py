import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from agents.search_agent import SearchAgent
from agents.title_abstract_filter import TitleAbstractFilterAgent
from agents.full_text_agent import FullTextAgent
from agents.prisma_checker import PRISMAChecker
from rewards.enhanced_reward_system import EnhancedRewardSystem
from utils.arxiv_interface import search_arxiv
from utils.full_text_parser import parse_arxiv_pdf
from utils.logger import get_logger

# Ensure parent directory is in path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logger = get_logger("prisma_trainer")

class PRISMAAgentTrainer:
    def __init__(self, model_dir: str = None, checklist_pdf_path: str = None):
        """
        Initialize the trainer with agents, PRISMA checker, and reward system.
        Args:
            model_dir: Directory to save/load models (defaults to env variable or 'models')
            checklist_pdf_path: Path to PRISMA checklist PDF (defaults to env variable or 'PRISMA_2020_checklist.pdf')
        """
        self.model_dir = model_dir or os.getenv("MODEL_DIR", "models")
        self.checklist_pdf_path = checklist_pdf_path or os.getenv("PRISMA_CHECKLIST_PATH", "PRISMA_2020_checklist.pdf")
        
        # Initialize agents
        self.search_agent = SearchAgent(state_dim=386, model_dir=self.model_dir)
        self.abstract_agent = TitleAbstractFilterAgent(model_dir=self.model_dir)
        self.fulltext_agent = FullTextAgent(model_dir=self.model_dir)
        
        # Initialize PRISMA checker and reward system
        self.prisma = PRISMAChecker(checklist_pdf_path=self.checklist_pdf_path)
        self.reward_system = EnhancedRewardSystem()

        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, training_data: list, epochs: int = 10):
        """
        Train agents sequentially with PRISMA checklist-based rewards.
        Args:
            training_data: List of dicts with query, papers, ground truth, etc.
            epochs: Number of training epochs
        """
        for epoch in range(epochs):
            total_search_reward = 0.0
            total_abstract_reward = 0.0
            total_fulltext_reward = 0.0
            total_prisma_score = 0.0
            num_samples = 0

            for data in training_data:
                query = data["query"]
                papers = data["papers"]
                human_feedback = data.get("human_feedback", {"relevance": 0.8, "quality": 0.7})

                if not papers:
                    logger.warning(f"No papers retrieved for query: {query}")
                    continue

                # Step 1: Search Agent
                try:
                    query_embedding = self.reward_system.embed_text(query)
                    search_state = np.concatenate([query_embedding, [len(papers), 0.0]])
                    search_action = self.search_agent.act(search_state, training=True)
                    search_reward = self.prisma.evaluate_search_reward(papers, query_embedding, human_feedback)
                    self.search_agent.remember(search_state, search_action, search_reward, search_state, True)
                    self.search_agent.train()
                    total_search_reward += search_reward
                except Exception as e:
                    logger.error(f"Search agent processing failed for query '{query}': {e}")
                    continue

                # Step 2: Title/Abstract Filter Agent
                filtered_papers = []
                results = []
                try:
                    for i, paper in enumerate(papers):
                        paper_embed = self.reward_system.embed_text(paper.summary)
                        abstract_action = self.abstract_agent.act(paper_embed, training=True)
                        abstract_reward = self.prisma.evaluate_abstract_reward(
                            paper.summary, abstract_action, data["ground_truth_labels"].get(i)
                        )
                        self.abstract_agent.remember(paper_embed, abstract_action, abstract_reward, paper_embed, True)
                        self.abstract_agent.train()
                        total_abstract_reward += abstract_reward
                        num_samples += 1

                        if abstract_action in [1, 2]:  # Maybe or Include
                            filtered_papers.append((paper, i))
                            results.append({
                                "Title": paper.title,
                                "Year": paper.published.year,
                                "URL": paper.entry_id,
                                "Decision": "Include" if abstract_action == 2 else "Maybe",
                                "Abstract": paper.summary,
                                "Score": abstract_reward,
                                "Authors": ", ".join([a.name for a in paper.authors])
                            })
                except Exception as e:
                    logger.error(f"Abstract agent processing failed for query '{query}': {e}")
                    continue

                # Step 3: Full Text Agent
                try:
                    for paper, idx in filtered_papers:
                        full_text = parse_arxiv_pdf(paper.entry_id) or paper.summary
                        full_text_embed = self.reward_system.embed_text(full_text)
                        fulltext_action = self.fulltext_agent.act(full_text_embed, training=True)
                        citation_count = paper.citation_count if hasattr(paper, 'citation_count') else 0
                        fulltext_reward = self.prisma.evaluate_fulltext_reward(
                            full_text, fulltext_action, data["ground_truth_labels"].get(idx), citation_count
                        )
                        self.fulltext_agent.remember(full_text_embed, fulltext_action, fulltext_reward, full_text_embed, True)
                        self.fulltext_agent.train()
                        total_fulltext_reward += fulltext_reward
                        num_samples += 1

                        # Update results with full-text decision
                        for res in results:
                            if res["Title"] == paper.title:
                                res["Score"] = round((res["Score"] + fulltext_reward) / 2, 3)
                                res["Decision"] = "Include" if fulltext_action == 1 else "Exclude"
                                break
                except Exception as e:
                    logger.error(f"Fulltext agent processing failed for query '{query}': {e}")
                    continue

                # Step 4: PRISMA Compliance
                try:
                    metadata = {
                        "query": query,
                        "modified_query": query,
                        "from_year": 2000,
                        "to_year": 2025,
                        "search_action": search_action,
                        "inclusion_criteria_clear": 1.0,
                        "exclusion_criteria_clear": 1.0
                    }
                    results_df = pd.DataFrame(results)
                    prisma_score = self.prisma.evaluate_prisma_score(papers, metadata, results_df)
                    total_prisma_score += prisma_score
                except Exception as e:
                    logger.error(f"PRISMA score evaluation failed for query '{query}': {e}")
                    continue

            # Compute average metrics
            avg_search_reward = total_search_reward / len(training_data) if training_data else 0.0
            avg_abstract_reward = total_abstract_reward / num_samples if num_samples else 0.0
            avg_fulltext_reward = total_fulltext_reward / num_samples if num_samples else 0.0
            avg_prisma_score = total_prisma_score / len(training_data) if training_data else 0.0

            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Search Reward={avg_search_reward:.3f}, "
                f"Abstract Reward={avg_abstract_reward:.3f}, "
                f"Fulltext Reward={avg_fulltext_reward:.3f}, "
                f"PRISMA Score={avg_prisma_score:.3f}"
            )

            # Save models
            try:
                self.search_agent.save_model()
                self.abstract_agent.save_model()
                self.fulltext_agent.save_model()
                logger.info(f"Saved models for epoch {epoch+1}")
            except Exception as e:
                logger.error(f"Failed to save models for epoch {epoch+1}: {e}")

if __name__ == "__main__":
    # Initialize trainer
    trainer = PRISMAAgentTrainer()

    # Prepare training data
    queries = ["scene graph", "3D scene understanding", "visual commonsense reasoning"]
    training_data = []
    for query in queries:
        try:
            papers = search_arxiv(query, 2000, 2025, max_results=10)
            if not papers:
                logger.warning(f"No papers retrieved for query: {query}")
                continue
            # Simulate ground truth based on keyword presence
            ground_truth = {
                i: 1 if any(term in paper.summary.lower() for term in ["scene graph", "3d scene", "commonsense"])
                else 0 for i, paper in enumerate(papers)
            }
            training_data.append({
                "query": query,
                "papers": papers,
                "search_action": trainer.search_agent.act(
                    np.concatenate([trainer.reward_system.embed_text(query), [len(papers), 0.0]]),
                    training=True
                ),
                "filter_decisions": [2 if gt == 1 else 0 for gt in ground_truth.values()],
                "ground_truth_labels": ground_truth,
                "human_feedback": {"relevance": 0.8, "quality": 0.7}
            })
        except Exception as e:
            logger.error(f"Data preparation failed for query '{query}': {e}")

    # Run training
    if training_data:
        trainer.train(training_data, epochs=10)
        logger.info("Training completed successfully")
    else:
        logger.error("No valid training data available. Exiting.")
        print("Error: No valid training data available.")
