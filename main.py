import sys
import os
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

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import centralized trainer
try:
    from trainers.centralized_trainer import CentralizedTrainer
except ImportError:
    CentralizedTrainer = None
    print("⚠ Warning: Centralized training module not found. Training mode disabled.")

# Configuration
MODEL_DIR = os.getenv("MODEL_DIR", "models")
CHECKLIST_PATH = os.getenv("PRISMA_CHECKLIST_PATH", "PRISMA_2020_checklist.pdf")
MAX_RESULTS = 30

logger = get_logger("prisma_main")

def modify_query(topic, action):
    """Map SearchAgent actions to query modifications."""
    if action == 0:
        return topic
    elif action == 1:
        return f"{topic} machine learning"
    elif action == 2:
        terms = topic.split()
        return " ".join(terms[:2]) if len(terms) > 2 else topic
    elif action == 3:
        return f"{topic} reinforcement learning" if "reinforcement" not in topic.lower() else topic
    elif action == 4:
        return f"{topic} artificial intelligence"
    return topic

def main():
    logger.info("Starting PRISMA literature review system")
    mode = input("Enter mode (train/infer): ").lower()
    
    # Initialize agents
    search_agent = SearchAgent(state_dim=386, model_dir=MODEL_DIR)
    abstract_agent = TitleAbstractFilterAgent(model_dir=MODEL_DIR)
    fulltext_agent = FullTextAgent(model_dir=MODEL_DIR)
    prisma_checker = PRISMAChecker(checklist_pdf_path=CHECKLIST_PATH)
    reward_system = EnhancedRewardSystem()

    if mode == "train":
        if CentralizedTrainer is None:
            logger.error("Training mode not available: CentralizedTrainer not found")
            print("❌ Training mode not available.")
            return

        logger.info("Entering centralized training mode")
        agents = [search_agent, abstract_agent, fulltext_agent]
        trainer = CentralizedTrainer(agents)

        query = input("Enter training query: ")
        try:
            papers = search_arxiv(query, 2000, 2025, MAX_RESULTS)
            if not papers:
                logger.error("No papers found for training")
                print("❌ No papers found for training.")
                return

            # Simplified experience gathering loop - replace with your actual environment interactions
            for epoch in range(10):
                logger.info(f"Epoch {epoch+1}/10")

                # Example loop to simulate agent experience generation
                for paper in papers:
                    # SearchAgent experience (simulate)
                    search_state = np.concatenate([reward_system.embed_text(query), [len(papers), 0.0]])
                    search_action = search_agent.act(search_state, training=True)
                    search_reward = prisma_checker.evaluate_search_reward(papers, reward_system.embed_text(query), prisma_checker.checklist_data)
                    search_agent.remember(search_state, search_action, search_reward, search_state, done=False)

                    # Abstract agent experience
                    abstract_embed = reward_system.embed_text(paper.summary)
                    abstract_action = abstract_agent.act(abstract_embed, training=True)
                    abstract_reward = prisma_checker.evaluate_abstract_reward(paper.summary, abstract_action, prisma_checker.checklist_data)
                    abstract_agent.remember(abstract_embed, abstract_action, abstract_reward, abstract_embed, done=False)

                    # Full text agent experience
                    full_text = parse_arxiv_pdf(paper.entry_id) or paper.summary
                    full_text_embed = reward_system.embed_text(full_text)
                    fulltext_action = fulltext_agent.act(full_text_embed, training=True)
                    citation_count = getattr(paper, 'citation_count', 0)
                    fulltext_reward = prisma_checker.evaluate_fulltext_reward(full_text, fulltext_action, None, citation_count, prisma_checker.checklist_data)
                    fulltext_agent.remember(full_text_embed, fulltext_action, fulltext_reward, full_text_embed, done=False)

                # Centralized training step
                trainer.centralized_training_step()
                logger.info("Centralized training step complete.")

            trainer.save_all_models()
            logger.info("Training completed successfully.")
            print("✅ Training completed successfully.")
            return

        except Exception as e:
            logger.error(f"Training failed: {e}")
            print(f"❌ Training failed: {e}")
            return

    # Inference mode
    topic = input("🔍 Enter research topic: ")
    try:
        from_year = int(input("📅 Start year: "))
        to_year = int(input("📅 End year: "))
    except ValueError:
        logger.error("Invalid year input, using default range 2000-2025")
        from_year, to_year = 2000, 2025

    # Step 1: Search Agent
    try:
        query_embedding = reward_system.embed_text(topic)
        papers = search_arxiv(topic, from_year, to_year, max_results=MAX_RESULTS)
        if not papers:
            logger.error("No papers found")
            print("❌ No papers found.")
            return
        logger.info(f"Retrieved {len(papers)} papers")
        search_state = np.concatenate([query_embedding, [len(papers), 0.0]])
        search_action = search_agent.act(search_state, training=False)
        modified_query = modify_query(topic, search_action)
        prisma_data = prisma_checker.checklist_data
        search_reward = prisma_checker.evaluate_search_reward(papers, query_embedding, prisma_data)
        logger.info(f"Search action {search_action}: Query='{modified_query}', Reward={search_reward:.3f}")
        if modified_query != topic:
            papers = search_arxiv(modified_query, from_year, to_year, max_results=MAX_RESULTS)
            logger.info(f"Retrieved {len(papers)} papers with modified query")
    except Exception as e:
        logger.error(f"Search evaluation failed: {e}")
        print("❌ Failed to retrieve papers.")
        return

    # Step 2: Title/Abstract Filter Agent
    filtered_papers = []
    results = []
    try:
        for i, paper in enumerate(papers):
            paper_embed = reward_system.embed_text(paper.summary)
            abstract_action = abstract_agent.act(paper_embed, training=False)
            abstract_reward = prisma_checker.evaluate_abstract_reward(paper.summary, abstract_action, prisma_data=prisma_data)
            if abstract_action in [1, 2]:  # Maybe or Include
                filtered_papers.append((paper, i))
            results.append({
                "Title": paper.title,
                "Year": paper.published.year,
                "URL": paper.entry_id,
                "Decision": "Include" if abstract_action == 2 else "Maybe" if abstract_action == 1 else "Exclude",
                "Abstract": paper.summary,
                "Score": abstract_reward,
                "Authors": ", ".join([a.name for a in paper.authors])
            })
    except Exception as e:
        logger.error(f"Abstract evaluation failed: {e}")
        print("❌ Failed to evaluate abstracts.")

    # Step 3: Full Text Agent
    try:
        for paper, idx in filtered_papers:
            full_text = parse_arxiv_pdf(paper.entry_id) or paper.summary
            full_text_embed = reward_system.embed_text(full_text)
            fulltext_action = fulltext_agent.act(full_text_embed, training=False)
            citation_count = getattr(paper, 'citation_count', 0)
            fulltext_reward = prisma_checker.evaluate_fulltext_reward(full_text, fulltext_action, None, citation_count, prisma_data)
            for res in results:
                if res["Title"] == paper.title:
                    res["Score"] = round((res["Score"] + fulltext_reward) / 2, 3)
                    res["Decision"] = "Include" if fulltext_action == 1 else "Exclude"
                    break
    except Exception as e:
        logger.error(f"Full-text evaluation failed: {e}")
        print("❌ Failed to evaluate full texts.")

    # Step 4: Save Results and Compute PRISMA Score
    try:
        df = pd.DataFrame(results).sort_values(by="Score", ascending=False).head(20)
        output_path = "results.csv"
        output_dir = os.path.dirname(output_path) or "."
        if not os.access(output_dir, os.W_OK):
            logger.error(f"No write permission for directory: {output_dir}")
            output_path = os.path.join(os.path.expanduser("~"), "results.csv")
            logger.info(f"Attempting to save to fallback path: {output_path}")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved top 20 papers to {output_path}")
        print(f"✅ Saved top 20 papers to {output_path}")

        metadata = {
            "query": modified_query,
            "modified_query": modified_query,
            "from_year": from_year,
            "to_year": to_year,
            "search_action": search_action,
            "inclusion_criteria_clear": 1.0,
            "exclusion_criteria_clear": 1.0
        }
        prisma_score = prisma_checker.evaluate_prisma_score(papers, metadata, df)
        print(f"📊 PRISMA Compliance Score: {prisma_score:.2f}")
    except PermissionError as e:
        logger.error(f"Permission denied when saving {output_path}: {e}")
        print(f"❌ Permission denied: Could not save {output_path}.")
    except Exception as e:
        logger.error(f"Failed to save results or compute PRISMA score: {e}")
        print(f"❌ Failed to save results: {e}")

if __name__ == "__main__":
    main()
