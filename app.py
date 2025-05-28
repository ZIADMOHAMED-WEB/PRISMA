import streamlit as st
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os

from agents.search_agent import SearchAgent
from agents.title_abstract_filter import TitleAbstractFilterAgent
from agents.full_text_agent import FullTextAgent
from agents.prisma_checker import PRISMAChecker
from rewards.enhanced_reward_system import EnhancedRewardSystem
from utils.arxiv_interface import search_arxiv
from utils.full_text_parser import parse_arxiv_pdf
from trainer.train_agents import PRISMAAgentTrainer
from utils.logger import get_logger

logger = get_logger("prisma_app")

st.set_page_config(page_title="PRISMA-MARL System", layout="wide")
st.title("📚 PRISMA-MARL System")
st.markdown("Automated systematic literature reviews with multi-agent RL and PRISMA 2020 feedback")

# Configuration
MODEL_DIR = os.getenv("MODEL_DIR", "E:\RL\prisma_marl_project\models")
CHECKLIST_PATH = os.getenv("PRISMA_CHECKLIST_PATH", "PRISMA_2020_checklist.pdf")

# Session state for persistent model access and caching
if "agents" not in st.session_state:
    st.session_state.agents = {
        "search": SearchAgent(state_dim=386, model_dir=MODEL_DIR),
        "abstract": TitleAbstractFilterAgent(model_dir=MODEL_DIR),
        "fulltext": FullTextAgent(model_dir=MODEL_DIR)
    }
if "prisma" not in st.session_state:
    st.session_state.prisma = PRISMAChecker(checklist_pdf_path=CHECKLIST_PATH)
if "reward" not in st.session_state:
    st.session_state.reward = EnhancedRewardSystem()
if "embedding_cache" not in st.session_state:
    st.session_state.embedding_cache = {}

# Check model status
model_status = all(os.path.exists(os.path.join(MODEL_DIR, f"{agent}_agent.pth")) for agent in ["search", "abstract", "fulltext"])
st.sidebar.header("🧠 System Status")
st.sidebar.write(f"Trained Models Loaded: {'✅ Yes' if model_status else '❌ No'}")
if not model_status:
    st.sidebar.warning("No trained models found. Train agents or check model directory.")

# Training Section
st.sidebar.header("🧠 Train Agents")
if st.sidebar.button("🎯 Start Training"):
    with st.spinner("Preparing training data..."):
        # Prepare training data (similar to train_agents.py)
        queries = ["scene graph", "3D scene understanding", "visual commonsense reasoning"]
        training_data = []
        for query in queries:
            try:
                papers = search_arxiv(query, 2000, 2025, max_results=5)
                if not papers:
                    logger.warning(f"No papers retrieved for query: {query}")
                    continue
                ground_truth = {
                    i: 1 if any(term in paper.summary.lower() for term in ["scene graph", "3d scene", "commonsense"])
                    else 0 for i, paper in enumerate(papers)
                }
                training_data.append({
                    "query": query,
                    "papers": papers,
                    "search_action": st.session_state.agents["search"].act(
                        np.concatenate([st.session_state.reward.embed_text(query), [len(papers), 0.0]]),
                        training=True
                    ),
                    "filter_decisions": [2 if gt == 1 else 0 for gt in ground_truth.values()],
                    "ground_truth_labels": ground_truth,
                    "human_feedback": {"relevance": 0.8, "quality": 0.7}
                })
            except Exception as e:
                logger.error(f"Data preparation failed for query '{query}': {e}")

    if training_data:
        with st.spinner("Training agents (10 epochs)..."):
            try:
                trainer = PRISMAAgentTrainer(model_dir=MODEL_DIR, checklist_pdf_path=CHECKLIST_PATH)
                trainer.train(training_data, epochs=10)
                st.success("✅ Training completed and models saved!")
                # Reload agents to use updated models
                st.session_state.agents = {
                    "search": SearchAgent(state_dim=386, model_dir=MODEL_DIR),
                    "abstract": TitleAbstractFilterAgent(model_dir=MODEL_DIR),
                    "fulltext": FullTextAgent(model_dir=MODEL_DIR)
                }
            except Exception as e:
                st.error(f"Training failed: {e}")
                logger.error(f"Training error: {e}")
    else:
        st.error("No valid training data. Check arXiv connectivity or query terms.")

# Literature Review Section
st.sidebar.header("🔍 Run Literature Review")
topic = st.sidebar.text_input("Research Topic:", "deep reinforcement learning")
from_year = st.sidebar.number_input("From Year:", min_value=2000, max_value=2025, value=2000)
to_year = st.sidebar.number_input("To Year:", min_value=2000, max_value=2025, value=2025)
max_results = st.sidebar.slider("Max Results:", min_value=5, max_value=30, value=10)

if st.sidebar.button("🚀 Start Review"):
    with st.spinner("Searching arXiv..."):
        try:
            papers = search_arxiv(topic, from_year, to_year, max_results)
        except Exception as e:
            st.error(f"Search failed: {e}")
            logger.error(f"Search error: {e}")
            papers = []

    if not papers:
        st.error("No papers found. Try a different topic or broader year range.")
    else:
        st.success(f"✅ Found {len(papers)} papers")

        results = []
        # Progress bar for inference
        for paper in tqdm(papers, desc="Processing Papers"):
            try:
                # Cache abstract embedding
                abstract_key = f"abstract_{paper.entry_id}"
                if abstract_key not in st.session_state.embedding_cache:
                    st.session_state.embedding_cache[abstract_key] = st.session_state.reward.embed_text(paper.summary)
                abstract_embed = st.session_state.embedding_cache[abstract_key]

                # TitleAbstractFilterAgent
                abstract_action = st.session_state.agents['abstract'].act(abstract_embed, training=False)
                abstract_reward = st.session_state.prisma.evaluate_abstract_reward(paper.summary, abstract_action, 1.0)

                # FullTextAgent (only for Include/Maybe)
                fulltext_action = None
                fulltext_reward = 0.0
                if abstract_action in [1, 2]:  # Maybe or Include
                    full_text = parse_arxiv_pdf(paper.entry_id) or paper.summary
                    fulltext_key = f"fulltext_{paper.entry_id}"
                    if fulltext_key not in st.session_state.embedding_cache:
                        st.session_state.embedding_cache[fulltext_key] = st.session_state.reward.embed_text(full_text)
                    fulltext_embed = st.session_state.embedding_cache[fulltext_key]
                    fulltext_action = st.session_state.agents['fulltext'].act(fulltext_embed, training=False)
                    fulltext_reward = st.session_state.prisma.evaluate_fulltext_reward(full_text, fulltext_action, 1.0, 0)

                # Compute score and decision
                score = (abstract_reward + fulltext_reward) / 2 if fulltext_action is not None else abstract_reward
                decision = "Include" if abstract_action == 2 else "Maybe" if abstract_action == 1 else "Exclude"
                if fulltext_action is not None:
                    decision = "Include" if fulltext_action == 1 else "Exclude"

                results.append({
                    "Title": paper.title,
                    "Year": paper.published.year,
                    "URL": paper.entry_id,
                    "Abstract": paper.summary,
                    "Authors": ", ".join([a.name for a in paper.authors]),
                    "Decision": decision,
                    "Score": round(score, 3)
                })
            except Exception as e:
                logger.error(f"Processing failed for paper {paper.entry_id}: {e}")
                continue

        if results:
            df = pd.DataFrame(results).sort_values(by="Score", ascending=False).head(10)
            st.subheader("📋 Top Papers")
            st.dataframe(df)

            # Download CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=f"review_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

            # PRISMA Compliance Score
            metadata = {
                "query": topic,
                "modified_query": topic,
                "from_year": from_year,
                "to_year": to_year,
                "search_action": 0,  # Default
                "inclusion_criteria_clear": 1.0,
                "exclusion_criteria_clear": 1.0
            }
            try:
                prisma_score = st.session_state.prisma.evaluate_prisma_score(papers, metadata, df)
                st.metric("📊 PRISMA Compliance Score", f"{prisma_score:.2f}")
            except Exception as e:
                st.error(f"PRISMA score calculation failed: {e}")
                logger.error(f"PRISMA score error: {e}")
        else:
            st.error("No results processed. Check logs for errors.")