import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from rewards.enhanced_reward_system import EnhancedRewardSystem
from utils.full_text_parser import parse_checklist_pdf
from utils.logger import get_logger
import os

logger = get_logger("prisma_checker")

class PRISMAChecker:
    def __init__(
        self,
        checklist_pdf_path: str = os.getenv("PRISMA_CHECKLIST_PATH", r"E:\RL\prisma_marl_project\PRISMA_2020_checklist.pdf")
    ):
        self.reward_system = EnhancedRewardSystem()
        self.checklist_pdf_path = checklist_pdf_path
        self.checklist_items = [
            "title_identifiable", "abstract_structured", "protocol_registered",
            "eligibility_criteria", "information_sources", "search_strategy_documented",
            "study_selection_process", "data_collection_process", "data_items_listed",
            "effect_measures", "synthesis_methods", "study_bias_assessment",
            "certainty_assessment", "results_study_selection", "results_study_characteristics",
            "results_synthesis", "results_risk_of_bias", "results_certainty_of_evidence",
            "limitations_discussed", "funding_reported", "inclusion_criteria_clear",
            "exclusion_criteria_clear", "quality_assessment_performed", "results_synthesized",
            "search_date_reported", "databases_searched", "grey_literature_included",
            "language_restrictions", "publication_restrictions", "data_availability_statement",
            "conflict_of_interest", "reviewer_agreement", "screening_process_described",
            "data_extraction_systematic", "study_flow_diagram", "sensitivity_analysis",
            "subgroup_analysis", "publication_bias_assessed", "certainty_assessment_method",
            "synthesis_exploration", "additional_analyses"
        ]
        self.checklist_data = parse_checklist_pdf(self.checklist_pdf_path)
        if not self.checklist_data:
            logger.warning(f"No checklist data parsed from {self.checklist_pdf_path}. Using fallback logic.")

    def evaluate_search_reward(
        self,
        all_agents_papers: List[List[Dict]],
        all_query_embeddings: List[np.ndarray],
        human_feedback: Optional[float] = None,
        prisma_data: Optional[Dict] = None
    ) -> float:
        try:
            if not all_agents_papers or any(not papers for papers in all_agents_papers):
                return -1.0
            prisma_data = prisma_data or self.checklist_data
            rewards = [
                self.reward_system.compute_search_reward(papers, query_embedding, prisma_data, human_feedback)
                for papers, query_embedding in zip(all_agents_papers, all_query_embeddings)
            ]
            joint_reward = sum(rewards) / len(rewards)
            return joint_reward
        except Exception as e:
            logger.error(f"Search reward evaluation failed: {e}")
            return -0.5

    def evaluate_abstract_reward(
        self,
        all_abstracts: List[str],
        all_decisions: List[int],
        all_ground_truths: Optional[List[Optional[int]]] = None,
        prisma_data: Optional[Dict] = None
    ) -> float:
        try:
            prisma_data = prisma_data or self.checklist_data
            rewards = []
            for i, abstract in enumerate(all_abstracts):
                decision = all_decisions[i]
                ground_truth = all_ground_truths[i] if all_ground_truths and i < len(all_ground_truths) else None
                paper_data = {"abstract": abstract}
                reward = self.reward_system.compute_filter_reward(paper_data, decision, prisma_data, ground_truth)
                rewards.append(reward)
            joint_reward = sum(rewards) / len(rewards) if rewards else -1.0
            return joint_reward
        except Exception as e:
            logger.error(f"Abstract reward evaluation failed: {e}")
            return -0.5

    def evaluate_fulltext_reward(
        self,
        all_full_texts: List[str],
        all_decisions: List[int],
        all_ground_truths: Optional[List[Optional[int]]] = None,
        all_citation_counts: Optional[List[int]] = None,
        prisma_data: Optional[Dict] = None
    ) -> float:
        try:
            prisma_data = prisma_data or self.checklist_data
            rewards = []
            for i, full_text in enumerate(all_full_texts):
                decision = all_decisions[i]
                ground_truth = all_ground_truths[i] if all_ground_truths and i < len(all_ground_truths) else None
                citation_count = all_citation_counts[i] if all_citation_counts and i < len(all_citation_counts) else 0
                paper_data = {"abstract": full_text, "citation_count": citation_count}
                reward = self.reward_system.compute_filter_reward(paper_data, decision, prisma_data, ground_truth)
                rewards.append(reward)
            joint_reward = sum(rewards) / len(rewards) if rewards else -1.0
            return joint_reward
        except Exception as e:
            logger.error(f"Fulltext reward evaluation failed: {e}")
            return -0.5

    def evaluate_prisma_score(
        self,
        papers: List[Dict],
        metadata: Dict,
        results_df: pd.DataFrame
    ) -> float:
        try:
            review_data = {}
            for item in self.checklist_items:
                if item in self.checklist_data:
                    review_data[item] = self.checklist_data[item]
                else:
                    # fallback rules
                    if item == "title_identifiable":
                        review_data[item] = 1.0 if metadata.get("query") else 0.0
                    elif item == "abstract_structured":
                        review_data[item] = 0.8 if results_df["Abstract"].str.contains("background|method|result", case=False, regex=True).any() else 0.5
                    elif item == "protocol_registered":
                        review_data[item] = 0.5
                    elif item == "eligibility_criteria":
                        review_data[item] = 1.0 if metadata.get("inclusion_criteria_clear") else 0.5
                    elif item == "information_sources":
                        review_data[item] = 1.0 if metadata.get("databases_searched") else 0.8
                    elif item == "search_strategy_documented":
                        review_data[item] = 1.0 if metadata.get("search_action") is not None else 0.0
                    elif item == "study_selection_process":
                        review_data[item] = 0.8 if results_df["Decision"].notnull().all() else 0.5
                    elif item == "data_collection_process":
                        review_data[item] = 0.7
                    elif item == "data_items_listed":
                        review_data[item] = 0.6
                    elif item == "effect_measures":
                        review_data[item] = 0.5
                    elif item == "synthesis_methods":
                        review_data[item] = 0.5
                    elif item == "study_bias_assessment":
                        review_data[item] = 0.8 if any("bias" in abstract.lower() for abstract in results_df["Abstract"]) else 0.5
                    elif item == "certainty_assessment":
                        review_data[item] = 0.7 if any("confidence" in abstract.lower() for abstract in results_df["Abstract"]) else 0.5
                    elif item == "results_study_selection":
                        review_data[item] = 1.0 if not results_df.empty else 0.0
                    elif item == "results_study_characteristics":
                        review_data[item] = 1.0 if results_df["Authors"].notnull().all() else 0.5
                    elif item == "results_synthesis":
                        review_data[item] = 0.6
                    elif item == "results_risk_of_bias":
                        review_data[item] = 0.5
                    elif item == "results_certainty_of_evidence":
                        review_data[item] = 0.5
                    elif item == "limitations_discussed":
                        review_data[item] = 0.7
                    elif item == "funding_reported":
                        review_data[item] = 0.5
                    elif item == "inclusion_criteria_clear":
                        review_data[item] = 1.0 if metadata.get("inclusion_criteria_clear") else 0.5
                    elif item == "exclusion_criteria_clear":
                        review_data[item] = 1.0 if metadata.get("exclusion_criteria_clear") else 0.5
                    elif item == "quality_assessment_performed":
                        review_data[item] = 0.8 if results_df["Score"].notnull().all() else 0.5
                    elif item == "results_synthesized":
                        review_data[item] = 0.6
                    elif item == "search_date_reported":
                        review_data[item] = 1.0 if metadata.get("from_year") and metadata.get("to_year") else 0.0
                    elif item == "databases_searched":
                        review_data[item] = 1.0
                    elif item == "grey_literature_included":
                        review_data[item] = 0.5
                    elif item == "language_restrictions":
                        review_data[item] = 0.5
                    elif item == "publication_restrictions":
                        review_data[item] = 0.5
                    elif item == "data_availability_statement":
                        review_data[item] = 0.5
                    elif item == "conflict_of_interest":
                        review_data[item] = 0.5
                    elif item == "reviewer_agreement":
                        review_data[item] = 0.5
                    elif item == "screening_process_described":
                        review_data[item] = 0.8
                    elif item == "data_extraction_systematic":
                        review_data[item] = 0.7
                    elif item == "study_flow_diagram":
                        review_data[item] = 0.5
                    elif item == "sensitivity_analysis":
                        review_data[item] = 0.5
                    elif item == "subgroup_analysis":
                        review_data[item] = 0.5
                    elif item == "publication_bias_assessed":
                        review_data[item] = 0.5
                    elif item == "certainty_assessment_method":
                        review_data[item] = 0.5
                    elif item == "synthesis_exploration":
                        review_data[item] = 0.5
                    elif item == "additional_analyses":
                        review_data[item] = 0.5

            prisma_score = self.reward_system.compute_prisma_reward(review_data)
            return np.clip(prisma_score, 0.0, 1.0)
        except Exception as e:
            logger.error(f"PRISMA score evaluation failed: {e}")
            return 0.0
