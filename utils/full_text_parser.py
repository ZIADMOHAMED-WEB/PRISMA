import PyPDF2
import pdfplumber
import requests
import os
import tempfile
import re
from typing import Dict, Optional
from utils.logger import get_logger

logger = get_logger("full_text_parser")

def parse_arxiv_pdf(identifier: str) -> str:
    """
    Parse a PDF from an arXiv URL or local file path and extract text.
    Args:
        identifier: arXiv URL (e.g., 'http://arxiv.org/abs/2503.07152v1') or local file path
    Returns:
        Extracted text or empty string if parsing fails
    """
    try:
        if identifier.startswith("http"):
            # Handle arXiv URL
            if not identifier.endswith(".pdf"):
                identifier = identifier.replace("abs", "pdf") + ".pdf"
            response = requests.get(identifier, timeout=10)
            response.raise_for_status()

            temp_file_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(response.content)
                    temp_file_path = temp_file.name

                text = extract_text_from_pdf(temp_file_path)
                if not text or len(text.split()) < 50:  # Validate text quality
                    logger.warning(f"Insufficient text extracted from {identifier}, trying fallback parser")
                    text = extract_text_with_fallback(temp_file_path)
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except Exception as e:
                        logger.error(f"Failed to delete temporary file {temp_file_path}: {e}")

            if text:
                logger.info(f"Successfully parsed PDF for {identifier}")
                return text.strip()
            else:
                logger.error(f"No usable text extracted from {identifier}")
                return ""
        else:
            # Handle local file path
            if not os.path.exists(identifier):
                logger.error(f"Local PDF file not found: {identifier}")
                return ""
            text = extract_text_from_pdf(identifier)
            if not text or len(text.split()) < 50:
                logger.warning(f"Insufficient text extracted from {identifier}, trying fallback parser")
                text = extract_text_with_fallback(identifier)
            if text:
                logger.info(f"Successfully parsed local PDF: {identifier}")
                return text.strip()
            else:
                logger.error(f"No usable text extracted from {identifier}")
                return ""
    except Exception as e:
        logger.error(f"Failed to parse PDF {identifier}: {e}")
        return ""

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using PyPDF2.
    Args:
        pdf_path: Path to the PDF file
    Returns:
        Extracted text or empty string if extraction fails
    """
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
    except Exception as e:
        logger.error(f"PyPDF2 failed to extract text from {pdf_path}: {e}")
        return ""

def extract_text_with_fallback(pdf_path: str) -> str:
    """
    Fallback text extraction using pdfplumber.
    Args:
        pdf_path: Path to the PDF file
    Returns:
        Extracted text or empty string if extraction fails
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
    except Exception as e:
        logger.error(f"pdfplumber failed to extract text from {pdf_path}: {e}")
        return ""

def parse_checklist_pdf(pdf_path: str) -> Dict:
    """
    Parse a PRISMA checklist PDF and extract checklist items.
    Args:
        pdf_path: Path to the checklist PDF
    Returns:
        Dictionary mapping checklist items to compliance (0.0, 0.5, or 1.0)
    """
    try:
        if not os.path.exists(pdf_path):
            logger.error(f"Checklist PDF not found: {pdf_path}")
            return {}

        text = extract_text_from_pdf(pdf_path)
        if not text or len(text.split()) < 50:
            logger.warning(f"Insufficient text extracted from {pdf_path}, trying fallback parser")
            text = extract_text_with_fallback(pdf_path)
        
        if not text:
            logger.error(f"No text extracted from checklist PDF: {pdf_path}")
            return {}

        # PRISMA 2020 checklist items
        checklist_items = [
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

        checklist_data = {item: 0.0 for item in checklist_items}
        # Enhanced regex for detecting completed items
        for item in checklist_items:
            # Match item name followed by 'X', 'Yes', '✓', or explicit completion indicators
            pattern = rf"{item.replace('_', ' ').title()}\s*[:\(\[]*\s*(X|Yes|✓|\[X\]|\[✓\]|Completed|True)"
            if re.search(pattern, text, re.IGNORECASE):
                checklist_data[item] = 1.0
            elif re.search(rf"{item.replace('_', ' ').title()}\s*[:\(\[]*\s*(No|Incomplete|False)", text, re.IGNORECASE):
                checklist_data[item] = 0.0
            else:
                checklist_data[item] = 0.5  # Default for unclear status

        logger.info(f"Successfully parsed checklist from {pdf_path} with {sum(1 for v in checklist_data.values() if v == 1.0)} items marked complete")
        return checklist_data
    except Exception as e:
        logger.error(f"Failed to parse checklist PDF {pdf_path}: {e}")
        return {}