import os
from typing import Dict, List

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


_ANALYZER = AnalyzerEngine()
_ANONYMIZER = AnonymizerEngine()


def mask_pii(text: str) -> str:
    """Mask PII entities in the input text using Presidio."""
    try:
        results = _ANALYZER.analyze(text=text, language="en")
        anonymized = _ANONYMIZER.anonymize(text=text, analyzer_results=results)
        return anonymized.text
    except Exception:
        return text


def validate_query(query: str) -> str:
    """Simple validation helper for user queries."""
    cleaned = " ".join(query.split())
    if not cleaned:
        raise ValueError("The question cannot be empty.")
    if len(cleaned) > 1000:
        raise ValueError("The question is too long. Please shorten it.")
    return cleaned


def validate_output(text: str) -> str:
    """Return the text after redacting PII from the model output."""
    return mask_pii(text)
