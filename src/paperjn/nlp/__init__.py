"""NLP utilities (text normalization, leakage prevention, embeddings, etc.)."""

from .leakage import LeakageAuditResult, audit_text_for_leakage, remove_blacklisted_terms
from .text import normalize_text, normalize_whitespace

__all__ = [
    "LeakageAuditResult",
    "audit_text_for_leakage",
    "remove_blacklisted_terms",
    "normalize_text",
    "normalize_whitespace",
]
