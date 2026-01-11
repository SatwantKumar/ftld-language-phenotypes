from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LeakageAuditResult:
    n_matches: int
    matches: list[str]


def _compile_blacklist(blacklist: list[str]) -> re.Pattern[str]:
    escaped = [re.escape(t.strip()) for t in blacklist if t and t.strip()]
    if not escaped:
        return re.compile(r"(?!x)x")  # never matches
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", flags=re.IGNORECASE)


def remove_blacklisted_terms(text: str, blacklist: list[str], replacement: str = "[REDACTED]") -> str:
    pattern = _compile_blacklist(blacklist)
    return pattern.sub(replacement, text)


def audit_text_for_leakage(text: str, blacklist: list[str]) -> LeakageAuditResult:
    pattern = _compile_blacklist(blacklist)
    matches = pattern.findall(text)
    # normalize matches for reporting
    matches_norm = [m.lower() for m in matches]
    return LeakageAuditResult(n_matches=len(matches_norm), matches=matches_norm)
