from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_text(text: str) -> str:
    return normalize_whitespace(text).lower()

