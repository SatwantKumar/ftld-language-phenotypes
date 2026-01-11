from __future__ import annotations

import json
import re
from typing import Any


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", flags=re.IGNORECASE | re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of a single JSON object from model text output."""
    if text is None:
        raise ValueError("No text to parse as JSON.")

    s = str(text).strip()
    m = _FENCE_RE.match(s)
    if m:
        s = m.group(1).strip()

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start : end + 1]

    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object at top level.")
    return obj

