from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path, *, override: bool = False) -> int:
    """Load KEY=VALUE pairs from a .env file into os.environ (best-effort).

    Returns the number of variables set.
    """
    env_path = Path(path)
    if not env_path.exists():
        return 0

    n_set = 0
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
            value = value[1:-1]

        if not override and os.environ.get(key):
            continue
        os.environ[key] = value
        n_set += 1

    return int(n_set)


def load_dotenv_candidates(candidates: list[Path], *, override: bool = False) -> int:
    n = 0
    for p in candidates:
        n += load_dotenv(p, override=override)
    return int(n)

