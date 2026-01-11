from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EmbeddingRunInfo:
    model_name: str
    embedding_dim: int
    device: str | None
    sentence_transformers_version: str | None
    transformers_version: str | None
    torch_version: str | None
    model_commit_hash: str | None


def compute_sentence_transformer_embeddings(
    texts: Sequence[str],
    model_name: str,
    *,
    normalize_embeddings: bool,
    batch_size: int,
    device: str | None,
) -> tuple[np.ndarray, EmbeddingRunInfo]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: sentence-transformers. Install with `pip install -e '.[embeddings]'`."
        ) from exc

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    model_commit_hash = None
    try:  # best-effort: commit hash if available
        first = model._first_module()
        cfg = getattr(getattr(first, "auto_model", None), "config", None)
        model_commit_hash = getattr(cfg, "_commit_hash", None)
    except Exception:
        model_commit_hash = None

    def _safe_version(mod_name: str) -> str | None:
        try:
            mod = __import__(mod_name)
            return getattr(mod, "__version__", None)
        except Exception:
            return None

    info = EmbeddingRunInfo(
        model_name=model_name,
        embedding_dim=int(embeddings.shape[1]),
        device=str(model.device) if getattr(model, "device", None) is not None else device,
        sentence_transformers_version=_safe_version("sentence_transformers"),
        transformers_version=_safe_version("transformers"),
        torch_version=_safe_version("torch"),
        model_commit_hash=model_commit_hash,
    )
    return embeddings, info
