from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from paperjn.nlp.leakage import audit_text_for_leakage, remove_blacklisted_terms
from paperjn.nlp.text import normalize_text, normalize_whitespace
from paperjn.stats.enrichment import TwoByTwo, compute_effect_sizes
from paperjn.stats.permutation import permutation_p_value_ge


def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    A_norm = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B_norm = np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return (A / A_norm) @ (B / B_norm).T


@dataclass(frozen=True)
class LiteratureReplicationResult:
    summary_path: Path
    assignments_path: Path
    figure_path: Path


def run_literature_replication(
    *,
    corpus: pd.DataFrame,
    text_column: str,
    is_ppa_column: str,
    doc_id_column: str,
    leakage_blacklist: list[str],
    literature_embeddings: np.ndarray,
    reference_centroids: np.ndarray,
    reference_is_language: np.ndarray,
    permutations: int,
    random_seed: int,
    out_summary: Path,
    out_assignments: Path,
    out_figure: Path,
) -> LiteratureReplicationResult:
    """Nearest-centroid assignment to curated reference prototypes + permutation test."""
    import matplotlib.pyplot as plt

    if text_column not in corpus.columns:
        raise ValueError(f"Missing literature text column: {text_column}")
    if is_ppa_column not in corpus.columns:
        raise ValueError(f"Missing literature is_ppa column: {is_ppa_column}")

    df = corpus.copy()
    df[is_ppa_column] = df[is_ppa_column].astype(bool)

    # Leakage audit (pre/post) on normalized text
    norm = df[text_column].astype(str).map(normalize_text)
    pre = norm.map(lambda t: audit_text_for_leakage(t, leakage_blacklist))
    pre_n = pre.map(lambda r: r.n_matches)

    redacted = norm.map(
        lambda t: normalize_whitespace(remove_blacklisted_terms(t, leakage_blacklist, replacement=" "))
    )
    post = redacted.map(lambda t: audit_text_for_leakage(t, leakage_blacklist))
    post_n = post.map(lambda r: r.n_matches)

    if int(post_n.sum()) != 0:
        raise RuntimeError("Literature leakage audit failed (post-redaction non-zero).")

    sim = _cosine_sim_matrix(literature_embeddings, reference_centroids)
    best = np.argmax(sim, axis=1)
    assigned_language = reference_is_language[best].astype(bool)
    assigned_sim = sim[np.arange(sim.shape[0]), best]

    out = pd.DataFrame(
        {
            doc_id_column: df[doc_id_column] if doc_id_column in df.columns else np.arange(len(df)),
            "assigned_ref": best.astype(int),
            "assigned_language": assigned_language,
            "assigned_similarity": assigned_sim,
            "is_ppa": df[is_ppa_column].astype(bool).to_numpy(),
        }
    )

    # Effect sizes
    is_ppa = out["is_ppa"].to_numpy(dtype=bool)
    is_lang = out["assigned_language"].to_numpy(dtype=bool)
    n_total = int(len(out))
    ppa_total = int(is_ppa.sum())
    n_lang = int(is_lang.sum())
    ppa_lang = int(is_ppa[is_lang].sum())
    a = ppa_lang
    b = n_lang - ppa_lang
    c = ppa_total - ppa_lang
    d = (n_total - n_lang) - c
    effects = compute_effect_sizes(TwoByTwo(a=a, b=b, c=c, d=d))

    rng = np.random.default_rng(random_seed)
    perm = np.zeros(permutations, dtype=np.float32)
    for i in range(permutations):
        perm_is = rng.permutation(is_ppa)
        perm[i] = float(np.mean(perm_is[is_lang])) if n_lang > 0 else 0.0
    p_perm = permutation_p_value_ge(float(effects.ppa_share_cluster), perm)

    summary = pd.DataFrame(
        [
            {
                "n_docs": n_total,
                "n_assigned_language": n_lang,
                "ppa_docs": ppa_total,
                "ppa_assigned_language": ppa_lang,
                "ppa_share_assigned_language": effects.ppa_share_cluster,
                "ppa_share_overall": effects.ppa_share_stage,
                "rr": effects.rr,
                "rr_ci95_low": effects.rr_ci95_low,
                "rr_ci95_high": effects.rr_ci95_high,
                "or": effects.or_,
                "or_ci95_low": effects.or_ci95_low,
                "or_ci95_high": effects.or_ci95_high,
                "perm_p_value": p_perm,
                "perm_B": int(permutations),
                "leakage_pre_total": int(pre_n.sum()),
                "leakage_post_total": int(post_n.sum()),
            }
        ]
    )

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_assignments.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_summary, index=False)
    out.to_csv(out_assignments, index=False)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.bar(
        ["Assigned language", "Overall"],
        [effects.ppa_share_cluster, effects.ppa_share_stage],
        color=["#4C72B0", "#DD8452"],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("PPA share")
    ax.set_title("Literature replication: PPA share by phenotype assignment")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    out_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_figure, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return LiteratureReplicationResult(summary_path=out_summary, assignments_path=out_assignments, figure_path=out_figure)

