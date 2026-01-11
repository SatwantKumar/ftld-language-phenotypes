#!/usr/bin/env python3
"""
Federated external validation on de-identified clinical notes.

This script is intended to be run locally at collaborator sites to validate the
locked FTLD-spectrum narrative phenotype construct(s) using de-identified note
text, while sharing only derived outputs (no raw text).

Primary external validation endpoint:
  - PPA enrichment in the assigned "language-dominant" construct, where external
    cases are assigned to the frozen Discovery cluster centroids (k=4) from the
    paper's Phase D run, using cosine similarity in L2-normalized embedding space.

Safety properties:
  - No OpenAI/LLM calls are made.
  - Outputs exclude note text by default (only IDs, hashes, lengths, assignments).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from paperjn.config import load_config  # noqa: E402
from paperjn.embeddings.sentence_transformers_backend import compute_sentence_transformer_embeddings  # noqa: E402
from paperjn.nlp.leakage import audit_text_for_leakage, remove_blacklisted_terms  # noqa: E402
from paperjn.nlp.text import normalize_text, normalize_whitespace  # noqa: E402
from paperjn.stats.enrichment import TwoByTwo, compute_effect_sizes  # noqa: E402
from paperjn.stats.permutation import permutation_p_value_ge  # noqa: E402


def _utc_stamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _coerce_bool01(x: object) -> bool:
    if isinstance(x, bool):
        return bool(x)
    if x is None:
        return False
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n", ""}:
        return False
    # fallback: non-empty string treated as True
    return True


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    A_norm = np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B_norm = np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return (A / A_norm) @ (B / B_norm).T


def _load_reference_centroids(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(npz_path)
    clusters = np.asarray(z["clusters"], dtype=np.int32)
    n_cases = np.asarray(z["n_cases"], dtype=np.int32)
    centroids = np.asarray(z["centroids"], dtype=np.float32)
    if centroids.ndim != 2 or centroids.shape[0] != clusters.shape[0]:
        raise ValueError("Reference centroids NPZ has unexpected shapes.")
    return clusters, n_cases, centroids


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    cols = {c.lower(): c for c in df.columns}
    # preferred names
    text_col = cols.get("note_text") or cols.get("text") or cols.get("clinical_text")
    pid_col = cols.get("patient_id") or cols.get("case_id") or cols.get("patient_pseudo_id")
    ppa_col = cols.get("is_ppa") or cols.get("ppa") or cols.get("diagnosis_ppa")
    site_col = cols.get("site_id")
    note_id_col = cols.get("note_id")

    missing = [k for k, v in {"patient_id": pid_col, "note_text": text_col, "is_ppa": ppa_col}.items() if v is None]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            "Expected at minimum: patient_id, note_text, is_ppa (case-insensitive)."
        )
    out = {"patient_id": pid_col, "note_text": text_col, "is_ppa": ppa_col}
    if site_col is not None:
        out["site_id"] = site_col
    if note_id_col is not None:
        out["note_id"] = note_id_col
    return out


def _clean_and_redact_text(series: pd.Series, blacklist: list[str]) -> tuple[pd.Series, pd.Series, pd.Series]:
    raw = series.fillna("").astype(str)
    norm = raw.map(lambda t: normalize_whitespace(normalize_text(t)))
    pre = norm.map(lambda t: audit_text_for_leakage(t, blacklist).n_matches)
    red = norm.map(lambda t: normalize_whitespace(remove_blacklisted_terms(t, blacklist, replacement=" ")))
    post = red.map(lambda t: audit_text_for_leakage(t, blacklist).n_matches)
    return red, pre, post


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="CSV with de-identified notes.")
    ap.add_argument("--out", required=True, type=Path, help="Output directory (derived outputs only).")
    ap.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "project.example.yaml",
        help="Project config YAML (used for embedding settings and leakage blacklist).",
    )
    ap.add_argument(
        "--ref-npz",
        type=Path,
        default=PROJECT_ROOT
        / "external_validation"
        / "reference"
        / "discovery_cluster_centroids__bge_small_en_v1_5__384d__k4__l2norm.npz",
        help="Frozen Discovery reference cluster centroids (NPZ).",
    )
    ap.add_argument("--language-cluster", type=int, default=0, help="Discovery language cluster ID (default=0).")
    ap.add_argument("--permutations", type=int, default=5000, help="Permutation count for fixed-cluster test.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for permutation test.")
    ap.add_argument(
        "--groupby",
        type=str,
        default="",
        help="Optional column to group by (e.g., patient_id) to pool multiple notes per patient.",
    )
    args = ap.parse_args()

    in_csv = Path(args.input).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    blacklist = list(cfg.leakage_blacklist or [])

    df = pd.read_csv(in_csv)
    col = _resolve_columns(df)

    df_use = df.copy()
    df_use["patient_id"] = df_use[col["patient_id"]].astype(str)
    df_use["is_ppa"] = df_use[col["is_ppa"]].map(_coerce_bool01).astype(bool)
    df_use["note_text_raw"] = df_use[col["note_text"]].fillna("").astype(str)
    if "site_id" in col:
        df_use["site_id"] = df_use[col["site_id"]].astype(str)
    if "note_id" in col:
        df_use["note_id"] = df_use[col["note_id"]].astype(str)

    df_use["n_chars_raw"] = df_use["note_text_raw"].map(lambda t: int(len(t)))

    text_clean, pre_n, post_n = _clean_and_redact_text(df_use["note_text_raw"], blacklist)
    df_use["note_text_clean"] = text_clean
    df_use["leakage_pre_n"] = pre_n.astype(int)
    df_use["leakage_post_n"] = post_n.astype(int)
    df_use["n_chars_clean"] = df_use["note_text_clean"].map(lambda t: int(len(t)))
    df_use["text_hash_sha256"] = df_use["note_text_clean"].map(_sha256)

    leakage_report = df_use[["patient_id", "leakage_pre_n", "leakage_post_n", "n_chars_clean"]].copy()
    leakage_report.to_csv(out_dir / "logs" / "leakage_audit.csv", index=False)
    if int(leakage_report["leakage_post_n"].sum()) != 0:
        raise RuntimeError(
            "Post-redaction leakage is non-zero; review blacklist and de-identification. "
            f"See: {out_dir / 'logs' / 'leakage_audit.csv'}"
        )

    # Embed per row (note-level).
    texts = df_use["note_text_clean"].astype(str).tolist()
    embeddings, info = compute_sentence_transformer_embeddings(
        texts,
        model_name=cfg.embeddings.model_name,
        normalize_embeddings=bool(cfg.embeddings.normalize_embeddings),
        batch_size=int(cfg.embeddings.batch_size),
        device=cfg.embeddings.device,
    )

    # Optionally pool multiple notes per patient (mean pooling + L2 normalize).
    groupby = str(args.groupby or "").strip()
    if groupby:
        if groupby not in df_use.columns:
            raise ValueError(f"--groupby={groupby!r} not found in input columns.")
        g = df_use[[groupby, "patient_id", "is_ppa", "site_id"]].copy() if "site_id" in df_use.columns else df_use[[groupby, "patient_id", "is_ppa"]].copy()
        g["row_idx"] = np.arange(len(g), dtype=np.int32)
        # check label consistency within group
        chk = g.groupby(groupby, as_index=False).agg(n=("row_idx", "size"), ppa_n=("is_ppa", "sum"))
        # If mixed PPA labels within a group, keep majority but record a warning
        mixed = chk[(chk["ppa_n"] != 0) & (chk["ppa_n"] != chk["n"])]
        if not mixed.empty:
            (out_dir / "logs" / "warnings.txt").write_text(
                "Warning: mixed is_ppa labels within some groups; majority label used.\n", encoding="utf-8"
            )

        pooled_rows = []
        pooled_emb = []
        for key, sub in g.groupby(groupby):
            idx = sub["row_idx"].to_numpy()
            v = embeddings[idx].mean(axis=0)
            v = v.astype(np.float32)
            v = v / (np.linalg.norm(v) + 1e-12)
            pooled_emb.append(v)
            # majority label
            is_ppa = bool(sub["is_ppa"].mean() >= 0.5)
            row = {"group_id": str(key), "patient_id": str(sub["patient_id"].iloc[0]), "is_ppa": is_ppa, "n_notes": int(len(sub))}
            if "site_id" in sub.columns:
                row["site_id"] = str(sub["site_id"].iloc[0])
            pooled_rows.append(row)
        df_eval = pd.DataFrame(pooled_rows)
        embeddings_eval = np.vstack(pooled_emb).astype(np.float32)
    else:
        df_eval = df_use.copy()
        embeddings_eval = embeddings

    # Assign to frozen Discovery centroids.
    clusters_ref, n_cases_ref, centroids = _load_reference_centroids(args.ref_npz)
    sim = _cosine_sim_matrix(embeddings_eval, centroids)
    best = np.argmax(sim, axis=1)
    assigned_cluster = clusters_ref[best].astype(int)
    assigned_sim = sim[np.arange(sim.shape[0]), best]

    df_out = df_eval[["patient_id", "is_ppa"]].copy()
    if "site_id" in df_eval.columns:
        df_out.insert(0, "site_id", df_eval["site_id"].astype(str))
    if "group_id" in df_eval.columns:
        df_out.insert(1, "group_id", df_eval["group_id"].astype(str))
        df_out["n_notes"] = df_eval["n_notes"].astype(int)
    df_out["assigned_cluster"] = assigned_cluster
    df_out["assigned_similarity"] = assigned_sim.astype(float)

    assignments_csv = out_dir / "tables" / "assignments.csv"
    df_out.to_csv(assignments_csv, index=False)

    # Cluster summary
    cl_summary = (
        df_out.groupby("assigned_cluster", as_index=False)
        .agg(n=("is_ppa", "size"), ppa_n=("is_ppa", "sum"), ppa_share=("is_ppa", "mean"))
        .sort_values("assigned_cluster")
    )
    cluster_summary_csv = out_dir / "tables" / "cluster_summary.csv"
    cl_summary.to_csv(cluster_summary_csv, index=False)

    # Primary endpoint: fixed language cluster enrichment test
    lang_id = int(args.language_cluster)
    is_lang = df_out["assigned_cluster"].astype(int).to_numpy() == lang_id
    is_ppa = df_out["is_ppa"].astype(bool).to_numpy()

    n_total = int(len(df_out))
    n_lang = int(is_lang.sum())
    ppa_total = int(is_ppa.sum())
    ppa_lang = int(is_ppa[is_lang].sum())

    a = float(ppa_lang)
    b = float(n_lang - ppa_lang)
    c = float(ppa_total - ppa_lang)
    d = float((n_total - n_lang) - (ppa_total - ppa_lang))
    effects = compute_effect_sizes(TwoByTwo(a=a, b=b, c=c, d=d))

    rng = np.random.default_rng(int(args.seed))
    perm = np.zeros(int(args.permutations), dtype=np.float32)
    for i in range(int(args.permutations)):
        perm_is = rng.permutation(is_ppa)
        perm[i] = float(np.mean(perm_is[is_lang])) if n_lang > 0 else 0.0
    p_perm = permutation_p_value_ge(float(effects.ppa_share_cluster), perm)

    primary = pd.DataFrame(
        [
            {
                "n_total": n_total,
                "ppa_total": ppa_total,
                "language_cluster_id": lang_id,
                "n_in_language_cluster": n_lang,
                "ppa_in_language_cluster": ppa_lang,
                "ppa_share_language_cluster": effects.ppa_share_cluster,
                "ppa_share_overall": effects.ppa_share_stage,
                "or": effects.or_,
                "or_ci95_low": effects.or_ci95_low,
                "or_ci95_high": effects.or_ci95_high,
                "perm_p_value_one_sided": p_perm,
                "perm_B": int(args.permutations),
                "seed": int(args.seed),
            }
        ]
    )
    primary_csv = out_dir / "tables" / "primary_endpoint_summary.csv"
    primary.to_csv(primary_csv, index=False)

    # Report
    report = out_dir / "REPORT.md"
    report.write_text(
        "\n".join(
            [
                "# External Validation (Federated) — Report",
                "",
                f"- Run timestamp (UTC): `{_utc_stamp()}`",
                f"- Input: `{in_csv}`",
                f"- Embedding model: `{info.model_name}` (dim={info.embedding_dim}, device={info.device})",
                f"- Reference centroids: `{Path(args.ref_npz).resolve()}`",
                f"- Language cluster (fixed): `{lang_id}`",
                "",
                "## Cohort summary",
                f"- Cases analyzed: {n_total}",
                f"- PPA overall: {ppa_total}/{n_total} ({effects.ppa_share_stage:.3f})",
                f"- Assigned to language cluster: {n_lang}/{n_total} ({n_lang/max(n_total,1):.3f})",
                f"- PPA in language cluster: {ppa_lang}/{n_lang} ({effects.ppa_share_cluster:.3f})" if n_lang else "- PPA in language cluster: NA (n_lang=0)",
                "",
                "## Primary endpoint (fixed-cluster permutation test; one-sided enrichment)",
                f"- OR: {effects.or_:.2f} (95% CI {effects.or_ci95_low:.2f}-{effects.or_ci95_high:.2f})",
                f"- Permutation p-value: {p_perm:.6f} (B={int(args.permutations)})",
                "",
                "## Files (derived outputs only)",
                f"- Assignments: `{assignments_csv}`",
                f"- Cluster summary: `{cluster_summary_csv}`",
                f"- Primary endpoint summary: `{primary_csv}`",
                f"- Leakage audit (counts only): `{out_dir / 'logs' / 'leakage_audit.csv'}`",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run_cfg = {
        "timestamp_utc": _utc_stamp(),
        "input_csv": str(in_csv),
        "output_dir": str(out_dir),
        "project_config": str(Path(args.config).resolve()),
        "leakage_blacklist_size": int(len(blacklist)),
        "embedding_run_info": asdict(info),
        "reference_npz": str(Path(args.ref_npz).resolve()),
        "language_cluster_id": int(lang_id),
        "permutations": int(args.permutations),
        "seed": int(args.seed),
        "groupby": groupby,
        "safety": {
            "writes_raw_text": False,
            "uses_llm_api": False,
        },
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    print(f"Wrote: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
