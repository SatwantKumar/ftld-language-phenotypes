from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from paperjn.pmcoa.insilico_raters import run_insilico_rater
from paperjn.pmcoa.phase_d_relabel import run_phase_d_relabel_with_rater_overrides
from paperjn.pmcoa.phase_e_pooling import run_phase_e_pool_after_replication
from paperjn.pmcoa.rater_validation import SYMPTOM_TAGS_V1, score_two_raters
from paperjn.utils.paths import ensure_dir


def _utc_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _find_results_root(path: Path) -> Path | None:
    for p in [path] + list(path.parents):
        if p.name == "results":
            return p
    return None


def _safe_int01(x: object) -> int | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return int(x) if int(x) in (0, 1) else None
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return None
        if float(x) in (0.0, 1.0):
            return int(float(x))
    s = str(x).strip()
    if s in {"0.0", "1.0"}:
        s = s[0]
    if s in {"0", "1"}:
        return int(s)
    return None


def _make_consensus(
    *, r1: pd.DataFrame, r2: pd.DataFrame, adjud: pd.DataFrame | None
) -> pd.DataFrame:
    """Consensus: agree -> value; disagree -> adjud value; else blank."""
    key = ["case_uid", "pmcid", "case_id", "packet_path"]
    out = r1[key].copy()
    r1i = r1.set_index("case_uid", drop=False)
    r2i = r2.set_index("case_uid", drop=False)
    adji = adjud.set_index("case_uid", drop=False) if adjud is not None and not adjud.empty else None

    def _value(df: pd.DataFrame, case_uid: str, col: str) -> int | None:
        if col not in df.columns:
            return None
        try:
            return _safe_int01(df.at[case_uid, col])
        except Exception:
            return None

    def _cons(col: str) -> list[object]:
        out_vals: list[object] = []
        for case_uid in out["case_uid"].astype(str).tolist():
            ai = _value(r1i, case_uid, col)
            bi = _value(r2i, case_uid, col)
            ci = _value(adji, case_uid, col) if adji is not None else None
            if ai is not None and bi is not None and ai == bi:
                out_vals.append(ai)
            elif ci is not None:
                out_vals.append(ci)
            else:
                out_vals.append("")
        return out_vals

    out["text_adequate"] = _cons("text_adequate")
    out["is_ftld_spectrum"] = _cons("is_ftld_spectrum")
    out["is_ppa"] = _cons("is_ppa")
    for tag in SYMPTOM_TAGS_V1:
        out[f"tag__{tag}"] = _cons(f"tag__{tag}")

    # Notes: keep empty to avoid leaking model-specific rationales; store in report instead.
    out["notes"] = ""
    return out


@dataclass(frozen=True)
class InsilicoStudyOutputs:
    run_dir: Path
    run_config_json: Path
    rater_a_csv: Path
    rater_b_csv: Path
    adjudicator_csv: Path
    consensus_csv: Path
    kappa_dir: Path
    report_md: Path
    phase_d_relabel_dir: Path
    phase_e_dir: Path


def run_insilico_rater_study(
    *,
    rater_sample_dir: Path,
    phase_d_run_dir: Path | None,
    out_dir: Path | None,
    model: str,
    temperature: float,
    endpoint: Literal["responses", "chat_completions"],
    request_delay_s: float,
    max_output_tokens: int,
    max_packet_chars: int | None,
    max_cases: int | None,
    exclude_non_ftld: bool,
) -> InsilicoStudyOutputs:
    rater_sample_dir = Path(rater_sample_dir).resolve()
    share_dir = rater_sample_dir / "share"
    internal_dir = rater_sample_dir / "internal"
    if not share_dir.exists():
        raise FileNotFoundError(f"Missing rater sample share/: {share_dir}")

    # Infer base Phase D run dir from the rater sample meta.json if not provided.
    if phase_d_run_dir is None:
        meta_path = internal_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError("phase_d_run_dir not provided and internal/meta.json missing.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        phase_d_run_dir = Path(meta["phase_d_run_dir"]).resolve()
    phase_d_run_dir = Path(phase_d_run_dir).resolve()

    ts = _utc_slug()
    if out_dir is None:
        results_root = _find_results_root(rater_sample_dir) or (rater_sample_dir.parent.parent / "results")
        out_dir = (results_root / "phase_f_insilico" / f"insilico_rater_study__{ts}").resolve()
    run_dir = ensure_dir(Path(out_dir).resolve())

    # Rater A/B (same model, independent persona label).
    rater_a_dir = ensure_dir(run_dir / "rater_A")
    rater_b_dir = ensure_dir(run_dir / "rater_B")
    rater_adj_dir = ensure_dir(run_dir / "rater_ADJUDICATOR")

    template_path = share_dir / "rater1_template.csv"

    out_a = run_insilico_rater(
        share_dir=share_dir,
        template_csv=template_path,
        out_dir=rater_a_dir,
        rater_label="VIRTUAL_BEHAVIORAL_NEUROLOGIST_A",
        model=str(model),
        temperature=float(temperature),
        endpoint=str(endpoint),  # type: ignore[arg-type]
        request_delay_s=float(request_delay_s),
        max_output_tokens=int(max_output_tokens),
        max_packet_chars=max_packet_chars,
        resume=True,
        max_cases=max_cases,
    )
    out_b = run_insilico_rater(
        share_dir=share_dir,
        template_csv=template_path,
        out_dir=rater_b_dir,
        rater_label="VIRTUAL_BEHAVIORAL_NEUROLOGIST_B",
        model=str(model),
        temperature=float(temperature),
        endpoint=str(endpoint),  # type: ignore[arg-type]
        request_delay_s=float(request_delay_s),
        max_output_tokens=int(max_output_tokens),
        max_packet_chars=max_packet_chars,
        resume=True,
        max_cases=max_cases,
    )

    r1 = pd.read_csv(out_a.filled_csv)
    r2 = pd.read_csv(out_b.filled_csv)

    # Adjudicator pass for disagreements only (still in-silico; clearly labeled).
    # Strategy: run adjudicator for any row where A and B disagree on is_ppa OR text_adequate.
    # If either is missing, also adjudicate.
    def _needs_adjud(row: pd.Series) -> bool:
        a_ok = _safe_int01(row.get("text_adequate_r1"))
        b_ok = _safe_int01(row.get("text_adequate_r2"))
        if a_ok is None or b_ok is None or a_ok != b_ok:
            return True
        if a_ok == 0 and b_ok == 0:
            return False
        a = _safe_int01(row.get("is_ppa_r1"))
        b = _safe_int01(row.get("is_ppa_r2"))
        return a is None or b is None or a != b

    merged = r1.merge(r2, on=["case_uid", "pmcid", "case_id", "packet_path"], how="inner", suffixes=("_r1", "_r2"))
    need = merged[merged.apply(_needs_adjud, axis=1)][["case_uid", "pmcid", "case_id", "packet_path"]].copy()
    if max_cases is not None:
        need = need.head(int(max_cases)).copy()

    out_adj_csv: Path | None = None
    r_adj_df: pd.DataFrame | None = None
    if not need.empty:
        # Build a minimal template CSV for adjudication.
        adjud_tmpl = need.copy()
        for c in ["text_adequate", "is_ftld_spectrum", "is_ppa"]:
            adjud_tmpl[c] = ""
        for tag in SYMPTOM_TAGS_V1:
            adjud_tmpl[f"tag__{tag}"] = ""
        adjud_tmpl["notes"] = ""
        adjud_template_csv = rater_adj_dir / "adjudicator_template.csv"
        adjud_tmpl.to_csv(adjud_template_csv, index=False)

        out_adj = run_insilico_rater(
            share_dir=share_dir,
            template_csv=adjud_template_csv,
            out_dir=rater_adj_dir,
            rater_label="VIRTUAL_BEHAVIORAL_NEUROLOGIST_ADJUDICATOR",
            model=str(model),
            temperature=float(temperature),
            endpoint=str(endpoint),  # type: ignore[arg-type]
            request_delay_s=float(request_delay_s),
            max_output_tokens=int(max_output_tokens),
            max_packet_chars=max_packet_chars,
            resume=True,
            max_cases=None,
        )
        out_adj_csv = out_adj.filled_csv
        r_adj_df = pd.read_csv(out_adj.filled_csv)

    consensus = _make_consensus(r1=r1, r2=r2, adjud=r_adj_df)
    consensus_csv = run_dir / "consensus__INSILICO_virtual_clinicians.csv"
    consensus.to_csv(consensus_csv, index=False)

    # Kappa (A vs B)
    kappa_dir = ensure_dir(run_dir / "kappa_A_vs_B")
    kappa_outputs = score_two_raters(
        rater1_csv=out_a.filled_csv,
        rater2_csv=out_b.filled_csv,
        out_dir=kappa_dir,
    )

    # Downstream: create a Phase D relabel sensitivity run and a Phase E run from it.
    phase_d_relabel_dir = ensure_dir(run_dir / "phase_d_relabel__INSILICO")
    relabel_outputs = run_phase_d_relabel_with_rater_overrides(
        base_phase_d_run_dir=phase_d_run_dir,
        rater_csv=consensus_csv,
        out_dir=phase_d_relabel_dir,
        exclude_non_ftld=bool(exclude_non_ftld),
        rater_source="insilico_gpt52",
    )
    # Phase E (post-replication pooled characterization)
    phase_e_dir = ensure_dir(run_dir / "phase_e_pool_after_replication__INSILICO")
    phase_e_forced = False
    try:
        run_phase_e_pool_after_replication(
            phase_d_run_dir=relabel_outputs.run_dir,
            out_dir=phase_e_dir,
            force=False,
        )
    except Exception:
        # For in-silico dry runs, always emit Phase E outputs even if the replication gate fails,
        # but record that the run is non-reviewer-proof.
        phase_e_forced = True
        run_phase_e_pool_after_replication(
            phase_d_run_dir=relabel_outputs.run_dir,
            out_dir=phase_e_dir,
            force=True,
        )

    # Report
    report_md = run_dir / "insilico_study_report.md"
    lines: list[str] = []
    lines.append("# In-silico Clinician Rating Study (GPT-5.2)")
    lines.append("")
    lines.append(f"- Rater sample dir: `{rater_sample_dir}`")
    lines.append(f"- Base Phase D run: `{phase_d_run_dir}`")
    lines.append(f"- Model: `{model}` (temperature={float(temperature)})")
    lines.append(f"- Endpoint: `{endpoint}`; request_delay_s={float(request_delay_s)}")
    lines.append(f"- Max cases: {int(max_cases) if max_cases is not None else 'ALL (from template)'}")
    lines.append("")
    lines.append("## Outputs")
    lines.append(f"- Rater A CSV: `{out_a.filled_csv}`")
    lines.append(f"- Rater B CSV: `{out_b.filled_csv}`")
    lines.append(f"- Adjudicator CSV: `{out_adj_csv}`" if out_adj_csv else "- Adjudicator CSV: (not needed; no disagreements)")
    lines.append(f"- Consensus CSV: `{consensus_csv}`")
    lines.append(f"- κ report: `{kappa_outputs.kappa_md}`")
    lines.append(f"- Phase D relabel run: `{relabel_outputs.run_dir}`")
    lines.append(f"- Phase E (from relabel): `{phase_e_dir}`" + (" (FORCED; gate failed)" if phase_e_forced else ""))
    lines.append("")
    lines.append("## Notes")
    lines.append("- This is *not* human validation; it is a software/LLM stability check and an end-to-end pipeline dry run.")
    lines.append("- Files are prefixed/labeled `INSILICO` to avoid confusion with human raters.")
    lines.append("")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    run_config_json = run_dir / "run_config.json"
    run_config = {
        "created_utc": ts,
        "rater_sample_dir": str(rater_sample_dir),
        "phase_d_run_dir": str(phase_d_run_dir),
        "model": str(model),
        "temperature": float(temperature),
        "endpoint": str(endpoint),
        "request_delay_s": float(request_delay_s),
        "max_output_tokens": int(max_output_tokens),
        "max_packet_chars": int(max_packet_chars) if max_packet_chars is not None else None,
        "max_cases": int(max_cases) if max_cases is not None else None,
        "exclude_non_ftld": bool(exclude_non_ftld),
        "phase_e_forced": bool(phase_e_forced),
        "outputs": {
            "rater_a_csv": str(out_a.filled_csv),
            "rater_b_csv": str(out_b.filled_csv),
            "adjudicator_csv": str(out_adj_csv) if out_adj_csv else None,
            "consensus_csv": str(consensus_csv),
            "kappa_dir": str(kappa_dir),
            "phase_d_relabel_dir": str(relabel_outputs.run_dir),
            "phase_e_dir": str(phase_e_dir),
        },
    }
    run_config_json.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    return InsilicoStudyOutputs(
        run_dir=run_dir,
        run_config_json=run_config_json,
        rater_a_csv=out_a.filled_csv,
        rater_b_csv=out_b.filled_csv,
        adjudicator_csv=out_adj_csv or (rater_adj_dir / "adjudicator_template.csv"),
        consensus_csv=consensus_csv,
        kappa_dir=kappa_outputs.out_dir,
        report_md=report_md,
        phase_d_relabel_dir=relabel_outputs.run_dir,
        phase_e_dir=phase_e_dir,
    )
