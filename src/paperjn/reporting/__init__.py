"""Reporting helpers (tables/figures/manuscript outputs)."""

from .plots import plot_ppa_enrichment_forest, plot_ppa_share_bars, plot_ppa_share_by_cluster
from .report import PaperRunSummary, write_paper_markdown_report

__all__ = [
    "PaperRunSummary",
    "plot_ppa_enrichment_forest",
    "plot_ppa_share_bars",
    "plot_ppa_share_by_cluster",
    "write_paper_markdown_report",
]
