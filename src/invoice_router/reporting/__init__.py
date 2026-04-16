"""Reporting package public API."""

from .summary import (
    build_issue_ledger,
    build_process_run_summary,
    compare_analysis_runs,
    compare_benchmark_summaries,
    describe_template_family,
    record_analysis_run,
    suggest_template_family_merges,
    suggest_template_family_retirements,
    suggest_template_family_splits,
    suggest_template_family_updates,
    summarize_failure_modes,
    summarize_family_benchmark_comparison,
    summarize_template_families,
    write_run_summary,
)

__all__ = [
    "build_issue_ledger",
    "build_process_run_summary",
    "compare_analysis_runs",
    "compare_benchmark_summaries",
    "describe_template_family",
    "record_analysis_run",
    "summarize_family_benchmark_comparison",
    "summarize_failure_modes",
    "summarize_template_families",
    "suggest_template_family_merges",
    "suggest_template_family_retirements",
    "suggest_template_family_splits",
    "suggest_template_family_updates",
    "write_run_summary",
]
