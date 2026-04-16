"""Benchmark and operational reporting helpers."""

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import TemplateLifecycleConfig
from ..domain.invoices.family_profiles import (
    merge_family_extraction_profiles,
    summarize_family_extraction_profile,
)
from ..infrastructure.persistence.postgres import PostgresClient, is_postgres_target
from ..infrastructure.persistence.storage import FingerprintDB
from ..models import (
    DocumentFamily,
    FingerprintRecord,
    ProcessingResult,
    Route,
    TemplateFamilyRecord,
    TemplateStatus,
)

_ISSUE_BUCKET_LABELS = {
    "routing_preprocessing": "routing / preprocessing",
    "party_block_extraction": "party block extraction",
    "summary_amount_resolution": "summary amount resolution",
    "line_item_detection": "line-item detection",
    "pdf_native_text_extraction": "PDF / native-text extraction",
    "scalar_reconciliation": "scalar reconciliation / validator mismatch",
    "runtime_error": "runtime error",
    "other": "other",
}

_ISSUE_BUCKET_WEIGHTS = {
    "routing_preprocessing": 1.40,
    "party_block_extraction": 1.20,
    "summary_amount_resolution": 1.00,
    "line_item_detection": 1.15,
    "pdf_native_text_extraction": 1.10,
    "scalar_reconciliation": 0.95,
    "runtime_error": 2.50,
    "other": 0.75,
}

_PARTY_FIELD_HINTS = (
    "invoice no",
    "invoice number",
    "invoice id",
    "invoice#",
    "customer",
    "customer name",
    "supplier",
    "vendor",
    "seller",
    "buyer",
    "bill to",
    "ship to",
    "remit to",
    "payer",
    "account name",
    "company name",
)

_GT_TRUST_CONFIG = TemplateLifecycleConfig()

_SUMMARY_FIELD_HINTS = (
    "total",
    "subtotal",
    "amount due",
    "balance due",
    "invoice total",
    "grand total",
    "tax",
    "vat",
    "amount payable",
    "net total",
)

_LINE_ITEM_FIELD_HINTS = (
    "line item",
    "line-item",
    "item count",
    "line items",
    "quantity",
    "description",
    "sku",
    "table",
    "row",
)

_PDF_FIELD_HINTS = (
    "pdf",
    "native text",
    "text extraction",
    "ocr",
)

_ROUTING_FIELD_HINTS = (
    "route",
    "routing",
    "preprocess",
    "preprocessing",
    "orientation",
    "deskew",
    "rotation",
    "rotate",
)

_MISSING_FIELD_RE = re.compile(r"^Missing extracted field:\s*(.+)$", re.IGNORECASE)
_MISMATCH_RE = re.compile(r"^Mismatch on\s+(.+?):", re.IGNORECASE)


def _normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _jaccard_similarity(values1: set[str], values2: set[str]) -> float:
    if not values1 or not values2:
        return 0.0
    union = values1 | values2
    if not union:
        return 0.0
    return len(values1 & values2) / len(union)


def _bucket_for_field_name(field_name: str) -> str:
    normalized = _normalize_text(field_name)
    if _contains_any(normalized, _PARTY_FIELD_HINTS):
        return "party_block_extraction"
    if _contains_any(normalized, _SUMMARY_FIELD_HINTS):
        return "summary_amount_resolution"
    if _contains_any(normalized, _LINE_ITEM_FIELD_HINTS):
        return "line_item_detection"
    if _contains_any(normalized, _PDF_FIELD_HINTS):
        return "pdf_native_text_extraction"
    if _contains_any(normalized, _ROUTING_FIELD_HINTS):
        return "routing_preprocessing"
    return "scalar_reconciliation"


def _bucket_for_error(error: str, invoice: Optional[Dict[str, Any]] = None) -> tuple[str, str]:
    text = str(error or "").strip()
    normalized = _normalize_text(text)
    if not text:
        return "other", "unknown"
    if normalized.startswith("runtime error") or "exception" in normalized:
        return "runtime_error", "runtime_error"
    if "line item count mismatch" in normalized:
        return "line_item_detection", "line_item_count_mismatch"
    if "line item arithmetic mismatch" in normalized:
        return "line_item_detection", "line_item_arithmetic_mismatch"
    if "subtotal mismatch" in normalized:
        return "summary_amount_resolution", "subtotal_mismatch"
    if "total mismatch" in normalized:
        return "summary_amount_resolution", "total_mismatch"
    if "currency mismatch" in normalized:
        return "scalar_reconciliation", "currency_mismatch"

    missing_match = _MISSING_FIELD_RE.match(text)
    if missing_match:
        field_name = missing_match.group(1).strip()
        return _bucket_for_field_name(field_name), "missing_field"

    mismatch_match = _MISMATCH_RE.match(text)
    if mismatch_match:
        field_name = mismatch_match.group(1).strip()
        return _bucket_for_field_name(field_name), "field_mismatch"

    if invoice is not None:
        if invoice.get("route") == "ERROR" or invoice.get("status") == "ERROR":
            return "routing_preprocessing", "runtime_error"
        if invoice.get("line_item_source") == "row_fallback":
            return "line_item_detection", "row_fallback"
        if invoice.get("table_detected") is False:
            return "line_item_detection", "table_not_detected"
        if invoice.get("summary_candidate_diagnostics"):
            return "summary_amount_resolution", "summary_candidate_conflict"
        if invoice.get("party_block_diagnostics"):
            return "party_block_extraction", "party_block_diagnostic"

    return "scalar_reconciliation", "other"


def _issue_weight(bucket: str, category: str, validation_score: Optional[float]) -> float:
    weight = _ISSUE_BUCKET_WEIGHTS.get(bucket, _ISSUE_BUCKET_WEIGHTS["other"])
    if category == "runtime_error":
        weight *= 1.25
    elif category in {"line_item_count_mismatch", "line_item_arithmetic_mismatch"}:
        weight *= 1.10
    elif category in {"missing_field", "field_mismatch"}:
        weight *= 1.0
    elif category in {"subtotal_mismatch", "total_mismatch"}:
        weight *= 1.05
    elif category == "currency_mismatch":
        weight *= 0.95

    if validation_score is not None:
        if validation_score < 0.5:
            weight *= 1.10
        elif validation_score < 0.8:
            weight *= 1.05
    return round(weight, 4)


def _issue_item(invoice: Dict[str, Any], error: str, *, source: str) -> Dict[str, Any]:
    bucket, category = _bucket_for_error(error, invoice)
    return {
        "invoice_name": invoice.get("invoice_name")
        or invoice.get("name")
        or invoice.get("invoice_path"),
        "invoice_path": invoice.get("invoice_path"),
        "bucket": bucket,
        "bucket_label": _ISSUE_BUCKET_LABELS.get(bucket, bucket.replace("_", " ")),
        "category": category,
        "weight": _issue_weight(bucket, category, invoice.get("validation_score")),
        "source": source,
        "detail": str(error),
        "attempted_route": invoice.get("attempted_route"),
        "route": invoice.get("route"),
        "status": invoice.get("status"),
    }


def _issues_from_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    per_invoice = summary.get("per_invoice") or []
    if per_invoice:
        for invoice in per_invoice:
            if invoice.get("validation_passed") is True:
                continue
            errors = invoice.get("validation_errors") or []
            if errors:
                for error in errors:
                    issues.append(_issue_item(invoice, error, source="validation_error"))
            else:
                fallback_error = (
                    "runtime error" if invoice.get("route") == "ERROR" else "validation failure"
                )
                issues.append(_issue_item(invoice, fallback_error, source="summary_fallback"))
        return issues

    failure_analysis = summary.get("failure_analysis") or {}
    error_counts = {}
    if (
        isinstance(summary.get("top_failure_categories"), dict)
        and summary["top_failure_categories"]
    ):
        error_counts = dict(summary["top_failure_categories"])
    elif isinstance(failure_analysis, dict):
        error_counts = dict(failure_analysis.get("discovery_error_category_counts") or {})
        if not error_counts and failure_analysis.get("failed_total"):
            error_counts = {"failed_total": int(failure_analysis["failed_total"])}

    if error_counts:
        pseudo_invoice = {
            "invoice_name": summary.get("dataset", "summary"),
            "invoice_path": summary.get("input_dir"),
            "validation_score": None,
            "attempted_route": None,
            "route": None,
            "status": "summary",
        }
        for category_name, count in sorted(error_counts.items()):
            bucket, category = _bucket_for_error(category_name, pseudo_invoice)
            if category_name == "failed_total":
                bucket, category = "scalar_reconciliation", "other"
            for _ in range(int(count)):
                issues.append(
                    {
                        "invoice_name": pseudo_invoice["invoice_name"],
                        "invoice_path": pseudo_invoice["invoice_path"],
                        "bucket": bucket,
                        "bucket_label": _ISSUE_BUCKET_LABELS.get(bucket, bucket.replace("_", " ")),
                        "category": category,
                        "weight": _issue_weight(bucket, category, None),
                        "source": "failure_summary",
                        "detail": str(category_name),
                        "attempted_route": None,
                        "route": None,
                        "status": "summary",
                    }
                )

    return issues


def build_issue_ledger(summary: Dict[str, Any]) -> Dict[str, Any]:
    issues = _issues_from_summary(summary)
    buckets: Dict[str, Dict[str, Any]] = {
        bucket: {
            "bucket": bucket,
            "bucket_label": label,
            "issue_count": 0,
            "weighted_issue_count": 0.0,
            "invoice_names": set(),
            "categories": Counter(),
        }
        for bucket, label in _ISSUE_BUCKET_LABELS.items()
    }

    for issue in issues:
        bucket = issue["bucket"]
        bucket_entry = buckets.setdefault(
            bucket,
            {
                "bucket": bucket,
                "bucket_label": issue.get("bucket_label", bucket.replace("_", " ")),
                "issue_count": 0,
                "weighted_issue_count": 0.0,
                "invoice_names": set(),
                "categories": Counter(),
            },
        )
        bucket_entry["issue_count"] += 1
        bucket_entry["weighted_issue_count"] += float(issue["weight"])
        if issue.get("invoice_name"):
            bucket_entry["invoice_names"].add(str(issue["invoice_name"]))
        bucket_entry["categories"][issue["category"]] += 1

    work_buckets = []
    total_weighted_issue_count = 0.0
    for bucket, entry in buckets.items():
        if entry["issue_count"] == 0:
            continue
        weighted_issue_count = round(float(entry["weighted_issue_count"]), 4)
        total_weighted_issue_count += weighted_issue_count
        work_buckets.append(
            {
                "bucket": bucket,
                "bucket_label": entry["bucket_label"],
                "issue_count": entry["issue_count"],
                "weighted_issue_count": weighted_issue_count,
                "invoice_names": sorted(entry["invoice_names"]),
                "categories": dict(sorted(entry["categories"].items())),
            }
        )

    work_buckets.sort(key=lambda item: (-item["weighted_issue_count"], item["bucket"]))
    issues.sort(
        key=lambda item: (
            item.get("invoice_name") or "",
            item["bucket"],
            item["category"],
            item["detail"],
        )
    )

    return {
        "version": 1,
        "issue_count": len(issues),
        "weighted_issue_count": round(total_weighted_issue_count, 4),
        "work_buckets": work_buckets,
        "issues": issues,
    }


def compare_benchmark_summaries(
    baseline_summary: Dict[str, Any],
    candidate_summary: Dict[str, Any],
    *,
    target_reduction: float = 0.75,
) -> Dict[str, Any]:
    baseline_gt = baseline_summary.get("ground_truth_sync") or {}
    candidate_gt = candidate_summary.get("ground_truth_sync") or {}
    baseline_ledger = build_issue_ledger(baseline_summary)
    candidate_ledger = build_issue_ledger(candidate_summary)

    provenance_warnings: List[str] = []
    baseline_status = baseline_gt.get("status")
    candidate_status = candidate_gt.get("status")
    if baseline_status and baseline_status not in {"in_sync", "self_contained"}:
        provenance_warnings.append(f"Baseline GT sync is `{baseline_status}`.")
    if candidate_status and candidate_status not in {"in_sync", "self_contained"}:
        provenance_warnings.append(f"Candidate GT sync is `{candidate_status}`.")
    baseline_source = baseline_gt.get("source_of_truth_dir")
    candidate_source = candidate_gt.get("source_of_truth_dir")
    if baseline_source and candidate_source and baseline_source != candidate_source:
        provenance_warnings.append(
            "Baseline and candidate use different GT source-of-truth directories."
        )

    comparison_is_trusted = not provenance_warnings
    baseline_latency = _latency_stats_from_summary(baseline_summary)
    candidate_latency = _latency_stats_from_summary(candidate_summary)
    baseline_memory = _peak_rss_stats_from_summary(baseline_summary)
    candidate_memory = _peak_rss_stats_from_summary(candidate_summary)
    if baseline_memory is None or candidate_memory is None:
        provenance_warnings.append(
            "Peak RSS comparison is unavailable because one or both benchmark summaries are missing memory measurements."
        )

    baseline_weighted = float(baseline_ledger["weighted_issue_count"])
    candidate_weighted = float(candidate_ledger["weighted_issue_count"])
    weighted_reduction = baseline_weighted - candidate_weighted
    weighted_reduction_ratio = (
        (weighted_reduction / baseline_weighted) if baseline_weighted else None
    )
    target_weighted_issue_count = (
        round(baseline_weighted * (1.0 - target_reduction), 4) if baseline_weighted else 0.0
    )
    progress_to_target_75 = (
        (weighted_reduction / (baseline_weighted * target_reduction))
        if baseline_weighted and target_reduction
        else None
    )
    remaining_weight_to_target = (
        round(max(candidate_weighted - target_weighted_issue_count, 0.0), 4)
        if baseline_weighted
        else 0.0
    )

    bucket_names = sorted(
        {bucket["bucket"] for bucket in baseline_ledger["work_buckets"]}
        | {bucket["bucket"] for bucket in candidate_ledger["work_buckets"]}
    )
    baseline_buckets = {bucket["bucket"]: bucket for bucket in baseline_ledger["work_buckets"]}
    candidate_buckets = {bucket["bucket"]: bucket for bucket in candidate_ledger["work_buckets"]}
    bucket_comparison = []
    for bucket in bucket_names:
        before = baseline_buckets.get(
            bucket,
            {
                "issue_count": 0,
                "weighted_issue_count": 0.0,
                "bucket_label": _ISSUE_BUCKET_LABELS.get(bucket, bucket.replace("_", " ")),
            },
        )
        after = candidate_buckets.get(
            bucket,
            {
                "issue_count": 0,
                "weighted_issue_count": 0.0,
                "bucket_label": _ISSUE_BUCKET_LABELS.get(bucket, bucket.replace("_", " ")),
            },
        )
        bucket_comparison.append(
            {
                "bucket": bucket,
                "bucket_label": before.get("bucket_label")
                or after.get("bucket_label")
                or bucket.replace("_", " "),
                "baseline_issue_count": before["issue_count"],
                "candidate_issue_count": after["issue_count"],
                "delta_issue_count": after["issue_count"] - before["issue_count"],
                "baseline_weighted_issue_count": round(float(before["weighted_issue_count"]), 4),
                "candidate_weighted_issue_count": round(float(after["weighted_issue_count"]), 4),
                "delta_weighted_issue_count": round(
                    float(after["weighted_issue_count"]) - float(before["weighted_issue_count"]), 4
                ),
            }
        )

    bucket_comparison.sort(
        key=lambda item: (
            -max(item["baseline_weighted_issue_count"], item["candidate_weighted_issue_count"]),
            item["bucket"],
        )
    )

    recommended_next_buckets = sorted(
        candidate_ledger["work_buckets"],
        key=lambda item: (-item["weighted_issue_count"], item["bucket"]),
    )

    return {
        "target_reduction": target_reduction,
        "provenance": {
            "comparison_is_trusted": comparison_is_trusted,
            "baseline_ground_truth_sync": baseline_gt,
            "candidate_ground_truth_sync": candidate_gt,
            "warnings": provenance_warnings,
        },
        "latency_comparison": {
            "baseline_avg_ms": baseline_latency["avg"] if baseline_latency else None,
            "candidate_avg_ms": candidate_latency["avg"] if candidate_latency else None,
            "delta_avg_ms": (
                candidate_latency["avg"] - baseline_latency["avg"]
                if baseline_latency and candidate_latency
                else None
            ),
            "baseline_max_ms": baseline_latency["max"] if baseline_latency else None,
            "candidate_max_ms": candidate_latency["max"] if candidate_latency else None,
            "delta_max_ms": (
                candidate_latency["max"] - baseline_latency["max"]
                if baseline_latency and candidate_latency
                else None
            ),
        },
        "memory_comparison": {
            "baseline_avg_peak_rss_mb": baseline_memory["avg"] if baseline_memory else None,
            "candidate_avg_peak_rss_mb": candidate_memory["avg"] if candidate_memory else None,
            "delta_avg_peak_rss_mb": (
                round(candidate_memory["avg"] - baseline_memory["avg"], 3)
                if baseline_memory and candidate_memory
                else None
            ),
            "baseline_max_peak_rss_mb": baseline_memory["max"] if baseline_memory else None,
            "candidate_max_peak_rss_mb": candidate_memory["max"] if candidate_memory else None,
            "delta_max_peak_rss_mb": (
                round(candidate_memory["max"] - baseline_memory["max"], 3)
                if baseline_memory and candidate_memory
                else None
            ),
        },
        "baseline": baseline_ledger,
        "candidate": candidate_ledger,
        "progress": {
            "baseline_weighted_issue_count": round(baseline_weighted, 4),
            "candidate_weighted_issue_count": round(candidate_weighted, 4),
            "weighted_reduction": round(weighted_reduction, 4),
            "weighted_reduction_ratio": round(weighted_reduction_ratio, 4)
            if weighted_reduction_ratio is not None
            else None,
            "progress_to_target_75": round(progress_to_target_75, 4)
            if progress_to_target_75 is not None
            else None,
            "target_weighted_issue_count": round(target_weighted_issue_count, 4),
            "remaining_weight_to_target": remaining_weight_to_target,
            "meets_target_75": candidate_weighted <= target_weighted_issue_count
            if baseline_weighted
            else candidate_weighted == 0.0,
        },
        "bucket_comparison": bucket_comparison,
        "recommended_next_buckets": recommended_next_buckets,
    }


def _matches_dataset(invoice_path: str, dataset_filter: Optional[str]) -> bool:
    if not dataset_filter:
        return True
    return dataset_filter in Path(invoice_path).parts


def _score_band(score: Optional[float], threshold: float) -> str:
    if score is None:
        return "unknown"
    if score < 0.5:
        return "<0.5"
    if score < 0.8:
        return "0.5-0.8"
    if score < threshold:
        return f"0.8-{threshold:.2f}"
    return f">={threshold:.2f}"


def _error_categories(result: ProcessingResult) -> List[str]:
    diagnostics = result.diagnostics
    if not diagnostics or not diagnostics.validation_errors:
        return ["unknown"]

    categories = []
    for err in diagnostics.validation_errors:
        if err.startswith("Missing extracted field:"):
            categories.append("missing_field")
        elif err.startswith("Mismatch on "):
            categories.append("field_mismatch")
        elif err.startswith("Line item count mismatch:"):
            categories.append("line_item_count_mismatch")
        elif err.startswith("Line item arithmetic mismatch"):
            categories.append("line_item_arithmetic_mismatch")
        elif err.startswith("Subtotal mismatch:"):
            categories.append("subtotal_mismatch")
        elif err.startswith("Total mismatch:"):
            categories.append("total_mismatch")
        elif err.startswith("Currency mismatch for country"):
            categories.append("country_currency_mismatch")
        else:
            categories.append("other")
    return categories


def summarize_failure_modes(
    db: FingerprintDB, dataset_filter: Optional[str] = None, discovery_threshold: float = 0.95
) -> Dict[str, Any]:
    failed_results = [
        result
        for result in db.get_failed_results()
        if _matches_dataset(result.invoice_path, dataset_filter)
    ]

    route_counts: Counter[str] = Counter()
    error_category_counts: Counter[str] = Counter()
    stage_status_counts: Counter[str] = Counter()
    discovery_mode_counts: Counter[str] = Counter()
    locate_error_category_counts: Counter[str] = Counter()
    extract_error_category_counts: Counter[str] = Counter()
    scalar_count_buckets: Counter[str] = Counter()
    score_band_counts: Counter[str] = Counter()
    line_item_mismatch = 0

    discovery_failures: List[ProcessingResult] = []
    for result in failed_results:
        attempted = result.attempted_route or (
            result.diagnostics.attempted_route if result.diagnostics else None
        )
        route_label = attempted.value if attempted else "UNKNOWN"
        route_counts[route_label] += 1
        if attempted == Route.DISCOVERY:
            discovery_failures.append(result)

    for result in discovery_failures:
        diagnostics = result.diagnostics
        if diagnostics and diagnostics.discovery_stage_status:
            stage_status_counts[diagnostics.discovery_stage_status] += 1
        if diagnostics and diagnostics.discovery_mode:
            discovery_mode_counts[diagnostics.discovery_mode] += 1
        if diagnostics and diagnostics.locate_error_category:
            locate_error_category_counts[diagnostics.locate_error_category] += 1
        if diagnostics and diagnostics.extract_error_category:
            extract_error_category_counts[diagnostics.extract_error_category] += 1
        for category in _error_categories(result):
            error_category_counts[category] += 1

        missing = diagnostics.scalar_field_missing_count if diagnostics else None
        mismatch = diagnostics.scalar_field_mismatch_count if diagnostics else None
        scalar_count_buckets[f"missing={missing or 0}, mismatch={mismatch or 0}"] += 1

        if diagnostics and any(
            err.startswith("Line item count mismatch:") for err in diagnostics.validation_errors
        ):
            line_item_mismatch += 1

        score_band_counts[
            _score_band(diagnostics.validation_score if diagnostics else None, discovery_threshold)
        ] += 1

    return {
        "dataset_filter": dataset_filter,
        "failed_total": len(failed_results),
        "failed_discovery": len(discovery_failures),
        "failed_apply": route_counts.get(Route.APPLY.value, 0),
        "attempted_route_counts": dict(route_counts),
        "discovery_mode_counts": dict(discovery_mode_counts),
        "discovery_stage_status_counts": dict(stage_status_counts),
        "discovery_locate_error_category_counts": dict(locate_error_category_counts),
        "discovery_extract_error_category_counts": dict(extract_error_category_counts),
        "discovery_error_category_counts": dict(error_category_counts),
        "discovery_scalar_count_buckets": dict(scalar_count_buckets),
        "discovery_line_item_mismatch_count": line_item_mismatch,
        "discovery_score_band_counts": dict(score_band_counts),
    }


def _latency_stats(latencies: List[int]) -> Optional[Dict[str, int]]:
    if not latencies:
        return None
    return {
        "min": min(latencies),
        "avg": sum(latencies) // len(latencies),
        "max": max(latencies),
    }


def _float_stats(values: List[float], *, precision: int = 3) -> Optional[Dict[str, float]]:
    if not values:
        return None
    return {
        "min": round(min(values), precision),
        "avg": round(sum(values) / len(values), precision),
        "max": round(max(values), precision),
    }


def _latency_stats_from_summary(summary: Dict[str, Any]) -> Optional[Dict[str, int]]:
    stats = summary.get("latency_stats_ms")
    if isinstance(stats, dict) and {"min", "avg", "max"} <= set(stats):
        return {
            "min": int(stats["min"]),
            "avg": int(stats["avg"]),
            "max": int(stats["max"]),
        }

    per_invoice = summary.get("per_invoice") or summary.get("invoices") or []
    latencies = [
        int(invoice["elapsed_ms"])
        for invoice in per_invoice
        if invoice.get("elapsed_ms") is not None
    ]
    if latencies:
        return _latency_stats(latencies)

    avg_time_ms = summary.get("avg_time_ms")
    if avg_time_ms is None:
        return None
    avg_value = int(avg_time_ms)
    return {"min": avg_value, "avg": avg_value, "max": avg_value}


def _peak_rss_stats_from_summary(summary: Dict[str, Any]) -> Optional[Dict[str, float]]:
    stats = summary.get("peak_rss_stats_mb")
    if isinstance(stats, dict) and {"min", "avg", "max"} <= set(stats):
        return {
            "min": round(float(stats["min"]), 3),
            "avg": round(float(stats["avg"]), 3),
            "max": round(float(stats["max"]), 3),
        }

    per_invoice = summary.get("per_invoice") or summary.get("invoices") or []
    values = [
        float(invoice["peak_rss_mb"])
        for invoice in per_invoice
        if invoice.get("peak_rss_mb") is not None
    ]
    return _float_stats(values)


def _build_family_rankings(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    family_rows: Dict[str, Dict[str, Any]] = {}

    for item in results:
        result = item.get("result")
        if result is None:
            continue

        diagnostics = result.diagnostics
        attempted_route = result.attempted_route or (
            diagnostics.attempted_route if diagnostics else None
        )
        family_id = result.template_family_id or (
            result.provenance.template_family_id if result.provenance else None
        )
        if not family_id:
            continue

        row = family_rows.setdefault(
            family_id,
            {
                "template_family_id": family_id,
                "invoice_count": 0,
                "accepted_count": 0,
                "review_count": 0,
                "error_count": 0,
                "apply_route_count": 0,
                "apply_success_count": 0,
                "family_match_count": 0,
                "reconciliation_failure_count": 0,
                "validator_policy_failure_count": 0,
                "validation_scores": [],
                "discovery_stage_status_counts": Counter(),
                "validation_error_counts": Counter(),
                "example_invoices": [],
            },
        )

        row["invoice_count"] += 1
        invoice_name = item.get("name")
        if (
            invoice_name
            and invoice_name not in row["example_invoices"]
            and len(row["example_invoices"]) < 3
        ):
            row["example_invoices"].append(invoice_name)

        if attempted_route == Route.APPLY:
            row["apply_route_count"] += 1
        if result.validation_passed is True:
            row["accepted_count"] += 1
            if attempted_route == Route.APPLY:
                row["apply_success_count"] += 1
        elif result.validation_passed is False or result.route_used == Route.REJECTED:
            row["review_count"] += 1

        if diagnostics:
            if diagnostics.family_match_outcome == "matched":
                row["family_match_count"] += 1
            if diagnostics.reconciliation_failure:
                row["reconciliation_failure_count"] += 1
            if diagnostics.validator_policy_failure:
                row["validator_policy_failure_count"] += 1
            if diagnostics.validation_score is not None:
                row["validation_scores"].append(float(diagnostics.validation_score))
            if diagnostics.discovery_stage_status:
                row["discovery_stage_status_counts"][diagnostics.discovery_stage_status] += 1
            if diagnostics.validation_error_counts:
                row["validation_error_counts"].update(diagnostics.validation_error_counts)

    family_rankings: List[Dict[str, Any]] = []
    for row in family_rows.values():
        invoice_count = max(int(row["invoice_count"]), 1)
        accepted_count = int(row["accepted_count"])
        review_count = int(row["review_count"])
        error_count = int(row["error_count"])
        apply_route_count = int(row["apply_route_count"])
        apply_success_count = int(row["apply_success_count"])
        unresolved_count = review_count + error_count
        pass_rate = accepted_count / invoice_count
        apply_reuse_rate = apply_success_count / apply_route_count if apply_route_count else 0.0
        avg_validation_score = (
            round(sum(row["validation_scores"]) / len(row["validation_scores"]), 4)
            if row["validation_scores"]
            else None
        )
        impact_score = round(
            (invoice_count * (1.0 - pass_rate) * 1.4)
            + (apply_route_count * 0.35)
            + (row["reconciliation_failure_count"] * 0.25)
            + (row["validator_policy_failure_count"] * 0.15),
            3,
        )
        family_rankings.append(
            {
                "template_family_id": row["template_family_id"],
                "invoice_count": invoice_count,
                "accepted_count": accepted_count,
                "review_count": review_count,
                "error_count": error_count,
                "unresolved_count": unresolved_count,
                "pass_rate": round(pass_rate, 4),
                "apply_route_count": apply_route_count,
                "apply_reuse_rate": round(apply_reuse_rate, 4) if apply_route_count else 0.0,
                "family_match_rate": round(row["family_match_count"] / invoice_count, 4),
                "avg_validation_score": avg_validation_score,
                "reconciliation_failure_count": int(row["reconciliation_failure_count"]),
                "validator_policy_failure_count": int(row["validator_policy_failure_count"]),
                "discovery_stage_status_counts": dict(row["discovery_stage_status_counts"]),
                "validation_error_counts": dict(row["validation_error_counts"]),
                "example_invoices": row["example_invoices"],
                "impact_score": impact_score,
            }
        )

    family_rankings.sort(
        key=lambda item: (
            item["unresolved_count"] == 0,
            -item["impact_score"],
            -item["invoice_count"],
            item["template_family_id"],
        )
    )
    return family_rankings


def build_process_run_summary(
    input_dir: str,
    results: List[Dict[str, Any]],
    total_ms: int,
    failure_summary: Optional[Dict[str, Any]] = None,
    discovery_only: bool = False,
    discovery_mode: Optional[str] = None,
    healing_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    route_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    error_categories: Counter[str] = Counter()
    attempted_route_counts: Counter[str] = Counter()
    discovery_stage_status_counts: Counter[str] = Counter()

    discovery_passed = 0
    discovery_rejected = 0
    apply_rejected = 0
    runtime_errors = 0
    apply_runtime_errors = 0
    discovery_runtime_errors = 0
    invoice_details: List[Dict[str, Any]] = []

    quality_scores: List[float] = []
    latencies_ms: List[int] = []
    peak_rss_values_mb: List[float] = []
    family_counts: Counter[str] = Counter()
    family_successes: Counter[str] = Counter()
    family_failures: Counter[str] = Counter()

    for item in results:
        route_counts[item["route"]] += 1
        status_counts[item["status"]] += 1
        if item.get("elapsed_ms") is not None:
            latencies_ms.append(int(item["elapsed_ms"]))
        if item.get("peak_rss_mb") is not None:
            peak_rss_values_mb.append(float(item["peak_rss_mb"]))

        result = item.get("result")
        error = item.get("error")

        attempted_route = None
        validation_score = None
        if result is not None:
            attempted_route = result.attempted_route or (
                result.diagnostics.attempted_route if result.diagnostics else None
            )
            if attempted_route is not None:
                attempted_route_counts[attempted_route.value] += 1
            family_id = result.template_family_id or (
                result.provenance.template_family_id if result.provenance else None
            )
            if family_id:
                family_counts[family_id] += 1
                if result.validation_passed is True:
                    family_successes[family_id] += 1
                elif result.validation_passed is False:
                    family_failures[family_id] += 1
            if attempted_route == Route.DISCOVERY and result.validation_passed is True:
                discovery_passed += 1
            elif attempted_route == Route.DISCOVERY and result.validation_passed is False:
                discovery_rejected += 1
            elif attempted_route == Route.APPLY and result.validation_passed is False:
                apply_rejected += 1

            if result.image_quality_score is not None:
                quality_scores.append(result.image_quality_score)
            if result.diagnostics:
                validation_score = result.diagnostics.validation_score
                if result.diagnostics.discovery_stage_status:
                    discovery_stage_status_counts[result.diagnostics.discovery_stage_status] += 1
        elif error is not None:
            runtime_errors += 1
            error_categories[str(error)] += 1
            attempted_route = getattr(error, "attempted_route", None)
            if attempted_route is not None:
                attempted_route_counts[attempted_route.value] += 1
            if attempted_route == Route.APPLY:
                apply_runtime_errors += 1
            elif attempted_route == Route.DISCOVERY:
                discovery_runtime_errors += 1

        invoice_details.append(
            {
                "name": item["name"],
                "route": item["route"],
                "attempted_route": attempted_route.value if attempted_route else None,
                "status": item["status"],
                "elapsed_ms": item["elapsed_ms"],
                "peak_rss_mb": item.get("peak_rss_mb"),
                "validation_score": validation_score,
                "template_family_id": result.template_family_id if result else None,
                "family_match_outcome": result.diagnostics.family_match_outcome
                if result and result.diagnostics
                else None,
                "determination_sources": list(result.provenance.determination_sources)
                if result and result.provenance
                else [],
                "gt_backed": bool(result.provenance.gt_backed)
                if result and result.provenance
                else False,
                "healing_reprocessed": bool(result.provenance.healing_reprocessed)
                if result and result.provenance
                else False,
            }
        )

    solved_families = {
        family_id for family_id, passed_count in family_successes.items() if passed_count > 0
    }
    family_review_queue = sorted(
        family_id for family_id, failed_count in family_failures.items() if failed_count > 0
    )
    invoices_covered_by_solved_families = sum(
        family_counts[family_id] for family_id in solved_families
    )
    attempted_apply = attempted_route_counts.get(Route.APPLY.value, 0)
    family_rankings = _build_family_rankings(results)
    highest_value_unstable_families = [
        {
            "template_family_id": row["template_family_id"],
            "invoice_count": row["invoice_count"],
            "pass_rate": row["pass_rate"],
            "unresolved_count": row["unresolved_count"],
            "impact_score": row["impact_score"],
        }
        for row in family_rankings
        if row["unresolved_count"] > 0
    ][:5]

    summary = {
        "input_dir": input_dir,
        "dataset": Path(input_dir).name,
        "discovery_only": discovery_only,
        "discovery_mode": discovery_mode,
        "total_invoices": len(results),
        "total_time_ms": total_ms,
        "avg_time_ms": total_ms // len(results) if results else 0,
        "latency_stats_ms": _latency_stats(latencies_ms),
        "peak_rss_stats_mb": _float_stats(peak_rss_values_mb),
        "route_counts": dict(route_counts),
        "status_counts": dict(status_counts),
        "attempted_route_counts": dict(attempted_route_counts),
        "discovery_passed": discovery_passed,
        "discovery_rejected": discovery_rejected,
        "apply_rejected": apply_rejected,
        "runtime_errors": runtime_errors,
        "apply_runtime_errors": apply_runtime_errors,
        "discovery_runtime_errors": discovery_runtime_errors,
        "runtime_error_categories": dict(error_categories),
        "quality_scores": {
            "min": min(quality_scores),
            "avg": sum(quality_scores) / len(quality_scores),
            "max": max(quality_scores),
        }
        if quality_scores
        else None,
        "family_counts": dict(family_counts),
        "family_metrics": {
            "family_match_rate": round(sum(family_counts.values()) / len(results), 4)
            if results
            else 0.0,
            "apply_rate": round(route_counts.get(Route.APPLY.value, 0) / len(results), 4)
            if results
            else 0.0,
            "false_apply_rate": round(apply_rejected / attempted_apply, 4)
            if attempted_apply
            else 0.0,
            "solved_family_count": len(solved_families),
            "invoice_coverage_by_solved_families": round(
                invoices_covered_by_solved_families / len(results), 4
            )
            if results
            else 0.0,
            "family_review_queue": family_review_queue,
            "unresolved_family_count": sum(
                1 for row in family_rankings if row["unresolved_count"] > 0
            ),
            "highest_value_unstable_families": highest_value_unstable_families,
        },
        "failure_analysis": failure_summary or {},
        "healing_summary": healing_summary or {},
        "discovery_stage_status_counts": dict(discovery_stage_status_counts),
        "family_rankings": family_rankings,
        "invoices": invoice_details,
    }
    return summary


def write_run_summary(summary: Dict[str, Any], output_dir: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_summary_{summary['dataset']}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    return out_path


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _classify_template_family_triage(
    *,
    status: str,
    apply_count: int,
    reject_count: int,
    field_override_count: int,
) -> Dict[str, Any]:
    invoice_count = max(0, int(apply_count) + int(reject_count))
    pass_rate = _safe_ratio(apply_count, invoice_count)
    reject_pressure = _safe_ratio(reject_count, invoice_count)
    unresolved_reviews = reject_count > 0 or status in {"degraded", "provisional"}

    triage_class = "low_value_tail"
    if (
        invoice_count >= 8
        and pass_rate >= 0.8
        and reject_pressure <= 0.2
        and status == "established"
    ):
        triage_class = "stable"
    elif invoice_count >= 250 and pass_rate < 0.35 and reject_pressure >= 0.65:
        triage_class = "over_broad_candidate"
    elif invoice_count >= 40 and reject_count >= 15 and pass_rate < 0.5:
        triage_class = "high_value_unstable"
    elif invoice_count >= 15 and reject_count >= max(8, apply_count * 2) and pass_rate < 0.4:
        triage_class = "degraded_reuse"
    elif (
        invoice_count <= 10
        and unresolved_reviews
        and (status == "degraded" or field_override_count >= 3 or reject_count >= apply_count)
    ):
        triage_class = "suspiciously_tiny"

    if triage_class == "stable":
        explanation = "Healthy reuse with low reject pressure and a solid pass rate."
    elif triage_class == "over_broad_candidate":
        explanation = "Large family with poor outcomes; prefer splitting or stricter reuse before expanding it."
    elif triage_class == "high_value_unstable":
        explanation = "Large unresolved family with enough volume to matter operationally."
    elif triage_class == "degraded_reuse":
        explanation = "Family is being reused often enough to matter, but rejects dominate successful applies."
    elif triage_class == "suspiciously_tiny":
        explanation = "Small unresolved family that may be acting like a micro-patch rather than reusable structure."
    else:
        explanation = (
            "Low-volume family that is currently lower priority than unstable high-value families."
        )

    triage_rank = {
        "over_broad_candidate": 0,
        "high_value_unstable": 1,
        "degraded_reuse": 2,
        "suspiciously_tiny": 3,
        "low_value_tail": 4,
        "stable": 5,
    }[triage_class]
    impact_score = round((reject_count * 2.0) + invoice_count + (field_override_count * 0.5), 2)
    return {
        "triage_class": triage_class,
        "triage_explanation": (
            f"{explanation} (invoices={invoice_count}, pass_rate={pass_rate * 100:.1f}%, rejects={reject_count})"
        ),
        "triage_rank": triage_rank,
        "invoice_count": invoice_count,
        "pass_rate": round(pass_rate, 4),
        "reject_pressure": round(reject_pressure, 4),
        "unresolved_count": int(reject_count),
        "impact_score": impact_score,
    }


def _healing_rollup(results: List[ProcessingResult]) -> Dict[str, int]:
    attempts = 0
    recoveries = 0
    for result in results:
        provenance = result.provenance
        if provenance and provenance.healing_reprocessed:
            attempts += 1
            if result.validation_passed is True or result.route_used == Route.APPLY:
                recoveries += 1
    return {
        "healed_attempt_count": attempts,
        "healed_recovery_count": recoveries,
    }


def summarize_template_families(db: FingerprintDB) -> Dict[str, Any]:
    families = db.get_active_template_families()
    status_counts: Counter[str] = Counter()
    family_rows: List[Dict[str, Any]] = []

    for family in families:
        status_counts[family.status.value] += 1
        representative = db.get_family_representative_fingerprint(family.template_family_id)
        family_results = db.get_results_for_fingerprints(
            [record.hash for record in db.get_fingerprints_for_family(family.template_family_id)]
        )
        healing_rollup = _healing_rollup(family_results)
        example_count = len(db.get_template_family_examples(family.template_family_id))
        versions = db.get_template_family_versions(family.template_family_id)
        extraction_profile = summarize_family_extraction_profile(family.extraction_profile)
        triage = _classify_template_family_triage(
            status=family.status.value,
            apply_count=int(family.apply_count),
            reject_count=int(family.reject_count),
            field_override_count=int(extraction_profile.get("field_override_count") or 0),
        )
        family_rows.append(
            {
                "template_family_id": family.template_family_id,
                "status": family.status.value,
                "confidence": round(float(family.confidence), 4),
                "apply_count": int(family.apply_count),
                "reject_count": int(family.reject_count),
                "gt_apply_count": int(family.gt_apply_count),
                "gt_reject_count": int(family.gt_reject_count),
                "gt_confidence": round(float(family.gt_confidence), 4),
                "gt_trust_qualified": bool(
                    family.gt_apply_count >= _GT_TRUST_CONFIG.establish_min_count
                    and float(family.gt_confidence) >= _GT_TRUST_CONFIG.establish_min_confidence
                ),
                "provider_name": family.provider_name,
                "country_code": family.country_code,
                "document_family": family.document_family.value,
                "extraction_profile": extraction_profile,
                "example_count": example_count,
                "version_count": len(versions),
                "last_change_reason": versions[-1].change_reason if versions else None,
                "representative_fingerprint_hash": representative.hash if representative else None,
                "representative_confidence": round(float(representative.confidence), 4)
                if representative
                else None,
                "representative_gt_apply_count": int(representative.gt_apply_count)
                if representative
                else 0,
                "representative_gt_reject_count": int(representative.gt_reject_count)
                if representative
                else 0,
                "representative_gt_confidence": round(float(representative.gt_confidence), 4)
                if representative
                else 0.0,
                **healing_rollup,
                **triage,
            }
        )

    family_rows.sort(
        key=lambda item: (
            item["triage_rank"],
            -item["impact_score"],
            item["status"] != "degraded",
            item["status"] != "provisional",
            item["status"] != "established",
            -item["reject_count"],
            item["template_family_id"],
        )
    )

    return {
        "total_families": len(families),
        "status_counts": dict(status_counts),
        "review_queue": [
            row["template_family_id"]
            for row in family_rows
            if row["status"] in {"degraded", "provisional"}
            or row["triage_class"]
            in {
                "over_broad_candidate",
                "high_value_unstable",
                "degraded_reuse",
                "suspiciously_tiny",
            }
        ],
        "families": family_rows,
    }


def _family_review_signals_for_result(result: ProcessingResult) -> List[str]:
    diagnostics = result.diagnostics
    if diagnostics is None:
        return []
    signals: List[str] = []
    for error in diagnostics.validation_errors[:2]:
        signals.append(str(error))
    return signals


def _representative_family_field_names(
    representative: Optional[FingerprintRecord],
) -> Dict[str, List[str]]:
    date_fields: List[str] = []
    amount_fields: List[str] = []
    if representative is None:
        return {"date_fields": date_fields, "amount_fields": amount_fields}
    for page in representative.layout_template.get("pages", []):
        for field_name in (page.get("fields") or {}).keys():
            lowered = str(field_name).lower()
            if "date" in lowered:
                date_fields.append(str(field_name))
            if any(token in lowered for token in ("total", "subtotal", "tax", "amount")):
                amount_fields.append(str(field_name))
    return {
        "date_fields": sorted(set(date_fields)),
        "amount_fields": sorted(set(amount_fields)),
    }


def _append_unique_name(target: List[str], invoice_name: str, *, limit: int = 3) -> None:
    if invoice_name and invoice_name not in target and len(target) < limit:
        target.append(invoice_name)


def _family_profile_suggestions(
    family: TemplateFamilyRecord,
    representative: Optional[FingerprintRecord],
    recent_results: List[ProcessingResult],
    *,
    min_support: int,
) -> List[Dict[str, Any]]:
    profile = family.extraction_profile or {}
    field_hints = _representative_family_field_names(representative)
    counters = {
        "date": {"count": 0, "examples": []},
        "summary_amount": {"count": 0, "examples": []},
        "line_item": {"count": 0, "examples": []},
    }

    for result in recent_results:
        diagnostics = result.diagnostics
        if diagnostics is None:
            continue
        invoice_name = Path(result.invoice_path).name
        date_hit = False
        summary_hit = False
        line_item_hit = False
        for error in diagnostics.validation_errors:
            lowered = str(error).lower()
            if (
                "missing extracted field:" in lowered or "mismatch on " in lowered
            ) and "date" in lowered:
                date_hit = True
            if (
                "total mismatch" in lowered
                or "subtotal mismatch" in lowered
                or (
                    ("missing extracted field:" in lowered or "mismatch on " in lowered)
                    and any(token in lowered for token in ("total", "subtotal", "tax", "amount"))
                )
            ):
                summary_hit = True
            if "line item count mismatch" in lowered or "line item arithmetic mismatch" in lowered:
                line_item_hit = True
        if diagnostics.table_detected is False or diagnostics.line_item_source == "row_fallback":
            line_item_hit = True

        if date_hit:
            counters["date"]["count"] += 1
            _append_unique_name(counters["date"]["examples"], invoice_name)
        if summary_hit:
            counters["summary_amount"]["count"] += 1
            _append_unique_name(counters["summary_amount"]["examples"], invoice_name)
        if line_item_hit:
            counters["line_item"]["count"] += 1
            _append_unique_name(counters["line_item"]["examples"], invoice_name)

    suggestions: List[Dict[str, Any]] = []
    ocr_profile = profile.get("ocr") or {}
    field_buffers = ocr_profile.get("field_buffer_multipliers") or {}
    table_profile = profile.get("table") or {}

    if counters["date"]["count"] >= min_support:
        patch: Dict[str, Any] = {
            "ocr": {
                "region_buffer_multiplier": round(
                    max(float(ocr_profile.get("region_buffer_multiplier") or 1.0), 1.25), 2
                ),
                "field_buffer_multipliers": {
                    "date": round(max(float(field_buffers.get("date") or 1.15), 1.35), 2),
                },
            }
        }
        if field_hints["date_fields"]:
            patch["ocr"]["field_type_overrides"] = {
                field_name: "date" for field_name in field_hints["date_fields"]
            }
        merged = merge_family_extraction_profiles(profile, patch)
        if merged != profile:
            suggestions.append(
                {
                    "suggestion_id": f"{family.template_family_id}:date_ocr_boost",
                    "template_family_id": family.template_family_id,
                    "family_status": family.status.value,
                    "kind": "date_ocr_boost",
                    "title": "Widen OCR for date fields",
                    "reason": f"{counters['date']['count']} recent invoice(s) had date extraction failures",
                    "support_count": counters["date"]["count"],
                    "priority": counters["date"]["count"]
                    + (1.0 if family.status.value == "degraded" else 0.0),
                    "example_invoices": counters["date"]["examples"],
                    "profile_patch": patch,
                }
            )

    if counters["summary_amount"]["count"] >= min_support:
        patch = {
            "ocr": {
                "field_buffer_multipliers": {
                    "currency": round(max(float(field_buffers.get("currency") or 0.9), 1.05), 2),
                },
            }
        }
        if field_hints["amount_fields"]:
            patch["ocr"]["field_type_overrides"] = {
                field_name: "currency" for field_name in field_hints["amount_fields"]
            }
        merged = merge_family_extraction_profiles(profile, patch)
        if merged != profile:
            suggestions.append(
                {
                    "suggestion_id": f"{family.template_family_id}:summary_amount_ocr_boost",
                    "template_family_id": family.template_family_id,
                    "family_status": family.status.value,
                    "kind": "summary_amount_ocr_boost",
                    "title": "Relax OCR around summary amounts",
                    "reason": f"{counters['summary_amount']['count']} recent invoice(s) had total, subtotal, or tax failures",
                    "support_count": counters["summary_amount"]["count"],
                    "priority": counters["summary_amount"]["count"]
                    + 0.5
                    + (1.0 if family.status.value == "degraded" else 0.0),
                    "example_invoices": counters["summary_amount"]["examples"],
                    "profile_patch": patch,
                }
            )

    if counters["line_item"]["count"] >= min_support:
        patch = {
            "table": {
                "enabled": True,
                "row_gap_multiplier": round(
                    min(float(table_profile.get("row_gap_multiplier") or 1.5), 1.2), 2
                ),
                "min_line_span_fraction": round(
                    min(float(table_profile.get("min_line_span_fraction") or 0.40), 0.28), 2
                ),
            },
            "postprocess": {
                "prefer_table_line_items": True,
            },
        }
        merged = merge_family_extraction_profiles(profile, patch)
        if merged != profile:
            suggestions.append(
                {
                    "suggestion_id": f"{family.template_family_id}:line_item_table_tuning",
                    "template_family_id": family.template_family_id,
                    "family_status": family.status.value,
                    "kind": "line_item_table_tuning",
                    "title": "Favor table extraction for line items",
                    "reason": f"{counters['line_item']['count']} recent invoice(s) had table or line-item failures",
                    "support_count": counters["line_item"]["count"],
                    "priority": counters["line_item"]["count"]
                    + 1.0
                    + (1.0 if family.status.value == "degraded" else 0.0),
                    "example_invoices": counters["line_item"]["examples"],
                    "profile_patch": patch,
                }
            )

    suggestions.sort(key=lambda item: (-item["priority"], item["kind"]))
    return suggestions


def suggest_template_family_updates(
    db: FingerprintDB,
    *,
    template_family_id: Optional[str] = None,
    recent_limit: int = 10,
    min_support: int = 2,
) -> Dict[str, Any]:
    families = db.get_active_template_families()
    suggestions: List[Dict[str, Any]] = []
    for family in families:
        if template_family_id and family.template_family_id != template_family_id:
            continue
        representative = db.get_family_representative_fingerprint(family.template_family_id)
        recent_results = db.get_recent_results_for_family(
            family.template_family_id, limit=recent_limit
        )
        suggestions.extend(
            _family_profile_suggestions(
                family,
                representative,
                recent_results,
                min_support=min_support,
            )
        )

    suggestions.sort(key=lambda item: (-item["priority"], item["template_family_id"], item["kind"]))
    family_ids = sorted({item["template_family_id"] for item in suggestions})
    return {
        "queue_count": len(suggestions),
        "family_count": len(family_ids),
        "family_ids": family_ids,
        "suggestions": suggestions,
    }


def _fingerprint_split_signature(record: FingerprintRecord) -> Dict[str, Any]:
    anchor_tokens: set[str] = set()
    field_names: set[str] = set()
    page_roles: List[str] = []
    table_enabled = False
    for page in record.page_fingerprints:
        if page.role is not None:
            page_roles.append(page.role.value)
        signature = page.stable_anchor_signature or {}
        for key in ("header_tokens", "summary_labels", "footer_tokens"):
            anchor_tokens.update(
                _normalize_text(value) for value in (signature.get(key) or []) if value
            )
        for values in (signature.get("keyword_hits") or {}).values():
            anchor_tokens.update(_normalize_text(value) for value in values if value)
    for page in record.layout_template.get("pages", []):
        fields = page.get("fields") or {}
        field_names.update(_normalize_text(name) for name in fields.keys() if name)
        if page.get("table"):
            table_enabled = True
    return {
        "anchor_tokens": {token for token in anchor_tokens if token},
        "field_names": {name for name in field_names if name},
        "page_roles": tuple(page_roles),
        "table_enabled": table_enabled,
        "multi_page": len(record.page_fingerprints) > 1,
    }


def _fingerprint_split_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    role_score = _jaccard_similarity(set(left["page_roles"]), set(right["page_roles"]))
    table_score = 1.0 if left["table_enabled"] == right["table_enabled"] else 0.0
    multi_page_score = 1.0 if left["multi_page"] == right["multi_page"] else 0.0
    return round(
        (_jaccard_similarity(left["anchor_tokens"], right["anchor_tokens"]) * 0.40)
        + (_jaccard_similarity(left["field_names"], right["field_names"]) * 0.35)
        + (role_score * 0.15)
        + (table_score * 0.05)
        + (multi_page_score * 0.05),
        3,
    )


def _cluster_family_fingerprints(
    fingerprints: List[FingerprintRecord],
    *,
    similarity_threshold: float,
) -> List[Dict[str, Any]]:
    ordered = sorted(
        fingerprints,
        key=lambda record: (-record.confidence, -record.apply_count, record.created_at),
    )
    clusters: List[Dict[str, Any]] = []
    for record in ordered:
        signature = _fingerprint_split_signature(record)
        best_cluster: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for cluster in clusters:
            score = _fingerprint_split_similarity(signature, cluster["prototype"])
            if score > best_score:
                best_score = score
                best_cluster = cluster
        if best_cluster is None or best_score < similarity_threshold:
            clusters.append(
                {
                    "cluster_id": f"cluster-{len(clusters) + 1}",
                    "prototype": signature,
                    "members": [record],
                }
            )
            continue
        best_cluster["members"].append(record)

    return sorted(
        clusters,
        key=lambda cluster: (
            -len(cluster["members"]),
            -sum(member.apply_count for member in cluster["members"]),
            cluster["cluster_id"],
        ),
    )


def suggest_template_family_splits(
    db: FingerprintDB,
    *,
    template_family_id: Optional[str] = None,
    min_cluster_size: int = 2,
    similarity_threshold: float = 0.72,
    result_limit: int = 50,
) -> Dict[str, Any]:
    suggestions: List[Dict[str, Any]] = []
    families = db.get_active_template_families()
    for family in families:
        if template_family_id and family.template_family_id != template_family_id:
            continue
        fingerprints = db.get_fingerprints_for_family(family.template_family_id)
        if len(fingerprints) < max(min_cluster_size, 2):
            continue

        clusters = _cluster_family_fingerprints(
            fingerprints,
            similarity_threshold=similarity_threshold,
        )
        if len(clusters) < 2:
            continue

        all_results = db.get_results_for_fingerprints(
            [record.hash for record in fingerprints], limit=result_limit
        )
        results_by_hash: Dict[str, List[ProcessingResult]] = {}
        for result in all_results:
            if result.fingerprint_hash:
                results_by_hash.setdefault(result.fingerprint_hash, []).append(result)

        examples = db.get_template_family_examples(family.template_family_id)
        examples_by_hash: Dict[str, List[Any]] = {}
        for example in examples:
            if example.fingerprint_hash:
                examples_by_hash.setdefault(example.fingerprint_hash, []).append(example)

        primary_cluster = clusters[0]
        primary_signature = primary_cluster["prototype"]
        for index, cluster in enumerate(clusters[1:], start=2):
            members = cluster["members"]
            member_hashes = [member.hash for member in members]
            cluster_results = [
                result
                for fingerprint_hash in member_hashes
                for result in results_by_hash.get(fingerprint_hash, [])
            ]
            cluster_examples = [
                example
                for fingerprint_hash in member_hashes
                for example in examples_by_hash.get(fingerprint_hash, [])
            ]
            if (
                len(members) < min_cluster_size
                and len(cluster_results) < min_cluster_size
                and len(cluster_examples) < min_cluster_size
            ):
                continue

            accepted = sum(1 for result in cluster_results if result.validation_passed is True)
            review = sum(
                1
                for result in cluster_results
                if result.validation_passed is False or result.route_used == Route.REJECTED
            )
            unknown = max(len(cluster_results) - accepted - review, 0)
            example_names = sorted(
                {
                    Path(example.invoice_path).name
                    for example in cluster_examples
                    if example.invoice_path
                }
            )[:3]
            similarity = _fingerprint_split_similarity(cluster["prototype"], primary_signature)
            anchor_tokens = sorted(cluster["prototype"]["anchor_tokens"])[:6]
            field_names = sorted(cluster["prototype"]["field_names"])[:6]
            suggested_family_id = f"{family.template_family_id}-split-{index - 1}"
            reason_parts = [
                f"{len(members)} fingerprint(s) differ structurally from the primary cluster"
            ]
            if review:
                reason_parts.append(f"{review} recent review outcome(s)")
            if accepted:
                reason_parts.append(f"{accepted} accepted invoice(s)")
            suggestions.append(
                {
                    "suggestion_id": f"{family.template_family_id}:{cluster['cluster_id']}",
                    "template_family_id": family.template_family_id,
                    "cluster_id": cluster["cluster_id"],
                    "title": f"Split out {cluster['cluster_id']} from {family.template_family_id}",
                    "reason": "; ".join(reason_parts),
                    "fingerprint_hashes": member_hashes,
                    "fingerprint_count": len(members),
                    "recent_result_count": len(cluster_results),
                    "status_counts": {
                        "accepted": accepted,
                        "review": review,
                        "unknown": unknown,
                    },
                    "example_count": len(cluster_examples),
                    "example_invoices": example_names,
                    "similarity_to_primary": similarity,
                    "structural_signals": {
                        "page_roles": list(cluster["prototype"]["page_roles"]),
                        "table_enabled": cluster["prototype"]["table_enabled"],
                        "anchor_tokens": anchor_tokens,
                        "field_names": field_names,
                    },
                    "proposed_family_id": suggested_family_id,
                    "priority": round((len(members) * 1.5) + review + (1.0 - similarity), 3),
                }
            )

    suggestions.sort(
        key=lambda item: (-item["priority"], item["template_family_id"], item["cluster_id"])
    )
    family_ids = sorted({item["template_family_id"] for item in suggestions})
    return {
        "queue_count": len(suggestions),
        "family_count": len(family_ids),
        "family_ids": family_ids,
        "suggestions": suggestions,
    }


def _family_anchor_tokens(family: TemplateFamilyRecord) -> set[str]:
    tokens: set[str] = set()
    for values in (family.anchor_summary.get("aggregate_keywords") or {}).values():
        tokens.update(_normalize_text(value) for value in values if value)
    tokens.update(
        _normalize_text(value)
        for value in (family.stable_anchor_regions.get("tokens") or [])
        if value
    )
    return {token for token in tokens if token}


def _family_merge_signature(
    family: TemplateFamilyRecord,
    representative: Optional[FingerprintRecord],
) -> Dict[str, Any]:
    field_names: set[str] = set()
    if representative is not None:
        for page in representative.layout_template.get("pages", []):
            field_names.update(
                _normalize_text(name) for name in (page.get("fields") or {}).keys() if name
            )
    page_roles = tuple(
        role.value if hasattr(role, "value") else str(role)
        for role in (family.page_role_expectations or [])
    ) or tuple(str(role) for role in (family.anchor_summary.get("page_roles") or []))
    profile = summarize_family_extraction_profile(family.extraction_profile)
    return {
        "anchor_tokens": _family_anchor_tokens(family),
        "field_names": {name for name in field_names if name},
        "page_roles": page_roles,
        "table_enabled": bool(profile.get("table_enabled")),
        "preferred_strategy": profile.get("preferred_strategy"),
        "provider_name": _normalize_text(family.provider_name) if family.provider_name else None,
        "country_code": (family.country_code or "").upper() or None,
        "document_family": family.document_family.value,
    }


def _family_merge_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    return round(
        (_jaccard_similarity(left["anchor_tokens"], right["anchor_tokens"]) * 0.45)
        + (_jaccard_similarity(left["field_names"], right["field_names"]) * 0.20)
        + (_jaccard_similarity(set(left["page_roles"]), set(right["page_roles"])) * 0.15)
        + (0.05 if left["table_enabled"] == right["table_enabled"] else 0.0)
        + (
            0.05
            if left["preferred_strategy"] == right["preferred_strategy"]
            and left["preferred_strategy"]
            else 0.0
        )
        + (
            0.05
            if left["provider_name"] and left["provider_name"] == right["provider_name"]
            else 0.0
        )
        + (0.05 if left["country_code"] and left["country_code"] == right["country_code"] else 0.0),
        3,
    )


def _family_strength_score(
    family: TemplateFamilyRecord,
    representative: Optional[FingerprintRecord],
    *,
    example_count: int,
) -> float:
    status_rank = {
        "established": 3.0,
        "provisional": 2.0,
        "degraded": 1.0,
        "retired": 0.0,
    }.get(family.status.value, 0.0)
    representative_bonus = 0.0
    if representative is not None:
        representative_bonus = (float(representative.confidence) * 5.0) + min(
            representative.apply_count, 10
        ) * 0.4
    return round(
        (status_rank * 10.0)
        + (float(family.confidence) * 8.0)
        + min(family.apply_count, 20) * 0.7
        - min(family.reject_count, 20) * 0.35
        + min(example_count, 10) * 0.25
        + representative_bonus,
        3,
    )


def suggest_template_family_merges(
    db: FingerprintDB,
    *,
    template_family_id: Optional[str] = None,
    similarity_threshold: float = 0.84,
) -> Dict[str, Any]:
    families = db.get_active_template_families()
    family_index: Dict[str, Dict[str, Any]] = {}
    for family in families:
        representative = db.get_family_representative_fingerprint(family.template_family_id)
        example_count = len(db.get_template_family_examples(family.template_family_id))
        family_index[family.template_family_id] = {
            "family": family,
            "representative": representative,
            "example_count": example_count,
            "signature": _family_merge_signature(family, representative),
            "strength": _family_strength_score(family, representative, example_count=example_count),
        }

    suggestions: List[Dict[str, Any]] = []
    family_ids = sorted(family_index.keys())
    for left_index, left_id in enumerate(family_ids):
        left = family_index[left_id]
        left_family = left["family"]
        for right_id in family_ids[left_index + 1 :]:
            if template_family_id and template_family_id not in {left_id, right_id}:
                continue
            right = family_index[right_id]
            right_family = right["family"]

            if (
                left_family.document_family != right_family.document_family
                and DocumentFamily.unknown
                not in {left_family.document_family, right_family.document_family}
            ):
                continue
            if (
                left_family.provider_name
                and right_family.provider_name
                and _normalize_text(left_family.provider_name)
                != _normalize_text(right_family.provider_name)
            ):
                continue
            if (
                left_family.country_code
                and right_family.country_code
                and str(left_family.country_code).upper() != str(right_family.country_code).upper()
            ):
                continue

            similarity = _family_merge_similarity(left["signature"], right["signature"])
            if similarity < similarity_threshold:
                continue

            target = left if left["strength"] >= right["strength"] else right
            source = right if target is left else left
            shared_anchor_tokens = sorted(
                left["signature"]["anchor_tokens"] & right["signature"]["anchor_tokens"]
            )[:6]
            shared_field_names = sorted(
                left["signature"]["field_names"] & right["signature"]["field_names"]
            )[:6]
            suggestions.append(
                {
                    "suggestion_id": f"{source['family'].template_family_id}:merge_into:{target['family'].template_family_id}",
                    "source_family_id": source["family"].template_family_id,
                    "target_family_id": target["family"].template_family_id,
                    "title": f"Merge {source['family'].template_family_id} into {target['family'].template_family_id}",
                    "reason": (
                        f"Families are highly similar ({similarity:.2f}) on anchors, fields, and page roles; "
                        f"{source['family'].template_family_id} is the weaker duplicate"
                    ),
                    "similarity": similarity,
                    "source_strength": source["strength"],
                    "target_strength": target["strength"],
                    "source_status": source["family"].status.value,
                    "target_status": target["family"].status.value,
                    "source_apply_count": source["family"].apply_count,
                    "target_apply_count": target["family"].apply_count,
                    "shared_signals": {
                        "anchor_tokens": shared_anchor_tokens,
                        "field_names": shared_field_names,
                        "page_roles": sorted(
                            set(left["signature"]["page_roles"])
                            & set(right["signature"]["page_roles"])
                        ),
                    },
                    "priority": round(
                        (similarity * 10.0)
                        + max(target["strength"] - source["strength"], 0.0) * 0.1,
                        3,
                    ),
                }
            )

    suggestions.sort(
        key=lambda item: (-item["priority"], item["source_family_id"], item["target_family_id"])
    )
    queued_family_ids = sorted(
        {
            family_id
            for item in suggestions
            for family_id in (item["source_family_id"], item["target_family_id"])
        }
    )
    return {
        "queue_count": len(suggestions),
        "family_count": len(queued_family_ids),
        "family_ids": queued_family_ids,
        "suggestions": suggestions,
    }


def suggest_template_family_retirements(
    db: FingerprintDB,
    *,
    template_family_id: Optional[str] = None,
    min_reject_count: int = 3,
) -> Dict[str, Any]:
    suggestions: List[Dict[str, Any]] = []
    families = db.get_active_template_families()
    for family in families:
        if template_family_id and family.template_family_id != template_family_id:
            continue
        fingerprints = db.get_fingerprints_for_family(family.template_family_id)
        recent_results = db.get_recent_results_for_family(family.template_family_id, limit=20)
        accepted = sum(1 for result in recent_results if result.validation_passed is True)
        review = sum(
            1
            for result in recent_results
            if result.validation_passed is False or result.route_used == Route.REJECTED
        )
        kind: Optional[str] = None
        reason: Optional[str] = None
        priority = 0.0
        if not fingerprints:
            kind = "orphaned_family"
            reason = "Family has no active fingerprints left"
            priority = 5.0 + review
        elif family.apply_count == 0 and family.reject_count >= min_reject_count and accepted == 0:
            kind = "repeated_rejection"
            reason = "Family has no successful applies and repeated rejects"
            priority = 4.0 + family.reject_count
        elif (
            family.status == TemplateStatus.degraded
            and family.apply_count == 0
            and accepted == 0
            and review >= min_reject_count
        ):
            kind = "degraded_without_success"
            reason = "Family is degraded and has not shown a successful recent invoice"
            priority = 3.5 + review
        if kind is None:
            continue

        example_names = [
            Path(example.invoice_path).name
            for example in db.get_template_family_examples(family.template_family_id)[-3:]
            if example.invoice_path
        ]
        suggestions.append(
            {
                "suggestion_id": f"{family.template_family_id}:{kind}",
                "template_family_id": family.template_family_id,
                "kind": kind,
                "title": f"Retire {family.template_family_id}",
                "reason": reason,
                "active_fingerprint_count": len(fingerprints),
                "apply_count": family.apply_count,
                "reject_count": family.reject_count,
                "recent_status_counts": {
                    "accepted": accepted,
                    "review": review,
                    "unknown": max(len(recent_results) - accepted - review, 0),
                },
                "example_invoices": example_names,
                "priority": round(priority, 3),
            }
        )

    suggestions.sort(key=lambda item: (-item["priority"], item["template_family_id"]))
    family_ids = sorted({item["template_family_id"] for item in suggestions})
    return {
        "queue_count": len(suggestions),
        "family_count": len(family_ids),
        "family_ids": family_ids,
        "suggestions": suggestions,
    }


def describe_template_family(
    db: FingerprintDB,
    template_family_id: str,
    *,
    example_limit: int = 5,
    version_limit: int = 5,
    recent_limit: int = 5,
) -> Optional[Dict[str, Any]]:
    family = db.get_template_family(template_family_id)
    if family is None:
        return None

    representative = db.get_family_representative_fingerprint(template_family_id)
    examples = db.get_template_family_examples(template_family_id)
    versions = db.get_template_family_versions(template_family_id)
    recent_results = db.get_recent_results_for_family(template_family_id, limit=recent_limit)
    family_results = db.get_results_for_fingerprints(
        [record.hash for record in db.get_fingerprints_for_family(template_family_id)]
    )
    healing_rollup = _healing_rollup(family_results)

    recent_outcomes: Counter[str] = Counter()
    recent_rows: List[Dict[str, Any]] = []
    for result in recent_results:
        if result.validation_passed is True:
            outcome = "accepted"
        elif result.validation_passed is False or (
            result.route_used and result.route_used == Route.REJECTED
        ):
            outcome = "review"
        else:
            outcome = "unknown"
        recent_outcomes[outcome] += 1
        recent_rows.append(
            {
                "invoice_name": Path(result.invoice_path).name,
                "invoice_path": result.invoice_path,
                "outcome": outcome,
                "route": result.route_used.value if result.route_used else None,
                "attempted_route": result.attempted_route.value if result.attempted_route else None,
                "validation_score": result.diagnostics.validation_score
                if result.diagnostics
                else None,
                "determination_sources": list(result.provenance.determination_sources)
                if result.provenance
                else [],
                "gt_backed": bool(result.provenance.gt_backed) if result.provenance else False,
                "healing_reprocessed": bool(result.provenance.healing_reprocessed)
                if result.provenance
                else False,
                "review_signals": _family_review_signals_for_result(result),
                "processed_at": result.processed_at,
            }
        )

    review_signals: List[str] = []
    if family.status.value in {"degraded", "provisional"}:
        review_signals.append(f"status is {family.status.value}")
    if family.reject_count > family.apply_count:
        review_signals.append("more rejects than successful applies")
    if recent_outcomes.get("review"):
        review_signals.append(f"{recent_outcomes['review']} recent review outcome(s)")

    suggestions = _family_profile_suggestions(
        family,
        representative,
        recent_results,
        min_support=2,
    )
    split_suggestions = suggest_template_family_splits(
        db,
        template_family_id=template_family_id,
    )["suggestions"]
    merge_suggestions = suggest_template_family_merges(
        db,
        template_family_id=template_family_id,
    )["suggestions"]
    retirement_suggestions = suggest_template_family_retirements(
        db,
        template_family_id=template_family_id,
    )["suggestions"]
    profile_summary = summarize_family_extraction_profile(family.extraction_profile)
    triage = _classify_template_family_triage(
        status=family.status.value,
        apply_count=int(family.apply_count),
        reject_count=int(family.reject_count),
        field_override_count=int(profile_summary.get("field_override_count") or 0),
    )

    return {
        "template_family_id": family.template_family_id,
        "status": family.status.value,
        "confidence": round(float(family.confidence), 4),
        "apply_count": int(family.apply_count),
        "reject_count": int(family.reject_count),
        "gt_apply_count": int(family.gt_apply_count),
        "gt_reject_count": int(family.gt_reject_count),
        "gt_confidence": round(float(family.gt_confidence), 4),
        "gt_trust_qualified": bool(
            family.gt_apply_count >= _GT_TRUST_CONFIG.establish_min_count
            and float(family.gt_confidence) >= _GT_TRUST_CONFIG.establish_min_confidence
        ),
        "provider_name": family.provider_name,
        "country_code": family.country_code,
        "document_family": family.document_family.value,
        "created_at": family.created_at,
        "updated_at": family.updated_at,
        "extraction_profile": family.extraction_profile,
        "extraction_profile_summary": profile_summary,
        **triage,
        "representative": {
            "fingerprint_hash": representative.hash if representative else None,
            "confidence": round(float(representative.confidence), 4) if representative else None,
            "apply_count": int(representative.apply_count) if representative else None,
            "reject_count": int(representative.reject_count) if representative else None,
            "gt_apply_count": int(representative.gt_apply_count) if representative else None,
            "gt_reject_count": int(representative.gt_reject_count) if representative else None,
            "gt_confidence": round(float(representative.gt_confidence), 4)
            if representative
            else None,
            "gt_trust_qualified": bool(
                representative is not None
                and representative.gt_apply_count >= _GT_TRUST_CONFIG.establish_min_count
                and float(representative.gt_confidence) >= _GT_TRUST_CONFIG.establish_min_confidence
            ),
            "status": representative.status.value if representative else None,
        },
        **healing_rollup,
        "example_count": len(examples),
        "examples": [
            {
                "invoice_path": example.invoice_path,
                "fingerprint_hash": example.fingerprint_hash,
                "created_at": example.created_at,
                "example_metadata": example.example_metadata,
            }
            for example in examples[-example_limit:]
        ],
        "version_count": len(versions),
        "versions": [
            {
                "version": version.version,
                "change_reason": version.change_reason,
                "created_at": version.created_at,
            }
            for version in versions[-version_limit:]
        ],
        "recent_outcome_counts": dict(recent_outcomes),
        "recent_results": recent_rows,
        "review_signals": review_signals,
        "suggestions": suggestions,
        "split_suggestions": split_suggestions,
        "merge_suggestions": merge_suggestions,
        "retirement_suggestions": retirement_suggestions,
    }


def _family_benchmark_status(
    result: Optional[ProcessingResult], error: Optional[Any] = None
) -> str:
    if error is not None:
        return "error"
    if result is None:
        return "missing"
    if result.validation_passed is True:
        return "accepted"
    if result.validation_passed is False or (
        result.route_used and result.route_used == Route.REJECTED
    ):
        return "review"
    return "unknown"


def _family_benchmark_rank(status: str) -> int:
    return {
        "error": -1,
        "missing": 0,
        "unknown": 1,
        "review": 2,
        "accepted": 3,
    }.get(status, 0)


def summarize_family_benchmark_comparison(
    template_family_id: str,
    invoice_paths: List[str],
    baseline_results: Dict[str, Optional[ProcessingResult]],
    candidate_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate_by_path = {entry["invoice_path"]: entry for entry in candidate_entries}
    baseline_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    baseline_scores: List[float] = []
    candidate_scores: List[float] = []
    improvements = 0
    regressions = 0
    unchanged = 0
    rows: List[Dict[str, Any]] = []

    for invoice_path in invoice_paths:
        baseline_result = baseline_results.get(invoice_path)
        candidate_entry = candidate_by_path.get(invoice_path, {"result": None, "error": "missing"})
        candidate_result = candidate_entry.get("result")
        candidate_error = candidate_entry.get("error")

        baseline_status = _family_benchmark_status(baseline_result)
        candidate_status = _family_benchmark_status(candidate_result, candidate_error)
        baseline_counts[baseline_status] += 1
        candidate_counts[candidate_status] += 1

        baseline_score = (
            baseline_result.diagnostics.validation_score
            if baseline_result and baseline_result.diagnostics
            else None
        )
        candidate_score = (
            candidate_result.diagnostics.validation_score
            if candidate_result and candidate_result.diagnostics
            else None
        )
        if baseline_score is not None:
            baseline_scores.append(float(baseline_score))
        if candidate_score is not None:
            candidate_scores.append(float(candidate_score))

        baseline_rank = _family_benchmark_rank(baseline_status)
        candidate_rank = _family_benchmark_rank(candidate_status)
        if candidate_rank > baseline_rank:
            improvements += 1
        elif candidate_rank < baseline_rank:
            regressions += 1
        else:
            unchanged += 1

        rows.append(
            {
                "invoice_name": Path(invoice_path).name,
                "invoice_path": invoice_path,
                "baseline_status": baseline_status,
                "baseline_score": round(float(baseline_score), 4)
                if baseline_score is not None
                else None,
                "candidate_status": candidate_status,
                "candidate_score": round(float(candidate_score), 4)
                if candidate_score is not None
                else None,
                "candidate_error": str(candidate_error) if candidate_error is not None else None,
            }
        )

    rows.sort(key=lambda item: (item["invoice_name"], item["invoice_path"]))

    baseline_avg_score = (sum(baseline_scores) / len(baseline_scores)) if baseline_scores else None
    candidate_avg_score = (
        (sum(candidate_scores) / len(candidate_scores)) if candidate_scores else None
    )

    return {
        "template_family_id": template_family_id,
        "invoice_count": len(invoice_paths),
        "baseline": {
            "status_counts": dict(baseline_counts),
            "avg_validation_score": round(float(baseline_avg_score), 4)
            if baseline_avg_score is not None
            else None,
        },
        "candidate": {
            "status_counts": dict(candidate_counts),
            "avg_validation_score": round(float(candidate_avg_score), 4)
            if candidate_avg_score is not None
            else None,
        },
        "progress": {
            "improvements": improvements,
            "regressions": regressions,
            "unchanged": unchanged,
            "accepted_delta": candidate_counts.get("accepted", 0)
            - baseline_counts.get("accepted", 0),
            "review_delta": candidate_counts.get("review", 0) - baseline_counts.get("review", 0),
            "error_delta": candidate_counts.get("error", 0) - baseline_counts.get("error", 0),
            "avg_validation_score_delta": (
                round(float(candidate_avg_score - baseline_avg_score), 4)
                if candidate_avg_score is not None and baseline_avg_score is not None
                else None
            ),
        },
        "invoices": rows,
    }


def _init_analysis_db_postgres(client: PostgresClient):
    statements = [
        """
        CREATE TABLE IF NOT EXISTS runs (
            id BIGSERIAL PRIMARY KEY,
            run_at TEXT NOT NULL,
            dataset TEXT NOT NULL,
            input_dir TEXT NOT NULL,
            command TEXT,
            git_branch TEXT,
            git_commit TEXT,
            discovery_mode TEXT,
            discovery_only INTEGER NOT NULL DEFAULT 0,
            total_invoices INTEGER NOT NULL,
            discovery_passed INTEGER NOT NULL,
            discovery_rejected INTEGER NOT NULL,
            apply_rejected INTEGER NOT NULL,
            runtime_errors INTEGER NOT NULL,
            total_time_ms INTEGER NOT NULL,
            failure_summary_json TEXT,
            summary_json TEXT NOT NULL,
            config_json TEXT,
            notes TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS run_invoices (
            id BIGSERIAL PRIMARY KEY,
            run_id BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            invoice_name TEXT NOT NULL,
            invoice_path TEXT,
            template_family_id TEXT,
            route TEXT,
            attempted_route TEXT,
            status TEXT NOT NULL,
            elapsed_ms INTEGER NOT NULL DEFAULT 0,
            validation_score DOUBLE PRECISION,
            validation_passed INTEGER,
            error_categories_json TEXT,
            validation_errors_json TEXT,
            extracted_data_json TEXT,
            ground_truth_json TEXT,
            normalized_data_json TEXT,
            diagnostics_json TEXT,
            UNIQUE(run_id, invoice_name)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_runs_dataset_run_at ON runs(dataset, id DESC)",
        "CREATE INDEX IF NOT EXISTS idx_run_invoices_run_id ON run_invoices(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_run_invoices_status ON run_invoices(status)",
        "CREATE INDEX IF NOT EXISTS idx_run_invoices_name ON run_invoices(invoice_name)",
        "ALTER TABLE run_invoices ADD COLUMN IF NOT EXISTS template_family_id TEXT",
    ]
    for statement in statements:
        client.execute(statement)


def record_analysis_run(
    summary: Dict[str, Any],
    results: List[Dict[str, Any]],
    output_dir: str,
    *,
    analysis_db_target: Optional[str] = None,
    command: Optional[str] = None,
    git_branch: Optional[str] = None,
    git_commit: Optional[str] = None,
    config_snapshot: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
) -> str:
    if not analysis_db_target:
        raise ValueError(
            "record_analysis_run requires an ANALYSIS_DATABASE_URL or explicit analysis_db_target"
        )
    if not is_postgres_target(analysis_db_target):
        raise ValueError(
            f"Analysis history requires a Postgres ANALYSIS_DATABASE_URL, got: {analysis_db_target}"
        )

    client = PostgresClient(analysis_db_target)
    _init_analysis_db_postgres(client)
    processed_timestamps = [
        result["result"].processed_at
        for result in results
        if result.get("result") is not None and result["result"].processed_at
    ]
    run_at = (
        max(processed_timestamps)
        if processed_timestamps
        else datetime.now(timezone.utc).isoformat()
    )
    run_row = client.fetchone(
        """
        INSERT INTO runs (
            run_at, dataset, input_dir, command, git_branch, git_commit,
            discovery_mode, discovery_only, total_invoices,
            discovery_passed, discovery_rejected, apply_rejected,
            runtime_errors, total_time_ms, failure_summary_json,
            summary_json, config_json, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        RETURNING id
        """,
        (
            run_at,
            summary["dataset"],
            summary["input_dir"],
            command,
            git_branch,
            git_commit,
            summary.get("discovery_mode"),
            1 if summary.get("discovery_only") else 0,
            summary["total_invoices"],
            summary["discovery_passed"],
            summary["discovery_rejected"],
            summary["apply_rejected"],
            summary["runtime_errors"],
            summary["total_time_ms"],
            json.dumps(summary.get("failure_analysis") or {}),
            json.dumps(summary),
            json.dumps(config_snapshot) if config_snapshot is not None else None,
            notes,
        ),
    )
    run_id = run_row["id"]
    for item in results:
        result = item.get("result")
        error = item.get("error")
        diagnostics = result.diagnostics.model_dump() if result and result.diagnostics else None
        validation_errors = (
            result.diagnostics.validation_errors if result and result.diagnostics else []
        )
        invoice_path = result.invoice_path if result else None
        validation_passed = None
        if result and result.validation_passed is not None:
            validation_passed = 1 if result.validation_passed else 0
        attempted_route = None
        if result and result.attempted_route:
            attempted_route = result.attempted_route.value
        elif error is not None:
            attempted = getattr(error, "attempted_route", None)
            attempted_route = attempted.value if attempted else None
        categories = _error_categories(result) if result else ["runtime_error"]
        client.execute(
            """
            INSERT INTO run_invoices (
                run_id, invoice_name, invoice_path, template_family_id, route, attempted_route, status,
                elapsed_ms, validation_score, validation_passed,
                error_categories_json, validation_errors_json,
                extracted_data_json, ground_truth_json, normalized_data_json, diagnostics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                item["name"],
                invoice_path,
                result.template_family_id if result else None,
                item["route"],
                attempted_route,
                item["status"],
                item["elapsed_ms"],
                diagnostics.get("validation_score") if diagnostics else None,
                validation_passed,
                json.dumps(categories),
                json.dumps(validation_errors if result else [str(error)]),
                json.dumps(result.extracted_data)
                if result and result.extracted_data is not None
                else None,
                json.dumps(result.ground_truth)
                if result and result.ground_truth is not None
                else None,
                json.dumps(result.normalized_data)
                if result and result.normalized_data is not None
                else None,
                json.dumps(diagnostics) if diagnostics is not None else None,
            ),
        )
    return analysis_db_target


def _select_run_row_postgres(
    client: PostgresClient,
    dataset: str,
    run_id: Optional[int] = None,
    *,
    offset: int = 0,
) -> Optional[Dict[str, Any]]:
    if run_id is not None:
        return client.fetchone(
            "SELECT * FROM runs WHERE dataset = ? AND id = ?",
            (dataset, run_id),
        )
    rows = client.fetchall(
        "SELECT * FROM runs WHERE dataset = ? ORDER BY id DESC LIMIT 1 OFFSET ?",
        (dataset, offset),
    )
    return rows[0] if rows else None


def compare_analysis_runs(
    db_path: str,
    dataset: str,
    *,
    baseline_run_id: Optional[int] = None,
    candidate_run_id: Optional[int] = None,
) -> Dict[str, Any]:
    if not is_postgres_target(db_path):
        raise ValueError(
            f"compare_analysis_runs requires a Postgres ANALYSIS_DATABASE_URL, got: {db_path}"
        )

    client = PostgresClient(db_path)
    candidate = _select_run_row_postgres(client, dataset, candidate_run_id, offset=0)
    if candidate is None:
        raise ValueError(f"No runs found for dataset '{dataset}'")

    baseline_offset = 1 if baseline_run_id is None and candidate_run_id is None else 0
    baseline = _select_run_row_postgres(client, dataset, baseline_run_id, offset=baseline_offset)
    if baseline is None:
        raise ValueError(f"No baseline run found for dataset '{dataset}'")
    if baseline["id"] == candidate["id"]:
        raise ValueError("Baseline run and candidate run must be different")

    def _load_invoice_rows(run_id: int) -> Dict[str, Dict[str, Any]]:
        rows = client.fetchall(
            """
            SELECT invoice_name, status, validation_score, validation_passed, validation_errors_json
            FROM run_invoices
            WHERE run_id = ?
            """,
            (run_id,),
        )
        return {row["invoice_name"]: row for row in rows}

    baseline_rows = _load_invoice_rows(int(baseline["id"]))
    candidate_rows = _load_invoice_rows(int(candidate["id"]))
    invoice_names = sorted(set(baseline_rows) | set(candidate_rows))

    regressions: List[Dict[str, Any]] = []
    improvements: List[Dict[str, Any]] = []
    unchanged = 0

    for invoice_name in invoice_names:
        before = baseline_rows.get(invoice_name)
        after = candidate_rows.get(invoice_name)
        if before is None or after is None:
            continue

        before_passed = before["validation_passed"]
        after_passed = after["validation_passed"]
        before_score = before["validation_score"] if before["validation_score"] is not None else 0.0
        after_score = after["validation_score"] if after["validation_score"] is not None else 0.0

        delta = {
            "invoice_name": invoice_name,
            "baseline_status": before["status"],
            "candidate_status": after["status"],
            "baseline_score": before["validation_score"],
            "candidate_score": after["validation_score"],
            "baseline_errors": json.loads(before["validation_errors_json"] or "[]"),
            "candidate_errors": json.loads(after["validation_errors_json"] or "[]"),
        }

        if before_passed == 1 and after_passed != 1:
            regressions.append(delta)
        elif before_passed != 1 and after_passed == 1:
            improvements.append(delta)
        elif before_passed != 1 and after_passed != 1 and after_score > before_score + 1e-9:
            improvements.append(delta)
        elif before_passed != 1 and after_passed != 1 and after_score + 1e-9 < before_score:
            regressions.append(delta)
        else:
            unchanged += 1

    return {
        "dataset": dataset,
        "baseline_run": {
            "id": baseline["id"],
            "run_at": baseline["run_at"],
            "discovery_passed": baseline["discovery_passed"],
            "discovery_rejected": baseline["discovery_rejected"],
            "runtime_errors": baseline["runtime_errors"],
        },
        "candidate_run": {
            "id": candidate["id"],
            "run_at": candidate["run_at"],
            "discovery_passed": candidate["discovery_passed"],
            "discovery_rejected": candidate["discovery_rejected"],
            "runtime_errors": candidate["runtime_errors"],
        },
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
        "issue_progress": compare_benchmark_summaries(
            json.loads(baseline["summary_json"] or "{}"),
            json.loads(candidate["summary_json"] or "{}"),
        ),
    }
