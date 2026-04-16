import json
from pathlib import Path

from click.testing import CliRunner

from invoice_router.cli.main import cli
from invoice_router.models import ProcessingDiagnostics, ProcessingResult, Provenance, Route
from invoice_router.pipeline import ProcessingStageError
from invoice_router.reporting import (
    build_issue_ledger,
    build_process_run_summary,
    compare_benchmark_summaries,
    summarize_family_benchmark_comparison,
)


def _sample_provenance(route: Route):
    return Provenance(
        request_id=f"req-{route.value.lower()}",
        route=route,
        fingerprint_hash="fp-1",
        template_status_at_time=None,
        template_confidence_at_time=None,
        ocr_engine="tesseract",
        inference_method="test",
        image_quality_score=[0.9],
        quality_flag=False,
        input_document_hash="doc-hash",
        per_page_hashes=["page-hash"],
        extraction_output_hash="out-hash",
        started_at="2026-04-13T10:00:00Z",
        completed_at="2026-04-13T10:00:01Z",
        latency_ms=1000,
    )


def _benchmark_summary(per_invoice, *, dataset="invoices-small"):
    normalized_invoices = []
    for index, item in enumerate(per_invoice):
        enriched = dict(item)
        enriched.setdefault("elapsed_ms", 1000 + (index * 10))
        enriched.setdefault("peak_rss_mb", round(128.0 + index, 3))
        normalized_invoices.append(enriched)

    total_invoices = len(normalized_invoices)
    discovery_passed = sum(
        1 for item in normalized_invoices if item.get("validation_passed") is True
    )
    discovery_rejected = sum(
        1 for item in normalized_invoices if item.get("validation_passed") is False
    )
    runtime_errors = sum(1 for item in normalized_invoices if item.get("route") == "ERROR")
    elapsed_values = [int(item["elapsed_ms"]) for item in normalized_invoices]
    peak_rss_values = [float(item["peak_rss_mb"]) for item in normalized_invoices]
    return {
        "dataset": dataset,
        "input_dir": str(Path("/tmp") / dataset),
        "total_invoices": total_invoices,
        "discovery_passed": discovery_passed,
        "discovery_rejected": discovery_rejected,
        "apply_rejected": 0,
        "runtime_errors": runtime_errors,
        "avg_time_ms": sum(elapsed_values) // len(elapsed_values) if elapsed_values else 0,
        "latency_stats_ms": {
            "min": min(elapsed_values),
            "avg": sum(elapsed_values) // len(elapsed_values),
            "max": max(elapsed_values),
        }
        if elapsed_values
        else None,
        "peak_rss_stats_mb": {
            "min": min(peak_rss_values),
            "avg": round(sum(peak_rss_values) / len(peak_rss_values), 3),
            "max": max(peak_rss_values),
        }
        if peak_rss_values
        else None,
        "per_invoice": normalized_invoices,
        "failure_analysis": {},
        "top_failure_categories": {},
        "ground_truth_sync": {
            "source_of_truth_dir": str(Path("/tmp") / "invoices-all"),
            "status": "in_sync",
            "checked_invoice_count": total_invoices,
            "matching_count": total_invoices,
            "mismatched_count": 0,
            "missing_local_count": 0,
            "missing_source_count": 0,
            "uses_external_source_of_truth": dataset in {"invoices-small", "invoices-test"},
        },
    }


def test_build_issue_ledger_groups_failures_into_work_buckets():
    summary = _benchmark_summary(
        [
            {
                "invoice_name": "party-1.png",
                "invoice_path": "/tmp/invoices-small/party-1.png",
                "route": "REJECTED",
                "attempted_route": "DISCOVERY",
                "status": "Failed/Rejected",
                "validation_passed": False,
                "validation_score": 0.95,
                "validation_errors": [
                    "Missing extracted field: Invoice No.",
                    "Mismatch on customer name: expected 'ACME', got 'ACMEE'",
                ],
                "party_block_diagnostics": [{"field": "invoice number"}],
                "summary_candidate_diagnostics": {},
                "table_detected": False,
                "line_item_source": "row_fallback",
            },
            {
                "invoice_name": "summary-1.png",
                "invoice_path": "/tmp/invoices-small/summary-1.png",
                "route": "REJECTED",
                "attempted_route": "DISCOVERY",
                "status": "Failed/Rejected",
                "validation_passed": False,
                "validation_score": 0.95,
                "validation_errors": [
                    "Total mismatch: expected '12.00', got '10.00'",
                    "Subtotal mismatch: expected '10.00', got '9.00'",
                ],
                "party_block_diagnostics": [],
                "summary_candidate_diagnostics": {"total": [{"value": "10.00"}]},
                "table_detected": True,
                "line_item_source": "header_table",
            },
            {
                "invoice_name": "pass.png",
                "invoice_path": "/tmp/invoices-small/pass.png",
                "route": "DISCOVERY",
                "attempted_route": "DISCOVERY",
                "status": "Passed",
                "validation_passed": True,
                "validation_score": 1.0,
                "validation_errors": [],
                "party_block_diagnostics": [],
                "summary_candidate_diagnostics": {},
                "table_detected": True,
                "line_item_source": "header_table",
            },
        ]
    )

    ledger = build_issue_ledger(summary)

    assert ledger["issue_count"] == 4
    assert ledger["weighted_issue_count"] == 4.5
    assert [bucket["bucket"] for bucket in ledger["work_buckets"]] == [
        "party_block_extraction",
        "summary_amount_resolution",
    ]
    assert ledger["work_buckets"][0]["issue_count"] == 2
    assert ledger["work_buckets"][1]["issue_count"] == 2
    assert ledger["work_buckets"][0]["categories"] == {"field_mismatch": 1, "missing_field": 1}
    assert ledger["work_buckets"][1]["categories"] == {"subtotal_mismatch": 1, "total_mismatch": 1}


def test_compare_benchmark_summaries_reports_weighted_progress_toward_target():
    baseline = _benchmark_summary(
        [
            {
                "invoice_name": "party-1.png",
                "invoice_path": "/tmp/invoices-small/party-1.png",
                "route": "REJECTED",
                "attempted_route": "DISCOVERY",
                "status": "Failed/Rejected",
                "validation_passed": False,
                "validation_score": 0.95,
                "validation_errors": ["Missing extracted field: Invoice No."],
                "party_block_diagnostics": [{"field": "invoice number"}],
                "summary_candidate_diagnostics": {},
                "table_detected": False,
                "line_item_source": "row_fallback",
            },
            {
                "invoice_name": "party-2.png",
                "invoice_path": "/tmp/invoices-small/party-2.png",
                "route": "REJECTED",
                "attempted_route": "DISCOVERY",
                "status": "Failed/Rejected",
                "validation_passed": False,
                "validation_score": 0.95,
                "validation_errors": ["Mismatch on customer name: expected 'ACME', got 'ACMEE'"],
                "party_block_diagnostics": [{"field": "customer name"}],
                "summary_candidate_diagnostics": {},
                "table_detected": False,
                "line_item_source": "row_fallback",
            },
            {
                "invoice_name": "summary-1.png",
                "invoice_path": "/tmp/invoices-small/summary-1.png",
                "route": "REJECTED",
                "attempted_route": "DISCOVERY",
                "status": "Failed/Rejected",
                "validation_passed": False,
                "validation_score": 0.95,
                "validation_errors": ["Total mismatch: expected '12.00', got '10.00'"],
                "party_block_diagnostics": [],
                "summary_candidate_diagnostics": {"total": [{"value": "10.00"}]},
                "table_detected": True,
                "line_item_source": "header_table",
            },
            {
                "invoice_name": "summary-2.png",
                "invoice_path": "/tmp/invoices-small/summary-2.png",
                "route": "REJECTED",
                "attempted_route": "DISCOVERY",
                "status": "Failed/Rejected",
                "validation_passed": False,
                "validation_score": 0.95,
                "validation_errors": ["Subtotal mismatch: expected '10.00', got '9.00'"],
                "party_block_diagnostics": [],
                "summary_candidate_diagnostics": {"subtotal": [{"value": "9.00"}]},
                "table_detected": True,
                "line_item_source": "header_table",
            },
            {
                "invoice_name": "pass.png",
                "invoice_path": "/tmp/invoices-small/pass.png",
                "route": "DISCOVERY",
                "attempted_route": "DISCOVERY",
                "status": "Passed",
                "validation_passed": True,
                "validation_score": 1.0,
                "validation_errors": [],
                "party_block_diagnostics": [],
                "summary_candidate_diagnostics": {},
                "table_detected": True,
                "line_item_source": "header_table",
            },
        ]
    )
    candidate = _benchmark_summary(
        [
            {
                "invoice_name": "summary-1.png",
                "invoice_path": "/tmp/invoices-small/summary-1.png",
                "route": "REJECTED",
                "attempted_route": "DISCOVERY",
                "status": "Failed/Rejected",
                "validation_passed": False,
                "validation_score": 0.95,
                "validation_errors": ["Total mismatch: expected '12.00', got '10.00'"],
                "party_block_diagnostics": [],
                "summary_candidate_diagnostics": {"total": [{"value": "10.00"}]},
                "table_detected": True,
                "line_item_source": "header_table",
            },
            {
                "invoice_name": "party-1.png",
                "invoice_path": "/tmp/invoices-small/party-1.png",
                "route": "DISCOVERY",
                "attempted_route": "DISCOVERY",
                "status": "Passed",
                "validation_passed": True,
                "validation_score": 1.0,
                "validation_errors": [],
                "party_block_diagnostics": [],
                "summary_candidate_diagnostics": {},
                "table_detected": True,
                "line_item_source": "header_table",
            },
            {
                "invoice_name": "party-2.png",
                "invoice_path": "/tmp/invoices-small/party-2.png",
                "route": "DISCOVERY",
                "attempted_route": "DISCOVERY",
                "status": "Passed",
                "validation_passed": True,
                "validation_score": 1.0,
                "validation_errors": [],
                "party_block_diagnostics": [],
                "summary_candidate_diagnostics": {},
                "table_detected": True,
                "line_item_source": "header_table",
            },
            {
                "invoice_name": "summary-2.png",
                "invoice_path": "/tmp/invoices-small/summary-2.png",
                "route": "DISCOVERY",
                "attempted_route": "DISCOVERY",
                "status": "Passed",
                "validation_passed": True,
                "validation_score": 1.0,
                "validation_errors": [],
                "party_block_diagnostics": [],
                "summary_candidate_diagnostics": {},
                "table_detected": True,
                "line_item_source": "header_table",
            },
            {
                "invoice_name": "pass.png",
                "invoice_path": "/tmp/invoices-small/pass.png",
                "route": "DISCOVERY",
                "attempted_route": "DISCOVERY",
                "status": "Passed",
                "validation_passed": True,
                "validation_score": 1.0,
                "validation_errors": [],
                "party_block_diagnostics": [],
                "summary_candidate_diagnostics": {},
                "table_detected": True,
                "line_item_source": "header_table",
            },
        ]
    )

    comparison = compare_benchmark_summaries(baseline, candidate)

    assert comparison["provenance"]["comparison_is_trusted"] is True
    assert comparison["provenance"]["warnings"] == []
    assert comparison["latency_comparison"]["baseline_avg_ms"] == 1020
    assert comparison["latency_comparison"]["candidate_avg_ms"] == 1020
    assert comparison["latency_comparison"]["delta_avg_ms"] == 0
    assert comparison["memory_comparison"]["baseline_avg_peak_rss_mb"] == 130.0
    assert comparison["memory_comparison"]["candidate_avg_peak_rss_mb"] == 130.0
    assert comparison["memory_comparison"]["delta_avg_peak_rss_mb"] == 0.0
    assert comparison["progress"]["baseline_weighted_issue_count"] == 4.5
    assert comparison["progress"]["candidate_weighted_issue_count"] == 1.05
    assert comparison["progress"]["weighted_reduction"] == 3.45
    assert comparison["progress"]["progress_to_target_75"] == 1.0222
    assert comparison["progress"]["meets_target_75"] is True
    assert [bucket["bucket"] for bucket in comparison["bucket_comparison"]] == [
        "party_block_extraction",
        "summary_amount_resolution",
    ]
    assert comparison["bucket_comparison"][0]["delta_issue_count"] == -2
    assert comparison["bucket_comparison"][1]["delta_issue_count"] == -1


def test_compare_benchmark_summaries_flags_gt_provenance_mismatch():
    baseline = _benchmark_summary([], dataset="invoices-small")
    candidate = _benchmark_summary([], dataset="invoices-small")
    candidate["ground_truth_sync"] = {
        **candidate["ground_truth_sync"],
        "status": "out_of_sync",
        "mismatched_count": 2,
    }

    comparison = compare_benchmark_summaries(baseline, candidate)

    assert comparison["provenance"]["comparison_is_trusted"] is False
    assert "Candidate GT sync is `out_of_sync`." in comparison["provenance"]["warnings"]


def test_compare_benchmark_summaries_reports_null_memory_when_missing():
    baseline = _benchmark_summary([], dataset="invoices-small")
    candidate = _benchmark_summary([], dataset="invoices-small")
    baseline["peak_rss_stats_mb"] = None

    comparison = compare_benchmark_summaries(baseline, candidate)

    assert comparison["memory_comparison"]["baseline_avg_peak_rss_mb"] is None
    assert comparison["memory_comparison"]["candidate_avg_peak_rss_mb"] is None
    assert any(
        "Peak RSS comparison is unavailable" in warning
        for warning in comparison["provenance"]["warnings"]
    )


def test_compare_benchmark_summaries_cli_writes_json_report(tmp_path):
    baseline = _benchmark_summary(
        [
            {
                "invoice_name": "party-1.png",
                "invoice_path": "/tmp/invoices-small/party-1.png",
                "route": "REJECTED",
                "attempted_route": "DISCOVERY",
                "status": "Failed/Rejected",
                "validation_passed": False,
                "validation_score": 0.95,
                "validation_errors": ["Missing extracted field: Invoice No."],
                "party_block_diagnostics": [{"field": "invoice number"}],
                "summary_candidate_diagnostics": {},
                "table_detected": False,
                "line_item_source": "row_fallback",
            },
            {
                "invoice_name": "party-2.png",
                "invoice_path": "/tmp/invoices-small/party-2.png",
                "route": "REJECTED",
                "attempted_route": "DISCOVERY",
                "status": "Failed/Rejected",
                "validation_passed": False,
                "validation_score": 0.95,
                "validation_errors": ["Mismatch on customer name: expected 'ACME', got 'ACMEE'"],
                "party_block_diagnostics": [{"field": "customer name"}],
                "summary_candidate_diagnostics": {},
                "table_detected": False,
                "line_item_source": "row_fallback",
            },
            {
                "invoice_name": "summary-1.png",
                "invoice_path": "/tmp/invoices-small/summary-1.png",
                "route": "REJECTED",
                "attempted_route": "DISCOVERY",
                "status": "Failed/Rejected",
                "validation_passed": False,
                "validation_score": 0.95,
                "validation_errors": ["Total mismatch: expected '12.00', got '10.00'"],
                "party_block_diagnostics": [],
                "summary_candidate_diagnostics": {"total": [{"value": "10.00"}]},
                "table_detected": True,
                "line_item_source": "header_table",
            },
            {
                "invoice_name": "summary-2.png",
                "invoice_path": "/tmp/invoices-small/summary-2.png",
                "route": "REJECTED",
                "attempted_route": "DISCOVERY",
                "status": "Failed/Rejected",
                "validation_passed": False,
                "validation_score": 0.95,
                "validation_errors": ["Subtotal mismatch: expected '10.00', got '9.00'"],
                "party_block_diagnostics": [],
                "summary_candidate_diagnostics": {"subtotal": [{"value": "9.00"}]},
                "table_detected": True,
                "line_item_source": "header_table",
            },
        ]
    )
    candidate = _benchmark_summary(
        [
            {
                "invoice_name": "party-1.png",
                "invoice_path": "/tmp/invoices-small/party-1.png",
                "route": "DISCOVERY",
                "attempted_route": "DISCOVERY",
                "status": "Passed",
                "validation_passed": True,
                "validation_score": 1.0,
                "validation_errors": [],
                "party_block_diagnostics": [],
                "summary_candidate_diagnostics": {},
                "table_detected": True,
                "line_item_source": "header_table",
            },
            {
                "invoice_name": "party-2.png",
                "invoice_path": "/tmp/invoices-small/party-2.png",
                "route": "DISCOVERY",
                "attempted_route": "DISCOVERY",
                "status": "Passed",
                "validation_passed": True,
                "validation_score": 1.0,
                "validation_errors": [],
                "party_block_diagnostics": [],
                "summary_candidate_diagnostics": {},
                "table_detected": True,
                "line_item_source": "header_table",
            },
            {
                "invoice_name": "summary-1.png",
                "invoice_path": "/tmp/invoices-small/summary-1.png",
                "route": "REJECTED",
                "attempted_route": "DISCOVERY",
                "status": "Failed/Rejected",
                "validation_passed": False,
                "validation_score": 0.95,
                "validation_errors": ["Total mismatch: expected '12.00', got '10.00'"],
                "party_block_diagnostics": [],
                "summary_candidate_diagnostics": {"total": [{"value": "10.00"}]},
                "table_detected": True,
                "line_item_source": "header_table",
            },
            {
                "invoice_name": "summary-2.png",
                "invoice_path": "/tmp/invoices-small/summary-2.png",
                "route": "DISCOVERY",
                "attempted_route": "DISCOVERY",
                "status": "Passed",
                "validation_passed": True,
                "validation_score": 1.0,
                "validation_errors": [],
                "party_block_diagnostics": [],
                "summary_candidate_diagnostics": {},
                "table_detected": True,
                "line_item_source": "header_table",
            },
        ]
    )
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    output_path = tmp_path / "comparison.json"
    baseline_path.write_text(json.dumps(baseline))
    candidate_path.write_text(json.dumps(candidate))

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "compare-benchmark-summaries",
            str(baseline_path),
            str(candidate_path),
            "--output-json",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["progress"]["meets_target_75"] is True
    assert "latency_comparison" in payload
    assert "memory_comparison" in payload
    assert payload["baseline_summary_path"] == str(baseline_path)
    assert payload["candidate_summary_path"] == str(candidate_path)
    assert output_path.exists()
    assert json.loads(output_path.read_text())["progress"]["weighted_reduction"] == 3.45


def test_build_process_run_summary_counts_rejects_and_runtime_errors(tmp_path):
    discovery_pass = ProcessingResult(
        invoice_path=str(tmp_path / "invoices-small" / "disc-pass.png"),
        fingerprint_hash="fp-disc-pass",
        template_family_id="family-1",
        extracted_data={"Invoice No.": "123"},
        ground_truth={"Invoice No.": "123"},
        provenance=_sample_provenance(Route.DISCOVERY),
        validation_passed=True,
        route_used=Route.DISCOVERY,
        attempted_route=Route.DISCOVERY,
        diagnostics=ProcessingDiagnostics(
            attempted_route=Route.DISCOVERY, validation_score=1.0, family_match_outcome="matched"
        ),
        image_quality_score=0.9,
        template_status_at_time=None,
        processed_at="2026-04-13T10:00:04Z",
    )
    discovery_reject = ProcessingResult(
        invoice_path=str(tmp_path / "invoices-small" / "disc-reject.png"),
        fingerprint_hash="fp-disc-reject",
        template_family_id="family-1",
        extracted_data={},
        ground_truth={"Invoice No.": "123"},
        provenance=_sample_provenance(Route.REJECTED),
        validation_passed=False,
        route_used=Route.REJECTED,
        attempted_route=Route.DISCOVERY,
        diagnostics=ProcessingDiagnostics(
            attempted_route=Route.DISCOVERY,
            discovery_mode="heuristic",
            discovery_stage_status="validation_failed_after_extract",
            validation_score=0.4,
            validation_errors=["Missing extracted field: Invoice No."],
            scalar_field_missing_count=1,
            scalar_field_mismatch_count=0,
        ),
        image_quality_score=0.8,
        template_status_at_time=None,
        processed_at="2026-04-13T10:00:05Z",
    )
    apply_reject = ProcessingResult(
        invoice_path=str(tmp_path / "invoices-small" / "apply-reject.png"),
        fingerprint_hash="fp-apply-reject",
        template_family_id="family-2",
        extracted_data={"Total": "9.99"},
        ground_truth={"Total": "10.99"},
        provenance=_sample_provenance(Route.REJECTED),
        validation_passed=False,
        route_used=Route.REJECTED,
        attempted_route=Route.APPLY,
        diagnostics=ProcessingDiagnostics(
            attempted_route=Route.APPLY,
            validation_score=0.6,
            validation_errors=["Mismatch on Total: expected '10.99', got '9.99'"],
            scalar_field_missing_count=0,
            scalar_field_mismatch_count=1,
        ),
        image_quality_score=0.85,
        template_status_at_time=None,
        processed_at="2026-04-13T10:00:06Z",
    )
    apply_error = ProcessingStageError(
        "apply_extraction", Route.APPLY, RuntimeError("Unknown argument: show_log")
    )

    summary = build_process_run_summary(
        input_dir=str(tmp_path / "invoices-small"),
        results=[
            {
                "name": "disc-pass.png",
                "route": "DISCOVERY",
                "status": "Passed",
                "elapsed_ms": 1000,
                "result": discovery_pass,
                "error": None,
            },
            {
                "name": "disc-reject.png",
                "route": "REJECTED",
                "status": "Failed/Rejected",
                "elapsed_ms": 900,
                "result": discovery_reject,
                "error": None,
            },
            {
                "name": "apply-reject.png",
                "route": "REJECTED",
                "status": "Failed/Rejected",
                "elapsed_ms": 800,
                "result": apply_reject,
                "error": None,
            },
            {
                "name": "apply-error.png",
                "route": "ERROR",
                "status": "Unknown argument: show_log",
                "elapsed_ms": 0,
                "result": None,
                "error": apply_error,
            },
        ],
        total_ms=2700,
        failure_summary={"failed_total": 2},
        discovery_only=False,
        discovery_mode="heuristic",
    )

    assert summary["route_counts"] == {"DISCOVERY": 1, "REJECTED": 2, "ERROR": 1}
    assert summary["discovery_passed"] == 1
    assert summary["discovery_rejected"] == 1
    assert summary["apply_rejected"] == 1
    assert summary["apply_runtime_errors"] == 1
    assert summary["runtime_error_categories"] == {"Unknown argument: show_log": 1}
    assert summary["failure_analysis"] == {"failed_total": 2}
    assert summary["discovery_mode"] == "heuristic"
    assert summary["latency_stats_ms"] == {"min": 0, "avg": 675, "max": 1000}
    assert summary["peak_rss_stats_mb"] is None
    assert summary["discovery_stage_status_counts"] == {"validation_failed_after_extract": 1}
    assert summary["family_counts"] == {"family-1": 2, "family-2": 1}
    assert summary["family_metrics"]["solved_family_count"] == 1
    assert summary["family_metrics"]["family_review_queue"] == ["family-1", "family-2"]
    assert summary["family_metrics"]["unresolved_family_count"] == 2
    assert (
        summary["family_metrics"]["highest_value_unstable_families"][0]["template_family_id"]
        == "family-2"
    )
    assert summary["family_rankings"][0]["template_family_id"] == "family-2"
    assert summary["family_rankings"][0]["unresolved_count"] == 1
    assert summary["invoices"][0]["template_family_id"] == "family-1"


def test_summarize_family_benchmark_comparison_reports_before_after(tmp_path):
    baseline = ProcessingResult(
        invoice_path=str(tmp_path / "invoices-test" / "one.png"),
        fingerprint_hash="fp-1",
        template_family_id="family-bench",
        extracted_data={"invoiceNumber": "1"},
        normalized_data={"invoice_number": "1"},
        ground_truth={"invoiceNumber": "1"},
        provenance=_sample_provenance(Route.APPLY),
        validation_passed=False,
        route_used=Route.REJECTED,
        attempted_route=Route.APPLY,
        diagnostics=ProcessingDiagnostics(
            attempted_route=Route.APPLY,
            validation_score=0.55,
            validation_errors=["Missing extracted field: Invoice Date"],
        ),
        image_quality_score=0.9,
        template_status_at_time=None,
        processed_at="2026-04-15T12:00:00Z",
    )
    candidate = ProcessingResult(
        invoice_path=str(tmp_path / "invoices-test" / "one.png"),
        fingerprint_hash="fp-1",
        template_family_id="family-bench",
        extracted_data={"invoiceNumber": "1"},
        normalized_data={"invoice_number": "1"},
        ground_truth={"invoiceNumber": "1"},
        provenance=_sample_provenance(Route.APPLY),
        validation_passed=True,
        route_used=Route.APPLY,
        attempted_route=Route.APPLY,
        diagnostics=ProcessingDiagnostics(
            attempted_route=Route.APPLY,
            validation_score=0.95,
            validation_errors=[],
        ),
        image_quality_score=0.9,
        template_status_at_time=None,
        processed_at="2026-04-15T12:05:00Z",
    )

    summary = summarize_family_benchmark_comparison(
        "family-bench",
        [baseline.invoice_path],
        {baseline.invoice_path: baseline},
        [
            {
                "invoice_path": baseline.invoice_path,
                "result": candidate,
                "error": None,
                "elapsed_ms": 100,
            }
        ],
    )

    assert summary["template_family_id"] == "family-bench"
    assert summary["baseline"]["status_counts"]["review"] == 1
    assert summary["candidate"]["status_counts"]["accepted"] == 1
    assert summary["progress"]["improvements"] == 1
    assert summary["progress"]["accepted_delta"] == 1
    assert summary["invoices"][0]["baseline_status"] == "review"
    assert summary["invoices"][0]["candidate_status"] == "accepted"


def test_baseline_fixture_matches_expected_counts():
    fixture_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "invoices_small_baseline_2026-04-13.json"
    )
    baseline = json.loads(fixture_path.read_text())

    assert baseline["route_counts"] == {"DISCOVERY": 8, "ERROR": 5}
    assert baseline["discovery_passed"] == 8
    assert baseline["discovery_rejected"] == 0
    assert baseline["apply_runtime_errors"] == 5
