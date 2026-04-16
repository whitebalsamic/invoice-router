import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from invoice_router.cli.main import cli
from invoice_router.models import ProcessingDiagnostics, ProcessingResult, Provenance, Route


def _sample_result(invoice_path: str, route: Route) -> ProcessingResult:
    return ProcessingResult(
        invoice_path=invoice_path,
        fingerprint_hash="fp-1",
        extracted_data={"Invoice No.": "1234"},
        ground_truth={"Invoice No.": "1234"},
        provenance=Provenance(
            request_id="req-1",
            route=route,
            fingerprint_hash="fp-1",
            template_status_at_time=None,
            template_confidence_at_time=None,
            ocr_engine="tesseract",
            inference_method="heuristic:test",
            image_quality_score=[0.9],
            quality_flag=False,
            input_document_hash="doc-hash",
            per_page_hashes=["page-hash"],
            extraction_output_hash="out-hash",
            started_at="2026-04-13T10:00:00Z",
            completed_at="2026-04-13T10:00:01Z",
            latency_ms=1000,
        ),
        validation_passed=True,
        route_used=route,
        attempted_route=route,
        diagnostics=ProcessingDiagnostics(
            attempted_route=route,
            validation_score=1.0,
            validation_errors=[],
        ),
        image_quality_score=0.9,
        template_status_at_time=None,
        processed_at="2026-04-13T10:00:01Z",
    )


def test_benchmark_structural_uses_curated_subset(monkeypatch, tmp_path):
    input_dir = tmp_path / "invoices-small"
    input_dir.mkdir()

    all_invoice_names = [
        "gokulraja_0001.png",
        "gokulraja_0002.png",
        "gokulraja_0005.png",
        "gokulraja_0007.png",
        "gokulraja_0008.png",
        "gokulraja_0010.png",
    ]
    for name in all_invoice_names:
        (input_dir / name).write_bytes(b"img")

    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )
    config = SimpleNamespace(
        validation=SimpleNamespace(discovery_threshold=0.95),
        discovery=SimpleNamespace(),
        processing=SimpleNamespace(worker_concurrency=4),
    )

    class FakeDB:
        def __init__(self, _db_path):
            pass

    processed = []

    def fake_process_single_invoice(invoice_path, *_args, **_kwargs):
        processed.append(Path(invoice_path).name)
        return _sample_result(invoice_path, Route.DISCOVERY)

    monkeypatch.setattr("invoice_router.cli.main.load_config", lambda: (settings, config))
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr(
        "invoice_router.cli.main.list_invoices",
        lambda _input_dir: [str(input_dir / name) for name in all_invoice_names],
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.process_single_invoice", fake_process_single_invoice
    )
    monkeypatch.setattr("invoice_router.cli.main.recreate_postgres_database", lambda _target: None)
    monkeypatch.setattr(
        "invoice_router.cli.main.summarize_failure_modes",
        lambda *_args, **_kwargs: {"failed_total": 0},
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.summarize_ground_truth_sync",
        lambda *_args, **_kwargs: {
            "status": "in_sync",
            "source_of_truth_dir": str(input_dir),
            "checked_invoice_count": 5,
            "matching_count": 5,
            "mismatched_count": 0,
            "missing_local_count": 0,
            "missing_source_count": 0,
            "uses_external_source_of_truth": False,
        },
    )

    runner = CliRunner()
    output_json = tmp_path / "benchmark.json"
    result = runner.invoke(
        cli,
        [
            "benchmark-structural",
            str(input_dir),
            "--output-json",
            str(output_json),
        ],
    )

    assert result.exit_code == 0, result.output
    assert processed == [
        "gokulraja_0005.png",
        "gokulraja_0001.png",
        "gokulraja_0008.png",
        "gokulraja_0007.png",
        "gokulraja_0010.png",
    ]

    payload = json.loads(output_json.read_text())
    assert payload["benchmark_variant"] == "structural"
    assert payload["benchmark_db_path"].startswith(
        "postgresql://invoice_router:invoice_router@localhost:5432/invoice_router_benchmark_invoices_small_structural"
    )
    assert "latency_stats_ms" in payload
    assert "peak_rss_stats_mb" in payload
    assert "peak_rss_mb" in payload["per_invoice"][0]
    assert payload["structural_subset"]["case_count"] == 5
    assert payload["ground_truth_sync"]["status"] == "in_sync"
    assert payload["structural_subset"]["bucket_order"] == [
        "party_block_extraction",
        "summary_amount_resolution",
        "routing_preprocessing",
        "scalar_reconciliation",
    ]


def test_sync_ground_truth_resolves_dataset_name_from_dataset_root(monkeypatch, tmp_path):
    dataset_root = tmp_path / "datasets"
    dataset_dir = dataset_root / "invoices-small"
    source_dir = dataset_root / "invoices-all"
    dataset_dir.mkdir(parents=True)
    source_dir.mkdir()

    monkeypatch.setenv("DATASET_ROOT", str(dataset_root))
    monkeypatch.setattr(
        "invoice_router.cli.main.load_config",
        lambda: (SimpleNamespace(dataset_root=str(dataset_root)), SimpleNamespace()),
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.summarize_ground_truth_sync",
        lambda input_dir, **_kwargs: {
            "dataset_dir": input_dir,
            "source_of_truth_dir": str(source_dir),
            "status": "in_sync",
            "checked_invoice_count": 0,
            "matching_count": 0,
            "mismatched_count": 0,
            "missing_local_count": 0,
            "missing_source_count": 0,
        },
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["sync-ground-truth", "invoices-small", "--check"])

    assert result.exit_code == 0, result.output
    assert f"Dataset: {dataset_dir}" in result.output


def test_benchmark_heuristic_reports_summary_progress(monkeypatch, tmp_path):
    input_dir = tmp_path / "invoices-small"
    input_dir.mkdir()
    invoices = [str(input_dir / f"{name}.png") for name in ["a", "b", "c", "d", "e"]]
    for invoice in invoices:
        Path(invoice).write_bytes(b"img")

    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )
    config = SimpleNamespace(
        validation=SimpleNamespace(discovery_threshold=0.95),
        discovery=SimpleNamespace(),
        processing=SimpleNamespace(worker_concurrency=2),
    )

    db_targets = []

    class FakeDB:
        def __init__(self, target):
            db_targets.append(target)
            pass

    monkeypatch.setattr("invoice_router.cli.main.load_config", lambda: (settings, config))
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr("invoice_router.cli.main.list_invoices", lambda _input_dir: invoices)
    monkeypatch.setattr(
        "invoice_router.cli.main.process_single_invoice",
        lambda invoice_path, *_args, **_kwargs: _sample_result(invoice_path, Route.DISCOVERY),
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.summarize_failure_modes",
        lambda *_args, **_kwargs: {"failed_total": 0},
    )
    monkeypatch.setattr("invoice_router.cli.main.recreate_postgres_database", lambda _target: None)
    monkeypatch.setattr(
        "invoice_router.cli.main.summarize_ground_truth_sync",
        lambda *_args, **_kwargs: {
            "status": "in_sync",
            "source_of_truth_dir": str(input_dir),
            "checked_invoice_count": 2,
            "matching_count": 2,
            "mismatched_count": 0,
            "missing_local_count": 0,
            "missing_source_count": 0,
            "uses_external_source_of_truth": False,
        },
    )

    runner = CliRunner()
    output_json = tmp_path / "benchmark.json"
    result = runner.invoke(
        cli,
        [
            "benchmark-heuristic",
            str(input_dir),
            "--output-json",
            str(output_json),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Benchmark progress: processed=5/5" in result.output
    assert len(db_targets) <= 1 + config.processing.worker_concurrency
    payload = json.loads(output_json.read_text())
    assert payload["ground_truth_sync"]["status"] == "in_sync"
    assert "latency_stats_ms" in payload
    assert "peak_rss_stats_mb" in payload
    assert "peak_rss_mb" in payload["per_invoice"][0]


def test_benchmark_structural_uses_isolated_postgres_target(monkeypatch, tmp_path):
    input_dir = tmp_path / "invoices-small"
    input_dir.mkdir()
    for name in [
        "gokulraja_0001.png",
        "gokulraja_0005.png",
        "gokulraja_0007.png",
        "gokulraja_0008.png",
        "gokulraja_0010.png",
    ]:
        (input_dir / name).write_bytes(b"img")

    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )
    config = SimpleNamespace(
        validation=SimpleNamespace(discovery_threshold=0.95),
        discovery=SimpleNamespace(),
        processing=SimpleNamespace(worker_concurrency=4),
    )

    created_targets = []
    db_targets = []

    class FakeDB:
        def __init__(self, target):
            db_targets.append(target)

    monkeypatch.setattr("invoice_router.cli.main.load_config", lambda: (settings, config))
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr(
        "invoice_router.cli.main.list_invoices",
        lambda _input_dir: [
            str(input_dir / name) for name in sorted(p.name for p in input_dir.iterdir())
        ],
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.process_single_invoice",
        lambda invoice_path, *_args, **_kwargs: _sample_result(invoice_path, Route.DISCOVERY),
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.summarize_failure_modes",
        lambda *_args, **_kwargs: {"failed_total": 0},
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.recreate_postgres_database",
        lambda target: created_targets.append(target),
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.summarize_ground_truth_sync",
        lambda *_args, **_kwargs: {
            "status": "in_sync",
            "source_of_truth_dir": str(input_dir),
            "checked_invoice_count": 5,
            "matching_count": 5,
            "mismatched_count": 0,
            "missing_local_count": 0,
            "missing_source_count": 0,
            "uses_external_source_of_truth": False,
        },
    )

    runner = CliRunner()
    output_json = tmp_path / "benchmark.json"
    result = runner.invoke(
        cli,
        [
            "benchmark-structural",
            str(input_dir),
            "--output-json",
            str(output_json),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(created_targets) == 1
    assert created_targets[0].startswith(
        "postgresql://invoice_router:invoice_router@localhost:5432/invoice_router_benchmark_invoices_small_structural"
    )
    assert db_targets == created_targets
    payload = json.loads(output_json.read_text())
    assert payload["benchmark_db_path"] == created_targets[0]
    assert "latency_stats_ms" in payload
    assert "peak_rss_stats_mb" in payload


def test_cli_version_flag_reports_public_command_name():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])

    assert result.exit_code == 0, result.output
    assert "invoice-router, version 0.1.0" in result.output


def test_sync_ground_truth_check_reports_status(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "invoice_router.cli.main.load_config",
        lambda: (SimpleNamespace(dataset_root=str(tmp_path)), SimpleNamespace()),
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.summarize_ground_truth_sync",
        lambda *_args, **_kwargs: {
            "dataset_dir": str(tmp_path / "invoices-small"),
            "source_of_truth_dir": str(tmp_path / "invoices-all"),
            "status": "out_of_sync",
            "checked_invoice_count": 3,
            "matching_count": 2,
            "mismatched_count": 1,
            "missing_local_count": 0,
            "missing_source_count": 0,
        },
    )

    dataset_dir = tmp_path / "invoices-small"
    dataset_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(cli, ["sync-ground-truth", str(dataset_dir), "--check"])

    assert result.exit_code == 0, result.output
    assert "Status: out_of_sync" in result.output
    assert "Mismatched: 1" in result.output


def test_sync_ground_truth_copies_from_source(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "invoice_router.cli.main.load_config",
        lambda: (SimpleNamespace(dataset_root=str(tmp_path)), SimpleNamespace()),
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.sync_ground_truth_from_source",
        lambda *_args, **_kwargs: {
            "dataset_dir": str(tmp_path / "invoices-small"),
            "source_of_truth_dir": str(tmp_path / "invoices-all"),
            "status": "in_sync",
            "copied_count": 1,
            "updated_count": 2,
            "unchanged_count": 9,
        },
    )

    dataset_dir = tmp_path / "invoices-small"
    dataset_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(cli, ["sync-ground-truth", str(dataset_dir)])

    assert result.exit_code == 0, result.output
    assert "Copied: 1" in result.output
    assert "Updated: 2" in result.output


def test_families_command_prints_family_summary(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )

    class FakeDB:
        def __init__(self, _target):
            pass

    monkeypatch.setattr(
        "invoice_router.cli.main.load_config", lambda: (settings, SimpleNamespace())
    )
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr(
        "invoice_router.cli.main.summarize_template_families",
        lambda _db: {
            "total_families": 2,
            "status_counts": {"degraded": 1, "established": 1},
            "review_queue": ["family-degraded"],
            "families": [
                {
                    "template_family_id": "family-degraded",
                    "status": "degraded",
                    "confidence": 0.42,
                    "apply_count": 2,
                    "reject_count": 4,
                    "triage_class": "suspiciously_tiny",
                    "extraction_profile": {
                        "preferred_strategy": "provider_template",
                        "field_override_count": 4,
                        "table_enabled": True,
                        "table_engine": "paddle",
                    },
                    "representative_fingerprint_hash": "abcdef123456",
                },
                {
                    "template_family_id": "family-established",
                    "status": "established",
                    "confidence": 0.99,
                    "apply_count": 8,
                    "reject_count": 0,
                    "triage_class": "stable",
                    "extraction_profile": {
                        "preferred_strategy": "provider_template",
                        "field_override_count": 2,
                        "table_enabled": False,
                        "table_engine": None,
                    },
                    "representative_fingerprint_hash": "fedcba654321",
                },
            ],
        },
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["families"])

    assert result.exit_code == 0, result.output
    assert "Template Families: 2" in result.output
    assert "Review queue: family-degraded" in result.output
    assert "family-degraded | status=degraded" in result.output
    assert "triage=suspiciously_tiny" in result.output
    assert "profile=provider_template/table:paddle/fields:4" in result.output


def test_family_show_prints_detail(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )

    class FakeDB:
        def __init__(self, _target):
            pass

    monkeypatch.setattr(
        "invoice_router.cli.main.load_config", lambda: (settings, SimpleNamespace())
    )
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr(
        "invoice_router.cli.main.describe_template_family",
        lambda *_args, **_kwargs: {
            "template_family_id": "family-detail",
            "status": "degraded",
            "confidence": 0.42,
            "apply_count": 2,
            "reject_count": 4,
            "provider_name": "Acme Ltd",
            "country_code": "GB",
            "document_family": "invoice",
            "created_at": "2026-04-15T12:00:00Z",
            "updated_at": "2026-04-15T12:10:00Z",
            "review_signals": ["status is degraded"],
            "triage_class": "high_value_unstable",
            "triage_explanation": "Large unresolved family with enough volume to matter operationally. (invoices=6, pass_rate=33.3%, rejects=4)",
            "extraction_profile_summary": {
                "preferred_strategy": "provider_template",
                "table_enabled": True,
                "table_engine": "paddle",
                "field_override_count": 2,
            },
            "representative": {
                "fingerprint_hash": "abcdef123456",
                "status": "established",
                "apply_count": 8,
                "reject_count": 1,
            },
            "example_count": 1,
            "examples": [
                {"invoice_path": str(tmp_path / "one.png"), "fingerprint_hash": "abcdef123456"}
            ],
            "version_count": 1,
            "versions": [
                {
                    "version": 3,
                    "change_reason": "manual_review",
                    "created_at": "2026-04-15T12:10:00Z",
                }
            ],
            "recent_results": [
                {
                    "invoice_name": "one.png",
                    "outcome": "review",
                    "route": "REJECTED",
                    "validation_score": 0.6,
                    "review_signals": ["mismatch on Total"],
                }
            ],
        },
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["family-show", "family-detail"])

    assert result.exit_code == 0, result.output
    assert "Template Family: family-detail" in result.output
    assert "Triage: high_value_unstable" in result.output
    assert "Review signals: status is degraded" in result.output
    assert "Recent versions:" in result.output
    assert "Recent outcomes:" in result.output


def test_family_update_invokes_manual_update(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )

    class FakeDB:
        def __init__(self, _target):
            pass

    seen = {}

    def _manual_update(_db, template_family_id, **kwargs):
        seen["family_id"] = template_family_id
        seen["kwargs"] = kwargs
        return SimpleNamespace(
            template_family_id=template_family_id,
            status=SimpleNamespace(value="established"),
            provider_name="Acme Ltd",
            country_code="GB",
            document_family=SimpleNamespace(value="invoice"),
        )

    profile_json = tmp_path / "profile.json"
    profile_json.write_text(json.dumps({"ocr": {"region_buffer_multiplier": 1.2}}))

    monkeypatch.setattr(
        "invoice_router.cli.main.load_config", lambda: (settings, SimpleNamespace())
    )
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr("invoice_router.cli.main.manually_update_template_family", _manual_update)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "family-update",
            "family-detail",
            "--status",
            "established",
            "--provider-name",
            "Acme Ltd",
            "--profile-json",
            str(profile_json),
            "--reason",
            "manual_review",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["family_id"] == "family-detail"
    assert seen["kwargs"]["reason"] == "manual_review"
    assert seen["kwargs"]["extraction_profile_updates"]["ocr"]["region_buffer_multiplier"] == 1.2
    assert "Updated template family: family-detail" in result.output


def test_family_suggestions_prints_review_queue(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )

    class FakeDB:
        def __init__(self, _target):
            pass

    monkeypatch.setattr(
        "invoice_router.cli.main.load_config", lambda: (settings, SimpleNamespace())
    )
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr(
        "invoice_router.cli.main.suggest_template_family_updates",
        lambda *_args, **_kwargs: {
            "queue_count": 1,
            "family_ids": ["family-detail"],
            "suggestions": [
                {
                    "template_family_id": "family-detail",
                    "kind": "date_ocr_boost",
                    "support_count": 3,
                    "priority": 4.0,
                    "title": "Widen OCR for date fields",
                    "reason": "3 recent invoice(s) had date extraction failures",
                    "example_invoices": ["one.png", "two.png"],
                    "profile_patch": {"ocr": {"region_buffer_multiplier": 1.25}},
                }
            ],
        },
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["family-suggestions"])

    assert result.exit_code == 0, result.output
    assert "Family suggestions: 1" in result.output
    assert "family-detail | kind=date_ocr_boost" in result.output
    assert 'patch: {"ocr": {"region_buffer_multiplier": 1.25}}' in result.output


def test_family_apply_suggestion_invokes_manual_update(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )

    class FakeDB:
        def __init__(self, _target):
            pass

    seen = {}

    def _manual_update(_db, template_family_id, **kwargs):
        seen["family_id"] = template_family_id
        seen["kwargs"] = kwargs
        return SimpleNamespace(template_family_id=template_family_id)

    monkeypatch.setattr(
        "invoice_router.cli.main.load_config", lambda: (settings, SimpleNamespace())
    )
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr(
        "invoice_router.cli.main.suggest_template_family_updates",
        lambda *_args, **_kwargs: {
            "queue_count": 1,
            "family_ids": ["family-detail"],
            "suggestions": [
                {
                    "template_family_id": "family-detail",
                    "kind": "date_ocr_boost",
                    "profile_patch": {"ocr": {"region_buffer_multiplier": 1.25}},
                }
            ],
        },
    )
    monkeypatch.setattr("invoice_router.cli.main.manually_update_template_family", _manual_update)

    runner = CliRunner()
    result = runner.invoke(cli, ["family-apply-suggestion", "family-detail", "date_ocr_boost"])

    assert result.exit_code == 0, result.output
    assert seen["family_id"] == "family-detail"
    assert seen["kwargs"]["extraction_profile_updates"]["ocr"]["region_buffer_multiplier"] == 1.25
    assert seen["kwargs"]["reason"] == "apply_suggestion:date_ocr_boost"
    assert "Applied suggestion: date_ocr_boost" in result.output


def test_family_splits_prints_candidates(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )

    class FakeDB:
        def __init__(self, _target):
            pass

    monkeypatch.setattr(
        "invoice_router.cli.main.load_config", lambda: (settings, SimpleNamespace())
    )
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr(
        "invoice_router.cli.main.suggest_template_family_splits",
        lambda *_args, **_kwargs: {
            "queue_count": 1,
            "family_ids": ["family-detail"],
            "suggestions": [
                {
                    "template_family_id": "family-detail",
                    "cluster_id": "cluster-2",
                    "fingerprint_count": 2,
                    "status_counts": {"review": 1},
                    "similarity_to_primary": 0.31,
                    "title": "Split out cluster-2 from family-detail",
                    "reason": "2 fingerprints differ structurally from the primary cluster",
                    "example_invoices": ["one.png"],
                    "proposed_family_id": "family-detail-split-1",
                    "fingerprint_hashes": ["fp-a", "fp-b"],
                }
            ],
        },
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["family-splits"])

    assert result.exit_code == 0, result.output
    assert "Family split suggestions: 1" in result.output
    assert "family-detail | cluster-2 | fingerprints=2" in result.output
    assert "new family: family-detail-split-1" in result.output


def test_family_split_invokes_lifecycle_split(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )
    config = SimpleNamespace(template_lifecycle=SimpleNamespace())

    class FakeDB:
        def __init__(self, _target):
            pass

    seen = {}

    def _split(_db, template_family_id, **kwargs):
        seen["family_id"] = template_family_id
        seen["kwargs"] = kwargs
        return {
            "source_family_id": template_family_id,
            "new_family_id": kwargs["new_template_family_id"],
            "moved_fingerprint_count": 2,
            "moved_example_count": 1,
            "moved_result_count": 3,
            "source_family": SimpleNamespace(
                status=SimpleNamespace(value="established"), apply_count=7, reject_count=1
            ),
            "new_family": SimpleNamespace(
                status=SimpleNamespace(value="provisional"), apply_count=2, reject_count=0
            ),
        }

    monkeypatch.setattr("invoice_router.cli.main.load_config", lambda: (settings, config))
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr(
        "invoice_router.cli.main.suggest_template_family_splits",
        lambda *_args, **_kwargs: {
            "queue_count": 1,
            "family_ids": ["family-detail"],
            "suggestions": [
                {
                    "template_family_id": "family-detail",
                    "cluster_id": "cluster-2",
                    "fingerprint_hashes": ["fp-a", "fp-b"],
                    "proposed_family_id": "family-detail-split-1",
                }
            ],
        },
    )
    monkeypatch.setattr("invoice_router.cli.main.split_template_family", _split)

    runner = CliRunner()
    result = runner.invoke(cli, ["family-split", "family-detail", "--cluster", "cluster-2"])

    assert result.exit_code == 0, result.output
    assert seen["family_id"] == "family-detail"
    assert seen["kwargs"]["fingerprint_hashes"] == ["fp-a", "fp-b"]
    assert seen["kwargs"]["new_template_family_id"] == "family-detail-split-1"
    assert "Split template family: family-detail -> family-detail-split-1" in result.output
    assert "Moved fingerprints/examples/results: 2/1/3" in result.output


def test_family_merges_prints_candidates(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )

    class FakeDB:
        def __init__(self, _target):
            pass

    monkeypatch.setattr(
        "invoice_router.cli.main.load_config", lambda: (settings, SimpleNamespace())
    )
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr(
        "invoice_router.cli.main.suggest_template_family_merges",
        lambda *_args, **_kwargs: {
            "queue_count": 1,
            "family_ids": ["family-a", "family-b"],
            "suggestions": [
                {
                    "source_family_id": "family-b",
                    "target_family_id": "family-a",
                    "similarity": 0.91,
                    "source_status": "provisional",
                    "target_status": "established",
                    "source_apply_count": 1,
                    "target_apply_count": 8,
                    "title": "Merge family-b into family-a",
                    "reason": "Families are highly similar",
                    "shared_signals": {"anchor_tokens": ["invoice date"]},
                }
            ],
        },
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["family-merges"])

    assert result.exit_code == 0, result.output
    assert "Family merge suggestions: 1" in result.output
    assert "family-b -> family-a | similarity=0.91" in result.output
    assert "shared anchors: invoice date" in result.output


def test_family_merge_invokes_lifecycle_merge(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )
    config = SimpleNamespace(template_lifecycle=SimpleNamespace())

    class FakeDB:
        def __init__(self, _target):
            pass

    seen = {}

    def _merge(_db, target_family_id, source_family_id, **kwargs):
        seen["target_family_id"] = target_family_id
        seen["source_family_id"] = source_family_id
        seen["kwargs"] = kwargs
        return {
            "moved_fingerprint_count": 2,
            "moved_example_count": 1,
            "moved_result_count": 3,
            "target_family": SimpleNamespace(
                status=SimpleNamespace(value="established"), apply_count=9, reject_count=1
            ),
            "source_family": SimpleNamespace(
                status=SimpleNamespace(value="retired"), apply_count=1, reject_count=2
            ),
        }

    monkeypatch.setattr("invoice_router.cli.main.load_config", lambda: (settings, config))
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr("invoice_router.cli.main.merge_template_families", _merge)

    runner = CliRunner()
    result = runner.invoke(cli, ["family-merge", "family-b", "family-a"])

    assert result.exit_code == 0, result.output
    assert seen["source_family_id"] == "family-b"
    assert seen["target_family_id"] == "family-a"
    assert seen["kwargs"]["reason"] == "manual_merge"
    assert "Merged template family: family-b -> family-a" in result.output
    assert "Moved fingerprints/examples/results: 2/1/3" in result.output


def test_family_retirements_prints_candidates(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )

    class FakeDB:
        def __init__(self, _target):
            pass

    monkeypatch.setattr(
        "invoice_router.cli.main.load_config", lambda: (settings, SimpleNamespace())
    )
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr(
        "invoice_router.cli.main.suggest_template_family_retirements",
        lambda *_args, **_kwargs: {
            "queue_count": 1,
            "family_ids": ["family-old"],
            "suggestions": [
                {
                    "template_family_id": "family-old",
                    "kind": "repeated_rejection",
                    "active_fingerprint_count": 1,
                    "apply_count": 0,
                    "reject_count": 4,
                    "title": "Retire family-old",
                    "reason": "Family has no successful applies and repeated rejects",
                    "example_invoices": ["old.png"],
                }
            ],
        },
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["family-retirements"])

    assert result.exit_code == 0, result.output
    assert "Family retirement suggestions: 1" in result.output
    assert "family-old | kind=repeated_rejection" in result.output
    assert "examples: old.png" in result.output


def test_family_retire_invokes_lifecycle_retire(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )

    class FakeDB:
        def __init__(self, _target):
            pass

    seen = {}

    def _retire(_db, template_family_id, **kwargs):
        seen["family_id"] = template_family_id
        seen["kwargs"] = kwargs
        return {
            "retired_fingerprint_count": 2,
            "family": SimpleNamespace(
                status=SimpleNamespace(value="retired"), apply_count=0, reject_count=4
            ),
        }

    monkeypatch.setattr(
        "invoice_router.cli.main.load_config", lambda: (settings, SimpleNamespace())
    )
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr("invoice_router.cli.main.retire_template_family", _retire)

    runner = CliRunner()
    result = runner.invoke(cli, ["family-retire", "family-old"])

    assert result.exit_code == 0, result.output
    assert seen["family_id"] == "family-old"
    assert seen["kwargs"]["reason"] == "manual_retire"
    assert seen["kwargs"]["retire_fingerprints"] is True
    assert "Retired template family: family-old" in result.output
    assert "Retired fingerprints: 2" in result.output


def test_family_benchmark_prints_before_after_summary(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )
    config = SimpleNamespace(processing=SimpleNamespace(worker_concurrency=1))

    invoice_path = tmp_path / "family-one.png"
    invoice_path.write_bytes(b"img")
    baseline_result = _sample_result(str(invoice_path), Route.REJECTED)
    baseline_result.validation_passed = False
    baseline_result.route_used = Route.REJECTED
    baseline_result.diagnostics.validation_score = 0.55
    candidate_result = _sample_result(str(invoice_path), Route.APPLY)
    candidate_result.validation_passed = True
    candidate_result.route_used = Route.APPLY
    candidate_result.diagnostics.validation_score = 0.95

    class FakeDB:
        def __init__(self, _target):
            self.target = _target

        def get_result(self, path):
            return baseline_result if path == str(invoice_path) else None

    monkeypatch.setattr("invoice_router.cli.main.load_config", lambda: (settings, config))
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr(
        "invoice_router.cli.main._collect_family_review_invoices",
        lambda *_args, **_kwargs: [str(invoice_path)],
    )
    monkeypatch.setattr(
        "invoice_router.cli.main._benchmark_database_target",
        lambda *_args, **_kwargs: (
            "postgresql://invoice_router:invoice_router@localhost:5432/invoice_router_family_review"
        ),
    )
    monkeypatch.setattr(
        "invoice_router.cli.main._seed_family_review_db", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.process_single_invoice",
        lambda inv, *_args, **_kwargs: candidate_result,
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.suggest_template_family_updates",
        lambda *_args, **_kwargs: {"queue_count": 1, "suggestions": [{"kind": "date_ocr_boost"}]},
    )

    runner = CliRunner()
    output_json = tmp_path / "family_benchmark.json"
    result = runner.invoke(
        cli,
        ["family-benchmark", "family-1", "--output-json", str(output_json)],
    )

    assert result.exit_code == 0, result.output
    assert "Family benchmark: family-1" in result.output
    assert "Baseline accepted/review/error: 0/1/0" in result.output
    assert "Candidate accepted/review/error: 1/0/0" in result.output
    assert "Improvements/regressions/unchanged: 1/0/0" in result.output
    assert output_json.exists()


def test_family_benchmark_reuses_worker_db_instances(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )
    config = SimpleNamespace(processing=SimpleNamespace(worker_concurrency=2))

    invoice_paths = []
    for idx in range(4):
        invoice_path = tmp_path / f"family-{idx}.png"
        invoice_path.write_bytes(b"img")
        invoice_paths.append(str(invoice_path))

    baseline_results = {path: _sample_result(path, Route.REJECTED) for path in invoice_paths}
    for result in baseline_results.values():
        result.validation_passed = False
        result.route_used = Route.REJECTED
        result.diagnostics.validation_score = 0.55
    candidate_result = _sample_result(invoice_paths[0], Route.APPLY)
    candidate_result.validation_passed = True
    candidate_result.route_used = Route.APPLY
    candidate_result.diagnostics.validation_score = 0.95

    db_targets = []

    class FakeDB:
        def __init__(self, target):
            db_targets.append(target)
            self.target = target

        def get_result(self, path):
            return baseline_results.get(path)

    monkeypatch.setattr("invoice_router.cli.main.load_config", lambda: (settings, config))
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr(
        "invoice_router.cli.main._collect_family_review_invoices",
        lambda *_args, **_kwargs: invoice_paths,
    )
    monkeypatch.setattr(
        "invoice_router.cli.main._benchmark_database_target",
        lambda *_args, **_kwargs: (
            "postgresql://invoice_router:invoice_router@localhost:5432/invoice_router_family_review"
        ),
    )
    monkeypatch.setattr(
        "invoice_router.cli.main._seed_family_review_db", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.process_single_invoice",
        lambda inv, *_args, **_kwargs: _sample_result(inv, Route.APPLY),
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.suggest_template_family_updates",
        lambda *_args, **_kwargs: {"queue_count": 0, "suggestions": []},
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["family-benchmark", "family-1"])

    assert result.exit_code == 0, result.output
    assert len(db_targets) <= 2 + config.processing.worker_concurrency
