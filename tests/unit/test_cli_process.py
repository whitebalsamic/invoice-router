from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from invoice_router.cli.main import cli
from invoice_router.models import ProcessingResult, Provenance, Route


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
        image_quality_score=0.9,
        template_status_at_time=None,
        processed_at="2026-04-13T10:00:01Z",
    )


def test_process_discovery_only_freezes_fingerprint_snapshot(monkeypatch, tmp_path):
    input_dir = tmp_path / "invoices-small"
    input_dir.mkdir()
    invoices = [str(input_dir / "a.png"), str(input_dir / "b.png")]
    for invoice in invoices:
        Path(invoice).write_bytes(b"img")

    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )
    config = SimpleNamespace(validation=SimpleNamespace(discovery_threshold=0.95))
    config.discovery = SimpleNamespace()
    config.processing = SimpleNamespace(worker_concurrency=1)

    class FakeDB:
        def __init__(self, _db_path):
            self.calls = []

        def clear_fingerprints(self):
            return None

        def clear_processing_results(self, _invoice_paths):
            return None

        def get_all_active_fingerprints(self):
            return []

        def get_failed_results_for_input_dir(self, *_args, **_kwargs):
            return []

    captured_overrides = []

    def fake_process_single_invoice(
        invoice_path, _settings, _config, _db, active_fingerprints_override=None
    ):
        captured_overrides.append(active_fingerprints_override)
        route = Route.DISCOVERY if active_fingerprints_override == [] else Route.APPLY
        return _sample_result(invoice_path, route)

    monkeypatch.setattr("invoice_router.cli.main.load_config", lambda: (settings, config))
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr("invoice_router.cli.main.list_invoices", lambda _input_dir: invoices)
    monkeypatch.setattr(
        "invoice_router.cli.main.process_single_invoice", fake_process_single_invoice
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.summarize_failure_modes",
        lambda *_args, **_kwargs: {"failed_total": 0},
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["process", str(input_dir), "--discovery-only"])

    assert result.exit_code == 0
    assert "Discovery strategy: heuristic" in result.output
    assert "Discovery-only mode active" in result.output
    assert "Process progress: processed=2/2" in result.output
    assert "Route APPLY" not in result.output
    assert len(captured_overrides) == 2
    assert captured_overrides[0] == []
    assert captured_overrides[1] == []


def test_process_discovery_mode_override(monkeypatch, tmp_path):
    input_dir = tmp_path / "invoices-small"
    input_dir.mkdir()
    invoice = str(input_dir / "a.png")
    Path(invoice).write_bytes(b"img")

    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )
    config = SimpleNamespace(
        validation=SimpleNamespace(discovery_threshold=0.95),
        discovery=SimpleNamespace(),
        processing=SimpleNamespace(worker_concurrency=1),
    )

    class FakeDB:
        def __init__(self, _db_path):
            pass

        def clear_fingerprints(self):
            return None

        def clear_processing_results(self, _invoice_paths):
            return None

        def get_all_active_fingerprints(self):
            return []

        def get_failed_results_for_input_dir(self, *_args, **_kwargs):
            return []

    def fake_process_single_invoice(
        invoice_path, _settings, passed_config, _db, active_fingerprints_override=None
    ):
        return _sample_result(invoice_path, Route.DISCOVERY)

    monkeypatch.setattr("invoice_router.cli.main.load_config", lambda: (settings, config))
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr("invoice_router.cli.main.list_invoices", lambda _input_dir: [invoice])
    monkeypatch.setattr(
        "invoice_router.cli.main.process_single_invoice", fake_process_single_invoice
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.summarize_failure_modes",
        lambda *_args, **_kwargs: {"failed_total": 0},
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["process", str(input_dir)])

    assert result.exit_code == 0
    assert "Discovery strategy: heuristic" in result.output


def test_process_reset_templates_implies_reset(monkeypatch, tmp_path):
    input_dir = tmp_path / "invoices-test"
    input_dir.mkdir()
    invoices = [str(input_dir / "a.png"), str(input_dir / "b.png")]
    for invoice in invoices:
        Path(invoice).write_bytes(b"img")

    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )
    config = SimpleNamespace(
        validation=SimpleNamespace(discovery_threshold=0.95),
        discovery=SimpleNamespace(),
        processing=SimpleNamespace(worker_concurrency=1),
    )

    class FakeDB:
        def __init__(self, _db_path):
            self.cleared_processing = None
            self.cleared_fingerprints = False

        def clear_fingerprints(self):
            self.cleared_fingerprints = True

        def clear_processing_results(self, invoice_paths):
            self.cleared_processing = list(invoice_paths)

        def get_all_active_fingerprints(self):
            return []

        def get_failed_results_for_input_dir(self, *_args, **_kwargs):
            return []

    fake_db = FakeDB(settings.database_url)

    monkeypatch.setattr("invoice_router.cli.main.load_config", lambda: (settings, config))
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", lambda _db_path: fake_db)
    monkeypatch.setattr("invoice_router.cli.main.list_invoices", lambda _input_dir: invoices)
    monkeypatch.setattr(
        "invoice_router.cli.main.process_single_invoice",
        lambda invoice_path, *_args, **_kwargs: _sample_result(invoice_path, Route.DISCOVERY),
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.summarize_failure_modes",
        lambda *_args, **_kwargs: {"failed_total": 0},
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["process", str(input_dir), "--reset-templates"])

    assert result.exit_code == 0
    assert fake_db.cleared_fingerprints is True
    assert fake_db.cleared_processing == invoices
    assert "Cleared all fingerprint templates." in result.output
    assert "Resetting cached processing results as part of --reset-templates." in result.output
    assert "Cleared cached results for 2 invoices." in result.output


def test_process_runs_auto_heal_post_pass(monkeypatch, tmp_path):
    input_dir = tmp_path / "invoices-small"
    input_dir.mkdir()
    invoice = str(input_dir / "a.png")
    Path(invoice).write_bytes(b"img")

    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )
    config = SimpleNamespace(
        validation=SimpleNamespace(discovery_threshold=0.95),
        discovery=SimpleNamespace(),
        processing=SimpleNamespace(worker_concurrency=1),
    )

    class FakeDB:
        def __init__(self, _db_path):
            pass

        def clear_fingerprints(self):
            return None

        def clear_processing_results(self, _invoice_paths):
            return None

        def get_all_active_fingerprints(self):
            return []

    monkeypatch.setattr("invoice_router.cli.main.load_config", lambda: (settings, config))
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr("invoice_router.cli.main.list_invoices", lambda _input_dir: [invoice])
    monkeypatch.setattr(
        "invoice_router.cli.main.process_single_invoice",
        lambda invoice_path, *_args, **_kwargs: _sample_result(invoice_path, Route.DISCOVERY),
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.summarize_failure_modes",
        lambda *_args, **_kwargs: {"failed_total": 0},
    )
    monkeypatch.setattr(
        "invoice_router.cli.main._heal_rejections_pass",
        lambda *_args, **_kwargs: {
            "candidates": 3,
            "attempted": 2,
            "recovered": 1,
            "still_failing": 1,
            "skipped_as_already_attempted": 1,
            "skipped_not_apply": 0,
            "skipped_not_gt_trusted": 0,
            "entries": [],
        },
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["process", str(input_dir)])

    assert result.exit_code == 0
    assert (
        "Healing summary: candidates=3 | attempted=2 | recovered=1 | still_failing=1 | skipped_as_already_attempted=1"
        in result.output
    )


def test_heal_rejections_command_retries_gt_trusted_apply_candidates(monkeypatch, tmp_path):
    input_dir = tmp_path / "invoices-small"
    input_dir.mkdir()
    invoice = input_dir / "rejected.png"
    invoice.write_bytes(b"img")

    settings = SimpleNamespace(
        database_url="postgresql://invoice_router:invoice_router@localhost:5432/invoice_router",
        output_dir=str(tmp_path / "output"),
    )
    config = SimpleNamespace(
        validation=SimpleNamespace(discovery_threshold=0.95),
        discovery=SimpleNamespace(),
        processing=SimpleNamespace(worker_concurrency=1),
    )
    previous = _sample_result(str(invoice), Route.REJECTED)
    previous.validation_passed = False
    previous.route_used = Route.REJECTED

    class FakeDB:
        def __init__(self, _db_path):
            pass

        def get_failed_results_for_input_dir(self, *_args, **_kwargs):
            return [previous]

    captured = {"force_reprocess": None, "retry_context": None}

    def fake_process_single_invoice(invoice_path, _settings, _config, _db, **kwargs):
        captured["force_reprocess"] = kwargs.get("force_reprocess")
        captured["retry_context"] = kwargs.get("retry_context")
        result = _sample_result(invoice_path, Route.APPLY)
        result.validation_passed = True
        return result

    monkeypatch.setattr("invoice_router.cli.main.load_config", lambda: (settings, config))
    monkeypatch.setattr("invoice_router.cli.main.FingerprintDB", FakeDB)
    monkeypatch.setattr(
        "invoice_router.cli.main.assess_healing_candidate",
        lambda *_args, **_kwargs: SimpleNamespace(
            route=Route.APPLY,
            current_family_id="family-1",
            trigger_family_id="family-1",
            trigger_fingerprint_hash="fp-1",
            trigger_match_type="fingerprint",
            trigger_score=1.0,
            gt_trust_scope="family",
            gt_trusted=True,
            gt_apply_count=6,
            gt_reject_count=1,
            gt_confidence=0.98,
        ),
    )
    monkeypatch.setattr(
        "invoice_router.cli.main.process_single_invoice", fake_process_single_invoice
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["heal-rejections", str(input_dir), "--show-per-invoice"])

    assert result.exit_code == 0
    assert captured["force_reprocess"] is True
    assert captured["retry_context"]["healing_origin"] == "manual_cli"
    assert "rejected.png: RECOVERED" in result.output
    assert (
        "Healing summary: candidates=1 | attempted=1 | recovered=1 | still_failing=0 | skipped_as_already_attempted=0"
        in result.output
    )
