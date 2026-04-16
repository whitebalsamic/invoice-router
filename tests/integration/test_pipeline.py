from unittest.mock import MagicMock

import numpy as np
import pytest

from invoice_router.config import load_config
from invoice_router.infrastructure.persistence.storage import FingerprintDB
from invoice_router.models import (
    DocumentContext,
    DocumentFamily,
    ExtractionStrategy,
    FingerprintRecord,
    PageFingerprint,
    PageRole,
    Route,
    SourceFormat,
    TemplateFamilyRecord,
    TemplateStatus,
    ValidationResult,
)
from invoice_router.pipeline import process_single_invoice
from invoice_router.provenance import get_utc_now


@pytest.fixture
def mock_pipeline_dependencies(monkeypatch, tmp_path, postgres_database_url):
    monkeypatch.setattr(
        "invoice_router.pipeline.pair_ground_truth",
        lambda _path: {"Invoice No.": "1234", "Total": 100.0},
    )

    # Mock preprocessor
    mock_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    monkeypatch.setattr("invoice_router.pipeline.normalize_page", lambda _path: [mock_img])

    # Create a dummy image file to avoid mocking open()
    dummy_img_path = tmp_path / "test_invoice.png"
    dummy_img_path.write_bytes(b"fake_image_data")

    # Mock OCR and Hashing
    monkeypatch.setattr(
        "invoice_router.pipeline.run_full_page_ocr",
        lambda _img, _engine: [("Invoice No.", 10, 10, 20, 10)],
    )

    # Mock discovery inference
    mock_template = {
        "version": "v3",
        "inference_method": "heuristic_discovery",
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "Invoice No.": {
                        "region": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.1},
                        "field_type": "string",
                        "anchor_direction": "right",
                    },
                    "Total": {
                        "region": {"x": 0.5, "y": 0.5, "width": 0.2, "height": 0.1},
                        "field_type": "currency",
                        "anchor_direction": "right",
                    },
                },
                "table": None,
                "label_confirmation_set": [],
            }
        ],
    }
    infer_template_mock = MagicMock(
        return_value=(
            mock_template,
            0.95,
            {"Invoice No.": "1234", "Total": 100.0, "line_items": []},
            {"located_field_count": 2, "extracted_field_count": 2},
        )
    )
    monkeypatch.setattr("invoice_router.pipeline.infer_template", infer_template_mock)

    # Mock Extract OCR
    extract_with_ocr_mock = MagicMock(return_value={"Invoice No.": "1234", "Total": 100.0})
    monkeypatch.setattr("invoice_router.pipeline.extract_with_ocr", extract_with_ocr_mock)

    # Database
    db = FingerprintDB(postgres_database_url)

    # Settings/Config
    monkeypatch.setenv("INVOICE_INPUT_DIR", "/fake")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("DATABASE_URL", postgres_database_url)
    settings, config = load_config("config.yaml")
    settings.output_dir = str(tmp_path / "output")

    # Mock Redis client
    mock_redis_client = MagicMock()
    mock_redis_client.hincrby.return_value = 1
    monkeypatch.setattr(
        "invoice_router.pipeline.get_redis_client", lambda _settings: mock_redis_client
    )

    return settings, config, db, tmp_path, infer_template_mock, extract_with_ocr_mock


def test_full_pipeline_discovery_and_apply(mock_pipeline_dependencies):
    settings, config, db, tmp_path, infer_template_mock, extract_with_ocr_mock = (
        mock_pipeline_dependencies
    )
    invoice_path = str(tmp_path / "test_invoice.png")

    # Run 1: Unknown fingerprint + GT present -> DISCOVERY
    res1 = process_single_invoice(invoice_path, settings, config, db)

    assert res1.route_used == Route.DISCOVERY
    assert res1.validation_passed is True
    assert res1.fingerprint_hash is not None
    assert res1.extracted_data["Invoice No."] == "1234"
    assert res1.normalized_data["invoice_number"] == "1234"
    assert res1.normalized_data["total"] == 100.0
    assert res1.diagnostics.discovery_mode == "heuristic"
    assert res1.provenance.extraction_strategy == ExtractionStrategy.ocr_structured
    assert res1.provenance.source_format == SourceFormat.image

    # Verify it was saved to the DB as provisional
    fp_record = db.get_all_active_fingerprints()[0]
    assert fp_record.status == TemplateStatus.provisional
    assert fp_record.hash == res1.fingerprint_hash

    # Clear processing results to simulate a new run of the same file (idempotency override)
    db.clear_processing_results()

    # Run 2: Known provisional fingerprint + GT present -> DISCOVERY again
    res2 = process_single_invoice(invoice_path, settings, config, db)

    assert res2.route_used == Route.DISCOVERY
    assert res2.validation_passed is True
    assert res2.fingerprint_hash == res1.fingerprint_hash
    assert infer_template_mock.call_count == 2
    assert extract_with_ocr_mock.call_count == 0


def test_family_profile_is_passed_into_apply_extraction(mock_pipeline_dependencies, monkeypatch):
    settings, config, db, tmp_path, _infer_template_mock, _extract_with_ocr_mock = (
        mock_pipeline_dependencies
    )
    invoice_path = str(tmp_path / "test_invoice.png")

    family = TemplateFamilyRecord(
        template_family_id="family-apply-1",
        document_family=DocumentFamily.invoice,
        extraction_profile={
            "preferred_strategy": "provider_template",
            "ocr": {
                "field_type_overrides": {"Invoice Date": "date"},
                "field_buffer_multipliers": {"date": 1.2},
            },
            "table": {"enabled": True, "ocr_engine": "paddle"},
        },
        created_at=get_utc_now(),
    )
    db.store_template_family(family)

    apply_record = FingerprintRecord(
        hash="known-fingerprint",
        layout_template={
            "pages": [
                {
                    "page_index": 0,
                    "fields": {
                        "Invoice Date": {
                            "region": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.1},
                            "field_type": "string",
                        },
                    },
                    "label_confirmation_set": [],
                }
            ]
        },
        template_family_id="family-apply-1",
        page_fingerprints=[PageFingerprint(page_index=0, visual_hash=1, visual_hash_hex="0" * 16)],
        confidence=1.0,
        status=TemplateStatus.established,
        version="v3",
        created_at=get_utc_now(),
        last_used=get_utc_now(),
    )
    monkeypatch.setattr(
        "invoice_router.pipeline.lookup_fingerprint", lambda *_args, **_kwargs: (apply_record, 1.0)
    )
    monkeypatch.setattr(
        "invoice_router.pipeline.validate_invoice",
        lambda *_args, **_kwargs: ValidationResult(
            passed=True,
            score=1.0,
            matched_fields=2,
            mismatched_fields=0,
            errors=[],
        ),
    )

    captured = {}

    def _extract_with_profile(_pages, _template, _buffer_px, _config, family_profile=None):
        captured["family_profile"] = family_profile
        return {"Invoice No.": "1234", "Total": 100.0, "Invoice Date": "2026-04-15"}

    monkeypatch.setattr("invoice_router.pipeline.extract_with_ocr", _extract_with_profile)

    res = process_single_invoice(invoice_path, settings, config, db)

    assert res.route_used == Route.APPLY
    assert captured["family_profile"]["ocr"]["field_type_overrides"]["Invoice Date"] == "date"


def test_rejected_discovery_preserves_attempted_route_and_diagnostics(
    mock_pipeline_dependencies, monkeypatch
):
    settings, config, db, tmp_path, infer_template_mock, extract_with_ocr_mock = (
        mock_pipeline_dependencies
    )
    invoice_path = str(tmp_path / "test_invoice.png")

    monkeypatch.setattr(
        "invoice_router.pipeline.validate_invoice",
        lambda *_args, **_kwargs: ValidationResult(
            passed=False,
            score=0.4,
            matched_fields=0,
            mismatched_fields=3,
            errors=[
                "Missing extracted field: Invoice No.",
                "Mismatch on Total: expected '100.0', got '95.0'",
                "Line item count mismatch: expected 2, got 0",
            ],
        ),
    )

    res = process_single_invoice(invoice_path, settings, config, db)

    assert res.route_used == Route.REJECTED
    assert res.attempted_route == Route.DISCOVERY
    assert res.validation_passed is False
    assert res.diagnostics is not None
    assert res.diagnostics.attempted_route == Route.DISCOVERY
    assert res.diagnostics.validation_score == 0.4
    assert res.diagnostics.validation_errors == [
        "Missing extracted field: Invoice No.",
        "Mismatch on Total: expected '100.0', got '95.0'",
        "Line item count mismatch: expected 2, got 0",
    ]
    assert res.diagnostics.discovery_stage_status == "validation_failed_after_extract"
    assert res.diagnostics.discovery_mode == "heuristic"
    assert res.diagnostics.extracted_field_count == 2
    assert res.diagnostics.located_field_count == 2
    assert res.diagnostics.scalar_field_missing_count == 1
    assert res.diagnostics.scalar_field_mismatch_count == 1
    assert res.diagnostics.gt_line_item_count is None
    assert res.diagnostics.extracted_line_item_count == 0

    stored = db.get_result(invoice_path)
    assert stored is not None
    assert stored.route_used == Route.REJECTED
    assert stored.attempted_route == Route.DISCOVERY
    assert stored.diagnostics is not None
    assert stored.diagnostics.discovery_stage_status == "validation_failed_after_extract"
    assert stored.diagnostics.validation_score == 0.4


def test_rejected_apply_preserves_attempted_route_and_line_item_counts(
    mock_pipeline_dependencies, monkeypatch
):
    settings, config, db, tmp_path, infer_template_mock, extract_with_ocr_mock = (
        mock_pipeline_dependencies
    )
    invoice_path = str(tmp_path / "test_invoice.png")

    apply_record = FingerprintRecord(
        hash="known-fingerprint",
        layout_template={"pages": [{"page_index": 0, "fields": {}, "label_confirmation_set": []}]},
        page_fingerprints=[PageFingerprint(page_index=0, visual_hash=1, visual_hash_hex="0" * 16)],
        confidence=1.0,
        apply_count=0,
        reject_count=0,
        status=TemplateStatus.established,
        version="v3",
        created_at=get_utc_now(),
        last_used=get_utc_now(),
    )
    monkeypatch.setattr(
        "invoice_router.pipeline.lookup_fingerprint", lambda *_args, **_kwargs: (apply_record, 1.0)
    )
    monkeypatch.setattr(
        "invoice_router.pipeline.pair_ground_truth",
        lambda _path: {
            "Invoice No.": "1234",
            "Total": 100.0,
            "line_items": [{"sku": "A"}, {"sku": "B"}, {"sku": "C"}],
        },
    )
    monkeypatch.setattr(
        "invoice_router.pipeline.extract_with_ocr",
        lambda *_args, **_kwargs: {
            "Invoice No.": "1234",
            "Total": 95.0,
            "line_items": [{"sku": "A"}],
        },
    )
    monkeypatch.setattr(
        "invoice_router.pipeline.validate_invoice",
        lambda *_args, **_kwargs: ValidationResult(
            passed=False,
            score=0.6,
            matched_fields=1,
            mismatched_fields=2,
            errors=[
                "Mismatch on Total: expected '100.0', got '95.0'",
                "Line item count mismatch: expected 3, got 1",
            ],
        ),
    )

    res = process_single_invoice(invoice_path, settings, config, db)

    assert res.route_used == Route.REJECTED
    assert res.attempted_route == Route.APPLY
    assert res.diagnostics is not None
    assert res.diagnostics.gt_line_item_count == 3
    assert res.diagnostics.extracted_line_item_count == 1
    assert res.diagnostics.validation_score == 0.6

    stored = db.get_result(invoice_path)
    assert stored is not None
    assert stored.attempted_route == Route.APPLY
    assert stored.diagnostics is not None
    assert stored.diagnostics.gt_line_item_count == 3
    assert stored.diagnostics.extracted_line_item_count == 1


def test_discovery_uses_ocr_table_fallback_when_heuristic_returns_no_line_items(
    mock_pipeline_dependencies, monkeypatch
):
    settings, config, db, tmp_path, infer_template_mock, extract_with_ocr_mock = (
        mock_pipeline_dependencies
    )
    invoice_path = str(tmp_path / "test_invoice.png")

    template_with_table = {
        "version": "v3",
        "inference_method": "heuristic_discovery",
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "Invoice No.": {
                        "region": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.1},
                        "field_type": "string",
                        "anchor_direction": "right_of_label",
                    },
                },
                "table": {
                    "region": {"x": 0.1, "y": 0.4, "width": 0.8, "height": 0.3},
                    "columns": {"Description": {"x_left": 0.1, "x_right": 0.6}},
                    "header_row_y": 0.4,
                },
                "label_confirmation_set": [],
            }
        ],
    }
    monkeypatch.setattr(
        "invoice_router.pipeline.infer_template",
        lambda *_args, **_kwargs: (
            template_with_table,
            0.95,
            {"Invoice No.": "1234"},
            {"located_field_count": 1, "extracted_field_count": 1},
        ),
    )
    monkeypatch.setattr(
        "invoice_router.pipeline.extract_with_ocr",
        lambda *_args, **_kwargs: {"line_items": [{"Description": "Recovered Row"}]},
    )
    monkeypatch.setattr(
        "invoice_router.pipeline.validate_invoice",
        lambda *_args, **_kwargs: ValidationResult(
            passed=True,
            score=1.0,
            matched_fields=2,
            mismatched_fields=0,
            errors=[],
        ),
    )

    res = process_single_invoice(invoice_path, settings, config, db)

    assert res.validation_passed is True
    assert res.extracted_data["line_items"] == [{"Description": "Recovered Row"}]


def test_discovery_uses_heuristic_backend_without_external_inference(
    mock_pipeline_dependencies, monkeypatch
):
    settings, config, db, tmp_path, infer_template_mock, extract_with_ocr_mock = (
        mock_pipeline_dependencies
    )
    invoice_path = str(tmp_path / "test_invoice.png")

    from invoice_router.extraction.template_inference import infer_template as real_infer_template

    monkeypatch.setattr("invoice_router.pipeline.infer_template", real_infer_template)
    monkeypatch.setattr(
        "invoice_router.pipeline.pair_ground_truth",
        lambda _path: {
            "Invoice No.": "INV-42",
            "Date": "2026-04-14",
            "Total": "100.00",
        },
    )
    monkeypatch.setattr(
        "invoice_router.pipeline.run_full_page_ocr",
        lambda _img, _engine: [
            ("Invoice", 10, 10, 50, 10),
            ("No.", 65, 10, 25, 10),
            ("INV-42", 100, 10, 45, 10),
            ("Date", 10, 35, 35, 10),
            ("2026-04-14", 100, 35, 75, 10),
            ("Total", 55, 70, 35, 10),
            ("100.00", 100, 70, 45, 10),
        ],
    )
    monkeypatch.setattr(
        "invoice_router.pipeline.validate_invoice",
        lambda *_args, **_kwargs: ValidationResult(
            passed=True,
            score=1.0,
            matched_fields=3,
            mismatched_fields=0,
            errors=[],
        ),
    )
    res = process_single_invoice(invoice_path, settings, config, db)

    assert res.route_used == Route.DISCOVERY
    assert res.validation_passed is True
    assert res.provenance.inference_method == "heuristic:heuristic"
    assert res.diagnostics.discovery_mode == "heuristic"
    assert res.diagnostics.matched_label_count is not None


def test_native_pdf_apply_without_ground_truth_uses_text_extraction(
    mock_pipeline_dependencies, monkeypatch
):
    settings, config, db, tmp_path, infer_template_mock, extract_with_ocr_mock = (
        mock_pipeline_dependencies
    )
    invoice_path = str(tmp_path / "test_invoice.pdf")
    (tmp_path / "test_invoice.pdf").write_bytes(b"%PDF-1.4")

    monkeypatch.setattr("invoice_router.pipeline.pair_ground_truth", lambda _path: None)
    monkeypatch.setattr(
        "invoice_router.pipeline.build_document_context",
        lambda *_args, **_kwargs: DocumentContext(
            source_format=SourceFormat.pdf_text,
            document_family=DocumentFamily.invoice,
            country_code="US",
            currency_code="USD",
            extraction_strategy=ExtractionStrategy.native_pdf,
        ),
    )
    monkeypatch.setattr("invoice_router.pipeline.resolve_provider", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "invoice_router.pipeline.lookup_fingerprint", lambda *_args, **_kwargs: (None, 0.0)
    )
    native_extract = MagicMock(return_value={"invoice_number": "INV-1", "total": "$42.00"})
    monkeypatch.setattr("invoice_router.pipeline.extract_from_text_pdf", native_extract)

    res = process_single_invoice(invoice_path, settings, config, db)

    assert res.route_used == Route.APPLY
    assert res.validation_passed is None
    assert res.extracted_data == {"invoice_number": "INV-1", "total": "$42.00"}
    assert res.normalized_data["invoice_number"] == "INV-1"
    assert res.normalized_data["total"] == 42.0
    assert res.provenance.extraction_strategy == ExtractionStrategy.native_pdf
    assert native_extract.call_count == 1
    assert extract_with_ocr_mock.call_count == 0


def test_family_match_can_drive_apply_without_exact_fingerprint(
    mock_pipeline_dependencies, monkeypatch
):
    settings, config, db, tmp_path, infer_template_mock, extract_with_ocr_mock = (
        mock_pipeline_dependencies
    )
    invoice_path = str(tmp_path / "test_invoice.png")

    family_record = TemplateFamilyRecord(
        template_family_id="family-1",
        provider_name=None,
        document_family=DocumentFamily.invoice,
        stable_anchor_regions={"tokens": ["invoice", "total"]},
        anchor_summary={"page_count": 1, "aggregate_keywords": {"summary": ["total"]}},
        page_role_expectations=[PageRole.summary_page],
        created_at=get_utc_now(),
    )
    representative = FingerprintRecord(
        hash="family-representative",
        layout_template={"pages": [{"page_index": 0, "fields": {}, "label_confirmation_set": []}]},
        template_family_id="family-1",
        page_fingerprints=[
            PageFingerprint(
                page_index=0, visual_hash=1, visual_hash_hex="0" * 16, role=PageRole.summary_page
            )
        ],
        confidence=0.95,
        apply_count=0,
        reject_count=0,
        status=TemplateStatus.established,
        version="v3",
        created_at=get_utc_now(),
        last_used=get_utc_now(),
    )

    monkeypatch.setattr("invoice_router.pipeline.pair_ground_truth", lambda _path: None)
    monkeypatch.setattr(
        "invoice_router.pipeline.lookup_fingerprint", lambda *_args, **_kwargs: (None, 0.0)
    )
    monkeypatch.setattr(db, "get_active_template_families", lambda: [family_record])
    monkeypatch.setattr(
        "invoice_router.pipeline.lookup_template_family",
        lambda *_args, **_kwargs: (family_record, representative, 0.82, 1),
    )

    res = process_single_invoice(invoice_path, settings, config, db)

    assert res.route_used == Route.APPLY
    assert res.attempted_route == Route.APPLY
    assert res.provenance.extraction_strategy == ExtractionStrategy.provider_template
    assert res.template_family_id == "family-1"
    assert extract_with_ocr_mock.call_count == 1


def test_discovery_reports_heuristic_mode(mock_pipeline_dependencies, monkeypatch):
    settings, config, db, tmp_path, infer_template_mock, extract_with_ocr_mock = (
        mock_pipeline_dependencies
    )
    invoice_path = str(tmp_path / "test_invoice.png")

    res = process_single_invoice(invoice_path, settings, config, db)

    assert res.route_used == Route.DISCOVERY
    assert infer_template_mock.call_count == 1
    assert res.diagnostics.discovery_mode == "heuristic"
    assert res.diagnostics.discovered_field_count == 2


def test_process_single_invoice_namespaces_output_artifacts_by_parent_directory(
    mock_pipeline_dependencies,
):
    settings, config, db, tmp_path, infer_template_mock, extract_with_ocr_mock = (
        mock_pipeline_dependencies
    )
    invoice_a = tmp_path / "set_a" / "test_invoice.png"
    invoice_b = tmp_path / "set_b" / "test_invoice.png"
    invoice_a.parent.mkdir(parents=True, exist_ok=True)
    invoice_b.parent.mkdir(parents=True, exist_ok=True)
    invoice_a.write_bytes(b"fake_image_data")
    invoice_b.write_bytes(b"fake_image_data")

    process_single_invoice(str(invoice_a), settings, config, db)
    process_single_invoice(str(invoice_b), settings, config, db)

    output_root = tmp_path / "output"
    assert (output_root / "set_a" / "test_invoice.json").exists()
    assert (output_root / "set_b" / "test_invoice.json").exists()
