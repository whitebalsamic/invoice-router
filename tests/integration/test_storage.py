from invoice_router.infrastructure.persistence.postgres import PostgresClient
from invoice_router.infrastructure.persistence.storage import FingerprintDB
from invoice_router.models import (
    DocumentFamily,
    FingerprintRecord,
    PageFingerprint,
    PageRole,
    ProcessingDiagnostics,
    ProcessingResult,
    Provenance,
    Route,
    TemplateFamilyExample,
    TemplateFamilyRecord,
    TemplateFamilyVersion,
    TemplateStatus,
)


def _sample_provenance():
    return Provenance(
        request_id="req-1",
        route=Route.DISCOVERY,
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
    )


def test_store_and_load_processing_diagnostics(tmp_path, postgres_database_url):
    db = FingerprintDB(postgres_database_url)
    result = ProcessingResult(
        invoice_path=str(tmp_path / "invoices-small" / "inv-1.png"),
        fingerprint_hash="fp-1",
        extracted_data={"Invoice No.": "1234", "line_items": [{"sku": "A"}]},
        normalized_data={"invoice_number": "1234", "line_item_count": 1},
        ground_truth={"Invoice No.": "1234", "line_items": [{"sku": "A"}, {"sku": "B"}]},
        provenance=_sample_provenance(),
        validation_passed=False,
        route_used=Route.REJECTED,
        attempted_route=Route.DISCOVERY,
        diagnostics=ProcessingDiagnostics(
            attempted_route=Route.DISCOVERY,
            discovery_mode="heuristic",
            discovery_stage_status="validation_failed_after_extract",
            locate_error_category=None,
            extract_error_category=None,
            locate_raw_response='{"fields":[]}',
            extract_raw_response='{"fields":{}}',
            validation_score=0.72,
            validation_errors=["Line item count mismatch: expected 2, got 1"],
            extracted_field_count=2,
            gt_line_item_count=2,
            extracted_line_item_count=1,
            scalar_field_missing_count=0,
            scalar_field_mismatch_count=0,
            discovered_field_count=4,
            located_field_count=4,
            label_confirmation_count=1,
        ),
        image_quality_score=0.9,
        template_status_at_time=None,
        processed_at="2026-04-13T10:00:01Z",
    )

    db.store_result(result)
    loaded = db.get_result(result.invoice_path)

    assert loaded is not None
    assert loaded.attempted_route == Route.DISCOVERY
    assert loaded.diagnostics is not None
    assert loaded.diagnostics.discovery_mode == "heuristic"
    assert loaded.diagnostics.discovery_stage_status == "validation_failed_after_extract"
    assert loaded.diagnostics.validation_score == 0.72
    assert loaded.diagnostics.gt_line_item_count == 2
    assert loaded.diagnostics.extracted_line_item_count == 1
    assert loaded.diagnostics.locate_raw_response == '{"fields":[]}'
    assert loaded.normalized_data == {"invoice_number": "1234", "line_item_count": 1}


def test_legacy_rows_without_diagnostics_still_deserialize(tmp_path, postgres_database_url):
    db = FingerprintDB(postgres_database_url)
    client = PostgresClient(postgres_database_url)
    client.execute("DELETE FROM processing_results")
    client.execute(
        """
        INSERT INTO processing_results (
            invoice_path, fingerprint_hash, extracted_data, ground_truth,
            ground_truth_valid, provenance, validation_passed, route_used,
            image_quality_score, template_status_at_time, processed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(tmp_path / "invoices-test" / "legacy.png"),
            "legacy-fp",
            '{"Invoice No.": "1234"}',
            '{"Invoice No.": "1234"}',
            1,
            _sample_provenance().model_dump_json(),
            1,
            Route.DISCOVERY.value,
            0.88,
            None,
            "2026-04-13T10:00:01Z",
        ),
    )

    loaded = db.get_result(str(tmp_path / "invoices-test" / "legacy.png"))

    assert loaded is not None
    assert loaded.route_used == Route.DISCOVERY
    assert loaded.attempted_route is None
    assert loaded.diagnostics is None


def test_fingerprint_db_requires_postgres_url():
    try:
        FingerprintDB("/tmp/test.db")
    except ValueError as exc:
        assert "requires a Postgres DATABASE_URL" in str(exc)
    else:
        raise AssertionError("Expected FingerprintDB to reject non-Postgres targets")


def test_store_and_load_template_family(postgres_database_url):
    db = FingerprintDB(postgres_database_url)
    family = TemplateFamilyRecord(
        template_family_id="family-acme-001",
        provider_name="Acme Ltd",
        country_code="GB",
        document_family=DocumentFamily.invoice,
        stable_anchor_regions={"seller_block": {"page": 0, "x": 0.1}},
        page_role_expectations=[PageRole.header_page, PageRole.summary_page],
        summary_area_anchors={"total": {"page": 0, "y": 0.82}},
        variable_region_masks=[{"page": 0, "region": "line_items"}],
        extraction_profile={
            "preferred_strategy": "provider_template",
            "table": {"enabled": True, "ocr_engine": "paddle"},
        },
        confidence=0.91,
        gt_confidence=0.97,
        gt_apply_count=6,
        gt_reject_count=1,
        status=TemplateStatus.established,
        created_at="2026-04-15T10:00:00Z",
        updated_at="2026-04-15T10:05:00Z",
    )

    db.store_template_family(family)
    loaded = db.get_template_family("family-acme-001")

    assert loaded is not None
    assert loaded.template_family_id == family.template_family_id
    assert loaded.document_family == DocumentFamily.invoice
    assert loaded.page_role_expectations == [PageRole.header_page, PageRole.summary_page]
    assert loaded.summary_area_anchors == {"total": {"page": 0, "y": 0.82}}
    assert loaded.extraction_profile["table"]["enabled"] is True
    assert loaded.apply_count == 0
    assert loaded.reject_count == 0
    assert loaded.gt_apply_count == 6
    assert loaded.gt_reject_count == 1
    assert loaded.gt_confidence == 0.97
    assert loaded.status == TemplateStatus.established


def test_link_fingerprint_to_family_and_store_examples(postgres_database_url):
    db = FingerprintDB(postgres_database_url)
    family = TemplateFamilyRecord(
        template_family_id="family-link-001",
        document_family=DocumentFamily.invoice,
        created_at="2026-04-15T11:00:00Z",
    )
    db.store_template_family(family)

    fingerprint = FingerprintRecord(
        hash="fp-family-1",
        layout_template={"pages": [{"fields": {}}]},
        page_fingerprints=[
            PageFingerprint(
                page_index=0, visual_hash=123, visual_hash_hex="7b", role=PageRole.header_page
            )
        ],
        created_at="2026-04-15T11:00:01Z",
    )
    db.store_fingerprint(
        fingerprint,
        visual_hashes=["7b"],
        page_fingerprints=[p.model_dump() for p in fingerprint.page_fingerprints],
    )
    db.link_fingerprint_to_family("fp-family-1", "family-link-001")

    stored_fingerprint = next(
        record for record in db.get_all_active_fingerprints() if record.hash == "fp-family-1"
    )
    assert stored_fingerprint.template_family_id == "family-link-001"

    example = TemplateFamilyExample(
        template_family_id="family-link-001",
        fingerprint_hash="fp-family-1",
        invoice_path="/tmp/invoices-small/example-1.png",
        example_metadata={"role": "seed"},
        created_at="2026-04-15T11:00:02Z",
    )
    example_id = db.add_template_family_example(example)
    versions_id = db.add_template_family_version(
        TemplateFamilyVersion(
            template_family_id="family-link-001",
            version=1,
            family_snapshot={"confidence": 0.5},
            change_reason="initial import",
            created_at="2026-04-15T11:00:03Z",
        )
    )

    examples = db.get_template_family_examples("family-link-001")
    versions = db.get_template_family_versions("family-link-001")

    assert example_id > 0
    assert versions_id > 0
    assert len(examples) == 1
    assert examples[0].fingerprint_hash == "fp-family-1"
    assert examples[0].example_metadata == {"role": "seed"}
    assert len(versions) == 1
    assert versions[0].version == 1
    assert versions[0].family_snapshot == {"confidence": 0.5}


def test_relink_family_members_updates_examples_and_processing_results(
    tmp_path, postgres_database_url
):
    db = FingerprintDB(postgres_database_url)
    source_family = TemplateFamilyRecord(
        template_family_id="family-source",
        document_family=DocumentFamily.invoice,
        created_at="2026-04-15T12:00:00Z",
    )
    target_family = TemplateFamilyRecord(
        template_family_id="family-target",
        document_family=DocumentFamily.invoice,
        created_at="2026-04-15T12:05:00Z",
    )
    db.store_template_family(source_family)
    db.store_template_family(target_family)

    fingerprint = FingerprintRecord(
        hash="fp-family-source",
        layout_template={"pages": [{"fields": {"Invoice No.": {"field_type": "string"}}}]},
        template_family_id="family-source",
        page_fingerprints=[
            PageFingerprint(
                page_index=0,
                visual_hash=123,
                visual_hash_hex="7b",
                role=PageRole.header_page,
                stable_anchor_signature={
                    "header_tokens": ["invoice"],
                    "summary_labels": [],
                    "footer_tokens": [],
                },
            )
        ],
        created_at="2026-04-15T12:00:01Z",
    )
    db.store_fingerprint(
        fingerprint,
        visual_hashes=["7b"],
        page_fingerprints=[p.model_dump() for p in fingerprint.page_fingerprints],
    )

    invoice_path = tmp_path / "invoice.png"
    invoice_path.write_bytes(b"img")
    db.add_template_family_example(
        TemplateFamilyExample(
            template_family_id="family-source",
            fingerprint_hash="fp-family-source",
            invoice_path=str(invoice_path),
            example_metadata={"role": "seed"},
            created_at="2026-04-15T12:00:02Z",
        )
    )
    result = ProcessingResult(
        invoice_path=str(invoice_path),
        fingerprint_hash="fp-family-source",
        template_family_id="family-source",
        extracted_data={"Invoice No.": "123"},
        ground_truth={"Invoice No.": "123"},
        provenance=Provenance(
            request_id="req-family-source",
            route=Route.APPLY,
            fingerprint_hash="fp-family-source",
            template_family_id="family-source",
            template_status_at_time=None,
            template_confidence_at_time=None,
            ocr_engine="tesseract",
            inference_method="test",
            image_quality_score=[0.9],
            quality_flag=False,
            input_document_hash="doc-hash",
            per_page_hashes=["page-hash"],
            extraction_output_hash="out-hash",
            started_at="2026-04-15T12:00:03Z",
            completed_at="2026-04-15T12:00:04Z",
            latency_ms=1000,
        ),
        validation_passed=True,
        route_used=Route.APPLY,
        attempted_route=Route.APPLY,
        diagnostics=ProcessingDiagnostics(
            attempted_route=Route.APPLY,
            template_family_id="family-source",
            validation_score=0.96,
        ),
        image_quality_score=0.9,
        template_status_at_time=None,
        processed_at="2026-04-15T12:00:04Z",
    )
    db.store_result(result)

    moved_fingerprints = db.relink_fingerprints_to_family(["fp-family-source"], "family-target")
    moved_examples = db.relink_template_family_examples(
        "family-source",
        "family-target",
        fingerprint_hashes=["fp-family-source"],
        invoice_paths=[str(invoice_path)],
    )
    moved_results = db.relink_processing_results_to_family(
        "family-source",
        "family-target",
        fingerprint_hashes=["fp-family-source"],
    )

    reloaded_result = db.get_result(str(invoice_path))
    target_examples = db.get_template_family_examples("family-target")
    stored_fingerprint = next(
        record for record in db.get_all_active_fingerprints() if record.hash == "fp-family-source"
    )

    assert moved_fingerprints == 1
    assert moved_examples == 1
    assert moved_results == 1
    assert stored_fingerprint.template_family_id == "family-target"
    assert len(target_examples) == 1
    assert target_examples[0].template_family_id == "family-target"
    assert reloaded_result is not None
    assert reloaded_result.template_family_id == "family-target"
    assert reloaded_result.provenance is not None
    assert reloaded_result.provenance.template_family_id == "family-target"
    assert reloaded_result.diagnostics is not None
    assert reloaded_result.diagnostics.template_family_id == "family-target"


def test_active_template_families_and_retired_fingerprints_are_filtered(postgres_database_url):
    db = FingerprintDB(postgres_database_url)
    db.store_template_family(
        TemplateFamilyRecord(
            template_family_id="family-active",
            document_family=DocumentFamily.invoice,
            status=TemplateStatus.established,
            created_at="2026-04-15T12:10:00Z",
        )
    )
    db.store_template_family(
        TemplateFamilyRecord(
            template_family_id="family-retired",
            document_family=DocumentFamily.invoice,
            status=TemplateStatus.retired,
            created_at="2026-04-15T12:11:00Z",
        )
    )
    active_fp = FingerprintRecord(
        hash="fp-active",
        layout_template={"pages": [{"fields": {}}]},
        template_family_id="family-active",
        page_fingerprints=[PageFingerprint(page_index=0, visual_hash=1, visual_hash_hex="1" * 16)],
        status=TemplateStatus.established,
        created_at="2026-04-15T12:12:00Z",
    )
    retired_fp = FingerprintRecord(
        hash="fp-retire-me",
        layout_template={"pages": [{"fields": {}}]},
        template_family_id="family-active",
        page_fingerprints=[PageFingerprint(page_index=0, visual_hash=2, visual_hash_hex="2" * 16)],
        status=TemplateStatus.provisional,
        created_at="2026-04-15T12:13:00Z",
    )
    db.store_fingerprint(
        active_fp,
        visual_hashes=["1" * 16],
        page_fingerprints=[p.model_dump() for p in active_fp.page_fingerprints],
    )
    db.store_fingerprint(
        retired_fp,
        visual_hashes=["2" * 16],
        page_fingerprints=[p.model_dump() for p in retired_fp.page_fingerprints],
    )

    retired_count = db.retire_fingerprints(["fp-retire-me"])

    active_families = db.get_active_template_families()
    active_fingerprints = db.get_all_active_fingerprints()

    assert retired_count == 1
    assert [family.template_family_id for family in active_families] == ["family-active"]
    assert [record.hash for record in active_fingerprints] == ["fp-active"]
