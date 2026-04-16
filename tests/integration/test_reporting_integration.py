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
from invoice_router.reporting import (
    build_process_run_summary,
    compare_analysis_runs,
    describe_template_family,
    record_analysis_run,
    suggest_template_family_merges,
    suggest_template_family_retirements,
    suggest_template_family_splits,
    suggest_template_family_updates,
    summarize_failure_modes,
    summarize_template_families,
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


def test_summarize_failure_modes_groups_by_route_and_error_type(tmp_path, postgres_database_url):
    db = FingerprintDB(postgres_database_url)
    dataset_dir = tmp_path / "invoices-small"
    dataset_dir.mkdir()

    discovery_one = ProcessingResult(
        invoice_path=str(dataset_dir / "disc-1.png"),
        fingerprint_hash="fp-disc-1",
        extracted_data={},
        ground_truth={"line_items": [{"sku": "A"}, {"sku": "B"}]},
        provenance=_sample_provenance(Route.REJECTED),
        validation_passed=False,
        route_used=Route.REJECTED,
        attempted_route=Route.DISCOVERY,
        diagnostics=ProcessingDiagnostics(
            attempted_route=Route.DISCOVERY,
            discovery_mode="heuristic",
            discovery_stage_status="locate_parse_failed",
            locate_error_category="discovery_locate_parse_error",
            validation_score=0.4,
            validation_errors=[
                "Missing extracted field: Invoice No.",
                "Line item count mismatch: expected 2, got 0",
            ],
            extracted_field_count=0,
            gt_line_item_count=2,
            extracted_line_item_count=0,
            scalar_field_missing_count=1,
            scalar_field_mismatch_count=0,
        ),
        image_quality_score=0.7,
        template_status_at_time=None,
        processed_at="2026-04-13T10:00:01Z",
    )
    discovery_two = ProcessingResult(
        invoice_path=str(dataset_dir / "disc-2.png"),
        fingerprint_hash="fp-disc-2",
        extracted_data={"Invoice No.": "123"},
        ground_truth={"Invoice No.": "456"},
        provenance=_sample_provenance(Route.REJECTED),
        validation_passed=False,
        route_used=Route.REJECTED,
        attempted_route=Route.DISCOVERY,
        diagnostics=ProcessingDiagnostics(
            attempted_route=Route.DISCOVERY,
            discovery_mode="heuristic",
            discovery_stage_status="validation_failed_after_extract",
            validation_score=0.76,
            validation_errors=["Mismatch on Invoice No.: expected '456', got '123'"],
            extracted_field_count=1,
            gt_line_item_count=0,
            extracted_line_item_count=0,
            scalar_field_missing_count=0,
            scalar_field_mismatch_count=1,
        ),
        image_quality_score=0.8,
        template_status_at_time=None,
        processed_at="2026-04-13T10:00:02Z",
    )
    apply_failure = ProcessingResult(
        invoice_path=str(dataset_dir / "apply-1.png"),
        fingerprint_hash="fp-apply-1",
        extracted_data={"Total": "9.99"},
        ground_truth={"Total": "10.99"},
        provenance=_sample_provenance(Route.REJECTED),
        validation_passed=False,
        route_used=Route.REJECTED,
        attempted_route=Route.APPLY,
        diagnostics=ProcessingDiagnostics(
            attempted_route=Route.APPLY,
            validation_score=0.7,
            validation_errors=["Mismatch on Total: expected '10.99', got '9.99'"],
            extracted_field_count=1,
            gt_line_item_count=0,
            extracted_line_item_count=0,
            scalar_field_missing_count=0,
            scalar_field_mismatch_count=1,
        ),
        image_quality_score=0.85,
        template_status_at_time=None,
        processed_at="2026-04-13T10:00:03Z",
    )

    for result in (discovery_one, discovery_two, apply_failure):
        db.store_result(result)

    summary = summarize_failure_modes(db, dataset_filter="invoices-small", discovery_threshold=0.95)

    assert summary["failed_total"] == 3
    assert summary["failed_discovery"] == 2
    assert summary["failed_apply"] == 1
    assert summary["attempted_route_counts"] == {"DISCOVERY": 2, "APPLY": 1}
    assert summary["discovery_mode_counts"] == {"heuristic": 2}
    assert summary["discovery_stage_status_counts"] == {
        "locate_parse_failed": 1,
        "validation_failed_after_extract": 1,
    }
    assert summary["discovery_locate_error_category_counts"] == {"discovery_locate_parse_error": 1}
    assert summary["discovery_extract_error_category_counts"] == {}
    assert summary["discovery_error_category_counts"] == {
        "missing_field": 1,
        "line_item_count_mismatch": 1,
        "field_mismatch": 1,
    }
    assert summary["discovery_line_item_mismatch_count"] == 1
    assert summary["discovery_score_band_counts"] == {"<0.5": 1, "0.5-0.8": 1}


def test_record_analysis_run_persists_run_and_invoice_details(tmp_path, postgres_database_url):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    discovery_pass = ProcessingResult(
        invoice_path=str(tmp_path / "invoices-small" / "disc-pass.png"),
        fingerprint_hash="fp-disc-pass",
        template_family_id="family-1",
        extracted_data={"Invoice No.": "123"},
        normalized_data={"invoiceNumber": "123"},
        ground_truth={"Invoice No.": "123"},
        provenance=_sample_provenance(Route.DISCOVERY),
        validation_passed=True,
        route_used=Route.DISCOVERY,
        attempted_route=Route.DISCOVERY,
        diagnostics=ProcessingDiagnostics(
            attempted_route=Route.DISCOVERY,
            discovery_stage_status="passed",
            validation_score=1.0,
        ),
        image_quality_score=0.9,
        template_status_at_time=None,
        processed_at="2026-04-14T10:00:00Z",
    )
    discovery_reject = ProcessingResult(
        invoice_path=str(tmp_path / "invoices-small" / "disc-reject.png"),
        fingerprint_hash="fp-disc-reject",
        template_family_id="family-2",
        extracted_data={"subtotal": "10.00"},
        normalized_data={"subtotal": 10.0},
        ground_truth={"subtotal": 12.0},
        provenance=_sample_provenance(Route.REJECTED),
        validation_passed=False,
        route_used=Route.REJECTED,
        attempted_route=Route.DISCOVERY,
        diagnostics=ProcessingDiagnostics(
            attempted_route=Route.DISCOVERY,
            discovery_stage_status="validation_failed_after_extract",
            validation_score=0.8,
            validation_errors=["Mismatch on subtotal: expected '12.0', got '10.0'"],
        ),
        image_quality_score=0.8,
        template_status_at_time=None,
        processed_at="2026-04-14T10:00:01Z",
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
        ],
        total_ms=1900,
        failure_summary={"failed_total": 1},
        discovery_only=False,
        discovery_mode="heuristic",
    )

    db_path = record_analysis_run(
        summary,
        [
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
        ],
        str(output_dir),
        command="invoice-router process ./invoices-small --reset",
        git_branch="codex/test",
        git_commit="abc123",
        config_snapshot={"discovery": {"strategy": "heuristic"}},
        notes="unit test",
        analysis_db_target=postgres_database_url,
    )

    client = PostgresClient(str(db_path))
    run_row = client.fetchone(
        "SELECT dataset, discovery_passed, discovery_rejected, command, git_branch, git_commit FROM runs"
    )
    assert run_row["dataset"] == "invoices-small"
    assert run_row["discovery_passed"] == 1
    assert run_row["discovery_rejected"] == 1
    assert run_row["command"] == "invoice-router process ./invoices-small --reset"
    assert run_row["git_branch"] == "codex/test"
    assert run_row["git_commit"] == "abc123"

    invoice_rows = client.fetchall(
        "SELECT invoice_name, template_family_id, status, validation_passed, error_categories_json, validation_errors_json FROM run_invoices ORDER BY invoice_name"
    )
    assert [row["invoice_name"] for row in invoice_rows] == ["disc-pass.png", "disc-reject.png"]
    assert [row["template_family_id"] for row in invoice_rows] == ["family-1", "family-2"]


def test_summarize_template_families_orders_review_queue(postgres_database_url):
    db = FingerprintDB(postgres_database_url)
    db.store_template_family(
        TemplateFamilyRecord(
            template_family_id="family-established",
            document_family=DocumentFamily.invoice,
            extraction_profile={
                "preferred_strategy": "provider_template",
                "ocr": {"field_type_overrides": {"Invoice Date": "date"}},
            },
            confidence=0.99,
            apply_count=8,
            reject_count=0,
            gt_confidence=0.98,
            gt_apply_count=6,
            status=TemplateStatus.established,
            created_at="2026-04-15T12:00:00Z",
        )
    )
    db.store_template_family(
        TemplateFamilyRecord(
            template_family_id="family-degraded",
            document_family=DocumentFamily.invoice,
            confidence=0.42,
            apply_count=2,
            reject_count=4,
            status=TemplateStatus.degraded,
            created_at="2026-04-15T12:00:00Z",
        )
    )
    db.store_template_family(
        TemplateFamilyRecord(
            template_family_id="family-big-unstable",
            document_family=DocumentFamily.invoice,
            confidence=0.18,
            apply_count=30,
            reject_count=170,
            status=TemplateStatus.degraded,
            created_at="2026-04-15T12:00:00Z",
        )
    )
    db.store_template_family(
        TemplateFamilyRecord(
            template_family_id="family-tiny-suspicious",
            document_family=DocumentFamily.invoice,
            extraction_profile={
                "preferred_strategy": "provider_template",
                "ocr": {
                    "field_type_overrides": {
                        "Invoice Date": "date",
                        "Invoice No": "string",
                        "Total": "currency",
                    }
                },
            },
            confidence=0.4,
            apply_count=1,
            reject_count=4,
            status=TemplateStatus.degraded,
            created_at="2026-04-15T12:00:00Z",
        )
    )

    summary = summarize_template_families(db)

    assert summary["total_families"] == 4
    assert summary["status_counts"]["established"] == 1
    assert summary["status_counts"]["degraded"] == 3
    assert summary["review_queue"][:3] == [
        "family-big-unstable",
        "family-tiny-suspicious",
        "family-degraded",
    ]
    assert summary["families"][0]["template_family_id"] == "family-big-unstable"
    assert summary["families"][0]["triage_class"] == "high_value_unstable"
    assert summary["families"][1]["triage_class"] == "suspiciously_tiny"
    assert summary["families"][-1]["triage_class"] == "stable"
    assert summary["families"][-1]["extraction_profile"]["field_override_count"] == 1
    assert summary["families"][-1]["example_count"] == 0
    assert summary["families"][-1]["gt_trust_qualified"] is True


def test_describe_template_family_includes_recent_activity(postgres_database_url, tmp_path):
    db = FingerprintDB(postgres_database_url)
    db.store_template_family(
        TemplateFamilyRecord(
            template_family_id="family-detail",
            document_family=DocumentFamily.invoice,
            provider_name="Acme Ltd",
            extraction_profile={"preferred_strategy": "provider_template"},
            status=TemplateStatus.degraded,
            created_at="2026-04-15T12:00:00Z",
        )
    )
    representative = FingerprintRecord(
        hash="fp-family-detail",
        layout_template={
            "pages": [
                {
                    "page_index": 0,
                    "fields": {
                        "Invoice Date": {"field_type": "string"},
                        "Total": {"field_type": "string"},
                    },
                    "label_confirmation_set": [],
                }
            ]
        },
        template_family_id="family-detail",
        page_fingerprints=[PageFingerprint(page_index=0, visual_hash=1, visual_hash_hex="0" * 16)],
        confidence=0.9,
        apply_count=3,
        reject_count=1,
        gt_apply_count=5,
        gt_confidence=0.99,
        status=TemplateStatus.established,
        created_at="2026-04-15T12:00:00Z",
    )
    db.store_fingerprint(
        representative,
        visual_hashes=["0" * 16],
        page_fingerprints=[p.model_dump() for p in representative.page_fingerprints],
    )
    db.add_template_family_example(
        TemplateFamilyExample(
            template_family_id="family-detail",
            fingerprint_hash="fp-1",
            invoice_path=str(tmp_path / "invoices-test" / "one.png"),
            example_metadata={"role": "seed"},
            created_at="2026-04-15T12:05:00Z",
        )
    )
    db.add_template_family_version(
        TemplateFamilyVersion(
            template_family_id="family-detail",
            version=1,
            family_snapshot={"status": "degraded"},
            change_reason="manual_review",
            created_at="2026-04-15T12:06:00Z",
        )
    )
    result = ProcessingResult(
        invoice_path=str(tmp_path / "invoices-test" / "one.png"),
        fingerprint_hash="fp-family-detail",
        template_family_id="family-detail",
        extracted_data={"invoiceNumber": "1"},
        normalized_data={"invoice_number": "1"},
        ground_truth={"invoiceNumber": "1"},
        provenance=_sample_provenance(Route.APPLY).model_copy(
            update={"healing_reprocessed": True, "determination_sources": ["healing"]}
        ),
        validation_passed=False,
        route_used=Route.REJECTED,
        attempted_route=Route.APPLY,
        diagnostics=ProcessingDiagnostics(
            attempted_route=Route.APPLY,
            validation_score=0.6,
            validation_errors=[
                "Mismatch on Invoice Date: expected '2026-04-15', got '2026-04-16'",
                "Mismatch on Total: expected '10.00', got '9.50'",
                "Line item count mismatch: expected 2, got 1",
            ],
            table_detected=False,
            line_item_source="row_fallback",
        ),
        image_quality_score=0.9,
        template_status_at_time=None,
        processed_at="2026-04-15T12:07:00Z",
    )
    db.store_result(result)
    db.store_result(
        ProcessingResult(
            invoice_path=str(tmp_path / "invoices-test" / "two.png"),
            fingerprint_hash="fp-family-detail",
            template_family_id="family-detail",
            extracted_data={"invoiceNumber": "2"},
            normalized_data={"invoice_number": "2"},
            ground_truth={"invoiceNumber": "2"},
            provenance=_sample_provenance(Route.APPLY),
            validation_passed=False,
            route_used=Route.REJECTED,
            attempted_route=Route.APPLY,
            diagnostics=ProcessingDiagnostics(
                attempted_route=Route.APPLY,
                validation_score=0.5,
                validation_errors=[
                    "Missing extracted field: Invoice Date",
                    "Line item count mismatch: expected 2, got 0",
                    "Mismatch on Total: expected '11.00', got '9.50'",
                ],
                table_detected=False,
                line_item_source="row_fallback",
            ),
            image_quality_score=0.9,
            template_status_at_time=None,
            processed_at="2026-04-15T12:08:00Z",
        )
    )

    detail = describe_template_family(db, "family-detail")

    assert detail is not None
    assert detail["template_family_id"] == "family-detail"
    assert detail["triage_class"] == "suspiciously_tiny"
    assert "micro-patch" in detail["triage_explanation"]
    assert detail["example_count"] == 1
    assert detail["version_count"] == 1
    assert detail["recent_outcome_counts"]["review"] == 2
    assert detail["recent_results"][0]["invoice_name"] == "two.png"
    assert "status is degraded" in detail["review_signals"]
    assert detail["gt_trust_qualified"] is False
    assert detail["representative"]["gt_trust_qualified"] is True
    assert detail["healed_attempt_count"] == 1
    suggestion_kinds = [item["kind"] for item in detail["suggestions"]]
    assert "date_ocr_boost" in suggestion_kinds
    assert "summary_amount_ocr_boost" in suggestion_kinds
    assert "line_item_table_tuning" in suggestion_kinds


def test_suggest_template_family_updates_builds_review_queue(postgres_database_url, tmp_path):
    db = FingerprintDB(postgres_database_url)
    db.store_template_family(
        TemplateFamilyRecord(
            template_family_id="family-suggest",
            document_family=DocumentFamily.invoice,
            status=TemplateStatus.degraded,
            created_at="2026-04-15T12:00:00Z",
        )
    )
    representative = FingerprintRecord(
        hash="fp-family-suggest",
        layout_template={
            "pages": [
                {
                    "page_index": 0,
                    "fields": {
                        "Invoice Date": {"field_type": "string"},
                        "Total Amount": {"field_type": "string"},
                    },
                    "label_confirmation_set": [],
                }
            ]
        },
        template_family_id="family-suggest",
        page_fingerprints=[PageFingerprint(page_index=0, visual_hash=1, visual_hash_hex="1" * 16)],
        confidence=0.8,
        status=TemplateStatus.established,
        created_at="2026-04-15T12:00:00Z",
    )
    db.store_fingerprint(
        representative,
        visual_hashes=["1" * 16],
        page_fingerprints=[p.model_dump() for p in representative.page_fingerprints],
    )
    for index in range(2):
        db.store_result(
            ProcessingResult(
                invoice_path=str(tmp_path / "invoices-test" / f"case-{index}.png"),
                fingerprint_hash=f"fp-{index}",
                template_family_id="family-suggest",
                extracted_data={"invoiceNumber": str(index)},
                normalized_data={"invoice_number": str(index)},
                ground_truth={"invoiceNumber": str(index)},
                provenance=_sample_provenance(Route.APPLY),
                validation_passed=False,
                route_used=Route.REJECTED,
                attempted_route=Route.APPLY,
                diagnostics=ProcessingDiagnostics(
                    attempted_route=Route.APPLY,
                    validation_score=0.55,
                    validation_errors=[
                        "Missing extracted field: Invoice Date",
                        "Mismatch on Total Amount: expected '10.00', got '9.50'",
                        "Line item count mismatch: expected 2, got 0",
                    ],
                    table_detected=False,
                    line_item_source="row_fallback",
                ),
                image_quality_score=0.9,
                template_status_at_time=None,
                processed_at=f"2026-04-15T12:0{index}:00Z",
            )
        )

    queue = suggest_template_family_updates(db)

    assert queue["queue_count"] == 3
    assert queue["family_ids"] == ["family-suggest"]
    kinds = [item["kind"] for item in queue["suggestions"]]
    assert "date_ocr_boost" in kinds
    assert "summary_amount_ocr_boost" in kinds
    assert "line_item_table_tuning" in kinds
    date_suggestion = next(
        item for item in queue["suggestions"] if item["kind"] == "date_ocr_boost"
    )
    assert date_suggestion["profile_patch"]["ocr"]["field_type_overrides"]["Invoice Date"] == "date"


def test_suggest_template_family_splits_returns_structural_cluster(postgres_database_url, tmp_path):
    db = FingerprintDB(postgres_database_url)
    db.store_template_family(
        TemplateFamilyRecord(
            template_family_id="family-split",
            document_family=DocumentFamily.invoice,
            status=TemplateStatus.established,
            created_at="2026-04-15T12:00:00Z",
        )
    )

    def _store_family_fp(hash_value, field_name, header_token, summary_label, created_at):
        record = FingerprintRecord(
            hash=hash_value,
            layout_template={
                "pages": [
                    {
                        "page_index": 0,
                        "fields": {field_name: {"field_type": "string"}},
                        "label_confirmation_set": [],
                    }
                ]
            },
            template_family_id="family-split",
            page_fingerprints=[
                PageFingerprint(
                    page_index=0,
                    visual_hash=1,
                    visual_hash_hex=(hash_value[-1] * 16),
                    role=PageRole.header_page,
                    stable_anchor_signature={
                        "header_tokens": [header_token],
                        "summary_labels": [summary_label] if summary_label else [],
                        "footer_tokens": [],
                        "keyword_hits": {"summary": [summary_label]} if summary_label else {},
                    },
                )
            ],
            confidence=0.9,
            apply_count=3,
            status=TemplateStatus.established,
            created_at=created_at,
        )
        db.store_fingerprint(
            record,
            visual_hashes=[hash_value[-1] * 16],
            page_fingerprints=[p.model_dump() for p in record.page_fingerprints],
        )

    _store_family_fp("fp-split-a1", "Invoice Date", "invoice date", "", "2026-04-15T12:00:01Z")
    _store_family_fp("fp-split-a2", "Invoice Date", "invoice date", "", "2026-04-15T12:00:02Z")
    _store_family_fp("fp-split-b1", "Total Due", "amount due", "total due", "2026-04-15T12:00:03Z")
    _store_family_fp("fp-split-b2", "Total Due", "amount due", "total due", "2026-04-15T12:00:04Z")

    for index, fingerprint_hash in enumerate(("fp-split-b1", "fp-split-b2"), start=1):
        invoice_path = tmp_path / f"split-{index}.png"
        invoice_path.write_bytes(b"img")
        db.add_template_family_example(
            TemplateFamilyExample(
                template_family_id="family-split",
                fingerprint_hash=fingerprint_hash,
                invoice_path=str(invoice_path),
                created_at=f"2026-04-15T12:00:0{index + 4}Z",
            )
        )

    queue = suggest_template_family_splits(db)

    assert queue["queue_count"] == 1
    assert queue["family_ids"] == ["family-split"]
    suggestion = queue["suggestions"][0]
    assert suggestion["template_family_id"] == "family-split"
    assert suggestion["fingerprint_count"] == 2
    assert suggestion["example_count"] == 2
    assert suggestion["proposed_family_id"] == "family-split-split-1"
    assert "total due" in suggestion["structural_signals"]["anchor_tokens"]
    assert set(suggestion["fingerprint_hashes"]) == {"fp-split-b1", "fp-split-b2"}


def test_suggest_template_family_merges_returns_duplicate_candidate(postgres_database_url):
    db = FingerprintDB(postgres_database_url)
    db.store_template_family(
        TemplateFamilyRecord(
            template_family_id="family-merge-strong",
            provider_name="Acme Ltd",
            country_code="GB",
            document_family=DocumentFamily.invoice,
            page_role_expectations=[PageRole.header_page],
            anchor_summary={"aggregate_keywords": {"invoice_number": ["invoice date"]}},
            extraction_profile={
                "preferred_strategy": "provider_template",
                "table": {"enabled": False},
            },
            status=TemplateStatus.established,
            confidence=0.98,
            apply_count=8,
            created_at="2026-04-15T12:40:00Z",
        )
    )
    db.store_template_family(
        TemplateFamilyRecord(
            template_family_id="family-merge-weak",
            provider_name="Acme Ltd",
            country_code="GB",
            document_family=DocumentFamily.invoice,
            page_role_expectations=[PageRole.header_page],
            anchor_summary={"aggregate_keywords": {"invoice_number": ["invoice date"]}},
            extraction_profile={
                "preferred_strategy": "provider_template",
                "table": {"enabled": False},
            },
            status=TemplateStatus.provisional,
            confidence=0.72,
            apply_count=1,
            created_at="2026-04-15T12:41:00Z",
        )
    )

    def _store_merge_fp(hash_value, family_id, created_at):
        record = FingerprintRecord(
            hash=hash_value,
            layout_template={
                "pages": [
                    {
                        "page_index": 0,
                        "fields": {
                            "Invoice Date": {"field_type": "date"},
                            "Invoice Number": {"field_type": "string"},
                        },
                        "label_confirmation_set": [],
                    }
                ]
            },
            template_family_id=family_id,
            page_fingerprints=[
                PageFingerprint(
                    page_index=0,
                    visual_hash=6,
                    visual_hash_hex=hash_value[-1] * 16,
                    role=PageRole.header_page,
                    stable_anchor_signature={
                        "header_tokens": ["invoice date", "invoice number"],
                        "summary_labels": [],
                        "footer_tokens": [],
                        "keyword_hits": {"invoice_number": ["invoice date"]},
                    },
                )
            ],
            confidence=0.9,
            apply_count=3,
            status=TemplateStatus.established,
            created_at=created_at,
        )
        db.store_fingerprint(
            record,
            visual_hashes=[hash_value[-1] * 16],
            page_fingerprints=[p.model_dump() for p in record.page_fingerprints],
        )

    _store_merge_fp("fp-merge-1", "family-merge-strong", "2026-04-15T12:42:00Z")
    _store_merge_fp("fp-merge-2", "family-merge-weak", "2026-04-15T12:43:00Z")

    queue = suggest_template_family_merges(db)

    assert queue["queue_count"] == 1
    suggestion = queue["suggestions"][0]
    assert suggestion["source_family_id"] == "family-merge-weak"
    assert suggestion["target_family_id"] == "family-merge-strong"
    assert suggestion["similarity"] >= 0.84
    assert "invoice date" in suggestion["shared_signals"]["anchor_tokens"]


def test_suggest_template_family_retirements_returns_dead_family(postgres_database_url):
    db = FingerprintDB(postgres_database_url)
    db.store_template_family(
        TemplateFamilyRecord(
            template_family_id="family-retire-me",
            document_family=DocumentFamily.invoice,
            status=TemplateStatus.degraded,
            confidence=0.3,
            apply_count=0,
            reject_count=4,
            created_at="2026-04-15T12:50:00Z",
        )
    )
    record = FingerprintRecord(
        hash="fp-retire-me",
        layout_template={"pages": [{"fields": {"Invoice Date": {"field_type": "date"}}}]},
        template_family_id="family-retire-me",
        page_fingerprints=[PageFingerprint(page_index=0, visual_hash=7, visual_hash_hex="7" * 16)],
        confidence=0.3,
        reject_count=4,
        status=TemplateStatus.degraded,
        created_at="2026-04-15T12:51:00Z",
    )
    db.store_fingerprint(
        record,
        visual_hashes=["7" * 16],
        page_fingerprints=[p.model_dump() for p in record.page_fingerprints],
    )

    queue = suggest_template_family_retirements(db)

    assert queue["queue_count"] == 1
    suggestion = queue["suggestions"][0]
    assert suggestion["template_family_id"] == "family-retire-me"
    assert suggestion["kind"] == "repeated_rejection"
    assert suggestion["active_fingerprint_count"] == 1
    assert suggestion["reject_count"] == 4


def test_compare_analysis_runs_reports_regressions_and_improvements(
    tmp_path, postgres_database_url
):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    baseline_pass = ProcessingResult(
        invoice_path=str(tmp_path / "invoices-test" / "one.png"),
        fingerprint_hash="fp-one-a",
        extracted_data={"invoiceNumber": "1"},
        normalized_data={"invoiceNumber": "1"},
        ground_truth={"invoiceNumber": "1"},
        provenance=_sample_provenance(Route.DISCOVERY),
        validation_passed=True,
        route_used=Route.DISCOVERY,
        attempted_route=Route.DISCOVERY,
        diagnostics=ProcessingDiagnostics(
            attempted_route=Route.DISCOVERY, discovery_stage_status="passed", validation_score=1.0
        ),
        image_quality_score=0.9,
        template_status_at_time=None,
        processed_at="2026-04-14T10:00:00Z",
    )
    baseline_fail = ProcessingResult(
        invoice_path=str(tmp_path / "invoices-test" / "two.png"),
        fingerprint_hash="fp-two-a",
        extracted_data={"subtotal": "10.00"},
        normalized_data={"subtotal": 10.0},
        ground_truth={"subtotal": 12.0},
        provenance=_sample_provenance(Route.REJECTED),
        validation_passed=False,
        route_used=Route.REJECTED,
        attempted_route=Route.DISCOVERY,
        diagnostics=ProcessingDiagnostics(
            attempted_route=Route.DISCOVERY,
            discovery_stage_status="validation_failed_after_extract",
            validation_score=0.7,
            validation_errors=["Mismatch on subtotal: expected '12.0', got '10.0'"],
        ),
        image_quality_score=0.8,
        template_status_at_time=None,
        processed_at="2026-04-14T10:00:01Z",
    )
    summary_a = build_process_run_summary(
        input_dir=str(tmp_path / "invoices-test"),
        results=[
            {
                "name": "one.png",
                "route": "DISCOVERY",
                "status": "Passed",
                "elapsed_ms": 1000,
                "result": baseline_pass,
                "error": None,
            },
            {
                "name": "two.png",
                "route": "REJECTED",
                "status": "Failed/Rejected",
                "elapsed_ms": 900,
                "result": baseline_fail,
                "error": None,
            },
        ],
        total_ms=1900,
        discovery_mode="heuristic",
    )
    db_path = record_analysis_run(
        summary_a,
        [
            {
                "name": "one.png",
                "route": "DISCOVERY",
                "status": "Passed",
                "elapsed_ms": 1000,
                "result": baseline_pass,
                "error": None,
            },
            {
                "name": "two.png",
                "route": "REJECTED",
                "status": "Failed/Rejected",
                "elapsed_ms": 900,
                "result": baseline_fail,
                "error": None,
            },
        ],
        str(output_dir),
        analysis_db_target=postgres_database_url,
    )

    candidate_fail = ProcessingResult(
        invoice_path=str(tmp_path / "invoices-test" / "one.png"),
        fingerprint_hash="fp-one-b",
        extracted_data={"invoiceNumber": "X"},
        normalized_data={"invoiceNumber": "X"},
        ground_truth={"invoiceNumber": "1"},
        provenance=_sample_provenance(Route.REJECTED),
        validation_passed=False,
        route_used=Route.REJECTED,
        attempted_route=Route.DISCOVERY,
        diagnostics=ProcessingDiagnostics(
            attempted_route=Route.DISCOVERY,
            discovery_stage_status="validation_failed_after_extract",
            validation_score=0.5,
            validation_errors=["Mismatch on invoiceNumber: expected '1', got 'X'"],
        ),
        image_quality_score=0.8,
        template_status_at_time=None,
        processed_at="2026-04-14T10:05:00Z",
    )
    candidate_pass = ProcessingResult(
        invoice_path=str(tmp_path / "invoices-test" / "two.png"),
        fingerprint_hash="fp-two-b",
        extracted_data={"subtotal": "12.00"},
        normalized_data={"subtotal": 12.0},
        ground_truth={"subtotal": 12.0},
        provenance=_sample_provenance(Route.DISCOVERY),
        validation_passed=True,
        route_used=Route.DISCOVERY,
        attempted_route=Route.DISCOVERY,
        diagnostics=ProcessingDiagnostics(
            attempted_route=Route.DISCOVERY, discovery_stage_status="passed", validation_score=1.0
        ),
        image_quality_score=0.9,
        template_status_at_time=None,
        processed_at="2026-04-14T10:05:01Z",
    )
    summary_b = build_process_run_summary(
        input_dir=str(tmp_path / "invoices-test"),
        results=[
            {
                "name": "one.png",
                "route": "REJECTED",
                "status": "Failed/Rejected",
                "elapsed_ms": 1000,
                "result": candidate_fail,
                "error": None,
            },
            {
                "name": "two.png",
                "route": "DISCOVERY",
                "status": "Passed",
                "elapsed_ms": 900,
                "result": candidate_pass,
                "error": None,
            },
        ],
        total_ms=1900,
        discovery_mode="heuristic",
    )
    record_analysis_run(
        summary_b,
        [
            {
                "name": "one.png",
                "route": "REJECTED",
                "status": "Failed/Rejected",
                "elapsed_ms": 1000,
                "result": candidate_fail,
                "error": None,
            },
            {
                "name": "two.png",
                "route": "DISCOVERY",
                "status": "Passed",
                "elapsed_ms": 900,
                "result": candidate_pass,
                "error": None,
            },
        ],
        str(output_dir),
        analysis_db_target=postgres_database_url,
    )

    comparison = compare_analysis_runs(str(db_path), "invoices-test")

    assert [item["invoice_name"] for item in comparison["regressions"]] == ["one.png"]
    assert [item["invoice_name"] for item in comparison["improvements"]] == ["two.png"]
