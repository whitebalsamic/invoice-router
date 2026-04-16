from invoice_router.config import AppConfig
from invoice_router.domain.templates.routing import (
    determine_extraction_strategy,
    hamming_distance,
    levenshtein_distance,
    lookup_fingerprint,
    lookup_template_family,
    route_invoice,
)
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
)


def test_hamming_distance():
    # 1 bit difference
    assert hamming_distance("0000000000000000", "0000000000000001") == 1
    # 4 bits difference
    assert hamming_distance("0000000000000000", "000000000000000F") == 4
    # Identical
    assert hamming_distance("FFFFFFFFFFFFFFFF", "FFFFFFFFFFFFFFFF") == 0


def test_levenshtein_distance():
    assert levenshtein_distance("Invoice No.", "Invoice No.") == 0
    assert levenshtein_distance("Invoice No.", "Invoice Num") == 2
    assert levenshtein_distance("Date:", "Date") == 1


def test_lookup_fingerprint_hamming_threshold():
    # Mock config
    config = AppConfig(
        **{
            "validation": {
                "apply_threshold": 0.9,
                "discovery_threshold": 0.95,
                "jaccard_threshold": 0.85,
            },
            "fingerprinting": {"visual_hash_hamming_threshold": 10},
            "template_lifecycle": {
                "establish_min_count": 5,
                "establish_min_confidence": 0.95,
                "degradation_threshold": 0.85,
                "degradation_window": 10,
                "rediscovery_attempts": 3,
            },
            "quality": {
                "blur_threshold": 100.0,
                "contrast_threshold": 0.3,
                "quality_threshold": 0.5,
                "quality_region_buffer_multiplier": 2,
            },
            "processing": {"batch_size": 50, "worker_concurrency": 4},
            "region_buffer_pixels": 5,
            "ocr": {"single_field_engine": "tesseract", "table_engine": "paddle"},
            "table_detection": {
                "min_line_span_fraction": 0.4,
                "column_gap_px": 20,
                "row_gap_multiplier": 1.5,
            },
            "discovery": {
                "inference_confidence_threshold": 0.6,
                "label_confirmation_threshold": 0.7,
                "label_position_tolerance": 0.05,
            },
            "field_mapping": {},
        }
    )

    # 0 bits difference
    p1 = PageFingerprint(page_index=0, visual_hash=0, visual_hash_hex="0000000000000000")

    # Create record with 0 bits diff
    r1 = FingerprintRecord(
        hash="hash1",
        layout_template={"pages": []},
        page_fingerprints=[p1],
        confidence=1.0,
        status=TemplateStatus.established,
        created_at="2026-01-01T00:00:00Z",
    )

    # Match against itself
    match, conf = lookup_fingerprint([p1], [[]], [(1000, 1000)], [r1], config)
    assert match is not None
    assert match.hash == "hash1"
    assert conf == 1.0

    # Test with distance > 10 (e.g. 16 bits difference "FFFF")
    p2 = PageFingerprint(page_index=0, visual_hash=0xFFFF, visual_hash_hex="000000000000FFFF")
    match_fail, _ = lookup_fingerprint([p2], [[]], [(1000, 1000)], [r1], config)
    assert match_fail is None


def test_lookup_fingerprint_requires_exact_visual_match_when_template_has_no_labels():
    config = AppConfig(
        **{
            "validation": {
                "apply_threshold": 0.9,
                "discovery_threshold": 0.95,
                "jaccard_threshold": 0.85,
            },
            "fingerprinting": {"visual_hash_hamming_threshold": 10},
            "template_lifecycle": {
                "establish_min_count": 5,
                "establish_min_confidence": 0.95,
                "degradation_threshold": 0.85,
                "degradation_window": 10,
                "rediscovery_attempts": 3,
            },
            "quality": {
                "blur_threshold": 100.0,
                "contrast_threshold": 0.3,
                "quality_threshold": 0.5,
                "quality_region_buffer_multiplier": 2,
            },
            "processing": {"batch_size": 50, "worker_concurrency": 8},
            "region_buffer_pixels": 5,
            "ocr": {"single_field_engine": "tesseract", "table_engine": "paddle"},
            "table_detection": {
                "min_line_span_fraction": 0.4,
                "column_gap_px": 20,
                "row_gap_multiplier": 1.5,
            },
            "discovery": {
                "inference_confidence_threshold": 0.6,
                "label_confirmation_threshold": 0.7,
                "label_position_tolerance": 0.05,
            },
            "field_mapping": {},
        }
    )

    stored = PageFingerprint(page_index=0, visual_hash=0, visual_hash_hex="b3731c4d1c4d1c73")
    near_match = PageFingerprint(page_index=0, visual_hash=0, visual_hash_hex="e3731c4d1c4d1c73")
    exact_match = PageFingerprint(page_index=0, visual_hash=0, visual_hash_hex="b3731c4d1c4d1c73")
    record = FingerprintRecord(
        hash="hash-near",
        layout_template={"pages": [{"page_index": 0, "label_confirmation_set": []}]},
        page_fingerprints=[stored],
        confidence=1.0,
        status=TemplateStatus.established,
        created_at="2026-01-01T00:00:00Z",
    )

    match_fail, conf_fail = lookup_fingerprint([near_match], [[]], [(1000, 1000)], [record], config)
    assert match_fail is None
    assert conf_fail == 0.0

    match_ok, conf_ok = lookup_fingerprint([exact_match], [[]], [(1000, 1000)], [record], config)
    assert match_ok is not None
    assert match_ok.hash == "hash-near"
    assert conf_ok == 1.0


def test_route_invoice_uses_provider_template_strategy_for_known_matches():
    context = DocumentContext(
        source_format=SourceFormat.pdf_text,
        extraction_strategy=ExtractionStrategy.provider_template,
    )
    record = FingerprintRecord(
        hash="hash1",
        layout_template={"pages": []},
        page_fingerprints=[],
        confidence=1.0,
        status=TemplateStatus.established,
        created_at="2026-01-01T00:00:00Z",
    )

    assert determine_extraction_strategy(context, record) == ExtractionStrategy.provider_template
    assert route_invoice(context, record, gt_present=False) == Route.APPLY


def test_route_invoice_allows_generic_native_pdf_apply_without_gt():
    context = DocumentContext(
        source_format=SourceFormat.pdf_text,
        extraction_strategy=ExtractionStrategy.native_pdf,
    )

    assert route_invoice(context, None, gt_present=False) == Route.APPLY


def test_route_invoice_rediscovery_for_provisional_match_when_gt_present():
    context = DocumentContext(
        source_format=SourceFormat.image,
        extraction_strategy=ExtractionStrategy.provider_template,
    )
    record = FingerprintRecord(
        hash="hash2",
        layout_template={"pages": []},
        page_fingerprints=[],
        confidence=0.6,
        status=TemplateStatus.provisional,
        created_at="2026-01-01T00:00:00Z",
    )

    assert route_invoice(context, record, gt_present=True) == Route.DISCOVERY


def test_route_invoice_keeps_provisional_apply_when_gt_is_not_discovery_ready():
    context = DocumentContext(
        source_format=SourceFormat.image,
        extraction_strategy=ExtractionStrategy.provider_template,
    )
    record = FingerprintRecord(
        hash="hash2",
        layout_template={"pages": []},
        page_fingerprints=[],
        confidence=0.6,
        status=TemplateStatus.provisional,
        created_at="2026-01-01T00:00:00Z",
    )

    assert route_invoice(context, record, gt_present=True, gt_discovery_ready=False) == Route.APPLY


def test_route_invoice_allows_family_representative_to_drive_apply_without_exact_fingerprint():
    context = DocumentContext(
        source_format=SourceFormat.image,
        extraction_strategy=ExtractionStrategy.discovery_fallback,
    )
    family_record = FingerprintRecord(
        hash="family-hash",
        layout_template={"pages": []},
        page_fingerprints=[],
        confidence=0.9,
        status=TemplateStatus.established,
        created_at="2026-01-01T00:00:00Z",
    )

    assert (
        determine_extraction_strategy(
            context,
            None,
            family_representative=family_record,
            family_match_score=0.82,
            family_apply_threshold=0.7,
        )
        == ExtractionStrategy.provider_template
    )
    assert (
        route_invoice(
            context,
            None,
            gt_present=False,
            family_representative=family_record,
            family_match_score=0.82,
            family_apply_threshold=0.7,
        )
        == Route.APPLY
    )


def test_route_invoice_rediscovery_for_provisional_family_representative_when_gt_present():
    context = DocumentContext(
        source_format=SourceFormat.image,
        extraction_strategy=ExtractionStrategy.discovery_fallback,
    )
    family_record = FingerprintRecord(
        hash="family-provisional",
        layout_template={"pages": []},
        page_fingerprints=[],
        confidence=0.6,
        status=TemplateStatus.provisional,
        created_at="2026-01-01T00:00:00Z",
    )

    assert (
        route_invoice(
            context,
            None,
            gt_present=True,
            family_representative=family_record,
            family_match_score=0.82,
            family_apply_threshold=0.7,
        )
        == Route.DISCOVERY
    )


def test_route_invoice_treats_non_discovery_ready_gt_like_missing_for_unknown_images():
    context = DocumentContext(
        source_format=SourceFormat.image,
        extraction_strategy=ExtractionStrategy.discovery_fallback,
    )

    assert route_invoice(context, None, gt_present=True, gt_discovery_ready=False) == Route.FAIL


def test_route_invoice_allows_native_pdf_apply_with_non_discovery_ready_gt():
    context = DocumentContext(
        source_format=SourceFormat.pdf_text,
        extraction_strategy=ExtractionStrategy.native_pdf,
    )

    assert route_invoice(context, None, gt_present=True, gt_discovery_ready=False) == Route.APPLY


def test_lookup_fingerprint_preserves_match_behavior_with_template_family_id():
    config = AppConfig(
        **{
            "validation": {
                "apply_threshold": 0.9,
                "discovery_threshold": 0.95,
                "jaccard_threshold": 0.85,
            },
            "fingerprinting": {"visual_hash_hamming_threshold": 10},
            "template_lifecycle": {
                "establish_min_count": 5,
                "establish_min_confidence": 0.95,
                "degradation_threshold": 0.85,
                "degradation_window": 10,
                "rediscovery_attempts": 3,
            },
            "quality": {
                "blur_threshold": 100.0,
                "contrast_threshold": 0.3,
                "quality_threshold": 0.5,
                "quality_region_buffer_multiplier": 2,
            },
            "processing": {"batch_size": 50, "worker_concurrency": 8},
            "region_buffer_pixels": 5,
            "ocr": {"single_field_engine": "tesseract", "table_engine": "paddle"},
            "table_detection": {
                "min_line_span_fraction": 0.4,
                "column_gap_px": 20,
                "row_gap_multiplier": 1.5,
            },
            "discovery": {
                "inference_confidence_threshold": 0.6,
                "label_confirmation_threshold": 0.7,
                "label_position_tolerance": 0.05,
            },
            "field_mapping": {},
        }
    )
    page = PageFingerprint(page_index=0, visual_hash=0, visual_hash_hex="b3731c4d1c4d1c73")
    record = FingerprintRecord(
        hash="hash-family",
        layout_template={"pages": [{"page_index": 0, "label_confirmation_set": []}]},
        template_family_id="family-1",
        page_fingerprints=[page],
        confidence=1.0,
        status=TemplateStatus.established,
        created_at="2026-01-01T00:00:00Z",
    )

    match, conf = lookup_fingerprint([page], [[]], [(1000, 1000)], [record], config)

    assert match is not None
    assert match.hash == "hash-family"
    assert conf == 1.0


def test_lookup_template_family_matches_anchor_summary_fallback():
    config = AppConfig(
        **{
            "validation": {
                "apply_threshold": 0.9,
                "discovery_threshold": 0.95,
                "jaccard_threshold": 0.85,
            },
            "fingerprinting": {"visual_hash_hamming_threshold": 10},
            "template_lifecycle": {
                "establish_min_count": 5,
                "establish_min_confidence": 0.95,
                "degradation_threshold": 0.85,
                "degradation_window": 10,
                "rediscovery_attempts": 3,
            },
            "quality": {
                "blur_threshold": 100.0,
                "contrast_threshold": 0.3,
                "quality_threshold": 0.5,
                "quality_region_buffer_multiplier": 2,
            },
            "processing": {"batch_size": 50, "worker_concurrency": 8},
            "region_buffer_pixels": 5,
            "ocr": {"single_field_engine": "tesseract", "table_engine": "paddle"},
            "table_detection": {
                "min_line_span_fraction": 0.4,
                "column_gap_px": 20,
                "row_gap_multiplier": 1.5,
            },
            "discovery": {
                "inference_confidence_threshold": 0.6,
                "label_confirmation_threshold": 0.7,
                "label_position_tolerance": 0.05,
                "family_anchor_threshold": 0.3,
                "family_apply_threshold": 0.5,
                "family_page_count_tolerance": 1,
            },
            "field_mapping": {},
        }
    )
    page = PageFingerprint(
        page_index=0,
        visual_hash=0,
        visual_hash_hex="b3731c4d1c4d1c73",
        role=PageRole.summary_page,
        stable_anchor_signature={
            "summary_labels": ["total", "tax"],
            "header_tokens": ["invoice"],
            "footer_tokens": [],
        },
    )
    family = TemplateFamilyRecord(
        template_family_id="family-1",
        provider_name="Acme",
        document_family=DocumentFamily.invoice,
        stable_anchor_regions={"tokens": ["invoice", "total", "tax"]},
        anchor_summary={
            "page_count": 1,
            "page_roles": ["summary_page"],
            "aggregate_keywords": {"summary": ["total", "tax"]},
        },
        page_role_expectations=[PageRole.summary_page],
        created_at="2026-01-01T00:00:00Z",
    )
    record = FingerprintRecord(
        hash="hash-family",
        layout_template={"pages": [{"page_index": 0, "label_confirmation_set": []}]},
        template_family_id="family-1",
        page_fingerprints=[page],
        confidence=0.9,
        status=TemplateStatus.established,
        created_at="2026-01-01T00:00:00Z",
    )
    context = DocumentContext(
        source_format=SourceFormat.image, extraction_strategy=ExtractionStrategy.discovery_fallback
    )

    matched_family, representative, score, candidate_count = lookup_template_family(
        [page], [record], [family], context, config
    )

    assert matched_family is not None
    assert matched_family.template_family_id == "family-1"
    assert representative is not None
    assert representative.hash == "hash-family"
    assert score >= 0.3
    assert candidate_count == 1


def test_lookup_template_family_prefers_representative_with_better_role_fit():
    config = AppConfig(
        **{
            "validation": {
                "apply_threshold": 0.9,
                "discovery_threshold": 0.95,
                "jaccard_threshold": 0.85,
            },
            "fingerprinting": {"visual_hash_hamming_threshold": 10},
            "template_lifecycle": {
                "establish_min_count": 5,
                "establish_min_confidence": 0.95,
                "degradation_threshold": 0.85,
                "degradation_window": 10,
                "rediscovery_attempts": 3,
            },
            "quality": {
                "blur_threshold": 100.0,
                "contrast_threshold": 0.3,
                "quality_threshold": 0.5,
                "quality_region_buffer_multiplier": 2,
            },
            "processing": {"batch_size": 50, "worker_concurrency": 8},
            "region_buffer_pixels": 5,
            "ocr": {"single_field_engine": "tesseract", "table_engine": "paddle"},
            "table_detection": {
                "min_line_span_fraction": 0.4,
                "column_gap_px": 20,
                "row_gap_multiplier": 1.5,
            },
            "discovery": {
                "inference_confidence_threshold": 0.6,
                "label_confirmation_threshold": 0.7,
                "label_position_tolerance": 0.05,
                "family_anchor_threshold": 0.3,
                "family_apply_threshold": 0.5,
                "family_page_count_tolerance": 1,
            },
            "field_mapping": {},
        }
    )
    new_page = PageFingerprint(
        page_index=0,
        visual_hash=0,
        visual_hash_hex="b3731c4d1c4d1c73",
        role=PageRole.summary_page,
        stable_anchor_signature={
            "summary_labels": ["total", "tax"],
            "header_tokens": ["invoice"],
            "footer_tokens": [],
        },
    )
    family = TemplateFamilyRecord(
        template_family_id="family-1",
        document_family=DocumentFamily.invoice,
        stable_anchor_regions={"tokens": ["invoice", "total", "tax"]},
        anchor_summary={"page_count": 1, "aggregate_keywords": {"summary": ["total", "tax"]}},
        page_role_expectations=[PageRole.summary_page],
        created_at="2026-01-01T00:00:00Z",
    )
    weaker_fit = FingerprintRecord(
        hash="hash-header",
        layout_template={"pages": [{"page_index": 0, "label_confirmation_set": []}]},
        template_family_id="family-1",
        page_fingerprints=[
            PageFingerprint(
                page_index=0,
                visual_hash=1,
                visual_hash_hex="0" * 16,
                role=PageRole.header_page,
                stable_anchor_signature={
                    "header_tokens": ["invoice"],
                    "summary_labels": [],
                    "footer_tokens": [],
                },
            )
        ],
        confidence=0.99,
        apply_count=10,
        status=TemplateStatus.established,
        created_at="2026-01-01T00:00:00Z",
    )
    better_fit = FingerprintRecord(
        hash="hash-summary",
        layout_template={"pages": [{"page_index": 0, "label_confirmation_set": []}]},
        template_family_id="family-1",
        page_fingerprints=[
            PageFingerprint(
                page_index=0,
                visual_hash=2,
                visual_hash_hex="1" * 16,
                role=PageRole.summary_page,
                stable_anchor_signature={
                    "header_tokens": ["invoice"],
                    "summary_labels": ["total", "tax"],
                    "footer_tokens": [],
                },
            )
        ],
        confidence=0.8,
        apply_count=3,
        status=TemplateStatus.established,
        created_at="2026-01-02T00:00:00Z",
    )
    context = DocumentContext(
        source_format=SourceFormat.image, extraction_strategy=ExtractionStrategy.discovery_fallback
    )

    matched_family, representative, score, candidate_count = lookup_template_family(
        [new_page],
        [weaker_fit, better_fit],
        [family],
        context,
        config,
    )

    assert matched_family is not None
    assert representative is not None
    assert representative.hash == "hash-summary"
    assert score >= 0.3
    assert candidate_count == 1


def test_lookup_template_family_allows_extra_line_item_pages_when_stable_regions_match():
    config = AppConfig(
        **{
            "validation": {
                "apply_threshold": 0.9,
                "discovery_threshold": 0.95,
                "jaccard_threshold": 0.85,
            },
            "fingerprinting": {"visual_hash_hamming_threshold": 10},
            "template_lifecycle": {
                "establish_min_count": 5,
                "establish_min_confidence": 0.95,
                "degradation_threshold": 0.85,
                "degradation_window": 10,
                "rediscovery_attempts": 3,
            },
            "quality": {
                "blur_threshold": 100.0,
                "contrast_threshold": 0.3,
                "quality_threshold": 0.5,
                "quality_region_buffer_multiplier": 2,
            },
            "processing": {"batch_size": 50, "worker_concurrency": 8},
            "region_buffer_pixels": 5,
            "ocr": {"single_field_engine": "tesseract", "table_engine": "paddle"},
            "table_detection": {
                "min_line_span_fraction": 0.4,
                "column_gap_px": 20,
                "row_gap_multiplier": 1.5,
            },
            "discovery": {
                "inference_confidence_threshold": 0.6,
                "label_confirmation_threshold": 0.7,
                "label_position_tolerance": 0.05,
                "family_anchor_threshold": 0.3,
                "family_apply_threshold": 0.5,
                "family_page_count_tolerance": 0,
            },
            "field_mapping": {},
        }
    )
    new_pages = [
        PageFingerprint(
            page_index=0,
            visual_hash=0,
            visual_hash_hex="0" * 16,
            role=PageRole.header_page,
            stable_anchor_signature={
                "header_tokens": ["invoice", "acme clinic"],
                "summary_labels": [],
                "footer_tokens": [],
                "keyword_hits": {"provider": ["acme clinic"], "invoice_number": ["invoice no"]},
            },
        ),
        PageFingerprint(
            page_index=1,
            visual_hash=1,
            visual_hash_hex="1" * 16,
            role=PageRole.line_item_page,
            stable_anchor_signature={
                "header_tokens": [],
                "summary_labels": [],
                "footer_tokens": [],
                "keyword_hits": {},
            },
        ),
        PageFingerprint(
            page_index=2,
            visual_hash=2,
            visual_hash_hex="2" * 16,
            role=PageRole.summary_page,
            stable_anchor_signature={
                "header_tokens": [],
                "summary_labels": ["total", "tax"],
                "footer_tokens": ["payment terms"],
                "keyword_hits": {"summary": ["total", "tax"], "footer": ["payment terms"]},
            },
        ),
    ]
    family = TemplateFamilyRecord(
        template_family_id="family-1",
        provider_name="Acme Clinic",
        document_family=DocumentFamily.invoice,
        stable_anchor_regions={
            "tokens": ["invoice", "acme clinic", "total", "tax", "payment terms"]
        },
        anchor_summary={
            "page_count": 2,
            "page_roles": ["header_page", "summary_page"],
            "aggregate_keywords": {
                "provider": ["acme clinic"],
                "invoice_number": ["invoice no"],
                "summary": ["total", "tax"],
                "footer": ["payment terms"],
            },
        },
        page_role_expectations=[PageRole.header_page, PageRole.summary_page],
        created_at="2026-01-01T00:00:00Z",
    )
    representative = FingerprintRecord(
        hash="hash-two-page",
        layout_template={
            "pages": [
                {"page_index": 0, "label_confirmation_set": []},
                {"page_index": 1, "label_confirmation_set": []},
            ]
        },
        template_family_id="family-1",
        page_fingerprints=[new_pages[0], new_pages[2]],
        confidence=0.9,
        apply_count=5,
        status=TemplateStatus.established,
        created_at="2026-01-01T00:00:00Z",
    )
    context = DocumentContext(
        source_format=SourceFormat.image, extraction_strategy=ExtractionStrategy.discovery_fallback
    )

    matched_family, matched_representative, score, candidate_count = lookup_template_family(
        new_pages,
        [representative],
        [family],
        context,
        config,
    )

    assert matched_family is not None
    assert matched_family.template_family_id == "family-1"
    assert matched_representative is not None
    assert matched_representative.hash == "hash-two-page"
    assert score >= 0.3
    assert candidate_count == 1


def test_lookup_template_family_weights_summary_and_identity_more_than_generic_overlap():
    config = AppConfig(
        **{
            "validation": {
                "apply_threshold": 0.9,
                "discovery_threshold": 0.95,
                "jaccard_threshold": 0.85,
            },
            "fingerprinting": {"visual_hash_hamming_threshold": 10},
            "template_lifecycle": {
                "establish_min_count": 5,
                "establish_min_confidence": 0.95,
                "degradation_threshold": 0.85,
                "degradation_window": 10,
                "rediscovery_attempts": 3,
            },
            "quality": {
                "blur_threshold": 100.0,
                "contrast_threshold": 0.3,
                "quality_threshold": 0.5,
                "quality_region_buffer_multiplier": 2,
            },
            "processing": {"batch_size": 50, "worker_concurrency": 8},
            "region_buffer_pixels": 5,
            "ocr": {"single_field_engine": "tesseract", "table_engine": "paddle"},
            "table_detection": {
                "min_line_span_fraction": 0.4,
                "column_gap_px": 20,
                "row_gap_multiplier": 1.5,
            },
            "discovery": {
                "inference_confidence_threshold": 0.6,
                "label_confirmation_threshold": 0.7,
                "label_position_tolerance": 0.05,
                "family_anchor_threshold": 0.3,
                "family_apply_threshold": 0.5,
                "family_page_count_tolerance": 1,
            },
            "field_mapping": {},
        }
    )
    page = PageFingerprint(
        page_index=0,
        visual_hash=0,
        visual_hash_hex="3" * 16,
        role=PageRole.summary_page,
        stable_anchor_signature={
            "header_tokens": ["invoice", "acme clinic"],
            "summary_labels": ["subtotal", "total", "tax"],
            "footer_tokens": [],
            "keyword_hits": {
                "provider": ["acme clinic"],
                "invoice_date": ["invoice date"],
                "summary": ["subtotal", "total", "tax"],
            },
        },
    )
    stronger_family = TemplateFamilyRecord(
        template_family_id="family-strong",
        provider_name="Acme Clinic",
        document_family=DocumentFamily.invoice,
        stable_anchor_regions={"tokens": ["invoice", "acme clinic", "subtotal", "total", "tax"]},
        anchor_summary={
            "page_count": 1,
            "aggregate_keywords": {
                "provider": ["acme clinic"],
                "invoice_date": ["invoice date"],
                "summary": ["subtotal", "total", "tax"],
            },
        },
        page_role_expectations=[PageRole.summary_page],
        created_at="2026-01-01T00:00:00Z",
    )
    weaker_family = TemplateFamilyRecord(
        template_family_id="family-weak",
        document_family=DocumentFamily.invoice,
        stable_anchor_regions={"tokens": ["invoice", "acme clinic", "subtotal", "total", "tax"]},
        anchor_summary={
            "page_count": 1,
            "aggregate_keywords": {
                "provider": ["other clinic"],
                "invoice_date": ["statement date"],
                "summary": ["balance due"],
            },
        },
        page_role_expectations=[PageRole.summary_page],
        created_at="2026-01-01T00:00:00Z",
    )
    strong_record = FingerprintRecord(
        hash="hash-strong",
        layout_template={"pages": [{"page_index": 0, "label_confirmation_set": []}]},
        template_family_id="family-strong",
        page_fingerprints=[page],
        confidence=0.8,
        apply_count=2,
        status=TemplateStatus.established,
        created_at="2026-01-01T00:00:00Z",
    )
    weak_record = FingerprintRecord(
        hash="hash-weak",
        layout_template={"pages": [{"page_index": 0, "label_confirmation_set": []}]},
        template_family_id="family-weak",
        page_fingerprints=[
            PageFingerprint(
                page_index=0,
                visual_hash=1,
                visual_hash_hex="4" * 16,
                role=PageRole.summary_page,
                stable_anchor_signature={
                    "header_tokens": ["invoice", "acme clinic"],
                    "summary_labels": ["balance due"],
                    "footer_tokens": [],
                    "keyword_hits": {"summary": ["balance due"]},
                },
            )
        ],
        confidence=0.95,
        apply_count=8,
        status=TemplateStatus.established,
        created_at="2026-01-01T00:00:00Z",
    )
    context = DocumentContext(
        source_format=SourceFormat.image, extraction_strategy=ExtractionStrategy.discovery_fallback
    )

    matched_family, representative, score, candidate_count = lookup_template_family(
        [page],
        [strong_record, weak_record],
        [stronger_family, weaker_family],
        context,
        config,
    )

    assert matched_family is not None
    assert matched_family.template_family_id == "family-strong"
    assert representative is not None
    assert representative.hash == "hash-strong"
    assert score >= 0.3
    assert candidate_count == 2


def test_lookup_template_family_is_more_conservative_for_poor_performing_families():
    config = AppConfig(
        **{
            "validation": {
                "apply_threshold": 0.9,
                "discovery_threshold": 0.95,
                "jaccard_threshold": 0.85,
            },
            "fingerprinting": {"visual_hash_hamming_threshold": 10},
            "template_lifecycle": {
                "establish_min_count": 5,
                "establish_min_confidence": 0.95,
                "degradation_threshold": 0.85,
                "degradation_window": 10,
                "rediscovery_attempts": 3,
            },
            "quality": {
                "blur_threshold": 100.0,
                "contrast_threshold": 0.3,
                "quality_threshold": 0.5,
                "quality_region_buffer_multiplier": 2,
            },
            "processing": {"batch_size": 50, "worker_concurrency": 8},
            "region_buffer_pixels": 5,
            "ocr": {"single_field_engine": "tesseract", "table_engine": "paddle"},
            "table_detection": {
                "min_line_span_fraction": 0.4,
                "column_gap_px": 20,
                "row_gap_multiplier": 1.5,
            },
            "discovery": {
                "inference_confidence_threshold": 0.6,
                "label_confirmation_threshold": 0.7,
                "label_position_tolerance": 0.05,
                "family_anchor_threshold": 0.3,
                "family_apply_threshold": 0.5,
                "family_page_count_tolerance": 1,
            },
            "field_mapping": {},
        }
    )
    page = PageFingerprint(
        page_index=0,
        visual_hash=0,
        visual_hash_hex="5" * 16,
        role=PageRole.summary_page,
        stable_anchor_signature={
            "header_tokens": ["invoice"],
            "summary_labels": ["total"],
            "footer_tokens": [],
            "keyword_hits": {"summary": ["total"]},
        },
    )
    healthy_family = TemplateFamilyRecord(
        template_family_id="family-healthy",
        document_family=DocumentFamily.invoice,
        stable_anchor_regions={"tokens": ["invoice", "total"]},
        anchor_summary={
            "page_count": 1,
            "aggregate_keywords": {"summary": ["total"]},
            "split_signals": {
                "member_count": 1,
                "unique_signature_count": 1,
                "dominant_signature_ratio": 1.0,
            },
        },
        page_role_expectations=[PageRole.summary_page],
        confidence=0.96,
        apply_count=12,
        reject_count=0,
        status=TemplateStatus.established,
        created_at="2026-01-01T00:00:00Z",
    )
    poor_family = TemplateFamilyRecord(
        template_family_id="family-poor",
        document_family=DocumentFamily.invoice,
        stable_anchor_regions={"tokens": ["invoice", "total"]},
        anchor_summary={
            "page_count": 1,
            "aggregate_keywords": {"summary": ["total"]},
            "split_signals": {
                "member_count": 3,
                "unique_signature_count": 3,
                "dominant_signature_ratio": 0.34,
            },
        },
        page_role_expectations=[PageRole.summary_page],
        confidence=0.58,
        apply_count=2,
        reject_count=9,
        status=TemplateStatus.degraded,
        created_at="2026-01-01T00:00:00Z",
    )
    healthy_record = FingerprintRecord(
        hash="hash-healthy",
        layout_template={"pages": [{"page_index": 0, "label_confirmation_set": []}]},
        template_family_id="family-healthy",
        page_fingerprints=[page],
        confidence=0.9,
        apply_count=4,
        status=TemplateStatus.established,
        created_at="2026-01-01T00:00:00Z",
    )
    poor_record = FingerprintRecord(
        hash="hash-poor",
        layout_template={"pages": [{"page_index": 0, "label_confirmation_set": []}]},
        template_family_id="family-poor",
        page_fingerprints=[page],
        confidence=0.9,
        apply_count=4,
        status=TemplateStatus.established,
        created_at="2026-01-01T00:00:00Z",
    )
    context = DocumentContext(
        source_format=SourceFormat.image, extraction_strategy=ExtractionStrategy.discovery_fallback
    )

    matched_family, representative, score, candidate_count = lookup_template_family(
        [page],
        [poor_record, healthy_record],
        [poor_family, healthy_family],
        context,
        config,
    )

    assert matched_family is not None
    assert matched_family.template_family_id == "family-healthy"
    assert representative is not None
    assert representative.hash == "hash-healthy"
    assert score >= 0.3
    assert candidate_count >= 1
