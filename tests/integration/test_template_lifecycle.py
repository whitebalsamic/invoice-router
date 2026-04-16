from invoice_router.config import TemplateLifecycleConfig
from invoice_router.domain.templates.lifecycle import (
    _rebuild_family_anchor_summary,
    family_has_gt_trust,
    manually_update_template_family,
    merge_template_families,
    record_template_family_rejection,
    resolve_gt_healing_authority,
    retire_template_family,
    split_template_family,
    update_template_confidence,
    update_template_family_confidence,
)
from invoice_router.infrastructure.persistence.storage import FingerprintDB
from invoice_router.models import (
    DocumentFamily,
    FingerprintRecord,
    PageFingerprint,
    PageRole,
    ProcessingResult,
    Route,
    TemplateFamilyExample,
    TemplateFamilyRecord,
    TemplateStatus,
)


def test_update_template_family_confidence_establishes_family_and_versions(postgres_database_url):
    db = FingerprintDB(postgres_database_url)
    family = TemplateFamilyRecord(
        template_family_id="family-life-001",
        document_family=DocumentFamily.invoice,
        confidence=0.95,
        apply_count=4,
        reject_count=0,
        status=TemplateStatus.provisional,
        created_at="2026-04-15T12:00:00Z",
    )
    db.store_template_family(family)

    update_template_family_confidence(
        db,
        "family-life-001",
        1.0,
        TemplateLifecycleConfig(),
        reason="apply_success",
    )

    updated = db.get_template_family("family-life-001")
    versions = db.get_template_family_versions("family-life-001")

    assert updated is not None
    assert updated.apply_count == 5
    assert updated.status == TemplateStatus.established
    assert versions[-1].change_reason == "apply_success"
    assert versions[-1].family_snapshot["apply_count"] == 5


def test_record_template_family_rejection_degrades_provisional_family(postgres_database_url):
    db = FingerprintDB(postgres_database_url)
    family = TemplateFamilyRecord(
        template_family_id="family-life-002",
        document_family=DocumentFamily.invoice,
        confidence=0.4,
        apply_count=0,
        reject_count=0,
        status=TemplateStatus.provisional,
        created_at="2026-04-15T12:00:00Z",
    )
    db.store_template_family(family)

    record_template_family_rejection(
        db,
        "family-life-002",
        TemplateLifecycleConfig(),
        reason="apply_rejection",
    )

    updated = db.get_template_family("family-life-002")
    versions = db.get_template_family_versions("family-life-002")

    assert updated is not None
    assert updated.reject_count == 1
    assert updated.status == TemplateStatus.degraded
    assert versions[-1].change_reason == "apply_rejection"
    assert versions[-1].family_snapshot["reject_count"] == 1


def test_gt_backed_family_updates_increment_gt_only_trust(postgres_database_url):
    db = FingerprintDB(postgres_database_url)
    family = TemplateFamilyRecord(
        template_family_id="family-life-gt",
        document_family=DocumentFamily.invoice,
        confidence=0.95,
        apply_count=4,
        reject_count=0,
        gt_confidence=0.95,
        gt_apply_count=4,
        gt_reject_count=0,
        status=TemplateStatus.provisional,
        created_at="2026-04-15T12:00:00Z",
    )
    db.store_template_family(family)

    update_template_family_confidence(
        db,
        "family-life-gt",
        1.0,
        TemplateLifecycleConfig(),
        reason="apply_success",
        gt_backed=True,
    )
    record_template_family_rejection(
        db,
        "family-life-gt",
        TemplateLifecycleConfig(),
        reason="apply_rejection",
        gt_backed=True,
    )

    updated = db.get_template_family("family-life-gt")
    assert updated is not None
    assert updated.gt_apply_count == 5
    assert updated.gt_reject_count == 1
    assert family_has_gt_trust(updated, TemplateLifecycleConfig()) is True


def test_non_gt_apply_does_not_change_gt_only_fingerprint_state(postgres_database_url):
    db = FingerprintDB(postgres_database_url)
    fingerprint = FingerprintRecord(
        hash="fp-no-gt",
        layout_template={"pages": [{"fields": {}}]},
        page_fingerprints=[PageFingerprint(page_index=0, visual_hash=7, visual_hash_hex="7" * 16)],
        confidence=0.4,
        apply_count=1,
        reject_count=0,
        gt_apply_count=1,
        gt_reject_count=0,
        gt_confidence=0.9,
        status=TemplateStatus.provisional,
        created_at="2026-04-15T12:00:00Z",
    )
    db.store_fingerprint(
        fingerprint,
        visual_hashes=["7" * 16],
        page_fingerprints=[p.model_dump() for p in fingerprint.page_fingerprints],
    )

    class FakeRedis:
        def __init__(self):
            self.counters = {("fingerprint_stats:fp-no-gt", "apply_count"): 1}

        def hincrby(self, key, field, amount):
            current = int(self.counters.get((key, field), 0)) + amount
            self.counters[(key, field)] = current
            return current

    update_template_confidence(
        db,
        FakeRedis(),
        "fp-no-gt",
        1.0,
        TemplateLifecycleConfig(),
        gt_backed=False,
    )

    updated = next(
        record for record in db.get_all_active_fingerprints() if record.hash == "fp-no-gt"
    )
    assert updated.apply_count == 2
    assert updated.gt_apply_count == 1
    assert updated.gt_confidence == 0.9


def test_resolve_gt_healing_authority_prefers_family_then_fingerprint(postgres_database_url):
    family = TemplateFamilyRecord(
        template_family_id="family-authority",
        document_family=DocumentFamily.invoice,
        gt_apply_count=5,
        gt_confidence=0.98,
        created_at="2026-04-15T12:00:00Z",
    )
    fingerprint = FingerprintRecord(
        hash="fp-authority",
        layout_template={"pages": [{"fields": {}}]},
        template_family_id="family-authority",
        page_fingerprints=[PageFingerprint(page_index=0, visual_hash=8, visual_hash_hex="8" * 16)],
        gt_apply_count=2,
        gt_confidence=0.91,
        created_at="2026-04-15T12:00:00Z",
    )
    family_only = resolve_gt_healing_authority(family, fingerprint, TemplateLifecycleConfig())
    fingerprint_only = resolve_gt_healing_authority(None, fingerprint, TemplateLifecycleConfig())

    assert family_only["scope"] == "family"
    assert family_only["trusted"] is True
    assert fingerprint_only["scope"] == "fingerprint"
    assert fingerprint_only["trusted"] is False


def test_manually_update_template_family_merges_profile_and_records_version(postgres_database_url):
    db = FingerprintDB(postgres_database_url)
    family = TemplateFamilyRecord(
        template_family_id="family-life-003",
        document_family=DocumentFamily.invoice,
        extraction_profile={
            "preferred_strategy": "provider_template",
            "table": {"enabled": True, "ocr_engine": "paddle"},
        },
        status=TemplateStatus.provisional,
        created_at="2026-04-15T12:00:00Z",
    )
    db.store_template_family(family)

    updated = manually_update_template_family(
        db,
        "family-life-003",
        status=TemplateStatus.established,
        provider_name="Acme Ltd",
        extraction_profile_updates={"ocr": {"region_buffer_multiplier": 1.2}},
        reason="manual_review",
    )

    versions = db.get_template_family_versions("family-life-003")

    assert updated is not None
    assert updated.status == TemplateStatus.established
    assert updated.provider_name == "Acme Ltd"
    assert updated.extraction_profile["table"]["ocr_engine"] == "paddle"
    assert updated.extraction_profile["ocr"]["region_buffer_multiplier"] == 1.2
    assert versions[-1].change_reason == "manual_review"
    assert versions[-1].family_snapshot["status"] == "established"


def test_split_template_family_moves_members_and_records_versions(tmp_path, postgres_database_url):
    db = FingerprintDB(postgres_database_url)
    family = TemplateFamilyRecord(
        template_family_id="family-life-004",
        document_family=DocumentFamily.invoice,
        extraction_profile={"preferred_strategy": "provider_template"},
        status=TemplateStatus.established,
        created_at="2026-04-15T12:00:00Z",
    )
    db.store_template_family(family)

    moved_invoice = tmp_path / "moved.png"
    moved_invoice.write_bytes(b"img")
    kept_invoice = tmp_path / "kept.png"
    kept_invoice.write_bytes(b"img")

    moved_fp = FingerprintRecord(
        hash="fp-split-moved",
        layout_template={"pages": [{"fields": {"Total": {"field_type": "currency"}}}]},
        template_family_id="family-life-004",
        page_fingerprints=[
            PageFingerprint(
                page_index=0,
                visual_hash=1,
                visual_hash_hex="1" * 16,
                role=PageRole.summary_page,
                stable_anchor_signature={
                    "summary_labels": ["total"],
                    "header_tokens": ["invoice"],
                    "footer_tokens": [],
                },
            )
        ],
        confidence=0.98,
        apply_count=6,
        reject_count=0,
        status=TemplateStatus.established,
        created_at="2026-04-15T12:00:01Z",
    )
    kept_fp = FingerprintRecord(
        hash="fp-split-keep",
        layout_template={"pages": [{"fields": {"Invoice Date": {"field_type": "date"}}}]},
        template_family_id="family-life-004",
        page_fingerprints=[
            PageFingerprint(
                page_index=0,
                visual_hash=2,
                visual_hash_hex="2" * 16,
                role=PageRole.header_page,
                stable_anchor_signature={
                    "summary_labels": [],
                    "header_tokens": ["invoice date"],
                    "footer_tokens": [],
                },
            )
        ],
        confidence=0.92,
        apply_count=3,
        reject_count=0,
        status=TemplateStatus.established,
        created_at="2026-04-15T12:00:02Z",
    )
    db.store_fingerprint(
        moved_fp,
        visual_hashes=["1" * 16],
        page_fingerprints=[p.model_dump() for p in moved_fp.page_fingerprints],
    )
    db.store_fingerprint(
        kept_fp,
        visual_hashes=["2" * 16],
        page_fingerprints=[p.model_dump() for p in kept_fp.page_fingerprints],
    )

    db.add_template_family_example(
        TemplateFamilyExample(
            template_family_id="family-life-004",
            fingerprint_hash="fp-split-moved",
            invoice_path=str(moved_invoice),
            created_at="2026-04-15T12:00:03Z",
        )
    )
    db.add_template_family_example(
        TemplateFamilyExample(
            template_family_id="family-life-004",
            fingerprint_hash="fp-split-keep",
            invoice_path=str(kept_invoice),
            created_at="2026-04-15T12:00:04Z",
        )
    )
    db.store_result(
        ProcessingResult(
            invoice_path=str(moved_invoice),
            fingerprint_hash="fp-split-moved",
            template_family_id="family-life-004",
            extracted_data={"Total": "10.00"},
            ground_truth={"Total": "10.00"},
            provenance=None,
            validation_passed=True,
            route_used=Route.APPLY,
            attempted_route=Route.APPLY,
            image_quality_score=0.9,
            template_status_at_time=None,
            processed_at="2026-04-15T12:00:05Z",
        )
    )

    split = split_template_family(
        db,
        "family-life-004",
        fingerprint_hashes=["fp-split-moved"],
        new_template_family_id="family-life-004-split-a",
        reason="cluster_cleanup",
    )

    assert split is not None
    assert split["new_family_id"] == "family-life-004-split-a"
    assert split["moved_fingerprint_count"] == 1
    assert split["moved_example_count"] == 1
    assert split["moved_result_count"] == 1

    moved_family = db.get_template_family("family-life-004-split-a")
    source_family = db.get_template_family("family-life-004")
    moved_versions = db.get_template_family_versions("family-life-004-split-a")
    source_versions = db.get_template_family_versions("family-life-004")
    moved_result = db.get_result(str(moved_invoice))
    moved_examples = db.get_template_family_examples("family-life-004-split-a")
    moved_record = next(
        record for record in db.get_all_active_fingerprints() if record.hash == "fp-split-moved"
    )

    assert moved_family is not None
    assert moved_family.apply_count >= 1
    assert source_family is not None
    assert source_family.template_family_id == "family-life-004"
    assert moved_record.template_family_id == "family-life-004-split-a"
    assert moved_result is not None
    assert moved_result.template_family_id == "family-life-004-split-a"
    assert len(moved_examples) == 1
    assert moved_examples[0].template_family_id == "family-life-004-split-a"
    assert moved_versions[-1].change_reason == "cluster_cleanup:split_from:family-life-004"
    assert source_versions[-1].change_reason == "cluster_cleanup:split_out:family-life-004-split-a"
    assert moved_family.anchor_summary["pages"]
    assert moved_family.anchor_summary["split_signals"]["member_count"] == 1


def test_rebuild_family_anchor_summary_tracks_generic_split_signals():
    records = [
        FingerprintRecord(
            hash="fp-split-signals-a",
            layout_template={"pages": [{"fields": {"Invoice Date": {"field_type": "date"}}}]},
            page_fingerprints=[
                PageFingerprint(
                    page_index=0,
                    visual_hash=10,
                    visual_hash_hex="a" * 16,
                    role=PageRole.header_page,
                    stable_anchor_signature={
                        "header_tokens": ["invoice date"],
                        "summary_labels": [],
                        "footer_tokens": [],
                        "keyword_hits": {"invoice_date": ["invoice date"]},
                    },
                )
            ],
            confidence=0.92,
            apply_count=4,
            created_at="2026-04-15T13:00:00Z",
        ),
        FingerprintRecord(
            hash="fp-split-signals-b",
            layout_template={"pages": [{"fields": {"Total Due": {"field_type": "currency"}}}]},
            page_fingerprints=[
                PageFingerprint(
                    page_index=0,
                    visual_hash=11,
                    visual_hash_hex="b" * 16,
                    role=PageRole.summary_page,
                    stable_anchor_signature={
                        "header_tokens": ["amount due"],
                        "summary_labels": ["total due"],
                        "footer_tokens": [],
                        "keyword_hits": {"summary": ["total due"]},
                    },
                )
            ],
            confidence=0.91,
            apply_count=4,
            created_at="2026-04-15T13:01:00Z",
        ),
    ]

    summary = _rebuild_family_anchor_summary(records)

    assert summary["page_count"] == 1
    assert summary["pages"]
    assert summary["split_signals"]["member_count"] == 2
    assert summary["split_signals"]["unique_signature_count"] == 2
    assert summary["split_signals"]["dominant_signature_ratio"] == 0.5


def test_merge_template_families_moves_members_and_retires_source(tmp_path, postgres_database_url):
    db = FingerprintDB(postgres_database_url)
    target_family = TemplateFamilyRecord(
        template_family_id="family-life-merge-target",
        document_family=DocumentFamily.invoice,
        extraction_profile={"preferred_strategy": "provider_template", "table": {"enabled": True}},
        status=TemplateStatus.established,
        confidence=0.97,
        apply_count=7,
        created_at="2026-04-15T12:20:00Z",
    )
    source_family = TemplateFamilyRecord(
        template_family_id="family-life-merge-source",
        document_family=DocumentFamily.invoice,
        extraction_profile={"ocr": {"region_buffer_multiplier": 1.2}},
        status=TemplateStatus.provisional,
        confidence=0.71,
        apply_count=1,
        created_at="2026-04-15T12:21:00Z",
    )
    db.store_template_family(target_family)
    db.store_template_family(source_family)

    target_fp = FingerprintRecord(
        hash="fp-merge-target",
        layout_template={"pages": [{"fields": {"Invoice Date": {"field_type": "date"}}}]},
        template_family_id="family-life-merge-target",
        page_fingerprints=[
            PageFingerprint(
                page_index=0,
                visual_hash=3,
                visual_hash_hex="3" * 16,
                role=PageRole.header_page,
                stable_anchor_signature={
                    "header_tokens": ["invoice date"],
                    "summary_labels": [],
                    "footer_tokens": [],
                },
            )
        ],
        confidence=0.97,
        apply_count=7,
        status=TemplateStatus.established,
        created_at="2026-04-15T12:22:00Z",
    )
    source_fp = FingerprintRecord(
        hash="fp-merge-source",
        layout_template={"pages": [{"fields": {"Invoice Date": {"field_type": "date"}}}]},
        template_family_id="family-life-merge-source",
        page_fingerprints=[
            PageFingerprint(
                page_index=0,
                visual_hash=4,
                visual_hash_hex="4" * 16,
                role=PageRole.header_page,
                stable_anchor_signature={
                    "header_tokens": ["invoice date"],
                    "summary_labels": [],
                    "footer_tokens": [],
                },
            )
        ],
        confidence=0.76,
        apply_count=2,
        status=TemplateStatus.provisional,
        created_at="2026-04-15T12:23:00Z",
    )
    db.store_fingerprint(
        target_fp,
        visual_hashes=["3" * 16],
        page_fingerprints=[p.model_dump() for p in target_fp.page_fingerprints],
    )
    db.store_fingerprint(
        source_fp,
        visual_hashes=["4" * 16],
        page_fingerprints=[p.model_dump() for p in source_fp.page_fingerprints],
    )

    source_invoice = tmp_path / "merge-source.png"
    source_invoice.write_bytes(b"img")
    db.add_template_family_example(
        TemplateFamilyExample(
            template_family_id="family-life-merge-source",
            fingerprint_hash="fp-merge-source",
            invoice_path=str(source_invoice),
            created_at="2026-04-15T12:24:00Z",
        )
    )
    db.store_result(
        ProcessingResult(
            invoice_path=str(source_invoice),
            fingerprint_hash="fp-merge-source",
            template_family_id="family-life-merge-source",
            extracted_data={"Invoice Date": "2026-04-15"},
            ground_truth={"Invoice Date": "2026-04-15"},
            provenance=None,
            validation_passed=True,
            route_used=Route.APPLY,
            attempted_route=Route.APPLY,
            image_quality_score=0.9,
            template_status_at_time=None,
            processed_at="2026-04-15T12:25:00Z",
        )
    )

    merged = merge_template_families(
        db,
        "family-life-merge-target",
        "family-life-merge-source",
        reason="duplicate_cleanup",
    )

    assert merged is not None
    assert merged["moved_fingerprint_count"] == 1
    assert merged["moved_example_count"] == 1
    assert merged["moved_result_count"] == 1

    merged_target = db.get_template_family("family-life-merge-target")
    retired_source = db.get_template_family("family-life-merge-source")
    moved_result = db.get_result(str(source_invoice))
    target_examples = db.get_template_family_examples("family-life-merge-target")
    target_versions = db.get_template_family_versions("family-life-merge-target")
    source_versions = db.get_template_family_versions("family-life-merge-source")
    moved_record = next(
        record for record in db.get_all_active_fingerprints() if record.hash == "fp-merge-source"
    )

    assert merged_target is not None
    assert merged_target.extraction_profile["table"]["enabled"] is True
    assert merged_target.extraction_profile["ocr"]["region_buffer_multiplier"] == 1.2
    assert retired_source is not None
    assert retired_source.status == TemplateStatus.retired
    assert moved_record.template_family_id == "family-life-merge-target"
    assert moved_result is not None
    assert moved_result.template_family_id == "family-life-merge-target"
    assert len(target_examples) == 1
    assert target_examples[0].template_family_id == "family-life-merge-target"
    assert (
        target_versions[-1].change_reason == "duplicate_cleanup:merge_from:family-life-merge-source"
    )
    assert (
        source_versions[-1].change_reason
        == "duplicate_cleanup:merged_into:family-life-merge-target"
    )


def test_retire_template_family_marks_family_and_fingerprints_retired(postgres_database_url):
    db = FingerprintDB(postgres_database_url)
    family = TemplateFamilyRecord(
        template_family_id="family-life-retire",
        document_family=DocumentFamily.invoice,
        status=TemplateStatus.degraded,
        reject_count=4,
        created_at="2026-04-15T12:30:00Z",
    )
    db.store_template_family(family)
    fingerprint = FingerprintRecord(
        hash="fp-life-retire",
        layout_template={"pages": [{"fields": {}}]},
        template_family_id="family-life-retire",
        page_fingerprints=[PageFingerprint(page_index=0, visual_hash=5, visual_hash_hex="5" * 16)],
        status=TemplateStatus.degraded,
        created_at="2026-04-15T12:31:00Z",
    )
    db.store_fingerprint(
        fingerprint,
        visual_hashes=["5" * 16],
        page_fingerprints=[p.model_dump() for p in fingerprint.page_fingerprints],
    )

    retired = retire_template_family(db, "family-life-retire", reason="dead_family_cleanup")

    assert retired is not None
    assert retired["retired_fingerprint_count"] == 1
    updated_family = db.get_template_family("family-life-retire")
    versions = db.get_template_family_versions("family-life-retire")
    active_fingerprints = db.get_all_active_fingerprints()

    assert updated_family is not None
    assert updated_family.status == TemplateStatus.retired
    assert not active_fingerprints
    assert versions[-1].change_reason == "dead_family_cleanup"
