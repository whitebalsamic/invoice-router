from types import SimpleNamespace

from invoice_router.domain.invoices.family_profiles import (
    build_family_extraction_profile,
    merge_family_extraction_profiles,
    resolve_table_detection_config,
    resolve_table_ocr_engine,
    summarize_family_extraction_profile,
)
from invoice_router.models import DocumentContext, DocumentFamily, SourceFormat


def test_merge_family_extraction_profiles_deep_merges_without_mutating_inputs():
    existing = {
        "ocr": {
            "field_type_overrides": {"Invoice Date": "date"},
            "field_buffer_multipliers": {"date": 1.15},
        },
        "table": {"enabled": False},
    }
    learned = {
        "ocr": {"field_buffer_multipliers": {"currency": 0.9}},
        "table": {"enabled": True, "ocr_engine": "paddle"},
    }

    merged = merge_family_extraction_profiles(existing, learned)

    assert merged == {
        "ocr": {
            "field_type_overrides": {"Invoice Date": "date"},
            "field_buffer_multipliers": {"date": 1.15, "currency": 0.9},
        },
        "table": {"enabled": True, "ocr_engine": "paddle"},
    }
    assert existing["ocr"]["field_buffer_multipliers"] == {"date": 1.15}
    assert learned["table"] == {"enabled": True, "ocr_engine": "paddle"}


def test_build_family_extraction_profile_marks_scanned_templates_and_tables():
    template = {
        "pages": [
            {
                "fields": {
                    "Invoice Date": {"field_type": "date"},
                    "Total": {"field_type": "currency"},
                },
                "table": {"region": {"x": 0.1, "y": 0.2, "width": 0.6, "height": 0.3}},
            }
        ]
    }
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
    )

    profile = build_family_extraction_profile(template, context, default_table_engine="tesseract")

    assert profile["preferred_strategy"] == "provider_template"
    assert profile["ocr"]["region_buffer_multiplier"] == 1.15
    assert profile["ocr"]["field_type_overrides"] == {
        "Invoice Date": "date",
        "Total": "currency",
    }
    assert profile["table"] == {
        "enabled": True,
        "ocr_engine": "tesseract",
        "row_gap_multiplier": 1.25,
        "min_line_span_fraction": 0.32,
    }
    assert profile["postprocess"] == {
        "refine_critical_fields": True,
        "prefer_table_line_items": True,
    }


def test_profile_summary_and_table_helpers_respect_family_overrides():
    profile = {
        "preferred_strategy": "provider_template",
        "ocr": {"field_type_overrides": {"Invoice Date": "date", "Total": "currency"}},
        "table": {
            "enabled": True,
            "ocr_engine": " paddle ",
            "row_gap_multiplier": 1.2,
        },
    }
    default_config = SimpleNamespace(row_gap_multiplier=1.5, min_line_span_fraction=0.4)

    summary = summarize_family_extraction_profile(profile)
    resolved_engine = resolve_table_ocr_engine("tesseract", profile)
    resolved_config = resolve_table_detection_config(default_config, profile)

    assert summary == {
        "preferred_strategy": "provider_template",
        "field_override_count": 2,
        "table_enabled": True,
        "table_engine": " paddle ",
    }
    assert resolved_engine == "paddle"
    assert resolved_config.row_gap_multiplier == 1.2
    assert resolved_config.min_line_span_fraction == 0.4
