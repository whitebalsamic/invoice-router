from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, Optional

from ...models import DocumentContext, SourceFormat

_DEFAULT_FIELD_BUFFER_MULTIPLIERS: Dict[str, float] = {
    "string": 1.0,
    "date": 1.15,
    "currency": 0.9,
    "qty": 0.9,
    "address": 1.35,
}


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def merge_family_extraction_profiles(
    existing: Optional[Dict[str, Any]], learned: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    return _deep_merge(existing or {}, learned or {})


def build_family_extraction_profile(
    template: Optional[Dict[str, Any]],
    document_context: Optional[DocumentContext],
    *,
    default_table_engine: str = "paddle",
) -> Dict[str, Any]:
    template = template or {}
    field_type_overrides: Dict[str, str] = {}
    has_table = False
    for page in template.get("pages", []):
        for field_name, field_config in (page.get("fields") or {}).items():
            field_type = field_config.get("field_type")
            if field_name and field_type:
                field_type_overrides[str(field_name)] = str(field_type)
        if page.get("table"):
            has_table = True

    scanned_like = bool(
        document_context
        and document_context.source_format in {SourceFormat.image, SourceFormat.pdf_scanned}
    )

    return {
        "version": 1,
        "preferred_strategy": "provider_template",
        "ocr": {
            "region_buffer_multiplier": 1.15 if scanned_like else 1.0,
            "field_type_overrides": field_type_overrides,
            "field_buffer_multipliers": dict(_DEFAULT_FIELD_BUFFER_MULTIPLIERS),
        },
        "table": {
            "enabled": has_table,
            "ocr_engine": default_table_engine,
            "row_gap_multiplier": 1.25 if has_table else 1.5,
            "min_line_span_fraction": 0.32 if has_table else 0.40,
        },
        "postprocess": {
            "refine_critical_fields": True,
            "prefer_table_line_items": has_table,
        },
    }


def summarize_family_extraction_profile(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    profile = profile or {}
    ocr = profile.get("ocr") or {}
    table = profile.get("table") or {}
    field_type_overrides = ocr.get("field_type_overrides") or {}
    return {
        "preferred_strategy": profile.get("preferred_strategy"),
        "field_override_count": len(field_type_overrides),
        "table_enabled": bool(table.get("enabled")),
        "table_engine": table.get("ocr_engine"),
    }


def resolve_table_ocr_engine(
    default_engine: Optional[str], family_profile: Optional[Dict[str, Any]]
) -> str:
    table_profile = (family_profile or {}).get("table") or {}
    configured_engine = table_profile.get("ocr_engine")
    if isinstance(configured_engine, str) and configured_engine.strip():
        return configured_engine.strip()
    if isinstance(default_engine, str) and default_engine.strip():
        return default_engine.strip()
    return "paddle"


def resolve_table_detection_config(
    default_config: Optional[Any], family_profile: Optional[Dict[str, Any]]
) -> Optional[Any]:
    table_profile = (family_profile or {}).get("table") or {}
    if not table_profile:
        return default_config

    base = {
        "row_gap_multiplier": getattr(default_config, "row_gap_multiplier", 1.5),
        "min_line_span_fraction": getattr(default_config, "min_line_span_fraction", 0.40),
    }
    if table_profile.get("row_gap_multiplier") is not None:
        base["row_gap_multiplier"] = table_profile["row_gap_multiplier"]
    if table_profile.get("min_line_span_fraction") is not None:
        base["min_line_span_fraction"] = table_profile["min_line_span_fraction"]
    return SimpleNamespace(**base)
