import re
from typing import Any, Dict, List, Optional

_GT_V2_PREFIX = "gt-v2"

_DOCUMENT_FIELD_MAP = {
    "invoiceNumber": "invoiceNumber",
    "invoiceDate": "invoiceDate",
    "sellerName": "sellerName",
    "customerName": "customerName",
    "currency": "currency_code",
    "country": "country_code",
}

_SUMMARY_FIELD_MAP = {
    "subtotal": "subtotal",
    "tax": "tax",
    "discount": "discount",
    "shipping": "shipping",
    "totalAmount": "totalAmount",
}

_LINE_ITEM_FIELD_MAP = {
    "description": "description",
    "quantity": "quantity",
    "unitPrice": "unit_price",
    "amount": "amount",
    "tax": "tax_amount",
    "taxRate": "tax_rate",
    "sku": "sku",
    "itemCode": "item_code",
}

_LEGACY_DOCUMENT_ALIASES = {
    "invoiceNumber": (
        "invoiceNumber",
        "invoice_number",
        "invoice no",
        "invoice no.",
        "invoice number",
        "invoice #",
        "invoice id",
    ),
    "invoiceDate": (
        "invoiceDate",
        "invoice_date",
        "date",
        "issue date",
    ),
    "sellerName": (
        "sellerName",
        "seller_name",
        "seller",
        "vendor",
        "supplier",
        "provider_name",
        "provider",
        "clinic",
        "hospital",
    ),
    "customerName": (
        "customerName",
        "customer_name",
        "customer",
        "client",
        "owner",
        "bill to",
        "billto",
    ),
    "currency": (
        "currency",
        "currency_code",
        "currencyCode",
    ),
    "country": (
        "country",
        "country_code",
        "countryCode",
    ),
}

_LEGACY_SUMMARY_ALIASES = {
    "subtotal": ("subtotal",),
    "tax": ("tax", "vat", "gst", "hst", "pst"),
    "discount": ("discount",),
    "shipping": (
        "shipping",
        "shipping and handling",
        "handling",
        "s&h",
        "s h",
    ),
    "totalAmount": (
        "totalAmount",
        "total",
        "amount due",
        "balance due",
    ),
}

_LEGACY_LINE_ITEM_ALIASES = {
    "description": ("description", "item", "service", "name"),
    "quantity": ("quantity", "qty", "units"),
    "unitPrice": ("unitPrice", "unit_price", "price", "rate"),
    "amount": ("amount", "line_total", "line total", "total"),
    "tax": ("tax", "vat", "gst", "hst", "pst"),
    "taxRate": ("taxRate", "tax_rate", "vat (%)", "vat%", "tax (%)", "tax rate"),
    "sku": ("sku",),
    "itemCode": ("itemCode", "item_code", "code"),
}


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _build_lookup(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    return {_normalize_key(key): value for key, value in data.items()}


def _coerce_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    text = re.sub(r"[^\d.,-]", "", text)
    if not text or not re.search(r"\d", text):
        return None

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        if len(text) - text.rfind(",") == 4:
            text = text.replace(",", "")
        else:
            text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return None


def _extract_legacy_value(field: Any) -> Any:
    if not isinstance(field, dict):
        return field
    if "value" in field:
        return field.get("value")
    if "raw" in field:
        return field.get("raw")
    return None


def _annotate_text_field(field: Any) -> Dict[str, Any]:
    if isinstance(field, dict) and str(field.get("status") or "").strip().lower() in {
        "present",
        "absent",
        "unclear",
        "derived",
    }:
        annotated = {"status": str(field.get("status") or "").strip().lower()}
        if field.get("raw") is not None:
            annotated["raw"] = field.get("raw")
        if field.get("evidence") is not None:
            annotated["evidence"] = field.get("evidence")
        if field.get("confidence") is not None:
            annotated["confidence"] = field.get("confidence")
        value = _extract_legacy_value(field)
        if value not in (None, "") and annotated["status"] in {"present", "derived"}:
            annotated["value"] = str(value)
        elif annotated["status"] == "present":
            annotated["status"] = "absent"
        return annotated

    value = _extract_legacy_value(field)
    if value in (None, ""):
        return {"status": "absent"}
    return {
        "status": "present",
        "value": str(value),
        "raw": value,
    }


def _annotate_number_field(field: Any) -> Dict[str, Any]:
    if isinstance(field, dict) and str(field.get("status") or "").strip().lower() in {
        "present",
        "absent",
        "unclear",
        "derived",
    }:
        annotated = {"status": str(field.get("status") or "").strip().lower()}
        if field.get("raw") is not None:
            annotated["raw"] = field.get("raw")
        if field.get("evidence") is not None:
            annotated["evidence"] = field.get("evidence")
        if field.get("confidence") is not None:
            annotated["confidence"] = field.get("confidence")
        value = _extract_legacy_value(field)
        parsed = _coerce_number(value)
        if parsed is not None and annotated["status"] in {"present", "derived"}:
            annotated["value"] = parsed
        elif value not in (None, "") and annotated["status"] == "present":
            annotated["status"] = "unclear"
        elif annotated["status"] == "present":
            annotated["status"] = "absent"
        return annotated

    value = _extract_legacy_value(field)
    parsed = _coerce_number(value)
    if parsed is None:
        if value in (None, ""):
            return {"status": "absent"}
        return {"status": "unclear", "raw": value}
    return {
        "status": "present",
        "value": parsed,
        "raw": value,
    }


def _select_legacy_field(
    section_lookup: Dict[str, Any], root_lookup: Dict[str, Any], aliases: tuple[str, ...]
) -> Any:
    for alias in aliases:
        normalized = _normalize_key(alias)
        if normalized in section_lookup:
            return section_lookup[normalized]
    for alias in aliases:
        normalized = _normalize_key(alias)
        if normalized in root_lookup:
            return root_lookup[normalized]
    return None


def _upgrade_legacy_line_items(line_items: Any) -> List[Dict[str, Any]]:
    if not isinstance(line_items, list):
        return []

    upgraded: List[Dict[str, Any]] = []
    for index, raw_item in enumerate(line_items, start=1):
        if not isinstance(raw_item, dict):
            continue
        item_lookup = _build_lookup(raw_item)
        upgraded_item: Dict[str, Any] = {
            "index": int(raw_item.get("index"))
            if isinstance(raw_item.get("index"), int)
            else index,
        }
        for field_name, aliases in _LEGACY_LINE_ITEM_ALIASES.items():
            field_value = _select_legacy_field(item_lookup, item_lookup, aliases)
            if field_name in {"quantity", "unitPrice", "amount", "tax", "taxRate"}:
                upgraded_item[field_name] = _annotate_number_field(field_value)
            else:
                upgraded_item[field_name] = _annotate_text_field(field_value)
        upgraded.append(upgraded_item)

    return upgraded


def upgrade_ground_truth_to_v2(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    if is_ground_truth_v2(data):
        return data

    root_lookup = _build_lookup(data)
    document_lookup = _build_lookup(data.get("document"))
    summary_lookup = _build_lookup(data.get("summary"))

    document: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    recognized_fields = 0

    for field_name, aliases in _LEGACY_DOCUMENT_ALIASES.items():
        field_value = _select_legacy_field(document_lookup, root_lookup, aliases)
        annotated = _annotate_text_field(field_value)
        if annotated.get("status") in {"present", "derived"}:
            recognized_fields += 1
        document[field_name] = annotated

    for field_name, aliases in _LEGACY_SUMMARY_ALIASES.items():
        field_value = _select_legacy_field(summary_lookup, root_lookup, aliases)
        annotated = _annotate_number_field(field_value)
        if annotated.get("status") in {"present", "derived"}:
            recognized_fields += 1
        summary[field_name] = annotated

    line_items_raw = data.get("lineItems")
    if line_items_raw is None:
        line_items_raw = data.get("line_items")
    line_items = _upgrade_legacy_line_items(line_items_raw)
    if line_items:
        recognized_fields += 1

    section_present = any(key in data for key in ("document", "summary", "lineItems", "line_items"))
    if recognized_fields == 0 and not section_present:
        return None

    upgraded: Dict[str, Any] = {
        "schemaVersion": "gt-v2",
        "document": document,
        "summary": summary,
        "lineItems": line_items,
    }

    if isinstance(data.get("sourceImage"), str):
        upgraded["sourceImage"] = data.get("sourceImage")
    if isinstance(data.get("annotator"), dict):
        upgraded["annotator"] = data.get("annotator")
    if isinstance(data.get("notes"), list):
        upgraded["notes"] = data.get("notes")

    return upgraded


def is_ground_truth_v2(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    schema_version = str(data.get("schemaVersion") or "").strip().lower()
    return schema_version.startswith(_GT_V2_PREFIX)


def _extract_annotated_value(field: Any) -> Any:
    if not isinstance(field, dict):
        return field

    status = str(field.get("status") or "").strip().lower()
    if not status:
        return field.get("value")

    if status == "present":
        return field.get("value")

    return None


def _canonicalize_section(section: Any, field_map: Dict[str, str]) -> Dict[str, Any]:
    canonical: Dict[str, Any] = {target: None for target in field_map.values()}
    if not isinstance(section, dict):
        return canonical

    for source_name, target_name in field_map.items():
        canonical[target_name] = _extract_annotated_value(section.get(source_name))
    return canonical


def _canonicalize_line_items(line_items: Any) -> List[Dict[str, Any]]:
    canonical_items: List[Dict[str, Any]] = []
    if not isinstance(line_items, list):
        return canonical_items

    for raw_item in line_items:
        if not isinstance(raw_item, dict):
            continue
        canonical_item: Dict[str, Any] = {"index": raw_item.get("index")}
        for source_name, target_name in _LINE_ITEM_FIELD_MAP.items():
            canonical_item[target_name] = _extract_annotated_value(raw_item.get(source_name))
        canonical_items.append(canonical_item)

    return canonical_items


def canonicalize_ground_truth(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict) or not is_ground_truth_v2(data):
        return data

    canonical: Dict[str, Any] = {}
    canonical.update(_canonicalize_section(data.get("document"), _DOCUMENT_FIELD_MAP))
    canonical.update(_canonicalize_section(data.get("summary"), _SUMMARY_FIELD_MAP))
    canonical["lineItems"] = _canonicalize_line_items(data.get("lineItems"))
    return canonical


def normalize_ground_truth(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None

    upgraded = upgrade_ground_truth_to_v2(data)
    if upgraded is None:
        return None
    return canonicalize_ground_truth(upgraded)


def is_ground_truth_discovery_ready(data: Dict[str, Any]) -> bool:
    canonical = normalize_ground_truth(data)
    if not canonical:
        return False

    has_identifier = (
        canonical.get("invoiceNumber") is not None or canonical.get("invoiceDate") is not None
    )
    has_total = canonical.get("totalAmount") is not None
    return has_total and has_identifier
