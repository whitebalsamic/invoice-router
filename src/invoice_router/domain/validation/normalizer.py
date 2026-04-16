import re
from typing import Any, Dict, List, Optional, Sequence

from ...extraction.heuristics.party import _trim_party_block_text
from ...models import DocumentContext
from ..invoices.country_rules import normalize_party_name
from .validator import normalize_amount, normalize_date, normalize_string


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = normalize_string(value)
    return text or None


DATE_LABEL_PREFIX_RE = re.compile(
    r"(?i)^\s*(?:date(?:\s+of\s+issue)?|invoice\s+date|issue\s+date|of\s+issue)\b(?:\s*[:#-]|\s+)?\s*"
)

INVOICE_LABEL_PREFIX_RE = re.compile(
    r"(?i)^\s*(?:invoice(?:\s*(?:no\.?|number|#))?|inv(?:\s+(?:no\.?|number)|\s*#|(?=\s*:)))\s*[:#-]?\s*"
)

SCI_NOTATION_RE = re.compile(r"(?i)\b\d+(?:[.,]\d+)?e[+-]?\d+\b")

PARTY_BLOCK_BOUNDARY_WORDS = {
    "address",
    "attn",
    "city",
    "country",
    "date",
    "description",
    "email",
    "fax",
    "invoice",
    "phone",
    "postal",
    "postcode",
    "province",
    "summary",
    "table",
    "tax",
    "total",
    "subtotal",
    "vat",
    "website",
    "www",
    "zip",
}


def _cleanup_label_prefixed_date_text(value: Any) -> str:
    text = normalize_string(value)
    if not text:
        return ""
    return DATE_LABEL_PREFIX_RE.sub("", text).strip(" :,-|")


def _cleanup_invoice_identifier_text(value: Any) -> str:
    text = normalize_string(value)
    if not text:
        return ""
    text = INVOICE_LABEL_PREFIX_RE.sub("", text).strip(" :,-|")
    text = re.sub(r"[|,;:\s-]+$", "", text).strip()
    tokens = [token for token in text.split() if token]
    if len(tokens) >= 2:
        prefix_tokens: List[str] = []
        for index, token in enumerate(tokens):
            normalized = re.sub(r"[^a-z0-9]", "", token.lower())
            if not normalized:
                continue
            if normalized in PARTY_BLOCK_BOUNDARY_WORDS or normalized in {
                "street",
                "st",
                "suite",
                "unit",
                "apt",
                "road",
                "rd",
                "lane",
                "ln",
                "drive",
                "dr",
                "floor",
            }:
                break
            if re.search(r"https?://|www\.", token, flags=re.IGNORECASE) or "@" in token:
                break
            if index > 0 and re.fullmatch(r"[|/,:;-]+", token):
                break
            prefix_tokens.append(token)
        if prefix_tokens and any(ch.isdigit() for ch in "".join(prefix_tokens)):
            text = " ".join(prefix_tokens).strip(" :,-|")
    return text


def _is_implausible_numeric_text(value: Any) -> bool:
    if value is None:
        return False
    text = re.sub(r"\s+", "", str(value))
    if not text:
        return False
    if SCI_NOTATION_RE.search(text):
        return True
    if re.search(r"[eE][+-]?\d+", text) and re.search(r"\d", text):
        return True
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 12 and not re.search(r"[.,]", text):
        return True
    if len(digits) >= 10 and digits and len(set(digits)) == 1:
        return True
    return False


def _cleanup_party_text(value: Any) -> Optional[str]:
    text = _clean(value)
    if text is None:
        return None
    trimmed = _trim_party_block_text(text)
    if not trimmed or not any(ch.isalpha() for ch in trimmed):
        return None
    return trimmed


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _pick_first(data: Dict[str, Any], candidates: List[str]) -> Any:
    normalized = {_normalize_key(str(key)): value for key, value in data.items()}
    for candidate in candidates:
        candidate_key = _normalize_key(candidate)
        if candidate_key in normalized:
            return normalized[candidate_key]
    return None


def _parse_percent(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = normalize_string(value)
    if not text:
        return None
    text = text.replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def _strip_numeric_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return re.sub(r"[^\d,.\-]", "", str(value))


def _iter_numeric_samples(value: Any) -> Sequence[str]:
    samples: List[str] = []
    if isinstance(value, dict):
        for nested in value.values():
            samples.extend(_iter_numeric_samples(nested))
    elif isinstance(value, list):
        for nested in value:
            samples.extend(_iter_numeric_samples(nested))
    else:
        text = _strip_numeric_text(value)
        if _is_implausible_numeric_text(text):
            return samples
        if re.search(r"\d", text) and ("," in text or "." in text):
            samples.append(text)
    return samples


def _score_numeric_token(token: str, decimal_separator: str, group_separator: str) -> float:
    score = 0.0
    if decimal_separator in token:
        trailing = token.rsplit(decimal_separator, 1)[1]
        if len(trailing) == 2 and trailing.isdigit():
            score += 2.0
        elif len(trailing) in (1, 3):
            score -= 0.5
    if group_separator in token:
        segments = token.split(decimal_separator, 1)[0].split(group_separator)
        if len(segments) > 1 and all(segment.isdigit() for segment in segments):
            if len(segments[0]) in (1, 2, 3) and all(len(segment) == 3 for segment in segments[1:]):
                score += 1.0
            else:
                score -= 1.0
    if decimal_separator in token and group_separator in token:
        rightmost = "," if token.rfind(",") > token.rfind(".") else "."
        if rightmost == decimal_separator:
            score += 2.5
        else:
            score -= 1.5
    return score


def _infer_numeric_convention(
    extracted_data: Dict[str, Any], document_context: Optional[DocumentContext]
) -> Dict[str, Any]:
    samples = list(_iter_numeric_samples(extracted_data))
    if not samples:
        decimal_separator = (
            ","
            if getattr(document_context, "country_code", None)
            in {"DE", "FR", "ES", "IT", "NL", "BE", "PT"}
            else "."
        )
        return {
            "decimal_separator": decimal_separator,
            "group_separator": "." if decimal_separator == "," else ",",
            "confidence": 0.0,
            "sample_count": 0,
            "samples": [],
        }

    dot_score = sum(_score_numeric_token(sample, ".", ",") for sample in samples)
    comma_score = sum(_score_numeric_token(sample, ",", ".") for sample in samples)
    if getattr(document_context, "country_code", None) in {
        "DE",
        "FR",
        "ES",
        "IT",
        "NL",
        "BE",
        "PT",
    }:
        comma_score += 0.75
    decimal_separator = "," if comma_score > dot_score else "."
    group_separator = "." if decimal_separator == "," else ","
    delta = abs(comma_score - dot_score)
    confidence = min(1.0, 0.45 + (delta / max(len(samples), 1)) * 0.15)
    return {
        "decimal_separator": decimal_separator,
        "group_separator": group_separator,
        "confidence": round(confidence, 3),
        "sample_count": len(samples),
        "samples": samples[:10],
    }


def _normalize_amount_with_convention(
    value: Any, numeric_convention: Dict[str, Any]
) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if _is_implausible_numeric_text(value):
        return None

    text = _strip_numeric_text(value)
    if not text or not re.search(r"\d", text):
        return None

    decimal_separator = numeric_convention.get("decimal_separator", ".")
    group_separator = numeric_convention.get("group_separator", ",")

    if decimal_separator in text and group_separator in text:
        text = text.replace(group_separator, "")
        text = text.replace(decimal_separator, ".")
    elif group_separator in text and decimal_separator != group_separator:
        parts = text.split(group_separator)
        if (
            len(parts) > 1
            and all(part.isdigit() for part in parts)
            and len(parts[-1]) == 3
            and all(len(part) == 3 for part in parts[1:])
        ):
            text = "".join(parts)
        elif decimal_separator == group_separator:
            text = text.replace(group_separator, ".")
    elif decimal_separator in text and decimal_separator != ".":
        text = text.replace(decimal_separator, ".")

    if text.count(".") > 1:
        head, tail = text.rsplit(".", 1)
        text = head.replace(".", "") + "." + tail

    try:
        return round(float(text), 2)
    except ValueError:
        fallback = normalize_amount(value)
        return round(fallback, 2) if fallback is not None else None


def _parse_percent_with_convention(
    value: Any, numeric_convention: Dict[str, Any]
) -> Optional[float]:
    if value is None:
        return None
    text = normalize_string(value)
    if not text:
        return None
    if _is_implausible_numeric_text(text):
        return None
    text = text.replace("%", "")
    return _normalize_amount_with_convention(text, numeric_convention)


def _build_field_provenance(
    field_name: str,
    raw_value: Any,
    parsed_value: Any,
    kind: str,
    source: str,
    numeric_convention: Dict[str, Any],
    supporting_fields: Optional[List[str]] = None,
    evidence: Optional[List[str]] = None,
) -> Dict[str, Any]:
    provenance: Dict[str, Any] = {
        "field": field_name,
        "kind": kind,
        "source": source,
        "raw_value": raw_value,
        "parsed_value": parsed_value,
    }
    if raw_value is not None and isinstance(raw_value, str) and re.search(r"[\d][,.]", raw_value):
        provenance["numeric_convention"] = {
            "decimal_separator": numeric_convention.get("decimal_separator"),
            "group_separator": numeric_convention.get("group_separator"),
            "confidence": numeric_convention.get("confidence"),
        }
    if supporting_fields:
        provenance["supporting_fields"] = supporting_fields
    if evidence:
        provenance["evidence"] = evidence
    return provenance


def _approx_equal(val1: Optional[float], val2: Optional[float], tolerance: float = 0.02) -> bool:
    if val1 is None or val2 is None:
        return False
    return abs(val1 - val2) <= tolerance


def _build_reconciliation_summary(
    normalized: Dict[str, Any],
    *,
    line_sum: Optional[float],
    line_net_sum: Optional[float],
    line_gross_sum: Optional[float],
    line_tax_sum: Optional[float],
    fully_priced_line_items: bool,
    row_tax_breakdown_reliable: bool,
    supports_invoice_gross_total: bool,
    field_provenance: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    subtotal = normalized.get("subtotal")
    tax = normalized.get("tax")
    shipping = normalized.get("shipping") or 0.0
    total = normalized.get("total")
    line_items = normalized.get("line_items") or []

    issues: List[Dict[str, Any]] = []

    if line_items and not fully_priced_line_items:
        issues.append(
            {
                "kind": "incomplete_line_items",
                "field": "subtotal",
                "message": "Line-item arithmetic is incomplete, so subtotal support is only partial.",
            }
        )
    if (
        line_sum is not None
        and subtotal is not None
        and fully_priced_line_items
        and not _approx_equal(line_sum, subtotal)
    ):
        issues.append(
            {
                "kind": "subtotal_mismatch",
                "field": "subtotal",
                "message": f"Subtotal mismatch: expected line sum {line_sum:.2f}, got {subtotal:.2f}",
            }
        )
    if (
        line_tax_sum is not None
        and tax is not None
        and row_tax_breakdown_reliable
        and not _approx_equal(line_tax_sum, tax)
    ):
        issues.append(
            {
                "kind": "tax_mismatch",
                "field": "tax",
                "message": f"Tax mismatch: expected line tax sum {line_tax_sum:.2f}, got {tax:.2f}",
            }
        )
    if subtotal is not None and tax is not None and total is not None:
        expected_total = round(subtotal + tax + shipping, 2)
        if not _approx_equal(expected_total, total):
            if shipping:
                message = f"Total mismatch: expected subtotal+tax+shipping {expected_total:.2f}, got {total:.2f}"
            else:
                message = (
                    f"Total mismatch: expected subtotal+tax {expected_total:.2f}, got {total:.2f}"
                )
            issues.append(
                {
                    "kind": "total_mismatch",
                    "field": "total",
                    "message": message,
                }
            )
    if (
        line_gross_sum is not None
        and total is not None
        and supports_invoice_gross_total
        and not _approx_equal(line_gross_sum, total)
    ):
        issues.append(
            {
                "kind": "gross_total_mismatch",
                "field": "total",
                "message": f"Total mismatch: expected gross line sum {line_gross_sum:.2f}, got {total:.2f}",
            }
        )

    derived_fields = sorted(
        field_name
        for field_name, provenance in field_provenance.items()
        if provenance.get("source")
        in {"derived", "resolved", "line_items", "summary_reconciliation"}
    )
    issue_types = [issue["kind"] for issue in issues]
    summary_formula_complete = (
        subtotal is not None
        and total is not None
        and (tax is not None or shipping not in (None, 0.0))
    )
    if any(issue_type.endswith("_mismatch") for issue_type in issue_types):
        status = "inconsistent"
    elif derived_fields or fully_priced_line_items or (subtotal is not None and total is not None):
        status = "reconciled"
    elif line_items or subtotal is not None or total is not None or tax is not None:
        status = "partial"
    else:
        status = "unavailable"

    return {
        "status": status,
        "issue_count": len(issues),
        "issue_types": issue_types,
        "issues": issues,
        "derived_fields": derived_fields,
        "supports": {
            "line_items_present": bool(line_items),
            "fully_priced_line_items": fully_priced_line_items,
            "row_tax_breakdown_reliable": row_tax_breakdown_reliable,
            "supports_invoice_gross_total": supports_invoice_gross_total,
            "summary_formula_complete": summary_formula_complete,
        },
        "line_metrics": {
            "subtotal": line_sum,
            "net_subtotal": line_net_sum,
            "tax": line_tax_sum if row_tax_breakdown_reliable else None,
            "gross_total": line_gross_sum if supports_invoice_gross_total else None,
        },
        "summary_metrics": {
            "subtotal": subtotal,
            "tax": tax,
            "shipping": shipping,
            "total": total,
        },
    }


def _normalize_line_item(row: Dict[str, Any], numeric_convention: Dict[str, Any]) -> Dict[str, Any]:
    description = _pick_first(row, ["description", "item", "service", "name"])
    quantity = _pick_first(row, ["qty", "quantity", "units"])
    unit_price = _pick_first(row, ["unit_price", "price", "rate", "unit net", "net unit price"])
    explicit_net_amount = _pick_first(row, ["net worth", "net amount", "net total"])
    plain_amount = _pick_first(row, ["amount", "total", "line_total"])
    gross_amount = _pick_first(row, ["gross worth", "gross amount", "gross total"])
    amount = (
        explicit_net_amount
        if explicit_net_amount is not None
        else plain_amount
        if plain_amount is not None
        else gross_amount
    )
    tax_rate = _pick_first(
        row, ["vat (%)", "vat%", "tax (%)", "tax rate", "gst (%)", "hst (%)", "pst (%)"]
    )

    quantity_value = (
        _normalize_amount_with_convention(quantity, numeric_convention)
        if quantity is not None
        else None
    )
    unit_price_value = (
        _normalize_amount_with_convention(unit_price, numeric_convention)
        if unit_price is not None
        else None
    )
    amount_value = (
        _normalize_amount_with_convention(amount, numeric_convention)
        if amount is not None
        else None
    )
    gross_amount_value = (
        _normalize_amount_with_convention(gross_amount, numeric_convention)
        if gross_amount is not None
        else None
    )
    net_amount_value = (
        _normalize_amount_with_convention(explicit_net_amount, numeric_convention)
        if explicit_net_amount is not None
        else None
    )
    tax_rate_value = _parse_percent_with_convention(tax_rate, numeric_convention)

    if gross_amount_value is not None and tax_rate_value not in (None, 0.0):
        derived_net_from_gross = round(gross_amount_value / (1.0 + (tax_rate_value / 100.0)), 2)
        if (
            amount_value is None
            or amount_value <= max(1.0, derived_net_from_gross * 0.25)
            or abs((amount_value * (1.0 + (tax_rate_value / 100.0))) - gross_amount_value)
            > max(0.05, gross_amount_value * 0.03)
        ):
            amount_value = derived_net_from_gross
        if net_amount_value is None:
            net_amount_value = amount_value

    if amount_value is None and quantity_value is not None and unit_price_value is not None:
        amount_value = round(quantity_value * unit_price_value, 2)
    elif (
        amount_value is not None
        and quantity_value is not None
        and unit_price_value is not None
        and quantity_value > 1.0
        and abs(amount_value - unit_price_value) <= 0.02
    ):
        # Some table parses duplicate the unit price into the total column.
        # When quantity is clearly >1 and amount equals the unit price, use row arithmetic.
        amount_value = round(quantity_value * unit_price_value, 2)
    elif (
        amount_value is not None
        and quantity_value is not None
        and unit_price_value is not None
        and quantity_value > 1.0
    ):
        expected_total = round(quantity_value * unit_price_value, 2)
        derived_unit_price = round(amount_value / quantity_value, 2) if quantity_value else None
        if (
            derived_unit_price is not None
            and abs(expected_total - amount_value) > max(0.02, amount_value * 0.015)
            and (
                abs(derived_unit_price - unit_price_value) <= max(3.0, unit_price_value * 0.1)
                or unit_price_value <= max(1.0, derived_unit_price * 0.25)
            )
        ):
            # When the total is plausible but the OCR price is slightly noisy, align unit price to row total.
            unit_price_value = derived_unit_price

    tax_amount_value = None
    if gross_amount_value is not None and net_amount_value is not None:
        tax_amount_value = round(gross_amount_value - net_amount_value, 2)
    elif tax_rate_value is not None and net_amount_value is not None:
        tax_amount_value = round(net_amount_value * (tax_rate_value / 100.0), 2)

    normalized = {
        "description": _clean(description),
        "quantity": quantity_value,
        "unit_price": unit_price_value,
        "amount": amount_value,
        "net_amount": net_amount_value,
        "gross_amount": gross_amount_value,
        "tax_rate": tax_rate_value,
        "tax_amount": tax_amount_value,
        "raw": row,
        "_provenance": {
            "quantity": _build_field_provenance(
                "quantity", quantity, quantity_value, "normalized", "line_item", numeric_convention
            ),
            "unit_price": _build_field_provenance(
                "unit_price",
                unit_price,
                unit_price_value,
                "normalized",
                "line_item",
                numeric_convention,
            ),
            "amount": _build_field_provenance(
                "amount", amount, amount_value, "normalized", "line_item", numeric_convention
            ),
            "net_amount": _build_field_provenance(
                "net_amount",
                explicit_net_amount,
                net_amount_value,
                "normalized",
                "line_item",
                numeric_convention,
            ),
            "gross_amount": _build_field_provenance(
                "gross_amount",
                gross_amount,
                gross_amount_value,
                "normalized",
                "line_item",
                numeric_convention,
            ),
            "tax_rate": _build_field_provenance(
                "tax_rate", tax_rate, tax_rate_value, "normalized", "line_item", numeric_convention
            ),
        },
    }

    text = (normalized["description"] or "").lower()
    if any(token in text for token in ("consult", "exam", "visit", "recheck", "evaluation")):
        normalized["category"] = "consultation"
    elif any(
        token in text
        for token in ("surgery", "surgical", "spay", "neuter", "anesthesia", "anaesthesia")
    ):
        normalized["category"] = "surgery"
    elif any(
        token in text
        for token in (
            "lab",
            "x-ray",
            "xray",
            "scan",
            "diagnostic",
            "test",
            "ultrasound",
            "radiograph",
            "bloodwork",
        )
    ):
        normalized["category"] = "diagnostics"
    elif any(
        token in text
        for token in (
            "tablet",
            "capsule",
            "drug",
            "med",
            "injection",
            "antibiotic",
            "prescription",
            "dose",
            "vaccine",
            "vaccination",
        )
    ):
        normalized["category"] = "medication"
    elif any(
        token in text
        for token in ("bandage", "consumable", "syringe", "catheter", "supplies", "supply")
    ):
        normalized["category"] = "consumables"
    elif any(token in text for token in ("hospital", "boarding", "overnight", "icu", "ward")):
        normalized["category"] = "hospitalization"
    else:
        normalized["category"] = "other"
    return normalized


def normalize_extracted_invoice(
    extracted_data: Optional[Dict[str, Any]],
    document_context: Optional[DocumentContext] = None,
) -> Dict[str, Any]:
    if not extracted_data:
        return {}

    invoice_number = _pick_first(
        extracted_data,
        [
            "invoice_number",
            "invoice no",
            "invoice no.",
            "invoice #",
            "invoice number",
            "invoiceNumber",
        ],
    )
    invoice_date = _pick_first(extracted_data, ["invoice_date", "date", "invoiceDate"])
    subtotal = _pick_first(extracted_data, ["subtotal"])
    tax = _pick_first(extracted_data, ["tax", "gst", "hst", "pst", "vat"])
    shipping = _pick_first(
        extracted_data, ["shipping", "shipping and handling", "handling", "s&h", "s h"]
    )
    total = _pick_first(extracted_data, ["totalAmount", "total", "amount due", "balance due"])
    provider_name = _pick_first(
        extracted_data,
        ["provider_name", "sellerName", "vendor", "seller", "supplier", "clinic", "hospital"],
    )
    customer_name = _pick_first(
        extracted_data,
        ["customer_name", "customerName", "customer", "bill to", "billto", "client", "owner"],
    )
    discount = _pick_first(extracted_data, ["discount"])

    line_items_raw = extracted_data.get("line_items") or extracted_data.get("lineItems") or []
    numeric_convention = _infer_numeric_convention(extracted_data, document_context)
    normalized_line_items = [
        _normalize_line_item(row, numeric_convention)
        for row in line_items_raw
        if isinstance(row, dict)
    ]

    raw_invoice_number = invoice_number
    raw_invoice_date = invoice_date
    raw_subtotal = subtotal
    raw_tax = tax
    raw_shipping = shipping
    raw_total = total
    raw_provider_name = provider_name
    raw_customer_name = customer_name
    raw_discount = discount

    normalized = {
        "invoice_number": _clean(_cleanup_invoice_identifier_text(invoice_number)),
        "invoice_date": (
            normalize_date(
                cleaned_invoice_date,
                country_code=document_context.country_code if document_context else None,
            )
            if (
                invoice_date is not None
                and (cleaned_invoice_date := _cleanup_label_prefixed_date_text(invoice_date))
            )
            else None
        ),
        "subtotal": _normalize_amount_with_convention(subtotal, numeric_convention)
        if subtotal is not None
        else None,
        "tax": _normalize_amount_with_convention(tax, numeric_convention)
        if tax is not None
        else None,
        "shipping": _normalize_amount_with_convention(shipping, numeric_convention)
        if shipping is not None
        else None,
        "total": _normalize_amount_with_convention(total, numeric_convention)
        if total is not None
        else None,
        "discount": _normalize_amount_with_convention(discount, numeric_convention)
        if discount is not None
        else None,
        "provider_name": normalize_party_name(_cleanup_party_text(provider_name))
        or (
            document_context.provider_match.provider_name
            if document_context and document_context.provider_match
            else None
        ),
        "customer_name": normalize_party_name(_cleanup_party_text(customer_name)),
        "country_code": document_context.country_code if document_context else None,
        "currency_code": document_context.currency_code if document_context else None,
        "document_family": document_context.document_family.value if document_context else None,
        "line_items": normalized_line_items,
        "numeric_convention": numeric_convention,
    }

    field_provenance = {
        "invoice_number": _build_field_provenance(
            "invoice_number",
            raw_invoice_number,
            normalized["invoice_number"],
            "normalized",
            "scalar",
            numeric_convention,
        ),
        "invoice_date": _build_field_provenance(
            "invoice_date",
            raw_invoice_date,
            normalized["invoice_date"],
            "normalized",
            "scalar",
            numeric_convention,
        ),
        "subtotal": _build_field_provenance(
            "subtotal",
            raw_subtotal,
            normalized["subtotal"],
            "normalized",
            "scalar",
            numeric_convention,
        ),
        "tax": _build_field_provenance(
            "tax", raw_tax, normalized["tax"], "normalized", "scalar", numeric_convention
        ),
        "shipping": _build_field_provenance(
            "shipping",
            raw_shipping,
            normalized["shipping"],
            "normalized",
            "scalar",
            numeric_convention,
        ),
        "total": _build_field_provenance(
            "total", raw_total, normalized["total"], "normalized", "scalar", numeric_convention
        ),
        "discount": _build_field_provenance(
            "discount",
            raw_discount,
            normalized["discount"],
            "normalized",
            "scalar",
            numeric_convention,
        ),
        "provider_name": _build_field_provenance(
            "provider_name",
            raw_provider_name,
            normalized["provider_name"],
            "normalized",
            "scalar",
            numeric_convention,
        ),
        "customer_name": _build_field_provenance(
            "customer_name",
            raw_customer_name,
            normalized["customer_name"],
            "normalized",
            "scalar",
            numeric_convention,
        ),
    }

    raw_line_amounts = [
        _normalize_amount_with_convention(item.get("raw", {}).get("amount"), numeric_convention)
        for item in normalized_line_items
    ]
    raw_line_sum = (
        round(sum(value for value in raw_line_amounts if value is not None), 2)
        if any(value is not None for value in raw_line_amounts)
        else None
    )
    normalized_line_sum = (
        round(
            sum(item["amount"] for item in normalized_line_items if item.get("amount") is not None),
            2,
        )
        if normalized_line_items
        else None
    )
    subtotal_value = normalized.get("subtotal")
    total_value = normalized.get("total")
    shipping_value = normalized.get("shipping") or 0.0
    tax_value = normalized.get("tax")

    raw_sum_supported_by_summary = (
        raw_line_sum is not None
        and subtotal_value is not None
        and abs(subtotal_value - raw_line_sum) <= 0.02
        and normalized_line_sum is not None
        and abs(normalized_line_sum - subtotal_value) > 0.02
        and (
            total_value is None
            or (
                tax_value is not None
                and abs(total_value - (raw_line_sum + tax_value + shipping_value)) <= 0.02
            )
            or (tax_value is None and total_value + 0.02 >= (raw_line_sum + shipping_value))
        )
    )
    if raw_sum_supported_by_summary:
        for item, raw_amount in zip(normalized_line_items, raw_line_amounts):
            if raw_amount is None or item.get("amount") is None:
                continue
            if abs(item["amount"] - raw_amount) <= 0.02:
                continue
            quantity = item.get("quantity")
            unit_price = item.get("unit_price")
            expected_total = (
                round((quantity or 0.0) * (unit_price or 0.0), 2)
                if quantity is not None and unit_price is not None
                else None
            )
            if expected_total is None or abs(item["amount"] - expected_total) > 0.02:
                continue
            item["amount"] = raw_amount
            item["_provenance"]["amount"] = _build_field_provenance(
                "amount",
                item.get("raw", {}).get("amount"),
                raw_amount,
                "resolved",
                "summary_consistency",
                numeric_convention,
                supporting_fields=["subtotal", "total", "tax", "shipping"],
                evidence=[
                    f"raw_line_sum={raw_line_sum:.2f}",
                    f"normalized_line_sum={normalized_line_sum:.2f}",
                ],
            )

    line_amounts = [
        item["amount"] for item in normalized_line_items if item.get("amount") is not None
    ]
    line_net_amounts = [
        item["net_amount"] for item in normalized_line_items if item.get("net_amount") is not None
    ]
    line_gross_amounts = [
        item["gross_amount"]
        for item in normalized_line_items
        if item.get("gross_amount") is not None
    ]
    line_tax_amounts = [
        item["tax_amount"] for item in normalized_line_items if item.get("tax_amount") is not None
    ]
    line_sum = round(sum(line_amounts), 2) if line_amounts else None
    line_net_sum = round(sum(line_net_amounts), 2) if line_net_amounts else None
    line_gross_sum = round(sum(line_gross_amounts), 2) if line_gross_amounts else None
    line_tax_sum = round(sum(line_tax_amounts), 2) if line_tax_amounts else None
    fully_priced_line_items = bool(normalized_line_items) and all(
        item.get("quantity") is not None
        and item.get("unit_price") is not None
        and item.get("amount") is not None
        for item in normalized_line_items
    )
    complete_row_tax_breakdown = bool(normalized_line_items) and len(line_tax_amounts) == len(
        normalized_line_items
    )
    complete_row_gross_breakdown = bool(normalized_line_items) and len(line_gross_amounts) == len(
        normalized_line_items
    )
    complete_row_net_breakdown = bool(normalized_line_items) and len(line_net_amounts) == len(
        normalized_line_items
    )
    row_tax_breakdown_reliable = complete_row_tax_breakdown and all(
        (item.get("tax_amount") is not None and item.get("tax_amount") >= -0.02)
        for item in normalized_line_items
    )
    if row_tax_breakdown_reliable and complete_row_gross_breakdown and complete_row_net_breakdown:
        if any(
            item.get("gross_amount") is not None
            and item.get("net_amount") is not None
            and item["gross_amount"] + 0.02 < item["net_amount"]
            for item in normalized_line_items
        ):
            row_tax_breakdown_reliable = False
        elif any(
            (item.get("tax_rate") or 0.0) > 0.0 and (item.get("net_amount") or 0.0) > 0.0
            for item in normalized_line_items
        ) and all((item.get("tax_amount") or 0.0) <= 0.02 for item in normalized_line_items):
            row_tax_breakdown_reliable = False
    supports_invoice_gross_total = complete_row_gross_breakdown and (
        complete_row_net_breakdown or row_tax_breakdown_reliable
    )
    scalar_total_supports_line_sum = line_sum is not None and (
        normalized["total"] is None
        or (
            normalized["tax"] is not None
            and abs(
                normalized["total"]
                - (line_sum + normalized["tax"] + (normalized["shipping"] or 0.0))
            )
            <= 0.02
        )
        or (
            normalized["tax"] is None
            and normalized["total"] + 0.02 >= (line_sum + (normalized["shipping"] or 0.0))
        )
    )
    scalar_summary_already_reconciled = (
        normalized["subtotal"] is not None
        and normalized["total"] is not None
        and abs(normalized["total"] - (normalized["subtotal"] + (normalized["shipping"] or 0.0)))
        <= 0.02
    )

    if normalized["subtotal"] is None and line_sum is not None:
        normalized["subtotal"] = line_sum or None
        field_provenance["subtotal"] = _build_field_provenance(
            "subtotal",
            raw_subtotal,
            normalized["subtotal"],
            "derived",
            "line_items",
            numeric_convention,
            supporting_fields=["line_items.amount"],
            evidence=[f"sum(line_items.amount)={line_sum:.2f}"],
        )
    elif normalized["subtotal"] is not None and line_sum is not None and fully_priced_line_items:
        if (
            abs(normalized["subtotal"] - line_sum) > 0.02
            and scalar_total_supports_line_sum
            and not scalar_summary_already_reconciled
        ):
            normalized["subtotal"] = line_sum
            field_provenance["subtotal"] = _build_field_provenance(
                "subtotal",
                raw_subtotal,
                normalized["subtotal"],
                "resolved",
                "line_items",
                numeric_convention,
                supporting_fields=["line_items.amount", "total", "tax"],
                evidence=[
                    f"sum(line_items.amount)={line_sum:.2f}",
                    "scalar total supported row subtotal",
                ],
            )
    elif (
        normalized["subtotal"] is not None
        and line_net_sum is not None
        and complete_row_net_breakdown
    ):
        if abs(normalized["subtotal"] - line_net_sum) > 0.02:
            normalized["subtotal"] = line_net_sum
            field_provenance["subtotal"] = _build_field_provenance(
                "subtotal",
                raw_subtotal,
                normalized["subtotal"],
                "resolved",
                "line_items",
                numeric_convention,
                supporting_fields=["line_items.net_amount"],
                evidence=[f"sum(line_items.net_amount)={line_net_sum:.2f}"],
            )

    if normalized["tax"] is None and line_tax_sum is not None:
        if row_tax_breakdown_reliable:
            normalized["tax"] = line_tax_sum or None
            field_provenance["tax"] = _build_field_provenance(
                "tax",
                raw_tax,
                normalized["tax"],
                "derived",
                "line_items",
                numeric_convention,
                supporting_fields=["line_items.tax_amount"],
                evidence=[f"sum(line_items.tax_amount)={line_tax_sum:.2f}"],
            )
    elif normalized["tax"] is not None and line_tax_sum is not None and row_tax_breakdown_reliable:
        if abs(normalized["tax"] - line_tax_sum) > 0.02:
            normalized["tax"] = line_tax_sum
            field_provenance["tax"] = _build_field_provenance(
                "tax",
                raw_tax,
                normalized["tax"],
                "resolved",
                "line_items",
                numeric_convention,
                supporting_fields=["line_items.tax_amount"],
                evidence=[f"sum(line_items.tax_amount)={line_tax_sum:.2f}"],
            )

    if (
        normalized["tax"] is None
        and normalized["subtotal"] is not None
        and normalized["total"] is not None
        and normalized["total"] + 0.02 >= normalized["subtotal"] + (normalized["shipping"] or 0.0)
    ):
        derived_tax = round(
            normalized["total"] - normalized["subtotal"] - (normalized["shipping"] or 0.0), 2
        )
        if derived_tax > 0.02:
            supporting_fields = ["subtotal", "total"]
            evidence = [f"total-subtotal={derived_tax:.2f}"]
            if normalized["shipping"] not in (None, 0.0):
                supporting_fields.insert(1, "shipping")
                evidence = [f"total-subtotal-shipping={derived_tax:.2f}"]
            normalized["tax"] = derived_tax
            field_provenance["tax"] = _build_field_provenance(
                "tax",
                raw_tax,
                normalized["tax"],
                "derived",
                "summary_reconciliation",
                numeric_convention,
                supporting_fields=supporting_fields,
                evidence=evidence,
            )

    if normalized["total"] is None and line_gross_sum is not None and supports_invoice_gross_total:
        normalized["total"] = line_gross_sum or None
        field_provenance["total"] = _build_field_provenance(
            "total",
            raw_total,
            normalized["total"],
            "derived",
            "line_items",
            numeric_convention,
            supporting_fields=["line_items.gross_amount"],
            evidence=[f"sum(line_items.gross_amount)={line_gross_sum:.2f}"],
        )
    elif (
        normalized["total"] is not None
        and line_gross_sum is not None
        and supports_invoice_gross_total
    ):
        if abs(normalized["total"] - line_gross_sum) > 0.02:
            normalized["total"] = line_gross_sum
            field_provenance["total"] = _build_field_provenance(
                "total",
                raw_total,
                normalized["total"],
                "resolved",
                "line_items",
                numeric_convention,
                supporting_fields=["line_items.gross_amount"],
                evidence=[f"sum(line_items.gross_amount)={line_gross_sum:.2f}"],
            )

    if (
        normalized["total"] is None
        and normalized["subtotal"] is not None
        and normalized["tax"] is not None
    ):
        normalized["total"] = (
            normalized["subtotal"] + normalized["tax"] + (normalized["shipping"] or 0.0)
        )
        supporting_fields = ["subtotal", "tax"]
        evidence = [f"subtotal+tax={normalized['total']:.2f}"]
        if normalized["shipping"] not in (None, 0.0):
            supporting_fields.append("shipping")
            evidence = [f"subtotal+tax+shipping={normalized['total']:.2f}"]
        field_provenance["total"] = _build_field_provenance(
            "total",
            raw_total,
            normalized["total"],
            "derived",
            "summary_reconciliation",
            numeric_convention,
            supporting_fields=supporting_fields,
            evidence=evidence,
        )

    category_totals: Dict[str, float] = {}
    for item in normalized_line_items:
        amount = item.get("amount")
        category = item.get("category") or "other"
        if amount is None:
            continue
        category_totals[category] = round(category_totals.get(category, 0.0) + amount, 2)

    normalized["category_totals"] = category_totals
    normalized["line_item_count"] = len(normalized_line_items)
    normalized["has_line_items"] = bool(normalized_line_items)
    normalized["invoiceNumber"] = normalized["invoice_number"]
    normalized["invoiceDate"] = normalized["invoice_date"]
    normalized["sellerName"] = normalized["provider_name"]
    normalized["customerName"] = normalized["customer_name"]
    normalized["totalAmount"] = normalized["total"]
    normalized["lineItems"] = normalized_line_items
    normalized["field_provenance"] = field_provenance
    normalized["reconciliation_summary"] = _build_reconciliation_summary(
        normalized,
        line_sum=line_sum,
        line_net_sum=line_net_sum,
        line_gross_sum=line_gross_sum,
        line_tax_sum=line_tax_sum,
        fully_priced_line_items=fully_priced_line_items,
        row_tax_breakdown_reliable=row_tax_breakdown_reliable,
        supports_invoice_gross_total=supports_invoice_gross_total,
        field_provenance=field_provenance,
    )
    return normalized


def merge_for_validation(
    extracted_data: Optional[Dict[str, Any]], normalized_data: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if extracted_data:
        merged.update(extracted_data)
    if normalized_data:
        for key, value in normalized_data.items():
            if value is not None:
                merged[key] = value
    return merged
