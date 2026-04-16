import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from dateutil.parser import parse as parse_date

from . import summary as _heuristic_summary

OCRToken = Tuple[str, int, int, int, int]


def _heuristic_party_module():
    from . import party as module

    return module


def _heuristic_table_module():
    from . import table as module

    return module


FIELD_SPECS: Dict[str, Dict[str, Any]] = {
    "invoiceNumber": {
        "aliases": ["invoice no", "invoice no.", "invoice number", "invoice #", "inv no", "inv #"],
        "field_type": "string",
        "anchor_direction": "right_of_label",
        "critical": True,
    },
    "invoiceDate": {
        "aliases": ["invoice date", "date", "issue date", "billing date"],
        "field_type": "date",
        "anchor_direction": "right_of_label",
        "critical": True,
    },
    "dueDate": {
        "aliases": ["due date", "payment due", "due"],
        "field_type": "date",
        "anchor_direction": "right_of_label",
        "critical": False,
    },
    "purchaseOrder": {
        "aliases": ["po", "po number", "purchase order", "purchase order number"],
        "field_type": "string",
        "anchor_direction": "right_of_label",
        "critical": False,
    },
    "subtotal": {
        "aliases": ["subtotal", "sub total", "net amount"],
        "field_type": "currency",
        "anchor_direction": "right_of_label",
        "critical": False,
    },
    "tax": {
        "aliases": ["tax", "vat", "gst", "sales tax"],
        "field_type": "currency",
        "anchor_direction": "right_of_label",
        "critical": False,
    },
    "shipping": {
        "aliases": ["shipping", "shipping and handling", "handling", "s&h", "s h"],
        "field_type": "currency",
        "anchor_direction": "right_of_label",
        "critical": False,
    },
    "total": {
        "aliases": ["total", "invoice total", "amount due", "balance due", "total due"],
        "field_type": "currency",
        "anchor_direction": "right_of_label",
        "critical": True,
    },
    "customerName": {
        "aliases": ["bill to", "billto", "customer", "sold to", "invoice to", "client", "owner"],
        "field_type": "address",
        "anchor_direction": "below_label",
        "critical": False,
    },
    "sellerName": {
        "aliases": [
            "seller",
            "vendor",
            "supplier",
            "from",
            "bill from",
            "billfrom",
            "remit to",
            "sold by",
        ],
        "field_type": "address",
        "anchor_direction": "below_label",
        "critical": False,
    },
}

TABLE_HEADER_ALIASES = {
    "description": ["description", "item", "product", "details", "service"],
    "quantity": ["qty", "quantity", "hours", "units"],
    "amount": ["amount", "total", "line total", "price", "unit price", "rate"],
}

OCR_LABEL_TRANSLATIONS = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "5": "s",
        "7": "t",
    }
)

GENERIC_LABEL_WORDS = {
    "company",
    "from",
    "date",
    "details",
    "number",
    "customer",
    "vendor",
}

ADDRESS_STOP_WORDS = {
    "invoice",
    "invoice no",
    "invoice number",
    "date",
    "due date",
    "subtotal",
    "tax",
    "total",
    "amount due",
    "balance due",
}

BUSINESS_SUFFIXES = {
    "ltd",
    "limited",
    "llc",
    "inc",
    "corp",
    "co",
    "company",
    "gmbh",
    "plc",
    "pty",
    "ag",
    "bv",
}

PARTY_ROLE_LABELS = {
    "vendor_block": {"vendor", "seller", "supplier", "from", "sold by"},
    "bill_to_block": {"bill to", "invoice to", "sold to", "customer", "client", "owner"},
    "ship_to_block": {"ship to", "deliver to", "delivery to"},
    "remit_to_block": {"remit to", "payment to", "pay to"},
}

SUMMARY_ROLE_LABELS = {
    "subtotal": {"subtotal", "sub total", "net amount"},
    "tax": {"tax", "sales tax", "vat", "gst"},
    "invoice_total": {"invoice total", "total"},
    "amount_due": {"amount due", "total due"},
    "balance_due": {"balance due"},
    "shipping": {"shipping", "shipping and handling", "handling", "s&h", "s h"},
}


@dataclass
class CandidateLabel:
    page_index: int
    field_name: str
    label_text: str
    score: float
    bbox: Tuple[int, int, int, int]


def normalize_ocr_text(value: str) -> str:
    lowered = value.lower().translate(OCR_LABEL_TRANSLATIONS)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", lowered)).strip()


def _join_tokens(tokens: Sequence[OCRToken]) -> str:
    return " ".join(
        token[0].strip()
        for token in sorted(tokens, key=lambda item: (item[2], item[1]))
        if token[0].strip()
    ).strip()


def _join_tokens_left_to_right(tokens: Sequence[OCRToken]) -> str:
    return " ".join(
        token[0].strip() for token in sorted(tokens, key=lambda item: item[1]) if token[0].strip()
    ).strip()


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _word_count(value: str) -> int:
    return len([token for token in normalize_ocr_text(value).split() if token])


def _position_prior(field_name: Optional[str], x: int, y: int, page_w: int, page_h: int) -> float:
    if not field_name or page_w <= 0 or page_h <= 0:
        return 0.0

    field_key = _normalize_key(field_name)
    x_ratio = x / page_w
    y_ratio = y / page_h

    if field_key in {"invoicenumber", "ponumber", "purchaseorder"}:
        return max(0.0, 0.16 - (y_ratio * 0.18)) + (0.04 if x_ratio < 0.7 else 0.0)
    if "date" in field_key:
        return max(0.0, 0.14 - (y_ratio * 0.16)) + (0.03 if x_ratio < 0.75 else 0.0)
    if field_key in {"total", "totalamount"} or ("amount" in field_key and "tax" not in field_key):
        return (0.12 if x_ratio > 0.55 else 0.0) + (0.14 if y_ratio > 0.55 else 0.0)
    if any(token in field_key for token in ("subtotal", "tax", "shipping", "discount")):
        return (0.08 if x_ratio > 0.5 else 0.0) + (0.10 if y_ratio > 0.5 else 0.0)
    if field_key == "sellername":
        return (0.12 if y_ratio < 0.28 else 0.0) + (0.05 if x_ratio < 0.55 else 0.0)
    if field_key == "customername":
        return (0.06 if 0.08 <= y_ratio <= 0.55 else 0.0) + (0.03 if x_ratio < 0.7 else 0.0)
    return 0.0


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(
                min(
                    curr[j - 1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + (ca != cb),
                )
            )
        prev = curr
    return prev[-1]


def _fuzzy_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    distance = _levenshtein(left, right)
    return max(0.0, 1.0 - (distance / max(len(left), len(right), 1)))


def group_tokens_into_rows(
    tokens: Sequence[OCRToken], row_gap_multiplier: float = 1.5
) -> List[List[OCRToken]]:
    if not tokens:
        return []

    sorted_tokens = sorted(tokens, key=lambda item: (item[2] + item[4] / 2.0, item[1]))
    heights = [max(h, 1) for _, _, _, _, h in sorted_tokens]
    median_height = float(np.median(heights)) if heights else 1.0
    row_gap = max(4.0, median_height * row_gap_multiplier)

    rows: List[List[OCRToken]] = []
    current_row = [sorted_tokens[0]]
    current_center = sorted_tokens[0][2] + sorted_tokens[0][4] / 2.0

    for token in sorted_tokens[1:]:
        center_y = token[2] + token[4] / 2.0
        if abs(center_y - current_center) <= row_gap:
            current_row.append(token)
            current_center = sum(t[2] + t[4] / 2.0 for t in current_row) / len(current_row)
        else:
            rows.append(sorted(current_row, key=lambda item: item[1]))
            current_row = [token]
            current_center = center_y

    rows.append(sorted(current_row, key=lambda item: item[1]))
    return rows


def _phrase_bbox(tokens: Sequence[OCRToken]) -> Tuple[int, int, int, int]:
    min_x = min(token[1] for token in tokens)
    min_y = min(token[2] for token in tokens)
    max_x = max(token[1] + token[3] for token in tokens)
    max_y = max(token[2] + token[4] for token in tokens)
    return min_x, min_y, max_x - min_x, max_y - min_y


def _resolve_field_output_name(default_field_name: str, gt_json: Optional[Dict[str, Any]]) -> str:
    if not gt_json:
        return default_field_name

    spec = FIELD_SPECS.get(default_field_name, {})
    candidates = [default_field_name, *spec.get("aliases", [])]
    normalized_candidates = {_normalize_key(candidate): candidate for candidate in candidates}

    for gt_key, gt_value in gt_json.items():
        if gt_value is None or isinstance(gt_value, (list, dict)):
            continue
        if _normalize_key(str(gt_key)) in normalized_candidates:
            return str(gt_key)
    return default_field_name


def _field_spec_for_name(field_name: str) -> Dict[str, Any]:
    if field_name in FIELD_SPECS:
        return FIELD_SPECS[field_name]

    normalized_name = _normalize_key(field_name)
    for default_name, spec in FIELD_SPECS.items():
        alias_set = {
            _normalize_key(default_name),
            *(_normalize_key(alias) for alias in spec["aliases"]),
        }
        if normalized_name in alias_set:
            return spec

    field_type = (
        "date"
        if "date" in normalized_name
        else "currency"
        if any(
            token in normalized_name
            for token in ("total", "amount", "tax", "subtotal", "shipping", "discount")
        )
        else "string"
    )
    anchor = (
        "below_label"
        if any(token in normalized_name for token in ("customer", "seller", "vendor", "billto"))
        else "right_of_label"
    )
    return {
        "aliases": [field_name],
        "field_type": field_type,
        "anchor_direction": anchor,
        "critical": normalized_name in {"invoicenumber", "invoicedate", "total"},
    }


def expected_scalar_fields(gt_json: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    expected: Dict[str, Dict[str, Any]] = {}
    if gt_json:
        for gt_key, gt_value in gt_json.items():
            if gt_value is None or isinstance(gt_value, (list, dict)):
                continue
            output_name = str(gt_key)
            spec = _field_spec_for_name(output_name)
            expected[output_name] = {
                **spec,
                "output_name": output_name,
            }

    for default_name, spec in FIELD_SPECS.items():
        output_name = _resolve_field_output_name(default_name, gt_json)
        expected.setdefault(
            output_name,
            {
                **spec,
                "output_name": output_name,
            },
        )
    return expected


def score_label_candidate(
    phrase_text: str,
    alias: str,
    *,
    has_trailing_colon: bool = False,
    label_x: int = 0,
    label_y: int = 0,
    page_w: int = 1,
    page_h: int = 1,
    field_name: Optional[str] = None,
) -> float:
    phrase_norm = normalize_ocr_text(phrase_text)
    alias_norm = normalize_ocr_text(alias)
    if not phrase_norm or not alias_norm:
        return 0.0

    similarity = _fuzzy_similarity(phrase_norm, alias_norm)
    phrase_tokens = phrase_norm.split()
    alias_tokens = alias_norm.split()
    phrase_token_set = set(phrase_tokens)
    alias_token_set = set(alias_tokens)
    token_overlap = (
        len(phrase_token_set & alias_token_set) / len(alias_token_set)
        if phrase_token_set and alias_token_set
        else 0.0
    )
    ordered_matches = sum(
        1
        for idx, token in enumerate(alias_tokens)
        if idx < len(phrase_tokens) and phrase_tokens[idx] == token
    )
    order_score = ordered_matches / max(len(alias_tokens), 1)

    score = (similarity * 0.55) + (token_overlap * 0.20) + (order_score * 0.12)
    if phrase_norm == alias_norm:
        score += 0.15
    elif phrase_norm.startswith(alias_norm) or alias_norm.startswith(phrase_norm):
        score += 0.08
    if has_trailing_colon:
        score += 0.05
    phrase_words = _word_count(phrase_text)
    alias_words = _word_count(alias)
    if phrase_words > alias_words + 2:
        score -= min(0.18, (phrase_words - alias_words) * 0.05)
    if alias_words > phrase_words:
        score -= min(0.26, (alias_words - phrase_words) * 0.12)
    alias_text_lower = alias.lower()
    phrase_text_lower = phrase_text.lower()
    if any(cue in alias_text_lower for cue in ("#", "no", "number")) and not any(
        cue in phrase_text_lower for cue in ("#", "no", "number")
    ):
        score -= 0.5
    normalized_field_name = _normalize_key(field_name) if field_name else ""
    if normalized_field_name == "invoicedate" and "due" in phrase_norm:
        score -= 0.85
    if normalized_field_name == "invoicedate" and any(
        token in phrase_norm for token in ("issue", "issued")
    ):
        score += 0.22
    if normalized_field_name == "duedate" and "invoice" in phrase_norm:
        score -= 0.3
    if normalized_field_name == "duedate" and any(
        token in phrase_norm for token in ("issue", "issued")
    ):
        score -= 0.4
    if field_name and _normalize_key(field_name) == "tax" and "tax id" in phrase_norm:
        return 0.0
    if (
        " to" in alias_text_lower
        and " to" not in phrase_text_lower
        and not phrase_text_lower.endswith("to:")
    ):
        score -= 0.2
    if phrase_norm in GENERIC_LABEL_WORDS and alias_norm not in {
        phrase_norm,
        f"invoice {phrase_norm}",
    }:
        score -= 0.18
    if page_w > 0 and label_x / page_w < 0.55:
        score += 0.05
    score += _position_prior(field_name, label_x, label_y, page_w, page_h)
    return min(score, 1.0)


def infer_field_type(field_name: str) -> str:
    return _field_spec_for_name(field_name)["field_type"]


def infer_value_region(
    label_bbox: Tuple[int, int, int, int],
    page_w: int,
    page_h: int,
    field_name: str,
    *,
    anchor_direction: Optional[str] = None,
    heuristic_config: Optional[Any] = None,
) -> Dict[str, float]:
    x, y, w, h = label_bbox
    spec = _field_spec_for_name(field_name)
    direction = anchor_direction or spec["anchor_direction"]
    max_width_fraction = getattr(heuristic_config, "max_region_width_fraction", 0.45)
    address_height_multiplier = getattr(heuristic_config, "address_region_height_multiplier", 4.0)

    gap_x = max(int(page_w * 0.01), 6)
    gap_y = max(int(page_h * 0.01), 6)

    if direction == "below_label":
        box_x = max(0, x - gap_x)
        box_y = min(page_h - 1, y + h + gap_y)
        box_w = min(int(page_w * 0.45), page_w - box_x)
        box_h = min(int(h * address_height_multiplier), page_h - box_y)
    else:
        box_x = min(page_w - 1, x + w + gap_x)
        box_y = max(0, y - int(h * 0.25))
        max_width = int(page_w * max_width_fraction)
        box_w = min(max_width, page_w - box_x)
        box_h = min(int(h * (1.6 if spec["field_type"] == "address" else 1.8)), page_h - box_y)

    box_w = max(box_w, min(int(page_w * 0.08), max(page_w - box_x, 1)))
    box_h = max(box_h, min(int(page_h * 0.03), max(page_h - box_y, 1)))

    return {
        "x": round(box_x / page_w, 6),
        "y": round(box_y / page_h, 6),
        "width": round(min(box_w / page_w, 1.0 - (box_x / page_w)), 6),
        "height": round(min(box_h / page_h, 1.0 - (box_y / page_h)), 6),
    }


def _iter_row_phrases(
    row: Sequence[OCRToken], max_tokens: int = 4
) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    phrases: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for start in range(len(row)):
        for end in range(start + 1, min(len(row), start + max_tokens) + 1):
            phrase_tokens = row[start:end]
            phrase = " ".join(token[0].strip() for token in phrase_tokens if token[0].strip())
            if not phrase:
                continue
            phrases.append((phrase, _phrase_bbox(phrase_tokens)))
    return phrases


def _row_bounds(row: Sequence[OCRToken]) -> Tuple[int, int, int, int]:
    return _phrase_bbox(row)


def _token_to_number(value: str) -> Optional[float]:
    text = re.sub(r"[^\d,.\-]", "", value)
    if not text or not re.search(r"\d", text):
        return None
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif text.count(",") == 1 and text.count(".") == 0:
        tail = text.rsplit(",", 1)[1]
        text = text.replace(",", ".") if len(tail) == 2 else text.replace(",", "")
    else:
        text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        return None


def _extract_amount_strings(text: str) -> List[str]:
    matches: List[str] = []
    for match in re.finditer(r"[$€£]?\s*-?\d[\d,]*(?:\.\d{2})?", text):
        value = match.group(0).strip()
        if value and any(ch.isdigit() for ch in value) and value not in matches:
            matches.append(value)
    return matches


def _extract_amount_candidates_from_row(row: Sequence[OCRToken]) -> List[str]:
    candidates: List[str] = []
    raw_tokens = [token[0].strip() for token in row if token[0].strip()]
    for index, raw in enumerate(raw_tokens):
        compact = raw.replace(" ", "")
        if index + 1 < len(raw_tokens):
            nxt = raw_tokens[index + 1].strip().replace(" ", "")
            if re.fullmatch(r"\d{3,}[,.]\d{2}", nxt):
                if re.fullmatch(r"[$€£]-?\d{1,3}", compact):
                    prefix = compact[0]
                    lead = re.sub(r"^[^0-9-]+", "", compact)
                    merged = f"{prefix}{lead}{nxt}"
                    if merged not in candidates:
                        candidates.append(merged)
                elif re.fullmatch(r"-?\d{1,3}", compact):
                    merged = f"{compact}{nxt}"
                    if merged not in candidates:
                        candidates.append(merged)
        if re.fullmatch(r"[$€£]?\s*-?\d[\d,]*(?:\.\d{2})?", raw) and raw not in candidates:
            candidates.append(raw)
    return candidates


def _appears_as_percentage(raw_text: str, row_text: str) -> bool:
    stripped = raw_text.strip()
    if not stripped:
        return False
    unprefixed = stripped.lstrip("$€£").strip()
    for candidate in {stripped, unprefixed}:
        if candidate and re.search(rf"(?<!\d){re.escape(candidate)}\s*%", row_text):
            return True
    return False


def _summary_amount_values(row_text: str) -> List[Dict[str, Any]]:
    unique_values: List[Dict[str, Any]] = []
    for candidate in _extract_amount_strings(row_text):
        if "%" in candidate or _appears_as_percentage(candidate, row_text):
            continue
        if _looks_like_non_summary_number(candidate, row_text):
            continue
        numeric = _token_to_number(candidate)
        if numeric is None or numeric < 0:
            continue
        if any(abs(existing["numeric"] - numeric) < 0.01 for existing in unique_values):
            continue
        unique_values.append({"raw": candidate, "numeric": numeric})
    return unique_values


def _looks_like_repeated_summary_triplet(row_text: str) -> bool:
    normalized = normalize_ocr_text(row_text)
    amount_values = _summary_amount_values(row_text)
    return (
        len(amount_values) == 3
        and "total" in normalized
        and any(keyword in normalized for keyword in ("summary", "vat", "tax", "subtotal"))
    )


def _extract_date_strings(text: str) -> List[str]:
    matches: List[str] = []
    for pattern in (
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
    ):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(0).strip()
            if value not in matches:
                matches.append(value)
    return matches


def _looks_like_valid_date(value: str) -> bool:
    if not value:
        return False
    try:
        parsed = parse_date(value, fuzzy=True)
    except Exception:
        return False
    return 1990 <= parsed.year <= 2100


def _looks_like_noise_line(text: str) -> bool:
    normalized = normalize_ocr_text(text)
    if not normalized:
        return True
    return bool(
        re.search(
            r"(www\.|https?://|@|phone|tel\b|fax\b|vat\b|iban\b|swift code|bic\b)", normalized
        )
    )


def _has_party_cues(text: str) -> bool:
    normalized = normalize_ocr_text(text)
    if not normalized:
        return False
    words = normalized.split()
    return any(
        label in normalized for labels in PARTY_ROLE_LABELS.values() for label in labels
    ) or any(word in BUSINESS_SUFFIXES for word in words)


def _has_summary_cues(text: str) -> bool:
    normalized = normalize_ocr_text(text)
    if not normalized:
        return False
    return any(label in normalized for labels in SUMMARY_ROLE_LABELS.values() for label in labels)


def _is_mixed_content_row(text: str) -> bool:
    normalized = normalize_ocr_text(text)
    if not normalized:
        return False

    words = normalized.split()
    if len(words) < 8:
        return False

    if not _has_summary_cues(text):
        return False

    has_party_cues = _has_party_cues(text)
    has_noise_cues = _looks_like_noise_line(text)
    has_numeric_cues = bool(
        _extract_amount_strings(text)
        or _extract_date_strings(text)
        or sum(ch.isdigit() for ch in text) >= 6
    )

    return has_numeric_cues and (has_party_cues or has_noise_cues)


def _table_header_match_count(text: str) -> int:
    normalized = normalize_ocr_text(text)
    if not normalized:
        return 0

    count = 0
    for aliases in TABLE_HEADER_ALIASES.values():
        if any(
            re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized)
            for alias in aliases
        ):
            count += 1
    return count


def _looks_tableish_row_text(text: str) -> bool:
    normalized = normalize_ocr_text(text)
    if not normalized:
        return False

    header_match_count = _table_header_match_count(text)
    amount_like_count = len(_extract_amount_strings(text))

    if header_match_count >= 2:
        return True
    if normalized in {"item", "items"}:
        return True
    if header_match_count >= 1 and any(
        word in normalized
        for word in ("gross", "net", "vat", "worth", "unit", "price", "quantity", "qty")
    ):
        return True
    if (
        re.match(r"^\s*[|]?\s*\d{1,3}[.:]?\b", text)
        and amount_like_count >= 1
        and any(ch.isalpha() for ch in text)
    ):
        return True
    return False


def _looks_metadataish_row_text(text: str) -> bool:
    normalized = normalize_ocr_text(text)
    if not normalized:
        return False

    if any(
        phrase in normalized
        for phrase in (
            "subtotal",
            "sales tax",
            "tax",
            "total due",
            "amount due",
            "balance due",
            "invoice total",
        )
    ):
        return False

    metadata_labels = ("invoice", "date", "due", "iban", "swift", "id")
    has_metadata_label = any(
        re.search(rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])", normalized)
        for label in metadata_labels
    )
    if has_metadata_label and bool(re.search(r"\d", text)):
        return True
    if "issue" in normalized and bool(re.search(r"\d", text)):
        return True
    return False


def _row_block_class(row: Sequence[OCRToken]) -> str:
    row_text = _join_tokens(row)
    normalized = normalize_ocr_text(row_text)
    if not normalized:
        return "other"
    if _looks_tableish_row_text(row_text):
        return "table"
    if normalized == "summary" or _is_summary_row_text(row_text):
        return "summary"
    if _has_party_cues(row_text):
        return "party"
    if (
        max(
            _party_line_role_score(row_text, "vendor_block"),
            _party_line_role_score(row_text, "bill_to_block"),
        )
        >= 0.22
    ):
        return "party"
    if _looks_metadataish_row_text(row_text):
        return "metadata"
    return "other"


def _should_split_row_block(
    current_rows: Sequence[Sequence[OCRToken]],
    next_row: Sequence[OCRToken],
    page_h: int,
) -> bool:
    if not current_rows:
        return False

    current_classes = [_row_block_class(row) for row in current_rows]
    current_strong_classes = [value for value in current_classes if value != "other"]
    current_class = current_strong_classes[-1] if current_strong_classes else "other"
    next_class = _row_block_class(next_row)

    if next_class == "other" or current_class == "other":
        return False
    if next_class == current_class:
        return False

    current_bbox = _row_bounds([token for row in current_rows for token in row])
    tall_block = len(current_rows) >= 4 or current_bbox[3] >= page_h * 0.18

    if current_class == "party" and next_class == "metadata":
        return len(current_rows) >= 3
    if current_class == "metadata" and next_class in {"party", "table", "summary"}:
        return True
    if current_class == "party" and next_class in {"table", "summary"}:
        return len(current_rows) >= 2
    if current_class == "table" and next_class in {"party", "summary"}:
        return True
    if current_class == "summary" and next_class in {"party", "table"}:
        return True
    if tall_block:
        return True
    return False


def _party_line_score(text: str) -> float:
    return _heuristic_party_module()._party_line_score(text)


def _clean_party_candidate(text: str, *, prefer_short_prefix: bool = False) -> str:
    return _heuristic_party_module()._clean_party_candidate(
        text,
        prefer_short_prefix=prefer_short_prefix,
    )


def _split_row_into_segments(row: Sequence[OCRToken]) -> List[List[OCRToken]]:
    return _heuristic_party_module()._split_row_into_segments(row)


def _party_segment_score(segment: Sequence[OCRToken], role: str, page_w: int) -> Tuple[float, str]:
    return _heuristic_party_module()._party_segment_score(segment, role, page_w)


def _normalize_party_words(value: str) -> List[str]:
    return _heuristic_party_module()._normalize_party_words(value)


def _cleanup_party_name_overlap(
    value: str,
    other_value: Optional[str],
    *,
    prefer_suffix: bool = False,
) -> str:
    return _heuristic_party_module()._cleanup_party_name_overlap(
        value,
        other_value,
        prefer_suffix=prefer_suffix,
    )


def _extract_vendor_name_from_row(row: Sequence[OCRToken], page_w: int) -> Optional[str]:
    return _heuristic_party_module()._extract_vendor_name_from_row(row, page_w)


def _is_summary_row_text(text: str) -> bool:
    if _is_mixed_content_row(text):
        return False
    normalized = normalize_ocr_text(text)
    if not normalized:
        return False
    return any(label in normalized for labels in SUMMARY_ROLE_LABELS.values() for label in labels)


def _is_banner_like_text(text: str) -> bool:
    return _heuristic_party_module()._is_banner_like_text(text)


def _is_contactish_or_address_line(text: str) -> bool:
    return _heuristic_party_module()._is_contactish_or_address_line(text)


def _party_line_role_score(text: str, role: str) -> float:
    return _heuristic_party_module()._party_line_role_score(text, role)


def _select_party_name_from_block(
    block_rows: Sequence[Sequence[OCRToken]], role: str
) -> Optional[str]:
    return _heuristic_party_module()._select_party_name_from_block(block_rows, role)


def _build_row_blocks(
    rows: Sequence[Sequence[OCRToken]],
    page_h: int,
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    current_rows: List[Sequence[OCRToken]] = []
    current_bbox: Optional[Tuple[int, int, int, int]] = None
    max_gap = max(28, int(page_h * 0.04))

    for row in rows:
        row_bbox = _row_bounds(row)
        row_text = _join_tokens(row)
        if not row_text.strip():
            continue
        if current_rows and current_bbox is not None:
            _, _, _, curr_h = current_bbox
            row_gap = row_bbox[1] - (current_bbox[1] + curr_h)
            if row_gap > max_gap or _should_split_row_block(current_rows, row, page_h):
                blocks.append({"rows": current_rows, "bbox": current_bbox})
                current_rows = []
                current_bbox = None
        if current_bbox is None:
            current_rows = [row]
            current_bbox = row_bbox
        else:
            min_x = min(current_bbox[0], row_bbox[0])
            min_y = min(current_bbox[1], row_bbox[1])
            max_x = max(current_bbox[0] + current_bbox[2], row_bbox[0] + row_bbox[2])
            max_y = max(current_bbox[1] + current_bbox[3], row_bbox[1] + row_bbox[3])
            current_rows.append(row)
            current_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

    if current_rows and current_bbox is not None:
        blocks.append({"rows": current_rows, "bbox": current_bbox})
    return blocks


def _block_role_scores(
    block: Dict[str, Any],
    page_w: int,
    page_h: int,
) -> Dict[str, float]:
    return _heuristic_party_module()._block_role_scores(block, page_w, page_h)


def _extract_labeled_parties_from_block(
    block_rows: Sequence[Sequence[OCRToken]],
    page_w: int,
    page_h: int,
) -> Dict[str, Tuple[str, Tuple[int, int, int, int], float]]:
    return _heuristic_party_module()._extract_labeled_parties_from_block(block_rows, page_w, page_h)


def _extract_party_blocks(
    rows: Sequence[Sequence[OCRToken]],
    page_w: int,
    page_h: int,
) -> Tuple[Dict[str, Tuple[str, Tuple[int, int, int, int]]], List[Dict[str, Any]]]:
    blocks = _build_row_blocks(rows, page_h)
    diagnostics: List[Dict[str, Any]] = []
    selected: Dict[str, Tuple[str, Tuple[int, int, int, int], float]] = {}

    for block in blocks:
        bbox = block["bbox"]
        block_rows = block["rows"]
        block_text = " ".join(_join_tokens(row) for row in block_rows)
        role_scores = _block_role_scores(block, page_w, page_h)
        chosen_role, chosen_score = max(role_scores.items(), key=lambda item: item[1])
        diagnostics.append(
            {
                "role": chosen_role,
                "score": round(chosen_score, 3),
                "bbox": bbox,
                "text": block_text[:240],
                "role_scores": {name: round(score, 3) for name, score in role_scores.items()},
            }
        )

        labeled_parties = _extract_labeled_parties_from_block(block_rows, page_w, page_h)
        for target_field, (party_name, party_bbox, label_score) in labeled_parties.items():
            current = selected.get(target_field)
            explicit_score = 5.0 + label_score
            if current is None or explicit_score > current[2]:
                selected[target_field] = (party_name, party_bbox, explicit_score)

        if chosen_role == "summary_block" or chosen_score < 0.35:
            continue

        party_name = _select_party_name_from_block(block_rows, chosen_role)
        if not party_name:
            continue

        target_field = (
            "sellerName" if chosen_role in {"vendor_block", "remit_to_block"} else "customerName"
        )
        current = selected.get(target_field)
        if current is None or chosen_score > current[2]:
            selected[target_field] = (party_name, bbox, chosen_score)

    return {field: (value, bbox) for field, (value, bbox, _score) in selected.items()}, diagnostics


def _extract_party_name_from_tokens(
    field_name: str,
    label_bbox: Tuple[int, int, int, int],
    rows: Sequence[Sequence[OCRToken]],
    page_w: int,
    page_h: int,
) -> Optional[str]:
    return _heuristic_party_module()._extract_party_name_from_tokens(
        field_name,
        label_bbox,
        rows,
        page_w,
        page_h,
    )


def _fallback_seller_name(
    rows: Sequence[Sequence[OCRToken]], page_w: int, page_h: int
) -> Optional[Tuple[str, Tuple[int, int, int, int]]]:
    return _heuristic_party_module()._fallback_seller_name(rows, page_w, page_h)


def _fallback_customer_name(
    rows: Sequence[Sequence[OCRToken]], page_w: int, page_h: int
) -> Optional[Tuple[str, Tuple[int, int, int, int]]]:
    return _heuristic_party_module()._fallback_customer_name(rows, page_w, page_h)


def _detect_table(
    rows: Sequence[Sequence[OCRToken]],
    page_w: int,
    page_h: int,
    heuristic_config: Optional[Any],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    return _heuristic_table_module()._detect_table(rows, page_w, page_h, heuristic_config)


def _detect_body_row_fallback(
    rows: Sequence[Sequence[OCRToken]],
    page_w: int,
    page_h: int,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    return _heuristic_table_module()._detect_body_row_fallback(rows, page_w, page_h)


def _invoice_candidate_score(
    value: str, bbox: Tuple[int, int, int, int], page_w: int, page_h: int
) -> float:
    normalized = value.strip()
    digits = sum(ch.isdigit() for ch in normalized)
    alpha = sum(ch.isalpha() for ch in normalized)
    score = digits * 0.18 + min(alpha, 4) * 0.06
    if digits == 0:
        score -= 0.6
    if len(normalized) < 3:
        score -= 0.5
    if len(re.sub(r"\D", "", normalized)) <= 3:
        score -= 0.25
    if re.fullmatch(r"\d{4}", re.sub(r"\D", "", normalized) or ""):
        score -= 0.75
    if re.fullmatch(r"\d{5}-\d{3,4}", normalized):
        score -= 1.3
    lowered = normalize_ocr_text(normalized)
    if re.fullmatch(r"\d+(?:st|nd|rd|th)", normalized.lower()):
        score -= 1.5
    if re.search(r"[$€£]", normalized):
        score -= 0.8
    if lowered in {"invoice", "number", "date", "total"}:
        score -= 0.8
    x, y, _, _ = bbox
    score += _position_prior("invoiceNumber", x, y, page_w, page_h)
    if x > page_w * 0.8:
        score -= 0.15
    return score


def _date_candidate_score(
    value: str, bbox: Tuple[int, int, int, int], page_w: int, page_h: int
) -> float:
    if not _looks_like_valid_date(value):
        return -1.0
    x, y, _, _ = bbox
    score = 0.5 + _position_prior("invoiceDate", x, y, page_w, page_h)
    if y > page_h * 0.55:
        score -= 0.35
    if x > page_w * 0.85:
        score -= 0.1
    return score


def _amount_candidate_score(
    field_name: str,
    value: str,
    bbox: Tuple[int, int, int, int],
    page_w: int,
    page_h: int,
    row_text: str,
) -> float:
    numeric = _token_to_number(value)
    if numeric is None:
        return -1.0
    x, y, _, _ = bbox
    score = 0.4 + _position_prior(field_name, x, y, page_w, page_h)
    if numeric < 0 and any(
        token in field_name.lower()
        for token in ("tax", "subtotal", "total", "amount", "shipping", "discount")
    ):
        score -= 1.4
    if numeric < 1.0:
        score -= 0.4
    elif numeric < 10.0:
        score -= 0.2
    if "." not in value and "," not in value and not re.search(r"[$€£]", value):
        score -= 0.18
    if re.fullmatch(r"\d{1,2},?", value.strip()):
        score -= 0.5
    if x > page_w * 0.7:
        score += 0.18
    elif x < page_w * 0.35:
        score -= 0.08
    if y > page_h * 0.65:
        score += 0.18
    row_norm = normalize_ocr_text(row_text)
    if "date" in row_norm and not any(
        word in row_norm for word in ("subtotal", "tax", "total", "amount due", "balance due")
    ):
        score -= 0.45
    if "subtotal" in row_norm and "subtotal" in field_name.lower():
        score += 0.28
    elif "tax" in row_norm and "tax" in field_name.lower():
        score += 0.28
    elif "total" in row_norm and "subtotal" not in row_norm and "tax" not in row_norm:
        if "total" in field_name.lower() or "amount" in field_name.lower():
            score += 0.34
    elif (
        any(word in row_norm for word in ("amount due", "balance due"))
        and "total" in field_name.lower()
    ):
        score += 0.3
    return score


def _fallback_field_value_and_bbox(
    field_name: str,
    rows: Sequence[Sequence[OCRToken]],
    page_w: int,
    page_h: int,
    structural_context: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]]]:
    normalized_name = _normalize_key(field_name)
    ranked_candidates: List[Tuple[float, str, Tuple[int, int, int, int]]] = []

    if normalized_name == "sellername":
        party_candidates = (structural_context or {}).get("party_candidates", {})
        if "sellerName" in party_candidates:
            return party_candidates["sellerName"]
        fallback_seller = _fallback_seller_name(rows, page_w, page_h)
        if fallback_seller:
            return fallback_seller
        return None, None

    if normalized_name == "customername":
        party_candidates = (structural_context or {}).get("party_candidates", {})
        if "customerName" in party_candidates:
            return party_candidates["customerName"]
        fallback_customer = _fallback_customer_name(rows, page_w, page_h)
        if fallback_customer:
            return fallback_customer
        return None, None

    if "date" in normalized_name:
        if normalized_name == "duedate":
            due_rows = [row for row in rows if "due" in normalize_ocr_text(_join_tokens(row))]
            if not due_rows:
                return None, None
        for row in rows:
            bbox = _row_bounds(row)
            if bbox[1] > page_h * 0.6:
                continue
            row_text = _join_tokens(row)
            if normalized_name == "duedate" and "due" not in normalize_ocr_text(row_text):
                continue
            for candidate in _extract_date_strings(row_text):
                ranked_candidates.append(
                    (
                        _date_candidate_score(candidate, bbox, page_w, page_h),
                        candidate,
                        bbox,
                    )
                )

    elif any(
        token in normalized_name
        for token in ("total", "subtotal", "tax", "amount", "shipping", "discount")
    ):
        summary_candidates = (structural_context or {}).get("summary_candidates", {})
        summary_concepts: List[str] = []
        if "subtotal" in normalized_name:
            summary_concepts = ["subtotal"]
        elif "tax" in normalized_name:
            summary_concepts = ["tax"]
        elif "shipping" in normalized_name:
            summary_concepts = ["shipping"]
        elif "total" in normalized_name or "amount" in normalized_name:
            summary_concepts = ["invoice_total", "total_due", "amount_due", "balance_due"]
        if any(token in normalized_name for token in ("subtotal", "tax", "shipping")) and not any(
            summary_candidates.get(concept) for concept in summary_concepts
        ):
            return None, None
        if (
            ("total" in normalized_name or "amount" in normalized_name)
            and not any(summary_candidates.get(concept) for concept in summary_concepts)
            and summary_candidates.get("subtotal")
            and summary_candidates.get("tax")
        ):
            subtotal_candidate = max(summary_candidates["subtotal"], key=lambda item: item["score"])
            tax_candidate = max(summary_candidates["tax"], key=lambda item: item["score"])
            subtotal_value = _token_to_number(subtotal_candidate["value"])
            tax_value = _token_to_number(tax_candidate["value"])
            if subtotal_value is not None and tax_value is not None:
                return f"{subtotal_value + tax_value:.2f}", subtotal_candidate["bbox"]
        if (
            ("total" in normalized_name or "amount" in normalized_name)
            and summary_candidates.get("subtotal")
            and summary_candidates.get("tax")
            and summary_candidates.get("shipping")
        ):
            subtotal_candidate = max(summary_candidates["subtotal"], key=lambda item: item["score"])
            tax_candidate = max(summary_candidates["tax"], key=lambda item: item["score"])
            shipping_candidate = max(summary_candidates["shipping"], key=lambda item: item["score"])
            subtotal_value = _token_to_number(subtotal_candidate["value"])
            tax_value = _token_to_number(tax_candidate["value"])
            shipping_value = _token_to_number(shipping_candidate["value"])
            if None not in (subtotal_value, tax_value, shipping_value):
                derived_total = round(subtotal_value + tax_value, 2)
                shipped_total = round(derived_total + shipping_value, 2)
                total_candidates = [
                    candidate
                    for concept in summary_concepts
                    for candidate in summary_candidates.get(concept, [])
                ]
                if any(
                    _token_to_number(candidate["value"]) is not None
                    and abs(_token_to_number(candidate["value"]) - shipped_total) <= 0.05
                    for candidate in total_candidates
                ):
                    return f"{shipped_total:.2f}", subtotal_candidate["bbox"]
        for concept in summary_concepts:
            for candidate in summary_candidates.get(concept, []):
                concept_bonus = 0.4
                if concept == "invoice_total":
                    concept_bonus = 0.62
                elif concept == "total_due":
                    concept_bonus = 0.54
                elif concept == "amount_due":
                    concept_bonus = 0.28
                elif concept == "balance_due":
                    concept_bonus = 0.22
                ranked_candidates.append(
                    (
                        candidate["score"] + concept_bonus,
                        candidate["value"],
                        candidate["bbox"],
                    )
                )
        for row in rows:
            row_text = _join_tokens(row)
            row_concepts = _summary_concepts_for_row(row_text)
            if not row_concepts:
                continue
            if not any(concept in row_concepts for concept in summary_concepts):
                continue
            if (
                _is_mixed_content_row(row_text)
                or _looks_tableish_row_text(row_text)
                or _looks_metadataish_row_text(row_text)
                or len(_extract_amount_strings(row_text)) >= 4
            ):
                continue
            row_bbox = _row_bounds(row)
            for token in row:
                if _looks_like_non_summary_number(token[0], row_text):
                    continue
                cleaned = _clean_extracted_candidate(field_name, token[0])
                if not cleaned or not re.search(r"\d", cleaned):
                    continue
                numeric = _token_to_number(cleaned)
                if numeric is None or numeric < 0:
                    continue
                bbox = _row_bounds([token])
                score = _amount_candidate_score(field_name, cleaned, bbox, page_w, page_h, row_text)
                ranked_candidates.append((score, cleaned, bbox))
            for candidate in _extract_amount_strings(row_text):
                if _looks_like_non_summary_number(candidate, row_text):
                    continue
                numeric = _token_to_number(candidate)
                if numeric is None or numeric < 0:
                    continue
                ranked_candidates.append(
                    (
                        _amount_candidate_score(
                            field_name, candidate, row_bbox, page_w, page_h, row_text
                        )
                        - 0.05,
                        candidate,
                        row_bbox,
                    )
                )

    elif "invoice" in normalized_name and (
        "number" in normalized_name
        or normalized_name.endswith("no")
        or "invoiceno" in normalized_name
    ):
        for row in rows:
            bbox = _row_bounds(row)
            if bbox[1] > page_h * 0.4:
                continue
            row_text = _join_tokens(row)
            row_norm = normalize_ocr_text(row_text)
            for token in row:
                text = token[0].strip()
                if not re.search(r"\d", text):
                    continue
                if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text):
                    continue
                cleaned = _clean_extracted_candidate(field_name, text)
                if not cleaned:
                    continue
                token_bbox = _row_bounds([token])
                score = _invoice_candidate_score(cleaned, token_bbox, page_w, page_h)
                if any(
                    label in row_norm for label in ("invoice", "invoice no", "invoice number", "no")
                ):
                    score += 0.18
                    digit_only = re.sub(r"\D", "", cleaned)
                    if re.fullmatch(r"\d{2,6}", digit_only) and len(digit_only) != 4:
                        score += 0.08
                ranked_candidates.append((score, cleaned, token_bbox))

    minimum_score = 0.25
    if "date" in normalized_name:
        minimum_score = 0.45
    elif "invoice" in normalized_name:
        minimum_score = 0.45
    elif any(
        token in normalized_name
        for token in ("total", "subtotal", "tax", "amount", "shipping", "discount")
    ):
        minimum_score = 0.6

    ranked_candidates = [
        candidate for candidate in ranked_candidates if candidate[0] >= minimum_score
    ]
    if ranked_candidates:
        _, best_value, best_bbox = max(ranked_candidates, key=lambda item: item[0])
        return best_value, best_bbox

    return None, None


def _clean_extracted_candidate(field_name: str, value: str) -> Optional[str]:
    value = value.strip()
    if not value:
        return None

    normalized_name = _normalize_key(field_name)
    if "date" in normalized_name:
        match = re.search(
            r"(?:\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4})",
            value,
            flags=re.IGNORECASE,
        )
        return match.group(0).strip() if match else None

    if any(
        token in normalized_name
        for token in ("total", "subtotal", "tax", "amount", "shipping", "discount")
    ):
        amount_matches = re.findall(r"[$€£]?\s*-?\d[\d,]*(?:\.\d{2})?", value)
        if amount_matches:
            return amount_matches[-1].strip()
        return None

    if "invoice" in normalized_name and (
        "number" in normalized_name
        or normalized_name.endswith("no")
        or "invoiceno" in normalized_name
    ):
        value = re.sub(
            r"^(?:invoice(?:\s+no\.?|\s+number)?|inv\s+no\.?|no\.?|number)[:#\s.-]+",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip()
        match = re.search(r"\b[A-Z0-9][A-Z0-9./-]{1,}\b", value, flags=re.IGNORECASE)
        return match.group(0).strip() if match and re.search(r"\d", match.group(0)) else None

    if "customer" in normalized_name or "seller" in normalized_name:
        lines = [
            segment.strip(" :")
            for segment in re.split(r"\n|(?<=\w)\s{2,}", value)
            if segment.strip()
        ]
        for line in lines:
            if _party_line_score(line) >= 0.1:
                return line
        return lines[0] if lines else value[:80].strip()

    return value[:80].strip()


def _extract_value_near_label(
    field_name: str,
    label_bbox: Tuple[int, int, int, int],
    rows: Sequence[Sequence[OCRToken]],
    page_w: int,
    page_h: int,
    anchor_direction: str,
) -> Optional[str]:
    normalized_name = _normalize_key(field_name)
    if normalized_name in {"customername", "sellername"}:
        party_name = _extract_party_name_from_tokens(field_name, label_bbox, rows, page_w, page_h)
        if party_name:
            return party_name

    x, y, w, h = label_bbox
    label_bottom = y + h
    label_right = x + w
    label_center_y = y + (h / 2.0)
    same_row_left_bound = 0.0
    same_row_right_bound = float(page_w)

    same_row_tokens: List[OCRToken] = []
    below_tokens: List[OCRToken] = []
    same_row_source: Optional[Sequence[OCRToken]] = None
    below_row_sources: List[Sequence[OCRToken]] = []
    for row in rows:
        row_center_y = sum(token[2] + token[4] / 2.0 for token in row) / len(row)
        if abs(row_center_y - label_center_y) <= max(h * 0.9, 14):
            best_label_matches: Dict[str, Tuple[Tuple[int, int, int, int], float]] = {}
            for phrase, phrase_bbox in _iter_row_phrases(row, max_tokens=3):
                phrase_text = phrase.strip()
                if not phrase_text:
                    continue
                for output_name, spec in FIELD_SPECS.items():
                    best_score = max(
                        score_label_candidate(
                            phrase_text,
                            alias,
                            has_trailing_colon=phrase_text.endswith(":"),
                            label_x=phrase_bbox[0],
                            label_y=phrase_bbox[1],
                            page_w=page_w,
                            page_h=page_h,
                            field_name=output_name,
                        )
                        for alias in spec["aliases"]
                    )
                    if best_score < 0.8:
                        continue
                    current = best_label_matches.get(output_name)
                    if current is None or best_score > current[1]:
                        best_label_matches[output_name] = (phrase_bbox, best_score)
                    break
            if best_label_matches:
                label_matches = sorted(
                    [
                        (output_name, bbox, score)
                        for output_name, (bbox, score) in best_label_matches.items()
                    ],
                    key=lambda item: item[1][0],
                )
                current_index = next(
                    (
                        idx
                        for idx, (_output_name, bbox, _score) in enumerate(label_matches)
                        if abs(bbox[0] - x) <= max(w, 12) and abs(bbox[1] - y) <= max(h, 12)
                    ),
                    None,
                )
                if current_index is None:
                    current_index = next(
                        (
                            idx
                            for idx, (output_name, _bbox, _score) in enumerate(label_matches)
                            if output_name == field_name
                        ),
                        None,
                    )
                if current_index is not None:
                    margin = max(page_w * 0.02, 18)
                    current_center = x + (w / 2.0)
                    if current_index > 0:
                        prev_bbox = label_matches[current_index - 1][1]
                        prev_center = prev_bbox[0] + (prev_bbox[2] / 2.0)
                        same_row_left_bound = max(
                            0.0, ((prev_center + current_center) / 2.0) - margin
                        )
                    if current_index + 1 < len(label_matches):
                        next_bbox = label_matches[current_index + 1][1]
                        next_center = next_bbox[0] + (next_bbox[2] / 2.0)
                        same_row_right_bound = min(
                            float(page_w), ((current_center + next_center) / 2.0) + margin
                        )
            same_row_source = row
            same_row_tokens.extend(
                [
                    token
                    for token in row
                    if (
                        (token[1] + token[3] / 2.0) > (label_right + max(w * 0.15, 6))
                        and same_row_left_bound
                        <= (token[1] + token[3] / 2.0)
                        <= same_row_right_bound
                    )
                ]
            )
        elif row_center_y > label_bottom and row_center_y <= label_bottom + max(
            h * 5.0, page_h * 0.12
        ):
            row_left = min(token[1] for token in row)
            if row_left <= x + max(w * 1.5, page_w * 0.2):
                sliced_row = [
                    token
                    for token in row
                    if same_row_left_bound <= (token[1] + token[3] / 2.0) <= same_row_right_bound
                ]
                if not sliced_row:
                    continue
                below_row_sources.append(sliced_row)
                below_tokens.extend(sliced_row)

    if any(
        token in normalized_name
        for token in ("total", "subtotal", "tax", "amount", "shipping", "discount")
    ):
        if "tax" in normalized_name:
            summary_concepts = ("tax",)
        elif "subtotal" in normalized_name:
            summary_concepts = ("subtotal",)
        elif "shipping" in normalized_name:
            summary_concepts = ("shipping",)
        else:
            summary_concepts = ("invoice_total", "total_due", "amount_due", "balance_due")
        if "tax" in normalized_name:
            field_hint = "tax"
        elif "subtotal" in normalized_name:
            field_hint = "subtotal"
        elif "shipping" in normalized_name:
            field_hint = "shipping"
        else:
            field_hint = "totalAmount"
        nearby_candidates: Dict[str, List[Tuple[float, str]]] = {
            concept: [] for concept in summary_concepts
        }
        for source_row in [same_row_source, *below_row_sources]:
            if not source_row:
                continue
            row_text = _join_tokens(source_row)
            row_concepts = set(_summary_concepts_for_row(row_text))
            if not row_concepts:
                continue
            candidate = _extract_summary_row_amount(source_row, page_w, page_h, field_hint)
            if not candidate:
                continue
            for concept in summary_concepts:
                if concept in row_concepts:
                    nearby_candidates[concept].append((candidate["score"], candidate["value"]))
        all_candidates: Dict[str, List[Tuple[float, str]]] = {
            concept: [] for concept in summary_concepts
        }
        for row in rows:
            row_text = _join_tokens(row)
            row_concepts = set(_summary_concepts_for_row(row_text))
            if not row_concepts:
                continue
            candidate = _extract_summary_row_amount(row, page_w, page_h, field_hint)
            if not candidate:
                continue
            for concept in summary_concepts:
                if concept in row_concepts:
                    all_candidates[concept].append((candidate["score"], candidate["value"]))
        for concept in summary_concepts:
            concept_candidates = all_candidates[concept] + nearby_candidates[concept]
            if concept_candidates:
                concept_candidates.sort(key=lambda item: item[0], reverse=True)
                return concept_candidates[0][1]
        if same_row_source or below_row_sources:
            return None

    amount_field = any(
        token in normalized_name
        for token in ("total", "subtotal", "tax", "amount", "shipping", "discount")
    )
    row_joiner = _join_tokens_left_to_right if "date" in normalized_name else _join_tokens
    candidate_rows: List[Sequence[OCRToken]] = []
    if anchor_direction == "below_label":
        candidate_rows.extend(below_row_sources)
        if same_row_source:
            candidate_rows.append(same_row_source)
    else:
        if same_row_source:
            candidate_rows.append(same_row_source)
        candidate_rows.extend(below_row_sources)

    if amount_field:
        candidate_rows = [
            row
            for row in candidate_rows
            if row
            and not (
                _is_mixed_content_row(_join_tokens(row))
                or _looks_tableish_row_text(_join_tokens(row))
                or _looks_metadataish_row_text(_join_tokens(row))
            )
        ]

    candidates: List[str] = []
    if amount_field and anchor_direction == "below_label":
        if candidate_rows:
            candidates.extend(row_joiner(row) for row in candidate_rows)
        elif below_tokens:
            candidates.append(row_joiner(below_tokens))
        elif same_row_tokens:
            candidates.append(row_joiner(same_row_tokens))
    elif amount_field:
        if candidate_rows:
            candidates.extend(row_joiner(row) for row in candidate_rows)
        elif same_row_tokens:
            candidates.append(row_joiner(same_row_tokens))
        elif below_tokens:
            candidates.append(row_joiner(below_tokens[: min(len(below_tokens), 8)]))
    elif anchor_direction == "below_label":
        if candidate_rows:
            candidates.extend(row_joiner(row) for row in candidate_rows)
        elif below_tokens:
            candidates.append(row_joiner(below_tokens))
        if same_row_tokens:
            candidates.append(row_joiner(same_row_tokens))
    else:
        if same_row_tokens:
            candidates.append(row_joiner(same_row_tokens))
        elif candidate_rows:
            candidates.extend(row_joiner(row) for row in candidate_rows)
        elif below_tokens:
            candidates.append(row_joiner(below_tokens[: min(len(below_tokens), 8)]))

    for candidate in candidates:
        cleaned = _clean_extracted_candidate(field_name, candidate)
        if cleaned:
            return cleaned
    return None


def _discover_page_fields(
    page_index: int,
    page_img: np.ndarray,
    ocr_tokens: Sequence[OCRToken],
    expected_fields: Dict[str, Dict[str, Any]],
    heuristic_config: Optional[Any],
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    List[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Dict[str, int],
    float,
]:
    page_h, page_w = page_img.shape[:2]
    rows = group_tokens_into_rows(ocr_tokens)
    min_label_score = getattr(heuristic_config, "min_label_score", 0.72)
    table_info, line_items = _detect_table(rows, page_w, page_h, heuristic_config)
    party_candidates, block_diagnostics = _extract_party_blocks(rows, page_w, page_h)
    summary_candidates = _collect_summary_candidates(rows, page_w, page_h)
    structural_context = {
        "party_candidates": party_candidates,
        "summary_candidates": summary_candidates,
    }

    matches: List[CandidateLabel] = []
    for row in rows:
        for phrase, bbox in _iter_row_phrases(row):
            phrase_text = phrase.strip()
            has_trailing_colon = phrase_text.endswith(":")
            for output_name, spec in expected_fields.items():
                for alias in spec.get("aliases", []):
                    score = score_label_candidate(
                        phrase_text,
                        alias,
                        has_trailing_colon=has_trailing_colon,
                        label_x=bbox[0],
                        label_y=bbox[1],
                        page_w=page_w,
                        page_h=page_h,
                        field_name=output_name,
                    )
                    if score < min_label_score:
                        continue
                    matches.append(
                        CandidateLabel(
                            page_index=page_index,
                            field_name=output_name,
                            label_text=phrase_text,
                            score=score,
                            bbox=bbox,
                        )
                    )

    matches.sort(key=lambda item: item.score, reverse=True)
    assigned_fields: set[str] = set()
    page_fields: Dict[str, Any] = {}
    extracted_values: Dict[str, Any] = {}
    label_confirmation_set: List[Dict[str, Any]] = []
    matched_label_count = 0
    critical_field_count = 0
    confidences: List[float] = []

    for match in matches:
        if match.field_name in assigned_fields:
            continue
        assigned_fields.add(match.field_name)
        spec = expected_fields[match.field_name]
        region = infer_value_region(
            match.bbox,
            page_w,
            page_h,
            match.field_name,
            anchor_direction=spec["anchor_direction"],
            heuristic_config=heuristic_config,
        )
        page_fields[match.field_name] = {
            "region": region,
            "field_type": spec["field_type"],
            "anchor_direction": spec["anchor_direction"],
            "label_text": match.label_text,
            "confidence": round(match.score, 3),
        }
        extracted_value = _extract_value_near_label(
            match.field_name,
            match.bbox,
            rows,
            page_w,
            page_h,
            spec["anchor_direction"],
        )
        match_label_norm = normalize_ocr_text(match.label_text)
        if (
            extracted_value
            and any(token in _normalize_key(match.field_name) for token in ("total", "amount"))
            and any(label in match_label_norm for label in ("amount due", "balance due"))
        ):
            preferred_total_candidates = (
                summary_candidates.get("invoice_total") or summary_candidates.get("total_due") or []
            )
            if preferred_total_candidates:
                extracted_value = preferred_total_candidates[0]["value"]
        if extracted_value:
            extracted_values[match.field_name] = extracted_value
            page_fields[match.field_name]["example_value"] = extracted_value
        x, y, w, h = match.bbox
        label_confirmation_set.append(
            {
                "label_text": match.label_text,
                "norm_cx": round((x + (w / 2.0)) / page_w, 6),
                "norm_cy": round((y + (h / 2.0)) / page_h, 6),
                "fuzzy_distance_threshold": 2,
            }
        )
        matched_label_count += 1
        if spec.get("critical"):
            critical_field_count += 1
        confidences.append(match.score)

    for output_name, spec in expected_fields.items():
        if output_name in extracted_values:
            continue
        fallback_value, fallback_bbox = _fallback_field_value_and_bbox(
            output_name,
            rows,
            page_w,
            page_h,
            structural_context=structural_context,
        )
        if not fallback_value or not fallback_bbox:
            continue
        region = infer_value_region(
            fallback_bbox,
            page_w,
            page_h,
            output_name,
            anchor_direction="none",
            heuristic_config=heuristic_config,
        )
        existing_field = page_fields.get(output_name, {})
        page_fields[output_name] = {
            **existing_field,
            "region": existing_field.get("region", region),
            "field_type": existing_field.get("field_type", spec["field_type"]),
            "anchor_direction": existing_field.get("anchor_direction", "none"),
            "confidence": max(existing_field.get("confidence", 0.0), 0.35),
            "example_value": fallback_value,
        }
        extracted_values[output_name] = fallback_value
        if spec.get("critical"):
            critical_field_count += 1
        confidences.append(0.35)

    critical_expected = sum(1 for spec in expected_fields.values() if spec.get("critical"))
    coverage = len(page_fields) / max(len(expected_fields), 1)
    critical_coverage = critical_field_count / max(critical_expected, 1)
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    page_confidence = min(
        1.0, (mean_confidence * 0.6) + (coverage * 0.15) + (critical_coverage * 0.25)
    )

    diagnostics = {
        "matched_label_count": matched_label_count,
        "critical_field_count": critical_field_count,
        "critical_field_total": critical_expected,
        "party_block_diagnostics": block_diagnostics,
        "summary_candidate_diagnostics": {
            key: [
                {
                    "value": item["value"],
                    "score": round(item["score"], 3),
                    "row_text": item["row_text"][:120],
                }
                for item in values[:3]
            ]
            for key, values in summary_candidates.items()
            if values
        },
        "line_item_source": "header_table"
        if table_info and table_info.get("header_row_y") is not None
        else "row_fallback"
        if line_items
        else None,
    }
    if line_items:
        extracted_values["line_items"] = line_items
    return (
        page_fields,
        extracted_values,
        label_confirmation_set,
        table_info,
        diagnostics,
        round(page_confidence, 3),
    )


def _looks_like_non_summary_number(raw_text: str, row_text: str) -> bool:
    return _heuristic_summary._looks_like_non_summary_number(raw_text, row_text)


def _summary_amount_values(row_text: str) -> List[Dict[str, Any]]:
    return _heuristic_summary._summary_amount_values(row_text)


def _looks_like_repeated_summary_triplet(row_text: str) -> bool:
    return _heuristic_summary._looks_like_repeated_summary_triplet(row_text)


def _summary_concepts_for_row(row_text: str) -> List[str]:
    return _heuristic_summary._summary_concepts_for_row(row_text)


def _extract_summary_row_amount(
    row: Sequence[OCRToken],
    page_w: int,
    page_h: int,
    field_hint: str,
) -> Optional[Dict[str, Any]]:
    return _heuristic_summary._extract_summary_row_amount(row, page_w, page_h, field_hint)


def _derive_tax_candidate_from_total_row(
    row: Sequence[OCRToken],
    page_w: int,
    page_h: int,
) -> Optional[Dict[str, Any]]:
    return _heuristic_summary._derive_tax_candidate_from_total_row(row, page_w, page_h)


def _collect_summary_candidates(
    rows: Sequence[Sequence[OCRToken]],
    page_w: int,
    page_h: int,
) -> Dict[str, List[Dict[str, Any]]]:
    return _heuristic_summary._collect_summary_candidates(rows, page_w, page_h)


def _reconcile_summary_amounts(extracted_values: Dict[str, Any]) -> None:
    _heuristic_summary._reconcile_summary_amounts(extracted_values)


def _reconcile_party_names(extracted_values: Dict[str, Any]) -> None:
    _heuristic_party_module()._reconcile_party_names(extracted_values)


def _merge_page_heuristic_diagnostics(
    page_diagnostics: Dict[str, Any],
    party_block_diagnostics: List[Dict[str, Any]],
    summary_candidate_diagnostics: Dict[str, List[Dict[str, Any]]],
) -> Optional[str]:
    party_block_diagnostics.extend(page_diagnostics.get("party_block_diagnostics", []))
    for key, values in page_diagnostics.get("summary_candidate_diagnostics", {}).items():
        summary_candidate_diagnostics.setdefault(key, []).extend(values)
    return page_diagnostics.get("line_item_source")


def _build_heuristic_diagnostics(
    layout_template: Dict[str, Any],
    extracted_values: Dict[str, Any],
    matched_label_count: int,
    critical_field_count: int,
    critical_field_total: int,
    template_confidence: float,
    table_detected: bool,
    party_block_diagnostics: List[Dict[str, Any]],
    summary_candidate_diagnostics: Dict[str, List[Dict[str, Any]]],
    line_item_source: Optional[str],
) -> Dict[str, Any]:
    return {
        "discovery_mode": "heuristic",
        "located_field_count": sum(
            len(page.get("fields", {})) for page in layout_template["pages"]
        ),
        "extracted_field_count": len(extracted_values),
        "locate_raw_response": None,
        "extract_raw_response": None,
        "matched_label_count": matched_label_count,
        "critical_field_count": critical_field_count,
        "critical_field_total": critical_field_total,
        "heuristic_confidence": round(template_confidence, 3),
        "table_detected": table_detected,
        "party_block_diagnostics": party_block_diagnostics,
        "summary_candidate_diagnostics": summary_candidate_diagnostics,
        "line_item_source": line_item_source,
    }


def infer_template_heuristic(
    pages: List[np.ndarray],
    gt_json: Optional[Dict[str, Any]],
    config: Any,
    ocr_results_per_page: Optional[List[List[OCRToken]]] = None,
) -> Tuple[Dict[str, Any], float, Optional[Dict[str, Any]], Dict[str, Any]]:
    expected_fields = expected_scalar_fields(gt_json)
    heuristic_config = getattr(config, "heuristic_discovery", None)
    layout_template: Dict[str, Any] = {
        "version": "v3",
        "inference_method": "heuristic_discovery",
        "discovery_mode": "heuristic",
        "pages": [],
    }

    matched_label_count = 0
    critical_field_count = 0
    critical_field_total = 0
    page_confidences: List[float] = []
    extracted_values: Dict[str, Any] = {}
    table_detected = False
    party_block_diagnostics: List[Dict[str, Any]] = []
    summary_candidate_diagnostics: Dict[str, List[Dict[str, Any]]] = {}
    line_item_source: Optional[str] = None

    for page_index, page_img in enumerate(pages):
        ocr_tokens = (
            ocr_results_per_page[page_index]
            if ocr_results_per_page and page_index < len(ocr_results_per_page)
            else []
        )
        (
            page_fields,
            page_extracted_values,
            label_confirmation_set,
            table_info,
            page_diagnostics,
            page_confidence,
        ) = _discover_page_fields(
            page_index,
            page_img,
            ocr_tokens,
            expected_fields,
            heuristic_config,
        )
        layout_template["pages"].append(
            {
                "page_index": page_index,
                "fields": page_fields,
                "label_confirmation_set": label_confirmation_set,
                "table": table_info,
            }
        )
        matched_label_count += page_diagnostics["matched_label_count"]
        critical_field_count += page_diagnostics["critical_field_count"]
        critical_field_total += page_diagnostics["critical_field_total"]
        page_line_item_source = _merge_page_heuristic_diagnostics(
            page_diagnostics,
            party_block_diagnostics,
            summary_candidate_diagnostics,
        )
        if page_line_item_source:
            line_item_source = page_line_item_source
        page_confidences.append(page_confidence)
        extracted_values.update(page_extracted_values)
        table_detected = table_detected or table_info is not None

    _reconcile_summary_amounts(extracted_values)
    _reconcile_party_names(extracted_values)

    template_confidence = sum(page_confidences) / len(page_confidences) if page_confidences else 0.0
    diagnostics = _build_heuristic_diagnostics(
        layout_template,
        extracted_values,
        matched_label_count,
        critical_field_count,
        critical_field_total,
        template_confidence,
        table_detected,
        party_block_diagnostics,
        summary_candidate_diagnostics,
        line_item_source,
    )
    return layout_template, round(template_confidence, 3), extracted_values, diagnostics
