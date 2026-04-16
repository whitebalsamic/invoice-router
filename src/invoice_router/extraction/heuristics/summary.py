import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

OCRToken = Tuple[str, int, int, int, int]

SUMMARY_ROLE_LABELS = {
    "subtotal": {"subtotal", "sub total", "net amount"},
    "tax": {"tax", "sales tax", "vat", "gst"},
    "invoice_total": {"invoice total", "total"},
    "amount_due": {"amount due", "total due"},
    "balance_due": {"balance due"},
    "shipping": {"shipping", "shipping and handling", "handling", "s&h", "s h"},
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


def normalize_ocr_text(value: str) -> str:
    lowered = value.lower().translate(OCR_LABEL_TRANSLATIONS)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", lowered)).strip()


def _join_tokens(tokens: Sequence[OCRToken]) -> str:
    return " ".join(
        token[0].strip()
        for token in sorted(tokens, key=lambda item: (item[2], item[1]))
        if token[0].strip()
    ).strip()


def _row_bounds(row: Sequence[OCRToken]) -> Tuple[int, int, int, int]:
    min_x = min(token[1] for token in row)
    min_y = min(token[2] for token in row)
    max_x = max(token[1] + token[3] for token in row)
    max_y = max(token[2] + token[4] for token in row)
    return min_x, min_y, max_x - min_x, max_y - min_y


def _position_prior(field_name: Optional[str], x: int, y: int, page_w: int, page_h: int) -> float:
    if not field_name or page_w <= 0 or page_h <= 0:
        return 0.0

    field_key = re.sub(r"[^a-z0-9]", "", field_name.lower())
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


def line_item_amount_total(extracted_values: Dict[str, Any]) -> Optional[float]:
    line_items = extracted_values.get("line_items")
    if not isinstance(line_items, list):
        return None

    amount_values = [
        _token_to_number(str(item.get("amount")))
        for item in line_items
        if isinstance(item, dict) and item.get("amount") is not None
    ]
    amount_values = [value for value in amount_values if value is not None]
    if not amount_values:
        return None
    return round(sum(amount_values), 2)


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


def _looks_like_non_summary_number(raw_text: str, row_text: str) -> bool:
    stripped = raw_text.strip()
    row_norm = normalize_ocr_text(row_text)
    if re.search(r"\d[-/]\d", stripped):
        return True
    if re.fullmatch(r"\d{3}[- ]\d{2}[- ]\d{4}", stripped):
        return True
    if any(
        label in row_norm for label in ("invoice", "invoice no", "invoice number", "id", "date")
    ):
        if "." not in stripped and "," not in stripped and not re.search(r"[$€£]", stripped):
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


def _clean_extracted_candidate(field_name: str, value: str) -> Optional[str]:
    value = value.strip()
    if not value:
        return None

    normalized_name = re.sub(r"[^a-z0-9]", "", field_name.lower())
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

    return value[:80].strip()


def _summary_concepts_for_row(row_text: str) -> List[str]:
    if (
        _is_mixed_content_row(row_text)
        or _looks_tableish_row_text(row_text)
        or _looks_metadataish_row_text(row_text)
    ) and not _looks_like_repeated_summary_triplet(row_text):
        return []
    normalized = normalize_ocr_text(row_text)
    concepts: List[str] = []
    for concept, labels in SUMMARY_ROLE_LABELS.items():
        if any(
            re.search(rf"(?<![a-z0-9]){re.escape(label)}(?![a-z0-9])", normalized)
            for label in labels
        ):
            concepts.append(concept)
    amount_like_count = len(_summary_amount_values(row_text))
    if _looks_like_repeated_summary_triplet(row_text):
        return ["subtotal", "tax", "invoice_total"]
    if amount_like_count >= 4:
        return []
    if "subtotal" in concepts:
        concepts = [
            concept
            for concept in concepts
            if concept not in {"invoice_total", "amount_due", "balance_due", "total_due"}
        ]
    if "tax" in concepts:
        concepts = [
            concept
            for concept in concepts
            if concept not in {"invoice_total", "amount_due", "balance_due", "total_due"}
        ]
    if (
        len(concepts) > 1
        and amount_like_count >= 2
        and not any(phrase in normalized for phrase in ("amount due", "total due", "balance due"))
    ):
        if not (
            {"subtotal", "tax"} & set(concepts)
            and "invoice_total" in concepts
            and amount_like_count == 3
        ):
            return []
    return concepts


def _extract_summary_row_amount(
    row: Sequence[OCRToken],
    page_w: int,
    page_h: int,
    field_hint: str,
) -> Optional[Dict[str, Any]]:
    row_text = _join_tokens(row)
    if (
        _is_mixed_content_row(row_text) or _looks_tableish_row_text(row_text)
    ) and not _looks_like_repeated_summary_triplet(row_text):
        return None
    row_bbox = _row_bounds(row)
    amount_strings = [candidate["raw"] for candidate in _summary_amount_values(row_text)]
    if len(amount_strings) >= 4:
        return None
    best_candidate: Optional[Dict[str, Any]] = None
    for token in row:
        raw_text = token[0].strip()
        if not raw_text:
            continue
        if "%" in raw_text:
            continue
        if _looks_like_non_summary_number(raw_text, row_text):
            continue
        cleaned = _clean_extracted_candidate(field_hint, raw_text)
        if not cleaned:
            continue
        numeric = _token_to_number(cleaned)
        if numeric is None:
            continue
        if numeric < 0:
            continue
        bbox = _row_bounds([token])
        score = _amount_candidate_score(field_hint, cleaned, bbox, page_w, page_h, row_text)
        if raw_text.startswith(("$", "£", "€")):
            score += 0.2
        if "." in raw_text:
            score += 0.1
        score += (token[1] / max(page_w, 1)) * 0.15
        candidate = {"value": cleaned, "bbox": bbox, "score": score, "row_text": row_text}
        if best_candidate is None or candidate["score"] > best_candidate["score"]:
            best_candidate = candidate

    if best_candidate:
        return best_candidate

    if not amount_strings:
        return None
    return {
        "value": amount_strings[-1],
        "bbox": row_bbox,
        "score": _amount_candidate_score(
            field_hint, amount_strings[-1], row_bbox, page_w, page_h, row_text
        ),
        "row_text": row_text,
    }


def _derive_tax_candidate_from_total_row(
    row: Sequence[OCRToken],
    page_w: int,
    page_h: int,
) -> Optional[Dict[str, Any]]:
    row_text = _join_tokens(row)
    normalized = normalize_ocr_text(row_text)
    if _is_mixed_content_row(row_text) or _looks_tableish_row_text(row_text):
        return None
    if not any(
        label in normalized for label in ("total", "amount due", "balance due", "total due")
    ):
        return None

    amount_strings = [
        candidate
        for candidate in _extract_amount_candidates_from_row(row)
        if "%" not in candidate and _token_to_number(candidate) is not None
    ]
    if len(amount_strings) < 3:
        return None

    subtotal_value, tax_value, total_value = amount_strings[-3:]
    subtotal_numeric = _token_to_number(subtotal_value)
    tax_numeric = _token_to_number(tax_value)
    total_numeric = _token_to_number(total_value)
    if subtotal_numeric is None or tax_numeric is None or total_numeric is None:
        return None
    if subtotal_numeric < 0 or tax_numeric < 0 or total_numeric < 0:
        return None
    if abs((subtotal_numeric + tax_numeric) - total_numeric) > max(1.0, total_numeric * 0.03):
        return None

    row_bbox = _row_bounds(row)
    cleaned_tax = _clean_extracted_candidate("tax", tax_value) or tax_value
    return {
        "value": cleaned_tax,
        "bbox": row_bbox,
        "score": _amount_candidate_score("tax", cleaned_tax, row_bbox, page_w, page_h, row_text)
        + 0.16,
        "row_text": row_text,
    }


def _derive_subtotal_candidate_from_total_row(
    row: Sequence[OCRToken],
    page_w: int,
    page_h: int,
) -> Optional[Dict[str, Any]]:
    row_text = _join_tokens(row)
    normalized = normalize_ocr_text(row_text)
    if _is_mixed_content_row(row_text) or _looks_tableish_row_text(row_text):
        return None
    if not any(
        label in normalized for label in ("total", "amount due", "balance due", "total due")
    ):
        return None

    amount_strings = [
        candidate
        for candidate in _extract_amount_candidates_from_row(row)
        if "%" not in candidate and _token_to_number(candidate) is not None
    ]
    if len(amount_strings) < 3:
        return None

    subtotal_value, tax_value, total_value = amount_strings[-3:]
    subtotal_numeric = _token_to_number(subtotal_value)
    tax_numeric = _token_to_number(tax_value)
    total_numeric = _token_to_number(total_value)
    if subtotal_numeric is None or tax_numeric is None or total_numeric is None:
        return None
    if subtotal_numeric < 0 or tax_numeric < 0 or total_numeric < 0:
        return None
    if abs((subtotal_numeric + tax_numeric) - total_numeric) > max(1.0, total_numeric * 0.03):
        return None

    row_bbox = _row_bounds(row)
    cleaned_subtotal = _clean_extracted_candidate("subtotal", subtotal_value) or subtotal_value
    return {
        "value": cleaned_subtotal,
        "bbox": row_bbox,
        "score": _amount_candidate_score(
            "subtotal", cleaned_subtotal, row_bbox, page_w, page_h, row_text
        )
        + 0.16,
        "row_text": row_text,
    }


def _collect_summary_candidates(
    rows: Sequence[Sequence[OCRToken]],
    page_w: int,
    page_h: int,
) -> Dict[str, List[Dict[str, Any]]]:
    collected: Dict[str, List[Dict[str, Any]]] = {concept: [] for concept in SUMMARY_ROLE_LABELS}
    for row in rows:
        row_text = _join_tokens(row)
        concepts = _summary_concepts_for_row(row_text)
        derived_subtotal = _derive_subtotal_candidate_from_total_row(row, page_w, page_h)
        if derived_subtotal is not None:
            collected["subtotal"].append(
                {
                    **derived_subtotal,
                    "label": "subtotal",
                }
            )
        derived_tax = _derive_tax_candidate_from_total_row(row, page_w, page_h)
        if derived_tax is not None:
            collected["tax"].append(
                {
                    **derived_tax,
                    "label": "tax",
                }
            )
        if not concepts:
            continue
        if _looks_like_repeated_summary_triplet(row_text):
            amount_values = _summary_amount_values(row_text)
            if len(amount_values) == 3:
                ordered_triplet = {
                    "subtotal": amount_values[0]["raw"],
                    "tax": amount_values[1]["raw"],
                    "invoice_total": amount_values[2]["raw"],
                }
                row_bbox = _row_bounds(row)
                for concept, raw_value in ordered_triplet.items():
                    field_hint = (
                        "tax"
                        if concept == "tax"
                        else "subtotal"
                        if concept == "subtotal"
                        else "totalAmount"
                    )
                    base_score = _amount_candidate_score(
                        field_hint, raw_value, row_bbox, page_w, page_h, row_text
                    )
                    bonus = 0.25 if concept == "invoice_total" else 0.2
                    collected[concept].append(
                        {
                            "value": raw_value,
                            "bbox": row_bbox,
                            "score": base_score + bonus,
                            "row_text": row_text,
                            "label": concept,
                        }
                    )
                continue
        for concept in concepts:
            if concept == "tax":
                field_hint = "tax"
            elif concept == "subtotal":
                field_hint = "subtotal"
            elif concept == "shipping":
                field_hint = "shipping"
            else:
                field_hint = "totalAmount"
            candidate = _extract_summary_row_amount(row, page_w, page_h, field_hint)
            if candidate is None:
                continue
            bonus = 0.0
            if concept in {"amount_due", "balance_due"}:
                bonus = 0.35
            elif concept in {"invoice_total", "total_due"}:
                bonus = 0.25
            elif concept in {"subtotal", "tax"}:
                bonus = 0.2
            collected[concept].append(
                {
                    **candidate,
                    "score": candidate["score"] + bonus,
                    "label": concept,
                }
            )
        if not collected["tax"] and any(
            concept in {"invoice_total", "amount_due", "balance_due", "total_due"}
            for concept in concepts
        ):
            amount_candidates = [candidate["raw"] for candidate in _summary_amount_values(row_text)]
            if len(amount_candidates) == 3:
                middle_amount = amount_candidates[1]
                tax_candidate = {
                    "value": middle_amount,
                    "bbox": _row_bounds(row),
                    "score": _amount_candidate_score(
                        "tax", middle_amount, _row_bounds(row), page_w, page_h, row_text
                    )
                    + 0.16,
                    "row_text": row_text,
                    "label": "tax",
                }
                if tax_candidate["score"] >= 0.7:
                    collected["tax"].append(tax_candidate)
    return collected


def _reconcile_summary_amounts(extracted_values: Dict[str, Any]) -> None:
    subtotal_fields = [key for key in extracted_values if "subtotal" in key.lower()]
    tax_fields = [key for key in extracted_values if "tax" in key.lower()]
    shipping_fields = [key for key in extracted_values if "shipping" in key.lower()]
    total_fields = [
        key
        for key in extracted_values
        if ("total" in key.lower() or "amount" in key.lower())
        and "subtotal" not in key.lower()
        and "tax" not in key.lower()
    ]
    if not subtotal_fields or not total_fields:
        return

    subtotal_value = _token_to_number(str(extracted_values[subtotal_fields[0]]))
    tax_value = _token_to_number(str(extracted_values[tax_fields[0]])) if tax_fields else None
    shipping_value = (
        _token_to_number(str(extracted_values[shipping_fields[0]])) if shipping_fields else 0.0
    )
    if subtotal_value is None:
        return

    line_item_subtotal = line_item_amount_total(extracted_values)

    if tax_value is None:
        total_numeric_candidates = [
            _token_to_number(str(extracted_values.get(field_name))) for field_name in total_fields
        ]
        total_numeric_candidates = [
            value for value in total_numeric_candidates if value is not None
        ]
        if total_numeric_candidates:
            candidate_total = max(total_numeric_candidates)
            derived_tax = round(candidate_total - subtotal_value - (shipping_value or 0.0), 2)
            if 0.0 < derived_tax <= max(candidate_total, subtotal_value):
                if tax_fields:
                    extracted_values[tax_fields[0]] = f"{derived_tax:.2f}"
                else:
                    extracted_values["tax"] = f"{derived_tax:.2f}"
                tax_value = derived_tax

    reconciled_total = subtotal_value + (tax_value or 0.0) + (shipping_value or 0.0)
    for field_name in total_fields:
        current_numeric = _token_to_number(str(extracted_values.get(field_name)))
        if current_numeric is None:
            if tax_value is not None:
                extracted_values[field_name] = f"{reconciled_total:.2f}"
            continue
        if tax_value is None:
            continue
        if current_numeric <= subtotal_value + 0.01:
            extracted_values[field_name] = f"{reconciled_total:.2f}"
            continue
        if (
            line_item_subtotal is not None
            and tax_value is not None
            and abs(current_numeric - line_item_subtotal) <= 0.02
        ):
            extracted_values[field_name] = (
                f"{line_item_subtotal + tax_value + (shipping_value or 0.0):.2f}"
            )
