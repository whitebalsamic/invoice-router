import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dateutil.parser import parse as parse_date

from ...extraction.heuristics.party import _trim_party_block_text

OCRToken = Tuple[str, int, int, int, int]

PARTY_LABEL_ALIASES = {
    "sellerName": ("seller", "vendor", "supplier", "from", "remit"),
    "customerName": ("bill to", "customer", "buyer", "client", "invoice to", "ship to"),
}

PARTY_STOP_WORDS = {
    "date",
    "invoice",
    "tax",
    "iban",
    "id",
    "qty",
    "description",
    "total",
    "subtotal",
    "amount",
    "due",
}

BUSINESS_SUFFIXES = {
    "ltd",
    "limited",
    "llc",
    "inc",
    "corp",
    "company",
    "plc",
    "corporation",
}

COUNTRY_NAME_WORDS = {
    "canada",
    "mexico",
    "usa",
    "unitedstates",
}


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _token_center(token: OCRToken) -> Tuple[float, float]:
    _, x, y, w, h = token
    return x + (w / 2.0), y + (h / 2.0)


def _token_line_key(token: OCRToken) -> int:
    _, _, y, _, h = token
    return int((y + (h / 2.0)) / max(h, 1))


def _tokens_in_region(
    ocr_tokens: Sequence[OCRToken],
    region: Dict[str, float],
    page_w: int,
    page_h: int,
) -> List[OCRToken]:
    x1 = region["x"] * page_w
    y1 = region["y"] * page_h
    x2 = x1 + (region["width"] * page_w)
    y2 = y1 + (region["height"] * page_h)

    selected: List[OCRToken] = []
    for token in ocr_tokens:
        cx, cy = _token_center(token)
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            selected.append(token)
    return selected


def _join_tokens(tokens: Sequence[OCRToken]) -> str:
    ordered = sorted(tokens, key=lambda t: (_token_line_key(t), t[1]))
    if not ordered:
        return ""

    parts: List[str] = []
    prev_line = None
    for token in ordered:
        text = token[0].strip()
        if not text:
            continue
        line = _token_line_key(token)
        if prev_line is not None and line != prev_line:
            parts.append("\n")
        elif parts and parts[-1] != "\n":
            parts.append(" ")
        parts.append(text)
        prev_line = line
    return "".join(parts).strip()


def _group_tokens_into_rows(tokens: Sequence[OCRToken]) -> List[List[OCRToken]]:
    if not tokens:
        return []
    ordered = sorted(tokens, key=lambda item: (item[2] + item[4] / 2.0, item[1]))
    rows: List[List[OCRToken]] = []
    current: List[OCRToken] = [ordered[0]]
    current_center = ordered[0][2] + ordered[0][4] / 2.0
    for token in ordered[1:]:
        center = token[2] + token[4] / 2.0
        if abs(center - current_center) <= max(token[4], current[-1][4], 18):
            current.append(token)
            current_center = sum(t[2] + t[4] / 2.0 for t in current) / len(current)
        else:
            rows.append(sorted(current, key=lambda item: item[1]))
            current = [token]
            current_center = center
    rows.append(sorted(current, key=lambda item: item[1]))
    return rows


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


def _find_label_neighbor_text(
    ocr_tokens: Sequence[OCRToken],
    label_text: Optional[str],
    page_w: int,
) -> str:
    if not label_text:
        return ""

    label_norm = _normalize_text(label_text)
    label_words = [
        _normalize_text(word) for word in re.split(r"\s+", str(label_text)) if _normalize_text(word)
    ]
    if not label_norm:
        return ""

    best: Optional[OCRToken] = None
    best_score: Optional[int] = None
    for token in ocr_tokens:
        text_norm = _normalize_text(token[0])
        if not text_norm:
            continue
        candidate_targets = [label_norm, *label_words]
        score = min(_levenshtein(text_norm, target) for target in candidate_targets if target)
        if best_score is None or score < best_score:
            best = token
            best_score = score

    if best is None or best_score is None:
        return ""

    best_text_norm = _normalize_text(best[0])
    accept_threshold = max(1, len(best_text_norm) // 3)
    if best_score > accept_threshold:
        return ""

    _, x, y, w, h = best
    row_tokens = []
    row_mid = y + (h / 2.0)
    for token in ocr_tokens:
        text, tx, ty, tw, th = token
        if not text.strip():
            continue
        token_mid = ty + (th / 2.0)
        if abs(token_mid - row_mid) > max(h, th):
            continue
        if tx <= x + w:
            continue
        if tx > page_w * 0.95:
            continue
        row_tokens.append(token)

    return _join_tokens(row_tokens)


DATE_PATTERNS = [
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
]

DATE_LABEL_PREFIX_RE = re.compile(
    r"(?i)^\s*(?:date(?:\s+of\s+issue)?|invoice\s+date|issue\s+date|of\s+issue)\b(?:\s*[:#-]|\s+)?\s*"
)

INVOICE_LABEL_PREFIX_RE = re.compile(
    r"(?i)^\s*(?:invoice(?:\s*(?:no\.?|number|#))?|inv(?:\s+(?:no\.?|number)|\s*#|(?=\s*:)))\s*[:#-]?\s*"
)

SCI_NOTATION_RE = re.compile(r"(?i)\b\d+(?:[.,]\d+)?e[+-]?\d+\b")
NUMERIC_GARBAGE_RE = re.compile(r"\d{12,}")

AMOUNT_PATTERN = r"[$€£]?\s*-?\d[\d,]*(?:\.\d{2})?"
INVOICE_NUMBER_PATTERN = r"\b[A-Z0-9][A-Z0-9./-]{2,}\b"


def _extract_date_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    if not text:
        return candidates
    for pattern in DATE_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(0).strip()
            if value not in candidates:
                candidates.append(value)
    return candidates


def _cleanup_label_prefixed_date_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    text = DATE_LABEL_PREFIX_RE.sub("", text).strip(" :,-|")
    return text


def _parse_date_candidate(value: str) -> Optional[str]:
    value = _cleanup_label_prefixed_date_text(value)
    try:
        return parse_date(value, fuzzy=True).strftime("%Y-%m-%d")
    except Exception:
        return None


def _to_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned_value = value.strip()
        if _is_implausible_numeric_candidate(cleaned_value):
            return None
    text = re.sub(r"[^\d.,-]", "", str(value).strip())
    if not text or not any(ch.isdigit() for ch in text):
        return None
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _is_implausible_numeric_candidate(value: Any) -> bool:
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
    if NUMERIC_GARBAGE_RE.search(digits) and not re.search(r"[.,]", text):
        return True
    if len(digits) >= 10 and digits and len(set(digits)) == 1:
        return True
    return False


def _extract_amount_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    if not text:
        return candidates
    split_decimal_pattern = re.compile(r"([$€£]?\s*-?\d[\d,]*)\s*([.,])\s*(\d{2})")
    split_decimal_candidates: List[str] = []
    for match in split_decimal_pattern.finditer(text):
        whole = re.sub(r"\s+", "", match.group(1))
        decimal = match.group(2)
        fraction = match.group(3)
        value = f"{whole}{decimal}{fraction}"
        if _is_implausible_numeric_candidate(value):
            continue
        if any(ch.isdigit() for ch in value) and value not in candidates:
            candidates.append(value)
            split_decimal_candidates.append(value)
    for match in re.finditer(AMOUNT_PATTERN, text):
        value = match.group(0).strip()
        compact = re.sub(r"\s+", "", value)
        before = text[match.start() - 1] if match.start() > 0 else ""
        after = text[match.end()] if match.end() < len(text) else ""
        if before.isalpha() or after.isalpha():
            continue
        if _is_implausible_numeric_candidate(compact):
            continue
        if any(
            split.startswith(compact)
            and re.sub(r"[^\d]", "", split).startswith(re.sub(r"[^\d]", "", compact))
            and split != compact
            for split in split_decimal_candidates
        ):
            continue
        if any(ch.isdigit() for ch in value) and value not in candidates:
            candidates.append(value)
    return candidates


def _resolve_date_value(current_value: Any, candidates: List[str]) -> Any:
    current_clean = _cleanup_label_prefixed_date_text(current_value)
    current_raw = re.sub(r"\s+", " ", str(current_value or "")).strip()
    current_had_label_prefix = bool(current_clean) and current_clean != current_raw
    current_is_valid = _parse_date_candidate(current_clean) if current_clean else None
    valid_candidates = []
    for candidate in candidates:
        cleaned_candidate = _cleanup_label_prefixed_date_text(candidate)
        if _parse_date_candidate(cleaned_candidate):
            valid_candidates.append(cleaned_candidate)
    if current_is_valid and valid_candidates:
        if (
            len(valid_candidates) == 1
            and valid_candidates[0] != current_clean
            and current_had_label_prefix
        ):
            return valid_candidates[0]
        return current_clean or current_value
    if len(valid_candidates) == 1:
        return valid_candidates[0]
    return current_clean or current_value


def _resolve_amount_value(current_value: Any, candidates: List[str]) -> Any:
    distinct = []
    for candidate in candidates:
        normalized = re.sub(r"\s+", "", candidate)
        if normalized not in {re.sub(r"\s+", "", existing) for existing in distinct}:
            distinct.append(candidate)
    if len(distinct) == 1:
        return distinct[0]
    return current_value


def _extract_invoice_number_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    if not text:
        return candidates
    text = _cleanup_invoice_number_text(text)
    for match in re.finditer(INVOICE_NUMBER_PATTERN, text, flags=re.IGNORECASE):
        value = match.group(0).strip()
        if sum(ch.isdigit() for ch in value) == 0:
            continue
        if value not in candidates:
            candidates.append(value)
    return candidates


def _cleanup_invoice_number_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    text = INVOICE_LABEL_PREFIX_RE.sub("", text).strip(" :,-|")
    text = re.sub(r"[|,;:\s-]+$", "", text).strip()
    tokens = [token for token in text.split() if token]
    if len(tokens) >= 2:
        prefix_tokens: List[str] = []
        for index, token in enumerate(tokens):
            normalized = _normalize_text(token)
            if not normalized:
                continue
            if (
                normalized
                in {
                    "invoice",
                    "invoices",
                    "date",
                    "due",
                    "total",
                    "tax",
                    "subtotal",
                    "amount",
                    "address",
                    "street",
                    "suite",
                    "apt",
                    "unit",
                    "city",
                    "state",
                    "province",
                    "zip",
                    "postal",
                    "phone",
                    "fax",
                    "email",
                    "www",
                }
                or normalized in PARTY_STOP_WORDS
            ):
                break
            if re.search(r"https?://|www\.", token, flags=re.IGNORECASE) or "@" in token:
                break
            if index > 0 and re.fullmatch(r"[|/,:;-]+", token):
                break
            prefix_tokens.append(token)
        if prefix_tokens and any(ch.isdigit() for ch in "".join(prefix_tokens)):
            text = " ".join(prefix_tokens).strip(" :,-|")
    return text


def _looks_suspicious_party_value(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    if _normalize_text(text) in COUNTRY_NAME_WORDS:
        return True
    words = [token for token in re.split(r"\s+", text) if token]
    normalized_words = {re.sub(r"[^a-z]", "", word.lower()) for word in words}
    if len(words) == 1 and not (normalized_words & BUSINESS_SUFFIXES):
        return True
    if len(words) > 8:
        return True
    if sum(ch.isdigit() for ch in text) >= 5:
        return True
    if "@" in text:
        return True
    return False


def _party_candidate_score(field_name: str, text: str) -> float:
    cleaned = re.sub(r"\s+", " ", text.strip(" :,-|"))
    if not cleaned:
        return -1.0
    if _normalize_text(cleaned) in COUNTRY_NAME_WORDS:
        return -1.0
    words = [word for word in re.split(r"\s+", cleaned) if word]
    if not words:
        return -1.0
    alpha_count = sum(ch.isalpha() for ch in cleaned)
    digit_count = sum(ch.isdigit() for ch in cleaned)
    score = 0.0
    if alpha_count >= 4:
        score += 1.0
    if digit_count:
        score -= min(1.2, digit_count * 0.2)
    if 1 <= len(words) <= 5:
        score += 0.8
    elif len(words) <= 8:
        score += 0.3
    else:
        score -= 1.0
    lower_words = {re.sub(r"[^a-z]", "", word.lower()) for word in words}
    if lower_words & BUSINESS_SUFFIXES:
        score += 0.8
    if field_name == "customerName" and len(words) <= 3:
        score += 0.3
    if field_name == "customerName" and 3 <= len(words) <= 5:
        score += 0.6
    if field_name == "customerName" and len(words) >= 4:
        score += 0.4
    if any(word in PARTY_STOP_WORDS for word in lower_words):
        score -= 1.2
    if cleaned.upper() == cleaned and alpha_count >= 4:
        score += 0.2
    return score


def _extract_party_from_labeled_rows(
    field_name: str,
    rows: Sequence[Sequence[OCRToken]],
) -> Optional[str]:
    aliases = PARTY_LABEL_ALIASES[field_name]
    best: Optional[Tuple[float, str]] = None
    for row in rows:
        row_text = _join_tokens(row)
        normalized_row = _normalize_text(row_text)
        if not any(alias.replace(" ", "") in normalized_row for alias in aliases):
            continue
        label_x = None
        normalized_tokens = [_normalize_text(token[0]) for token in row]
        for alias in aliases:
            alias_words = [_normalize_text(word) for word in alias.split()]
            if not alias_words:
                continue
            for index in range(0, max(len(row) - len(alias_words) + 1, 0)):
                window = normalized_tokens[index : index + len(alias_words)]
                if window == alias_words:
                    last_token = row[index + len(alias_words) - 1]
                    label_x = last_token[1] + last_token[3]
                    break
            if label_x is not None:
                break
        right_tokens = [token for token in row if label_x is None or token[1] >= label_x]
        contiguous_tokens: List[OCRToken] = []
        for token in right_tokens:
            token_word = _normalize_text(token[0])
            if token_word in PARTY_STOP_WORDS:
                break
            contiguous_tokens.append(token)
        contiguous_phrase = _trim_party_block_text(_join_tokens(contiguous_tokens))
        contiguous_score = _party_candidate_score(field_name, contiguous_phrase)
        if contiguous_score >= 1.25:
            if best is None or contiguous_score > best[0]:
                best = (contiguous_score, contiguous_phrase.strip(" :,-|"))
        max_phrase_tokens = 8 if field_name == "customerName" else 5
        for start in range(len(right_tokens)):
            for end in range(start + 1, min(len(right_tokens), start + max_phrase_tokens) + 1):
                phrase = _trim_party_block_text(_join_tokens(right_tokens[start:end]))
                score = _party_candidate_score(field_name, phrase)
                if score < 1.1:
                    continue
                if best is None or score > best[0]:
                    best = (score, phrase.strip(" :,-|"))
    return best[1] if best else None


def _is_year_like_invoice_candidate(value: str) -> bool:
    text = re.sub(r"\D", "", value)
    if len(text) != 4:
        return False
    year = int(text)
    return 1900 <= year <= 2100


def _invoice_number_score(candidate: str, current_value: Any) -> Tuple[float, int, int]:
    text = candidate.strip()
    digit_count = sum(ch.isdigit() for ch in text)
    alpha_count = sum(ch.isalpha() for ch in text)
    lower_count = sum(ch.islower() for ch in text)
    separator_count = sum(ch in "-/." for ch in text)
    score = 0.0

    score += min(digit_count, 8) * 2.0
    score -= alpha_count * 0.5
    score -= lower_count * 1.5
    score += separator_count * 0.25
    if len(text) <= 4 and digit_count < 4:
        score -= 3.0
    if len(text) <= 6 and alpha_count > 0:
        score -= 2.5
    if len(text) <= 6 and digit_count < max(alpha_count, 1):
        score -= 4.0
    if digit_count >= max(alpha_count, 1):
        score += 2.0
    if digit_count >= 6 and alpha_count == 0:
        score += 2.0
    elif digit_count >= 3 and alpha_count == 0:
        score += 1.0
    if alpha_count and text.upper() == text:
        score += 1.0
    if _is_year_like_invoice_candidate(text):
        score -= 8.0

    current_text = str(current_value).strip() if current_value is not None else ""
    current_digits = re.sub(r"\D", "", current_text)
    candidate_digits = re.sub(r"\D", "", text)
    if current_digits and candidate_digits:
        prefix_match = 0
        for a, b in zip(candidate_digits, current_digits):
            if a != b:
                break
            prefix_match += 1
        score += prefix_match * 0.75
        score -= abs(len(candidate_digits) - len(current_digits)) * 0.5

    return score, digit_count, -alpha_count


def _resolve_invoice_number_value(
    current_value: Any,
    region_candidates: Sequence[str],
    neighbor_candidates: Sequence[str],
) -> Any:
    candidates: List[Tuple[str, float]] = []
    seen = set()
    current_text = _cleanup_invoice_number_text(current_value)
    comparison_value = current_text or current_value
    if current_text:
        current_score, _, _ = _invoice_number_score(current_text, comparison_value)
        candidates.append((current_text, 0.75 if current_score >= 8.0 else 0.0))
        seen.add(current_text)
    for candidate in neighbor_candidates:
        candidate = _cleanup_invoice_number_text(candidate)
        if candidate in seen:
            continue
        candidates.append((candidate, 0.5))
        seen.add(candidate)
    for candidate in region_candidates:
        candidate = _cleanup_invoice_number_text(candidate)
        if candidate in seen:
            continue
        candidates.append((candidate, 0.0))
        seen.add(candidate)

    if not candidates:
        return current_text or current_value

    ranked = sorted(
        candidates,
        key=lambda item: (
            (_invoice_number_score(item[0], comparison_value)[0] + item[1],)
            + _invoice_number_score(item[0], comparison_value)[1:]
        ),
        reverse=True,
    )
    best, best_bonus = ranked[0]
    best_score, _, _ = _invoice_number_score(best, comparison_value)
    best_score += best_bonus
    if len(ranked) == 1:
        return best if best_score >= 3.0 else (current_text or current_value)

    next_candidate, next_bonus = ranked[1]
    next_score, _, _ = _invoice_number_score(next_candidate, comparison_value)
    next_score += next_bonus
    if (
        re.search(r"\d", best)
        and not re.search(r"[A-Za-z]", best)
        and re.search(r"[A-Za-z]", next_candidate)
        and best_score >= next_score + 0.5
    ):
        return best
    if best_score >= max(3.0, next_score + 1.5):
        return best
    return current_text or current_value


def _dedupe_amount_candidates(candidates: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for candidate in candidates:
        normalized = re.sub(r"\s+", "", candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


def _resolve_amount_in_context(
    field_name: str,
    current_value: Any,
    candidates: Sequence[str],
    current_amounts: Dict[str, Optional[float]],
) -> Any:
    deduped = _dedupe_amount_candidates(candidates)
    if len(deduped) <= 1:
        return deduped[0] if deduped else current_value

    parsed = [(candidate, _to_number(candidate)) for candidate in deduped]
    parsed = [(candidate, value) for candidate, value in parsed if value is not None]
    if len(parsed) <= 1:
        return current_value

    field_name_lower = field_name.lower()
    subtotal_val = current_amounts.get("subtotal")
    tax_val = current_amounts.get("tax")
    total_val = current_amounts.get("total")
    current_numeric = _to_number(current_value)

    def _is_implausibly_small_summary_value(value: Optional[float]) -> bool:
        if value is None:
            return True
        reference = max(
            subtotal_val or 0.0,
            total_val or 0.0,
            tax_val or 0.0,
        )
        if reference >= 100.0 and value <= 20.0:
            return True
        if "total" in field_name_lower or "amount" in field_name_lower:
            if (
                subtotal_val is not None
                and subtotal_val >= 50.0
                and value + 0.01 < subtotal_val * 0.5
            ):
                return True
        if "tax" in field_name_lower:
            if (
                subtotal_val is not None
                and subtotal_val >= 100.0
                and value <= max(20.0, subtotal_val * 0.02)
            ):
                return True
        return False

    if "total" in field_name_lower or "amount" in field_name_lower:
        if subtotal_val is not None and tax_val is not None:
            target = subtotal_val + tax_val
            best = min(parsed, key=lambda item: abs(item[1] - target))
            if current_numeric is None or abs(best[1] - target) + 0.01 < abs(
                current_numeric - target
            ):
                return best[0]
        if subtotal_val is not None and _is_implausibly_small_summary_value(current_numeric):
            plausible = [item for item in parsed if item[1] >= subtotal_val + 0.01]
            if plausible:
                return min(plausible, key=lambda item: item[1])[0]
        if subtotal_val is not None and (
            current_numeric is None or current_numeric <= subtotal_val + 0.01
        ):
            larger = [item for item in parsed if item[1] > subtotal_val + 0.01]
            if larger:
                return min(larger, key=lambda item: item[1])[0]
        return current_value

    if (
        "tax" in field_name_lower
        and subtotal_val is not None
        and total_val is not None
        and total_val >= subtotal_val
    ):
        target = total_val - subtotal_val
        best = min(parsed, key=lambda item: abs(item[1] - target))
        if current_numeric is None or abs(best[1] - target) + 0.01 < abs(current_numeric - target):
            return best[0]
        return current_value

    if (
        "tax" in field_name_lower
        and subtotal_val is not None
        and _is_implausibly_small_summary_value(current_numeric)
    ):
        plausible = [
            item
            for item in parsed
            if 0.0 < item[1] <= subtotal_val * 0.5
            and item[1] > max(current_numeric or 0.0, 0.0) + 0.01
        ]
        if plausible:
            return min(plausible, key=lambda item: item[1])[0]

    if (
        "subtotal" in field_name_lower
        and tax_val is not None
        and total_val is not None
        and total_val >= tax_val
    ):
        target = total_val - tax_val
        best = min(parsed, key=lambda item: abs(item[1] - target))
        if current_numeric is None or abs(best[1] - target) + 0.01 < abs(current_numeric - target):
            return best[0]
        return current_value

    return current_value


def refine_critical_fields(
    extracted_data: Optional[Dict[str, Any]],
    template: Optional[Dict[str, Any]],
    ocr_results_per_page: Sequence[Sequence[OCRToken]],
    page_dimensions: Sequence[Tuple[int, int]],
) -> Dict[str, Any]:
    if not extracted_data or not template:
        return extracted_data or {}

    refined = deepcopy(extracted_data)
    amount_candidates_by_field: Dict[str, List[str]] = {}
    all_rows: List[List[OCRToken]] = []
    for page in template.get("pages", []):
        page_index = page.get("page_index", 0)
        if page_index >= len(ocr_results_per_page) or page_index >= len(page_dimensions):
            continue
        ocr_tokens = ocr_results_per_page[page_index]
        all_rows.extend(_group_tokens_into_rows(ocr_tokens))
        page_w, page_h = page_dimensions[page_index]
        for field_name, field_config in page.get("fields", {}).items():
            if field_name not in refined:
                continue

            field_name_lower = field_name.lower()
            is_date_field = "date" in field_name_lower
            is_amount_field = any(
                token in field_name_lower for token in ("total", "subtotal", "tax", "amount")
            )
            is_invoice_number_field = "invoice" in field_name_lower and "number" in field_name_lower
            is_party_field = field_name in PARTY_LABEL_ALIASES
            if (
                not is_date_field
                and not is_amount_field
                and not is_invoice_number_field
                and not is_party_field
            ):
                continue

            region = field_config.get("region")
            label_text = field_config.get("label_text")
            region_text = (
                _join_tokens(_tokens_in_region(ocr_tokens, region, page_w, page_h))
                if region
                else ""
            )
            neighbor_text = _find_label_neighbor_text(ocr_tokens, label_text, page_w)

            if is_date_field:
                candidates = _extract_date_candidates(region_text)
                candidates.extend(
                    candidate
                    for candidate in _extract_date_candidates(neighbor_text)
                    if candidate not in candidates
                )
                refined[field_name] = _resolve_date_value(refined.get(field_name), candidates)
            elif is_amount_field:
                candidates = _extract_amount_candidates(region_text)
                candidates.extend(
                    candidate
                    for candidate in _extract_amount_candidates(neighbor_text)
                    if candidate not in candidates
                )
                amount_candidates_by_field[field_name] = candidates
                refined[field_name] = _resolve_amount_value(refined.get(field_name), candidates)
            elif is_invoice_number_field:
                region_candidates = _extract_invoice_number_candidates(region_text)
                neighbor_candidates = _extract_invoice_number_candidates(neighbor_text)
                refined[field_name] = _resolve_invoice_number_value(
                    refined.get(field_name),
                    region_candidates,
                    neighbor_candidates,
                )
            elif is_party_field and _looks_suspicious_party_value(refined.get(field_name)):
                candidate = _extract_party_from_labeled_rows(field_name, all_rows)
                if candidate:
                    refined[field_name] = candidate

    current_amounts = {
        "subtotal": next(
            (_to_number(value) for key, value in refined.items() if "subtotal" in key.lower()), None
        ),
        "tax": next(
            (_to_number(value) for key, value in refined.items() if "tax" in key.lower()), None
        ),
        "total": next(
            (
                _to_number(value)
                for key, value in refined.items()
                if ("total" in key.lower() or "amount" in key.lower())
                and "subtotal" not in key.lower()
                and "tax" not in key.lower()
            ),
            None,
        ),
    }
    for field_name, candidates in amount_candidates_by_field.items():
        refined[field_name] = _resolve_amount_in_context(
            field_name, refined.get(field_name), candidates, current_amounts
        )
        if "subtotal" in field_name.lower():
            current_amounts["subtotal"] = _to_number(refined.get(field_name))
        elif "tax" in field_name.lower():
            current_amounts["tax"] = _to_number(refined.get(field_name))
        elif "total" in field_name.lower() or "amount" in field_name.lower():
            current_amounts["total"] = _to_number(refined.get(field_name))

    subtotal_val = current_amounts.get("subtotal")
    tax_val = current_amounts.get("tax")
    total_val = current_amounts.get("total")
    if subtotal_val is not None and tax_val is not None:
        reconciled_total = round(subtotal_val + tax_val, 2)
        for field_name in list(refined.keys()):
            field_lower = field_name.lower()
            if (
                ("total" in field_lower or "amount" in field_lower)
                and "subtotal" not in field_lower
                and "tax" not in field_lower
            ):
                current_value = _to_number(refined.get(field_name))
                if current_value is None or current_value + 0.01 < subtotal_val:
                    refined[field_name] = f"{reconciled_total:.2f}"
        current_amounts["total"] = reconciled_total
    elif subtotal_val is not None and total_val is not None and total_val >= subtotal_val:
        reconciled_tax = round(total_val - subtotal_val, 2)
        for field_name in list(refined.keys()):
            if "tax" in field_name.lower():
                current_value = _to_number(refined.get(field_name))
                if current_value is None or abs(current_value - reconciled_tax) > 0.02:
                    refined[field_name] = f"{reconciled_tax:.2f}"
        current_amounts["tax"] = reconciled_tax

    for party_field in PARTY_LABEL_ALIASES:
        if party_field in refined and _looks_suspicious_party_value(refined.get(party_field)):
            candidate = _extract_party_from_labeled_rows(party_field, all_rows)
            if candidate:
                refined[party_field] = candidate

    return refined
