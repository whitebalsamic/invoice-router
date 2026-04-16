import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .discovery import (
    ADDRESS_STOP_WORDS,
    BUSINESS_SUFFIXES,
    FIELD_SPECS,
    PARTY_ROLE_LABELS,
    SUMMARY_ROLE_LABELS,
    OCRToken,
    _extract_amount_strings,
    _is_mixed_content_row,
    _is_summary_row_text,
    _iter_row_phrases,
    _looks_like_noise_line,
    _row_block_class,
    _row_bounds,
    normalize_ocr_text,
    score_label_candidate,
)

COUNTRY_NAME_WORDS = {
    "canada",
    "mexico",
    "usa",
    "united states",
}

PARTY_BLOCK_BOUNDARY_WORDS = {
    "address",
    "addr",
    "attention",
    "attn",
    "bill",
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

PARTY_BLOCK_ADDRESS_WORDS = {
    "apt",
    "ave",
    "avenue",
    "blvd",
    "drive",
    "dr",
    "floor",
    "lane",
    "ln",
    "plaza",
    "rd",
    "road",
    "room",
    "route",
    "st",
    "street",
    "suite",
    "ste",
    "unit",
}

PARTY_STOP_WORD_WORDS = PARTY_BLOCK_BOUNDARY_WORDS | PARTY_BLOCK_ADDRESS_WORDS


def _trim_party_block_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip(" :,-|")
    if not text:
        return ""

    words = [word for word in text.split() if word]
    kept: List[str] = []
    for idx, word in enumerate(words):
        normalized = normalize_ocr_text(word)
        if not normalized:
            continue
        if normalized in COUNTRY_NAME_WORDS:
            break
        if normalized in PARTY_STOP_WORD_WORDS or normalized in ADDRESS_STOP_WORDS:
            break
        if re.search(r"https?://|www\.", word, flags=re.IGNORECASE) or "@" in word:
            break
        if re.fullmatch(r"\d{5}(?:-\d{4})?", normalized):
            break
        if re.fullmatch(r"\d{1,5}", normalized):
            lookahead = [normalize_ocr_text(next_word) for next_word in words[idx + 1 : idx + 4]]
            if any(
                next_word in PARTY_STOP_WORD_WORDS
                or next_word in ADDRESS_STOP_WORDS
                or next_word
                in {"country", "city", "state", "province", "postal", "postcode", "zip"}
                for next_word in lookahead
            ):
                break
        kept.append(word)

    return re.sub(r"\s+", " ", " ".join(kept)).strip(" :,-|")


def _join_party_tokens(tokens: Sequence[OCRToken]) -> str:
    return " ".join(
        token[0].strip() for token in sorted(tokens, key=lambda item: item[1]) if token[0].strip()
    ).strip()


def _party_line_score(text: str) -> float:
    normalized = normalize_ocr_text(text)
    if not normalized:
        return -1.0
    if any(marker in normalized for marker in ("tax id", "iban", "swift", "routing", "account no")):
        return -1.0
    if normalized in COUNTRY_NAME_WORDS:
        return -1.0
    if normalized in ADDRESS_STOP_WORDS or normalized.endswith(" to"):
        return -0.6
    if not any(ch.isalpha() for ch in normalized):
        return -0.5
    words = normalized.split()
    if len(words) == 1 and words[0] in BUSINESS_SUFFIXES:
        return -0.45
    if (
        len(words) <= 3
        and all(word.isalpha() and len(word) <= 3 for word in words)
        and text.upper() == text
        and not any(word in BUSINESS_SUFFIXES for word in words)
    ):
        return -0.5
    if len(words) > 8:
        return -0.2
    if _looks_like_noise_line(normalized):
        return -0.5
    if sum(ch.isdigit() for ch in text) >= 5:
        return -0.3
    score = 0.2
    if any(word in BUSINESS_SUFFIXES for word in words):
        score += 0.35
    if len(words) in (2, 3, 4):
        score += 0.15
    if text.upper() == text and len(words) <= 4:
        score += 0.08
    return score


def _clean_party_candidate(text: str, *, prefer_short_prefix: bool = False) -> str:
    if _is_mixed_content_row(text):
        return ""

    original_text = text
    text = re.sub(r"\b\S+@\S+\b", " ", text)
    text = re.sub(r"\b(?:e-?mail|email)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^(?:\s*(?:tax\s+invoice|commercial\s+invoice|invoice|invoices)\b[:\s-]*)+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    if "|" in text:
        if prefer_short_prefix and text.lstrip().startswith("|"):
            post_pipe = [word.strip(" :") for word in re.split(r"\|", text) if word.strip(" :")]
            if post_pipe:
                first_words = [word for word in post_pipe[0].split() if word]
                if first_words:
                    first_word = first_words[0].strip(" :")
                    if first_word[:1].isalpha() and not any(ch.isdigit() for ch in first_word):
                        return first_word
        segments = [segment.strip(" :") for segment in text.split("|") if segment.strip(" :")]
        if segments:
            if prefer_short_prefix and len(normalize_ocr_text(segments[0]).split()) <= 3:
                return segments[0]
            text = segments[0]
    text = _trim_party_block_text(text)
    text = re.split(
        r"\b(?:invoice|invoices|date|due|www\.|https?://)\b", text, maxsplit=1, flags=re.IGNORECASE
    )[0].strip(" :")
    text = re.split(r"\b\d{3,}\b", text, maxsplit=1)[0].strip(" :")
    if prefer_short_prefix:
        words = [word for word in text.split() if word]
        if "|" in original_text and words:
            first_word = words[0].strip(" :")
            if first_word[:1].isalpha() and not any(ch.isdigit() for ch in first_word):
                return first_word
        if len(words) > 1:
            leading_word = words[0]
            trailing_words = words[1:]
            trailing_norm = normalize_ocr_text(" ".join(trailing_words))
            trailing_has_business_suffix = any(
                word in BUSINESS_SUFFIXES for word in trailing_norm.split()
            )
            trailing_looks_like_company = sum(token.isupper() for token in trailing_words) >= 2
            trailing_all_caps = all(
                not token.isalpha() or token.upper() == token for token in trailing_words
            )
            if (
                leading_word[:1].isalpha()
                and leading_word[:1].isupper()
                and leading_word != leading_word.upper()
                and trailing_all_caps
                and trailing_looks_like_company
            ):
                return leading_word
            if (
                "|" in original_text
                and leading_word[:1].isalpha()
                and leading_word[:1].isupper()
                and (trailing_has_business_suffix or trailing_looks_like_company)
            ):
                return leading_word
    cleaned = text.strip(" :")
    cleaned_words = [word for word in cleaned.split() if word]
    for size in range(min(4, len(cleaned_words) // 2), 0, -1):
        phrase = cleaned_words[:size]
        if phrase and cleaned_words[size : size * 2] == phrase:
            cleaned = " ".join(phrase + cleaned_words[size * 2 :]).strip()
            break
    if normalize_ocr_text(cleaned) in COUNTRY_NAME_WORDS:
        return ""
    return cleaned


def _split_row_into_segments(row: Sequence[OCRToken]) -> List[List[OCRToken]]:
    if not row:
        return []

    sorted_row = sorted(row, key=lambda token: token[1])
    widths = [max(token[3], 1) for token in sorted_row]
    median_width = float(np.median(widths)) if widths else 1.0
    gap_threshold = max(54.0, median_width * 2.6)

    segments: List[List[OCRToken]] = [[sorted_row[0]]]
    previous = sorted_row[0]
    for token in sorted_row[1:]:
        gap = token[1] - (previous[1] + previous[3])
        if gap >= gap_threshold:
            segments.append([token])
        else:
            segments[-1].append(token)
        previous = token
    return segments


def _is_banner_like_text(text: str) -> bool:
    normalized = normalize_ocr_text(text)
    return normalized in {"invoice", "tax invoice", "commercial invoice"}


def _is_contactish_or_address_line(text: str) -> bool:
    normalized = normalize_ocr_text(text)
    if not normalized:
        return True
    if _looks_like_noise_line(text):
        return True
    if re.search(r"\b\d{3}[- ]?\d{3}[- ]?\d{4}\b", text):
        return True
    if re.search(r"\b[a-z]\d[a-z]\b", normalized):
        return True
    if sum(ch.isdigit() for ch in text) >= 5:
        return True
    return False


def _party_line_role_score(text: str, role: str) -> float:
    cleaned = _clean_party_candidate(
        text,
        prefer_short_prefix=role in {"bill_to_block", "ship_to_block"},
    )
    score = _party_line_score(cleaned)
    if not cleaned or _is_banner_like_text(cleaned):
        return -1.0
    if re.search(r"\d", cleaned):
        score -= 0.35
    if len(normalize_ocr_text(cleaned).split()) <= 2 and not re.search(r"\d", cleaned):
        score += 0.18
    if role in {"bill_to_block", "ship_to_block"} and "|" in text:
        score += 0.2
    if role == "vendor_block" and any(
        suffix in normalize_ocr_text(cleaned).split() for suffix in BUSINESS_SUFFIXES
    ):
        score += 0.15
    normalized_text = normalize_ocr_text(text)
    cleaned_norm = normalize_ocr_text(cleaned)
    trimmed_prefix = (
        bool(cleaned_norm)
        and normalized_text.startswith(cleaned_norm)
        and len(cleaned_norm.split()) <= 4
    )
    if re.search(r"\d", text) and any(
        token in normalized_text.split()
        for token in {
            "street",
            "st",
            "road",
            "rd",
            "drive",
            "dr",
            "suite",
            "ste",
            "box",
            "oval",
            "plains",
            "course",
            "neck",
        }
    ):
        score -= 0.15 if trimmed_prefix else 0.55
    if re.search(r",\s*[A-Z]{2}\b", text) or re.search(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b", text):
        score -= 0.15 if trimmed_prefix else 0.45
    if _is_contactish_or_address_line(cleaned):
        score -= 0.55
    return score


def _party_segment_score(segment: Sequence[OCRToken], role: str, page_w: int) -> Tuple[float, str]:
    segment_text = _join_party_tokens(segment)
    cleaned = _clean_party_candidate(
        segment_text,
        prefer_short_prefix=role in {"bill_to_block", "ship_to_block"},
    )
    if not cleaned:
        return -1.0, ""

    score = _party_line_role_score(segment_text, role)
    seg_x = min(token[1] for token in segment)
    seg_right = max(token[1] + token[3] for token in segment)
    left_ratio = seg_x / max(page_w, 1)
    right_ratio = seg_right / max(page_w, 1)
    if role in {"vendor_block", "remit_to_block"}:
        score += max(0.0, 0.16 - (left_ratio * 0.22))
    else:
        score += max(0.0, (right_ratio - 0.42) * 0.28)
        if left_ratio < 0.2:
            score -= 0.08
        normalized_words = normalize_ocr_text(cleaned).split()
        if len(normalized_words) == 1 and normalized_words[0] in BUSINESS_SUFFIXES:
            score -= 1.0
        if any(token[0].strip() == "|" for token in segment) and len(normalized_words) <= 3:
            score += 0.45
    return score, cleaned


def _normalize_party_words(value: str) -> List[str]:
    return [word for word in normalize_ocr_text(value).split() if word]


def _cleanup_party_name_overlap(
    value: str,
    other_value: Optional[str],
    *,
    prefer_suffix: bool = False,
) -> str:
    words = [word for word in value.split() if word]
    if not words:
        return value

    while len(words) > 1 and normalize_ocr_text(words[0]) in {
        "and",
        "corporation",
        "corp",
        *BUSINESS_SUFFIXES,
    }:
        words = words[1:]

    if other_value and normalize_ocr_text(other_value) != normalize_ocr_text(value):
        other_words = set(_normalize_party_words(other_value))
        while (
            len(words) > 2
            and normalize_ocr_text(words[0]) in other_words
            and any(token in {",", "and"} for token in words[1:])
        ):
            words = words[1:]
        while (
            len(words) > 2
            and normalize_ocr_text(words[-1]) in other_words
            and any(token in {",", "and"} for token in words[:-1])
        ):
            words = words[:-1]
        while len(words) > 1 and normalize_ocr_text(words[0]) in {
            "and",
            "corporation",
            "corp",
            *BUSINESS_SUFFIXES,
        }:
            words = words[1:]
        cleaned_words: List[str] = []
        for idx, word in enumerate(words):
            normalized = normalize_ocr_text(word)
            if (
                idx > 0
                and normalized in other_words
                and normalized in {*(BUSINESS_SUFFIXES), "group", "sons", "corporation", "corp"}
                and len(words) - len(cleaned_words) > 1
            ):
                continue
            cleaned_words.append(word)
        if cleaned_words:
            words = cleaned_words

    if prefer_suffix and len(words) >= 2:
        last_norm = normalize_ocr_text(words[-1])
        prev_norm = normalize_ocr_text(words[-2])
        if prev_norm in BUSINESS_SUFFIXES and last_norm not in BUSINESS_SUFFIXES:
            words = [words[-1], words[-2]]
        elif words[-1] == "-" and re.fullmatch(r"[A-Z]{1,4}", words[-2] or ""):
            words = [*words[:-2], "-", words[-2]]

    return " ".join(words).strip()


def _extract_vendor_name_from_row(row: Sequence[OCRToken], page_w: int) -> Optional[str]:
    if _is_mixed_content_row(_join_party_tokens(row)):
        return None

    candidate_tokens: List[OCRToken] = []
    for token in row:
        text = token[0].strip()
        normalized = normalize_ocr_text(text)
        if not text:
            continue
        if token[1] > page_w * 0.55:
            break
        if normalized in {"invoice", "date", "due"} or "#" in text:
            break
        if re.search(r"\d", text):
            break
        if _looks_like_noise_line(text):
            break
        candidate_tokens.append(token)

    candidate = _join_party_tokens(candidate_tokens).strip()
    words = normalize_ocr_text(candidate).split()
    if len(words) >= 2 and any(word in BUSINESS_SUFFIXES for word in words):
        return candidate
    return None


def _find_party_label_matches(
    rows: Sequence[Sequence[OCRToken]],
    page_w: int,
    page_h: int,
) -> Dict[str, Tuple[Tuple[int, int, int, int], float]]:
    best_label_matches: Dict[str, Tuple[Tuple[int, int, int, int], float]] = {}
    for row in rows[:4]:
        for phrase, phrase_bbox in _iter_row_phrases(row, max_tokens=3):
            phrase_text = phrase.strip()
            if not phrase_text:
                continue
            for output_name in ("sellerName", "customerName"):
                spec = FIELD_SPECS[output_name]
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
    return best_label_matches


def _extract_labeled_parties_from_block(
    block_rows: Sequence[Sequence[OCRToken]],
    page_w: int,
    page_h: int,
) -> Dict[str, Tuple[str, Tuple[int, int, int, int], float]]:
    label_matches = _find_party_label_matches(block_rows, page_w, page_h)
    extracted: Dict[str, Tuple[str, Tuple[int, int, int, int], float]] = {}
    for field_name, (label_bbox, label_score) in label_matches.items():
        party_name = _extract_party_name_from_tokens(
            field_name, label_bbox, block_rows, page_w, page_h
        )
        if not party_name:
            continue
        cleaned = _clean_party_candidate(
            party_name,
            prefer_short_prefix=(field_name == "customerName"),
        )
        if not cleaned:
            continue
        extracted[field_name] = (cleaned, label_bbox, label_score)
    return extracted


def _select_party_name_from_block(
    block_rows: Sequence[Sequence[OCRToken]], role: str
) -> Optional[str]:
    if not block_rows:
        return None

    page_w = max(token[1] + token[3] for row in block_rows for token in row)
    if role == "vendor_block":
        for row in block_rows[:2]:
            vendor_name = _extract_vendor_name_from_row(row, page_w)
            if vendor_name:
                return vendor_name

    best_name = None
    best_score = -1.0
    for row in block_rows[:4]:
        for segment in _split_row_into_segments(row):
            score, candidate = _party_segment_score(segment, role, page_w)
            if candidate and score > best_score:
                best_name = candidate
                best_score = score
    return best_name if best_score >= 0.12 else None


def _block_role_scores(
    block: Dict[str, object],
    page_w: int,
    page_h: int,
) -> Dict[str, float]:
    rows = block["rows"]
    bbox = block["bbox"]
    x, y, w, h = bbox
    x_ratio = x / max(page_w, 1)
    y_ratio = y / max(page_h, 1)
    text = " ".join(_join_party_tokens(row) for row in rows)
    normalized = normalize_ocr_text(text)
    amount_like_count = len(_extract_amount_strings(text))
    itemish_row_count = sum(
        1 for row in rows if any(re.fullmatch(r"\d{1,3}", token[0].strip()) for token in row[:2])
    )
    row_classes = [_row_block_class(row) for row in rows]
    partyish_row_count = sum(1 for value in row_classes if value == "party")
    tableish_row_count = sum(1 for value in row_classes if value == "table")
    summaryish_row_count = sum(1 for value in row_classes if value == "summary")
    metadataish_row_count = sum(1 for value in row_classes if value == "metadata")
    strong_class_count = len({value for value in row_classes if value != "other"})
    scores = {
        "vendor_block": 0.0,
        "bill_to_block": 0.0,
        "ship_to_block": 0.0,
        "remit_to_block": 0.0,
        "summary_block": 0.0,
    }

    if any(
        label in normalized
        for label in SUMMARY_ROLE_LABELS["subtotal"]
        | SUMMARY_ROLE_LABELS["tax"]
        | SUMMARY_ROLE_LABELS["amount_due"]
        | SUMMARY_ROLE_LABELS["invoice_total"]
        | SUMMARY_ROLE_LABELS["balance_due"]
    ):
        scores["summary_block"] += 1.2
    if _is_summary_row_text(text):
        scores["summary_block"] += 0.5

    for role, labels in PARTY_ROLE_LABELS.items():
        if any(label in normalized for label in labels):
            scores[role] += 1.1

    if y_ratio < 0.25:
        scores["vendor_block"] += 0.45
    if x_ratio < 0.55 and y_ratio < 0.35:
        scores["vendor_block"] += 0.25
    if y_ratio < 0.25 and any(
        suffix in normalize_ocr_text(_join_party_tokens(row)).split()
        for row in rows[:2]
        for suffix in BUSINESS_SUFFIXES
    ):
        scores["vendor_block"] += 0.6
    if y_ratio > 0.75:
        scores["bill_to_block"] += 0.5
        scores["ship_to_block"] += 0.2
    if x_ratio < 0.75 and 0.1 <= y_ratio <= 0.8:
        scores["bill_to_block"] += 0.15
    if x_ratio > 0.5 and y_ratio > 0.45:
        scores["summary_block"] += 0.35

    if any(
        _party_line_role_score(_join_party_tokens(row), "vendor_block") >= 0.25 for row in rows[:2]
    ):
        scores["vendor_block"] += 0.3
    if any(
        _party_line_role_score(_join_party_tokens(row), "bill_to_block") >= 0.2 for row in rows[:2]
    ):
        scores["bill_to_block"] += 0.25

    if amount_like_count >= 2 and 0.25 <= y_ratio <= 0.8:
        scores["vendor_block"] -= 0.9
        scores["bill_to_block"] -= 0.7
    if itemish_row_count >= 2:
        scores["vendor_block"] -= 0.75
        scores["bill_to_block"] -= 0.6

    if partyish_row_count and tableish_row_count:
        scores["vendor_block"] -= 0.95
        scores["bill_to_block"] -= 0.85
        scores["summary_block"] -= 0.45
    if summaryish_row_count and tableish_row_count:
        scores["vendor_block"] -= 0.85
        scores["bill_to_block"] -= 0.75
        scores["summary_block"] -= 0.55
    if summaryish_row_count and partyish_row_count:
        scores["vendor_block"] -= 0.65
        scores["bill_to_block"] -= 0.65
        scores["summary_block"] -= 0.35
    if metadataish_row_count >= 2 and tableish_row_count:
        scores["vendor_block"] -= 0.45
        scores["bill_to_block"] -= 0.35
    if strong_class_count >= 3 and len(rows) >= 5:
        scores["vendor_block"] -= 0.65
        scores["bill_to_block"] -= 0.55
        scores["summary_block"] -= 0.4
    if h / max(page_h, 1) > 0.32 and strong_class_count >= 2:
        scores["vendor_block"] -= 0.35
        scores["bill_to_block"] -= 0.25
        scores["summary_block"] -= 0.25

    if any(_is_banner_like_text(_join_party_tokens(row)) for row in rows):
        scores["vendor_block"] -= 0.6
        scores["bill_to_block"] -= 0.6

    if any(_is_contactish_or_address_line(_join_party_tokens(row)) for row in rows[:1]):
        scores["vendor_block"] -= 0.15
        scores["bill_to_block"] -= 0.15

    if w / max(page_w, 1) > 0.7 and y_ratio > 0.85:
        scores["bill_to_block"] += 0.3

    return scores


def _extract_party_name_from_tokens(
    field_name: str,
    label_bbox: Tuple[int, int, int, int],
    rows: Sequence[Sequence[OCRToken]],
    page_w: int,
    page_h: int,
) -> Optional[str]:
    x, y, w, h = label_bbox
    label_bottom = y + h
    label_center_y = y + (h / 2.0)
    left_bound = 0.0
    right_bound = float(page_w)

    for row in rows:
        row_center_y = sum(token[2] + token[4] / 2.0 for token in row) / len(row)
        if abs(row_center_y - label_center_y) > max(h * 0.9, 16):
            continue
        best_label_matches = _find_party_label_matches([row], page_w, page_h)
        if not best_label_matches:
            break
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
                left_bound = max(0.0, ((prev_center + current_center) / 2.0) - margin)
            if current_index + 1 < len(label_matches):
                next_bbox = label_matches[current_index + 1][1]
                next_center = next_bbox[0] + (next_bbox[2] / 2.0)
                right_bound = min(float(page_w), ((current_center + next_center) / 2.0) + margin)
        break

    block_rows: List[Sequence[OCRToken]] = []
    for row in rows:
        sliced_row = [
            token for token in row if left_bound <= (token[1] + token[3] / 2.0) <= right_bound
        ]
        if not sliced_row:
            continue
        row_left, row_top, _, row_height = _row_bounds(sliced_row)
        row_bottom = row_top + row_height
        if row_bottom <= label_bottom:
            continue
        if row_top > label_bottom + max(h * 6.0, page_h * 0.18):
            break
        if row_left > x + max(w * 1.75, page_w * 0.2):
            continue
        row_text = _join_party_tokens(sliced_row)
        row_norm = normalize_ocr_text(row_text)
        if row_norm in ADDRESS_STOP_WORDS:
            break
        if any(
            marker in row_norm for marker in ("tax id", "iban", "swift", "routing", "account no")
        ):
            break
        if _is_summary_row_text(row_text):
            break
        block_rows.append(sliced_row)

    best_name = None
    best_score = -1.0
    role = "vendor_block" if field_name == "sellerName" else "bill_to_block"
    for row_index, row in enumerate(block_rows[:4]):
        row_best_name = None
        row_best_score = -1.0
        for segment in _split_row_into_segments(row):
            score, candidate = _party_segment_score(segment, role, page_w)
            if field_name == "sellerName" and min(token[2] for token in segment) < page_h * 0.2:
                score += 0.1
            if candidate and score > row_best_score:
                row_best_name = candidate
                row_best_score = score
        if row_best_name and row_best_score > best_score:
            best_name = row_best_name
            best_score = row_best_score
            if row_index == 0 and row_best_score >= 0.25:
                return best_name
            continue

        row_text = _join_party_tokens(row)
        score = _party_line_role_score(row_text, role)
        if field_name == "sellerName" and min(token[2] for token in row) < page_h * 0.2:
            score += 0.1
        if score > best_score:
            best_name = _clean_party_candidate(
                row_text, prefer_short_prefix=(field_name == "customerName")
            )
            best_score = score
            if row_index == 0 and best_name and best_score >= 0.25:
                return best_name
    return best_name if best_score >= 0.1 else None


def _fallback_seller_name(
    rows: Sequence[Sequence[OCRToken]],
    page_w: int,
    page_h: int,
) -> Optional[Tuple[str, Tuple[int, int, int, int]]]:
    candidates: List[Tuple[float, str, Tuple[int, int, int, int]]] = []
    for row in rows:
        bbox = _row_bounds(row)
        x, y, w, _ = bbox
        if y > page_h * 0.22:
            continue
        if x > page_w * 0.55:
            continue
        row_text = _join_party_tokens(row)
        score = _party_line_score(row_text)
        score += max(0.0, 0.12 - (y / max(page_h, 1)) * 0.3)
        if x < page_w * 0.35:
            score += 0.05
        if score >= 0.2:
            candidates.append((score, _clean_party_candidate(row_text), bbox))
    if not candidates:
        return None
    best_score, best_text, best_bbox = max(candidates, key=lambda item: item[0])
    return (best_text, best_bbox) if best_score >= 0.28 else None


def _fallback_customer_name(
    rows: Sequence[Sequence[OCRToken]],
    page_w: int,
    page_h: int,
) -> Optional[Tuple[str, Tuple[int, int, int, int]]]:
    candidates: List[Tuple[float, str, Tuple[int, int, int, int]]] = []
    for row in rows:
        bbox = _row_bounds(row)
        x, y, _, _ = bbox
        if y < page_h * 0.58 or x > page_w * 0.6:
            continue
        row_text = _join_party_tokens(row)
        candidate_text = _clean_party_candidate(row_text, prefer_short_prefix=True)
        score = _party_line_score(candidate_text)
        if len(normalize_ocr_text(candidate_text).split()) <= 3:
            score += 0.18
        if "|" in row_text:
            score += 0.12
        score += max(0.0, 0.16 - (y / max(page_h, 1)) * 0.18)
        if score >= 0.22:
            candidates.append((score, candidate_text, bbox))
    if not candidates:
        return None
    best_score, best_text, best_bbox = max(candidates, key=lambda item: item[0])
    return (best_text, best_bbox) if best_score >= 0.3 else None


def _reconcile_party_names(extracted_values: Dict[str, str]) -> None:
    seller_key = next((key for key in extracted_values if "seller" in key.lower()), None)
    customer_key = next((key for key in extracted_values if "customer" in key.lower()), None)
    seller_value = (
        str(extracted_values[seller_key])
        if seller_key and extracted_values.get(seller_key)
        else None
    )
    customer_value = (
        str(extracted_values[customer_key])
        if customer_key and extracted_values.get(customer_key)
        else None
    )

    if seller_key and seller_value:
        extracted_values[seller_key] = _cleanup_party_name_overlap(seller_value, customer_value)
    if customer_key and customer_value:
        extracted_values[customer_key] = _cleanup_party_name_overlap(
            customer_value, seller_value, prefer_suffix=True
        )
