import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .discovery import (
    TABLE_HEADER_ALIASES,
    OCRToken,
    _is_summary_row_text,
    _join_tokens,
    _row_bounds,
    _token_to_number,
    normalize_ocr_text,
    score_label_candidate,
)

_NUMERIC_TOKEN_RE = re.compile(r"^[\$€£]?-?\d[\d.,]*$")
_PERCENT_TOKEN_RE = re.compile(r"^-?\d+(?:[.,]\d+)?%$")


def _looks_like_leading_item_id(row_text: str) -> bool:
    first = (row_text or "").strip().split()
    if not first:
        return False
    token = first[0]
    if re.fullmatch(r"[A-Za-z]\.", token):
        return True
    if re.fullmatch(r"\d{1,2}[.:]?", token):
        return True
    return False


def _merge_row_bbox(
    existing_bbox: Optional[Tuple[int, int, int, int]],
    row_x: int,
    row_top: int,
    row_right: int,
    row_bottom: int,
) -> Tuple[int, int, int, int]:
    if existing_bbox is None:
        return (row_x, row_top, row_right, row_bottom)
    min_x = min(existing_bbox[0], row_x)
    min_y = min(existing_bbox[1], row_top)
    max_x = max(existing_bbox[2], row_right)
    max_y = max(existing_bbox[3], row_bottom)
    return (min_x, min_y, max_x, max_y)


def _repair_merged_total_token(
    raw_value: str, quantity_text: Optional[str], unit_price_text: Optional[str]
) -> str:
    quantity_value = _token_to_number(quantity_text) if quantity_text else None
    unit_price_value = _token_to_number(unit_price_text) if unit_price_text else None
    numeric_value = _token_to_number(raw_value)
    if quantity_value is None or unit_price_value is None or numeric_value is None:
        return raw_value

    compact = re.sub(r"[^\d]", "", raw_value)
    if len(compact) < 4 or any(separator in raw_value for separator in (".", ",")):
        return raw_value

    expected_total = quantity_value * unit_price_value
    if expected_total <= 0 or numeric_value <= (expected_total * 1.5):
        return raw_value

    best_candidate = numeric_value
    best_error = abs(numeric_value - expected_total)
    for start in range(1, len(compact) - 1):
        candidate_digits = compact[start:]
        candidate_value = float(candidate_digits)
        candidate_error = abs(candidate_value - expected_total)
        if candidate_error < best_error:
            best_candidate = candidate_value
            best_error = candidate_error

    if best_candidate == numeric_value:
        return raw_value

    tolerance = max(25.0, expected_total * 0.15)
    if best_error > tolerance:
        return raw_value

    if best_candidate.is_integer():
        return str(int(best_candidate))
    return f"{best_candidate:.2f}"


def _extract_row_measurements(
    row: Sequence[OCRToken],
    description_tokens: Sequence[OCRToken],
    quantity_tokens: Sequence[OCRToken],
) -> Dict[str, Optional[str]]:
    quantity_text = _join_tokens(quantity_tokens)
    quantity_value = _token_to_number(quantity_text) if quantity_text else None
    value_tokens = [token for token in row if token not in description_tokens]

    monetary_tokens: List[str] = []
    percent_tokens: List[str] = []
    sorted_value_tokens = sorted(value_tokens, key=lambda item: item[1])
    idx = 0
    while idx < len(sorted_value_tokens):
        token = sorted_value_tokens[idx]
        raw = token[0].strip().replace(" ", "")
        if not raw:
            idx += 1
            continue
        if _PERCENT_TOKEN_RE.fullmatch(raw):
            percent_tokens.append(raw)
            idx += 1
            continue

        numeric = _token_to_number(raw) if _NUMERIC_TOKEN_RE.fullmatch(raw) else None
        if (
            numeric is not None
            and quantity_value is not None
            and quantity_tokens
            and token in quantity_tokens
            and abs(numeric - quantity_value) <= 0.001
        ):
            idx += 1
            continue

        if idx + 1 < len(sorted_value_tokens):
            next_token = sorted_value_tokens[idx + 1]
            next_raw = next_token[0].strip().replace(" ", "")
            next_numeric = (
                _token_to_number(next_raw) if _NUMERIC_TOKEN_RE.fullmatch(next_raw) else None
            )
            if (
                re.fullmatch(r"-?\d{1,3}", raw)
                and next_numeric is not None
                and next_numeric >= 100
                and not (
                    quantity_value is not None
                    and quantity_tokens
                    and next_token in quantity_tokens
                    and abs(next_numeric - quantity_value) <= 0.001
                )
            ):
                if next_raw[:1] in "$€£":
                    merged = f"{next_raw[0]}{raw}{next_raw[1:]}"
                else:
                    merged = f"{raw}{next_raw}"
                if _token_to_number(merged) is not None:
                    monetary_tokens.append(merged)
                    idx += 2
                    continue

        if numeric is not None:
            monetary_tokens.append(raw)
        idx += 1

    result: Dict[str, Optional[str]] = {
        "qty": quantity_text if quantity_text and re.search(r"\d", quantity_text) else None,
        "price": None,
        "amount": None,
        "gross amount": None,
        "tax rate": percent_tokens[-1] if percent_tokens else None,
    }
    if not monetary_tokens:
        return result

    if percent_tokens and len(monetary_tokens) >= 2:
        if len(monetary_tokens) >= 3:
            result["gross amount"] = monetary_tokens[-1]
            result["amount"] = monetary_tokens[-2]
            result["price"] = monetary_tokens[-3]
            return result

        first_value = _token_to_number(monetary_tokens[0])
        second_value = _token_to_number(monetary_tokens[1])
        if quantity_value not in (None, 0) and first_value is not None and second_value is not None:
            if quantity_value > 1 and abs((first_value * quantity_value) - second_value) <= max(
                0.05, second_value * 0.03
            ):
                result["price"] = monetary_tokens[0]
                result["amount"] = monetary_tokens[1]
                return result
            if abs(quantity_value - 1.0) <= 0.001:
                result["price"] = monetary_tokens[0]
                result["amount"] = monetary_tokens[0]
                result["gross amount"] = monetary_tokens[1]
                return result

        result["gross amount"] = monetary_tokens[-1]
        result["amount"] = monetary_tokens[-2]
        if quantity_value not in (None, 0):
            amount_value = _token_to_number(result["amount"])
            if amount_value is not None:
                result["price"] = f"{(amount_value / quantity_value):.2f}"
        return result

    if len(monetary_tokens) >= 2:
        result["price"] = monetary_tokens[0]
        repaired_total = _repair_merged_total_token(
            monetary_tokens[-1], result["qty"], result["price"]
        )
        result["amount"] = repaired_total
        return result

    result["amount"] = monetary_tokens[-1]
    return result


def _line_item_has_measurements(item: Dict[str, Any]) -> bool:
    return any(
        item.get(key) is not None for key in ("qty", "amount", "price", "gross amount", "tax rate")
    )


def _detect_table(
    rows: Sequence[Sequence[OCRToken]],
    page_w: int,
    page_h: int,
    heuristic_config: Optional[Any],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    if not getattr(heuristic_config, "enable_table_detection", False):
        return None, []

    min_table_header_score = getattr(heuristic_config, "min_table_header_score", 0.75)
    best_index = None
    best_score = 0.0
    best_matches: Dict[str, OCRToken] = {}

    for idx, row in enumerate(rows):
        row_text = normalize_ocr_text(_join_tokens(row))
        if not row_text:
            continue
        matches: Dict[str, OCRToken] = {}
        score = 0.0
        for canonical_name, aliases in TABLE_HEADER_ALIASES.items():
            for token in row:
                token_text = normalize_ocr_text(token[0])
                if not token_text:
                    continue
                if any(
                    score_label_candidate(token_text, alias, label_x=token[1], page_w=page_w) >= 0.8
                    for alias in aliases
                ):
                    matches[canonical_name] = token
                    score += 0.34
                    break
        if len(matches) >= 2 and score > best_score:
            best_score = score
            best_index = idx
            best_matches = matches

    if best_index is None or best_score < min_table_header_score:
        return _detect_body_row_fallback(rows, page_w, page_h)

    header_row = rows[best_index]
    header_top = min(token[2] for token in header_row)
    header_bottom = max(token[2] + token[4] for token in header_row)
    description_boundary = (
        best_matches["quantity"][1] - max(page_w * 0.015, 10)
        if "quantity" in best_matches
        else best_matches["amount"][1] - max(page_w * 0.02, 12)
        if "amount" in best_matches
        else page_w
    )
    quantity_left = (
        best_matches["quantity"][1] - max(page_w * 0.015, 10)
        if "quantity" in best_matches
        else description_boundary
    )
    quantity_boundary = (
        best_matches["amount"][1] - max(page_w * 0.02, 12) if "amount" in best_matches else page_w
    )

    line_items: List[Dict[str, Any]] = []
    row_boxes: List[Tuple[int, int, int, int]] = []
    previous_row_bottom = header_bottom
    previous_item_bbox: Optional[Tuple[int, int, int, int]] = None
    for row in rows[best_index + 1 :]:
        row_text = _join_tokens(row)
        row_norm = normalize_ocr_text(row_text)
        if not row_norm:
            continue
        if any(
            stop_word in row_norm
            for stop_word in ("subtotal", "total", "tax", "amount due", "balance due")
        ):
            break
        row_top = min(token[2] for token in row)
        row_bottom = max(token[2] + token[4] for token in row)
        if row_top - previous_row_bottom > max(page_h * 0.04, 40):
            break
        if len(row) < 2:
            continue
        min_x = min(token[1] for token in row)
        max_x = max(token[1] + token[3] for token in row)
        if row_bottom <= header_bottom:
            continue

        description_tokens = [
            token for token in row if (token[1] + token[3] / 2.0) < description_boundary
        ]
        quantity_tokens = []
        if "quantity" in best_matches:
            quantity_tokens = [
                token
                for token in row
                if quantity_left <= (token[1] + token[3] / 2.0) < quantity_boundary
            ]
        description = _join_tokens(description_tokens) or row_text
        measurements = _extract_row_measurements(row, description_tokens, quantity_tokens)
        quantity_text = measurements.get("qty")
        amount_text = measurements.get("amount")
        has_leading_item_id = _looks_like_leading_item_id(row_text)
        is_continuation_row = (
            line_items
            and not amount_text
            and not quantity_text
            and not has_leading_item_id
            and row_top - previous_row_bottom <= max(page_h * 0.025, 34)
        )
        if is_continuation_row:
            existing_description = line_items[-1].get("description", "")
            merged_description = f"{existing_description} {description}".strip()
            line_items[-1]["description"] = re.sub(r"\s+", " ", merged_description).strip()
            if previous_item_bbox is not None:
                min_x = min(previous_item_bbox[0], min_x)
                min_y = min(previous_item_bbox[1], row_top)
                max_x_box = max(previous_item_bbox[2], max_x)
                max_y_box = max(previous_item_bbox[3], row_bottom)
                previous_item_bbox = (min_x, min_y, max_x_box, max_y_box)
                row_boxes[-1] = previous_item_bbox
            previous_row_bottom = row_bottom
            continue

        if (
            line_items
            and not has_leading_item_id
            and _line_item_has_measurements(measurements)
            and not _line_item_has_measurements(line_items[-1])
            and row_top - previous_row_bottom <= max(page_h * 0.03, 42)
        ):
            existing_description = line_items[-1].get("description", "")
            merged_description = f"{existing_description} {description}".strip()
            line_items[-1]["description"] = re.sub(r"\s+", " ", merged_description).strip()
            for key, value in measurements.items():
                if value is not None:
                    line_items[-1][key] = value
            previous_item_bbox = _merge_row_bbox(
                previous_item_bbox, min_x, row_top, max_x, row_bottom
            )
            row_boxes[-1] = previous_item_bbox
            previous_row_bottom = row_bottom
            continue

        line_item: Dict[str, Any] = {"description": description}
        for key, value in measurements.items():
            if value is not None:
                line_item[key] = value

        if normalize_ocr_text(description) and any(ch.isalpha() for ch in description):
            line_items.append(line_item)
            previous_item_bbox = (min_x, row_top, max_x, row_bottom)
            row_boxes.append(previous_item_bbox)
            previous_row_bottom = row_bottom

    if not line_items:
        return None, []

    first_row_y = min(box[1] for box in row_boxes)
    last_row_y = max(box[3] for box in row_boxes)
    table_region = {
        "x": round(min(box[0] for box in row_boxes) / page_w, 6),
        "y": round(first_row_y / page_h, 6),
        "width": round(
            (max(box[2] for box in row_boxes) - min(box[0] for box in row_boxes)) / page_w, 6
        ),
        "height": round((last_row_y - first_row_y) / page_h, 6),
    }

    columns = {}
    if "description" in best_matches:
        description_token = best_matches["description"]
        columns["description"] = {
            "x_left": round(max(0.0, description_token[1] / page_w), 6),
            "x_right": round(
                1.0 if "amount" not in best_matches else best_matches["amount"][1] / page_w, 6
            ),
        }
    if "amount" in best_matches:
        amount_token = best_matches["amount"]
        columns["amount"] = {
            "x_left": round(amount_token[1] / page_w, 6),
            "x_right": 1.0,
        }

    return {
        "region": table_region,
        "columns": columns or {"description": {"x_left": 0.0, "x_right": 1.0}},
        "header_row_y": round(header_top / page_h, 6),
    }, line_items


def _detect_body_row_fallback(
    rows: Sequence[Sequence[OCRToken]],
    page_w: int,
    page_h: int,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_start_y = page_h
    for row in rows:
        row_text = _join_tokens(row)
        if _is_summary_row_text(row_text):
            summary_start_y = min(summary_start_y, _row_bounds(row)[1])

    line_items: List[Dict[str, Any]] = []
    row_boxes: List[Tuple[int, int, int, int]] = []
    current_item: Optional[Dict[str, Any]] = None
    current_bbox: Optional[Tuple[int, int, int, int]] = None
    previous_row_bottom = 0

    def flush_current() -> None:
        nonlocal current_item, current_bbox
        if current_item and current_bbox:
            description = re.sub(r"\s+", " ", current_item.get("description", "")).strip()
            if description and any(ch.isalpha() for ch in description):
                current_item["description"] = description
                line_items.append(current_item)
                x, y, w, h = current_bbox
                row_boxes.append((x, y, x + w, y + h))
        current_item = None
        current_bbox = None

    for row in rows:
        row_text = _join_tokens(row)
        row_norm = normalize_ocr_text(row_text)
        row_bbox = _row_bounds(row)
        row_x, row_y, row_w, row_h = row_bbox
        row_bottom = row_y + row_h

        if not row_norm:
            continue
        if row_y < page_h * 0.22 or row_y >= summary_start_y:
            continue
        if _is_summary_row_text(row_text) or any(stop in row_norm for stop in ("invoice", "date")):
            flush_current()
            continue

        left_tokens = [token for token in row if (token[1] + token[3] / 2.0) < page_w * 0.72]
        if not left_tokens:
            continue

        leading_id_index = next(
            (
                idx
                for idx, token in enumerate(left_tokens[:4])
                if re.fullmatch(r"\d{1,3}", token[0].strip())
            ),
            None,
        )
        has_leading_id = leading_id_index is not None
        qty_tokens = [
            token
            for token in row
            if page_w * 0.45 <= (token[1] + token[3] / 2.0) <= page_w * 0.62
            and re.fullmatch(r"\d+(?:\.\d+)?", token[0].strip())
        ]

        description_tokens = list(left_tokens)
        if has_leading_id and description_tokens:
            description_tokens = description_tokens[leading_id_index + 1 :]
        description_tokens = [
            token
            for token in description_tokens
            if token[0].strip() not in {"|", "'"} and not re.fullmatch(r"\d{1,3}", token[0].strip())
        ]
        description_text = _join_tokens(description_tokens).strip()

        starts_new_item = (
            has_leading_id
            and any(ch.isalpha() for ch in description_text)
            and "@" not in description_text
            and not _is_summary_row_text(description_text)
        )
        continuation = (
            current_item is not None
            and not starts_new_item
            and any(ch.isalpha() for ch in row_text)
            and row_y - previous_row_bottom <= max(42, int(page_h * 0.03))
        )

        if starts_new_item:
            flush_current()
            current_item = {"description": description_text}
            current_bbox = row_bbox
        elif continuation and current_item is not None:
            current_item["description"] = (
                f"{current_item.get('description', '')} {description_text}".strip()
            )
            if current_bbox is not None:
                min_x = min(current_bbox[0], row_x)
                min_y = min(current_bbox[1], row_y)
                max_x = max(current_bbox[0] + current_bbox[2], row_x + row_w)
                max_y = max(current_bbox[1] + current_bbox[3], row_bottom)
                current_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
        else:
            flush_current()
            continue

        if current_item is not None and qty_tokens and "qty" not in current_item:
            current_item["qty"] = qty_tokens[-1][0].strip()

        previous_row_bottom = row_bottom

    flush_current()

    if len(line_items) < 2 or not row_boxes:
        return None, []

    first_row_y = min(box[1] for box in row_boxes)
    last_row_y = max(box[3] for box in row_boxes)
    table_region = {
        "x": round(min(box[0] for box in row_boxes) / page_w, 6),
        "y": round(first_row_y / page_h, 6),
        "width": round(
            (max(box[2] for box in row_boxes) - min(box[0] for box in row_boxes)) / page_w, 6
        ),
        "height": round((last_row_y - first_row_y) / page_h, 6),
    }
    return {
        "region": table_region,
        "columns": {
            "description": {"x_left": 0.0, "x_right": 0.72},
            "amount": {"x_left": 0.72, "x_right": 1.0},
        },
        "header_row_y": None,
    }, line_items
