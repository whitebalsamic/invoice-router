from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytesseract

from ..domain.invoices.family_profiles import (
    resolve_table_detection_config,
    resolve_table_ocr_engine,
)
from ..domain.templates.fingerprinting import run_full_page_ocr
from ..models import BoundingBox


def get_tesseract_config(field_type: str) -> str:
    """Return Tesseract configuration tailored for the field type."""
    if field_type in ["string", "date"]:
        return "--psm 7 --oem 3"
    elif field_type == "currency":
        return "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.,$€£-"
    elif field_type == "qty":
        return "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.,"
    elif field_type == "address":
        return "--psm 6 --oem 3"
    return "--psm 7 --oem 3"


def expand_region(
    region: Dict[str, float], buffer_px: int, page_w: int, page_h: int
) -> Tuple[int, int, int, int]:
    """Convert normalized region to pixels and apply buffer."""
    x = max(0, int(region["x"] * page_w) - buffer_px)
    y = max(0, int(region["y"] * page_h) - buffer_px)
    x2 = min(page_w, int((region["x"] + region["width"]) * page_w) + buffer_px)
    y2 = min(page_h, int((region["y"] + region["height"]) * page_h) + buffer_px)
    return x, y, x2 - x, y2 - y


def extract_field(
    page_img: np.ndarray, region: Dict[str, float], field_type: str, buffer_px: int
) -> str:
    """Extract a single field using Tesseract."""
    h, w = page_img.shape[:2]
    rx, ry, rw, rh = expand_region(region, buffer_px, w, h)

    if rw <= 0 or rh <= 0:
        return ""

    crop = page_img[ry : ry + rh, rx : rx + rw]
    config = get_tesseract_config(field_type)
    text = pytesseract.image_to_string(crop, config=config)
    return text.strip()


def _group_tokens_into_rows(
    tokens: List[Tuple[str, int, int, int, int]],
    row_gap_multiplier: float,
) -> List[List[Tuple[str, int, int, int, int]]]:
    """Group OCR tokens into horizontal rows using token vertical centers."""
    if not tokens:
        return []

    sorted_tokens = sorted(tokens, key=lambda item: (item[2] + item[4] / 2.0, item[1]))
    heights = [max(h, 1) for _, _, _, _, h in sorted_tokens]
    median_height = float(np.median(heights)) if heights else 1.0
    row_gap = max(4.0, median_height * row_gap_multiplier)

    rows: List[List[Tuple[str, int, int, int, int]]] = []
    current_row = [sorted_tokens[0]]
    current_center = sorted_tokens[0][2] + sorted_tokens[0][4] / 2.0

    for token in sorted_tokens[1:]:
        _, _, y, _, h = token
        center_y = y + h / 2.0
        if abs(center_y - current_center) <= row_gap:
            current_row.append(token)
            centers = [t[2] + t[4] / 2.0 for t in current_row]
            current_center = sum(centers) / len(centers)
        else:
            rows.append(sorted(current_row, key=lambda item: item[1]))
            current_row = [token]
            current_center = center_y

    rows.append(sorted(current_row, key=lambda item: item[1]))
    return rows


def _row_boxes_from_table_ocr(
    page_img: np.ndarray,
    table_region: Dict[str, float],
    ocr_engine: str,
    table_detection_config: Optional[Any],
    buffer_px: int,
) -> List[BoundingBox]:
    """Detect row boxes inside the table region using the configured OCR engine."""
    page_h, page_w = page_img.shape[:2]
    rx, ry, rw, rh = expand_region(table_region, buffer_px, page_w, page_h)
    if rw <= 0 or rh <= 0:
        return []

    crop = page_img[ry : ry + rh, rx : rx + rw]
    tokens = run_full_page_ocr(crop, ocr_engine)
    if not tokens:
        return []

    row_gap_multiplier = getattr(table_detection_config, "row_gap_multiplier", 1.5)
    min_line_span_fraction = getattr(table_detection_config, "min_line_span_fraction", 0.40)
    rows = _group_tokens_into_rows(tokens, row_gap_multiplier)

    row_boxes: List[BoundingBox] = []
    min_span_px = rw * min_line_span_fraction
    for row in rows:
        min_x = min(x for _, x, _, _, _ in row)
        max_x = max(x + w for _, x, _, w, _ in row)
        span = max_x - min_x
        if span < min_span_px and len(row) < 2:
            continue

        min_y = min(y for _, _, y, _, _ in row)
        max_y = max(y + h for _, _, y, _, h in row)
        abs_x = rx + min_x
        abs_y = ry + min_y
        abs_w = max_x - min_x
        abs_h = max_y - min_y
        if abs_w <= 0 or abs_h <= 0:
            continue

        row_boxes.append(
            BoundingBox(
                x=abs_x / page_w,
                y=abs_y / page_h,
                width=abs_w / page_w,
                height=abs_h / page_h,
            )
        )

    return row_boxes


def extract_table(
    page_img: np.ndarray,
    table_info: Dict[str, Any],
    buffer_px: int,
    ocr_engine: str = "paddle",
    table_detection_config: Optional[Any] = None,
    family_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Extract table rows and columns using the configured table OCR engine."""
    table_detection_config = resolve_table_detection_config(table_detection_config, family_profile)
    rows = _row_boxes_from_table_ocr(
        page_img,
        table_info["region"],
        ocr_engine,
        table_detection_config,
        buffer_px,
    )

    results = []
    for row_bb in rows:
        row_dict = {}
        for col_name, col_config in table_info["columns"].items():
            col_x = col_config["x_left"]
            col_w = col_config["x_right"] - col_x

            cell_region = {"x": col_x, "y": row_bb.y, "width": col_w, "height": row_bb.height}

            ctype = "string"
            col_lower = col_name.lower()
            if "qty" in col_lower or "quantity" in col_lower:
                ctype = "qty"
            elif "price" in col_lower or "amount" in col_lower or "total" in col_lower:
                ctype = "currency"

            val = extract_field(page_img, cell_region, ctype, buffer_px)
            row_dict[col_name] = val
        results.append(row_dict)

    return results


def extract_with_ocr(
    pages: List[np.ndarray],
    template: Dict[str, Any],
    buffer_px: int,
    config: Optional[Any] = None,
    family_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Perform full extraction given a layout template using OCR-only extraction."""
    extracted = {}
    line_items = []
    family_profile = family_profile or {}
    ocr_profile = family_profile.get("ocr") or {}
    configured_table_engine = getattr(getattr(config, "ocr", None), "table_engine", None)
    table_engine = resolve_table_ocr_engine(configured_table_engine, family_profile)
    table_detection_config = getattr(config, "table_detection", None)
    region_buffer_multiplier = float(ocr_profile.get("region_buffer_multiplier") or 1.0)
    field_type_overrides = ocr_profile.get("field_type_overrides") or {}
    field_buffer_multipliers = ocr_profile.get("field_buffer_multipliers") or {}

    for page_info in template.get("pages", []):
        page_idx = page_info.get("page_index", 0)
        if page_idx >= len(pages):
            continue

        page_img = pages[page_idx]

        for fname, fconfig in page_info.get("fields", {}).items():
            field_type = field_type_overrides.get(fname) or fconfig.get("field_type", "string")
            field_buffer_px = max(
                1,
                int(
                    round(
                        buffer_px
                        * region_buffer_multiplier
                        * float(field_buffer_multipliers.get(field_type, 1.0))
                    )
                ),
            )
            val = extract_field(page_img, fconfig["region"], field_type, field_buffer_px)
            extracted[fname] = val

        if page_info.get("table"):
            table_data = extract_table(
                page_img,
                page_info["table"],
                buffer_px,
                ocr_engine=table_engine,
                table_detection_config=table_detection_config,
                family_profile=family_profile,
            )
            line_items.extend(table_data)

    if line_items:
        extracted["line_items"] = line_items

    return extracted
