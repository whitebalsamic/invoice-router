from typing import Any, Dict

import cv2
import numpy as np

from ..models import ProcessingResult


def add_diagonal_pattern(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple,
    thickness: int = 1,
    spacing: int = 10,
):
    """Draw a diagonal pattern inside a bounding box for color-blind accessibility."""
    # Draw diagonals
    for d in range(-h, w, spacing):
        start_x = max(0, d)
        start_y = max(0, -d)
        end_x = min(w, d + h)
        end_y = min(h, h - (end_x - d))

        cv2.line(img, (x + start_x, y + start_y), (x + end_x, y + end_y), color, thickness)


def annotate_invoice(
    page_image: np.ndarray, page_index: int, result: ProcessingResult, template: Dict[str, Any]
) -> np.ndarray:
    """Draw bounding boxes with patterns for the extracted fields on the image."""
    img = page_image.copy()
    h, w = img.shape[:2]

    # We need to map from result to template regions
    # The template is stored inside the processing result's provenance or Fingerprint DB.
    # Since we only get ProcessingResult and the matched layout_template, we can draw the boxes directly.

    if not template or "pages" not in template:
        return img

    pages = template["pages"]
    page_info = next((p for p in pages if p.get("page_index", 0) == page_index), None)
    if not page_info:
        return img

    # Blue box for regular fields
    for field_name, fconfig in page_info.get("fields", {}).items():
        region = fconfig["region"]
        rx = int(region["x"] * w)
        ry = int(region["y"] * h)
        rw = int(region["width"] * w)
        rh = int(region["height"] * h)

        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
        add_diagonal_pattern(img, rx, ry, rw, rh, (255, 0, 0), 1, 8)

        # Add label
        cv2.putText(
            img, field_name, (rx, max(10, ry - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )

    # Green box for table region
    if page_info.get("table"):
        t_region = page_info["table"]["region"]
        rx = int(t_region["x"] * w)
        ry = int(t_region["y"] * h)
        rw = int(t_region["width"] * w)
        rh = int(t_region["height"] * h)

        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        # add_diagonal_pattern(img, rx, ry, rw, rh, (0, 255, 0), 1, 15)

        for col_name, cconfig in page_info["table"]["columns"].items():
            cx = int(cconfig["x_left"] * w)
            cw = int((cconfig["x_right"] - cconfig["x_left"]) * w)
            cv2.rectangle(img, (cx, ry), (cx + cw, ry + rh), (0, 200, 0), 1)
            cv2.putText(img, col_name, (cx, ry + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

    return img


def save_annotated_image(img: np.ndarray, output_path: str):
    """Save the annotated BGR image to file."""
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr_img)
