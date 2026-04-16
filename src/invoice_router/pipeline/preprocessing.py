from pathlib import Path
from typing import List

import cv2
import fitz  # pymupdf
import numpy as np
import pytesseract


def _resize_if_needed(image: np.ndarray, max_dim: int = 2048) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image

    scale = max_dim / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_image(file_path: str) -> List[np.ndarray]:
    """
    Load image or PDF file.
    Returns a list of RGB numpy arrays (one per page).
    PDFs are loaded at 2x zoom.
    Images exceeding 2048x2048 are resized proportionally.
    Max 3 pages per invoice.
    """
    path = Path(file_path)
    pages = []

    if path.suffix.lower() == ".pdf":
        with fitz.open(file_path) as doc:
            # Process at most 3 pages
            for i in range(min(len(doc), 3)):
                page = doc[i]
                # 2x zoom
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)

                # Convert to numpy array while allowing the pixmap to be released immediately.
                if pix.n - pix.alpha < 4:
                    img = (
                        np.frombuffer(pix.samples, dtype=np.uint8)
                        .reshape(pix.h, pix.w, pix.n)
                        .copy()
                    )
                    if pix.alpha:
                        img = img[:, :, :3]
                else:
                    rgb_pix = fitz.Pixmap(fitz.csRGB, pix)
                    img = (
                        np.frombuffer(rgb_pix.samples, dtype=np.uint8)
                        .reshape(rgb_pix.h, rgb_pix.w, 3)
                        .copy()
                    )
                    rgb_pix = None
                pix = None
                pages.append(_resize_if_needed(img))
    else:
        # Load directly as BGR, then convert to RGB
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Failed to load image: {file_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pages.append(_resize_if_needed(img))

    return pages


def deskew_and_normalize(image: np.ndarray) -> np.ndarray:
    """
    Combined deskew and orientation normalization.
    """
    # 1. Skew detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    skew_angle = 0.0
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45.0 <= angle <= 45.0:
                angles.append(angle)

        if len(angles) >= 5:
            skew_angle = np.median(angles)

    # 2. Deskew
    if abs(skew_angle) >= 0.5:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        image = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )

    return _choose_best_upright_orientation(image)


def _readability_score(image: np.ndarray) -> float:
    """
    Score OCR readability to distinguish upright pages from upside-down pages.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    max_dim = 1200
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    data = pytesseract.image_to_data(
        gray,
        config="--psm 6 --oem 3",
        output_type=pytesseract.Output.DICT,
    )
    score = 0.0
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        token = str(text or "").strip()
        if not token:
            continue
        try:
            confidence = float(conf)
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence < 0:
            continue

        alpha_count = sum(ch.isalpha() for ch in token)
        digit_count = sum(ch.isdigit() for ch in token)
        if alpha_count == 0 and digit_count == 0:
            continue

        token_score = max(confidence, 0.0) / 100.0
        if alpha_count >= 2:
            token_score += 0.5
        elif digit_count >= 2:
            token_score += 0.2
        if token.isupper() and alpha_count >= 3:
            token_score += 0.1
        score += token_score
    return score


def _choose_best_upright_orientation(image: np.ndarray) -> np.ndarray:
    upright_score = _readability_score(image)
    rotated = cv2.rotate(image, cv2.ROTATE_180)
    rotated_score = _readability_score(rotated)
    if rotated_score > upright_score * 1.1 and rotated_score - upright_score > 1.0:
        return rotated
    return image


def normalize_page(file_path: str) -> List[np.ndarray]:
    """
    Wrapper: load + deskew + normalize.
    """
    pages = load_image(file_path)
    return [deskew_and_normalize(page) for page in pages]
