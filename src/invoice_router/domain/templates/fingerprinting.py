import hashlib
import importlib.util
import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import imagehash
import numpy as np
from PIL import Image, ImageDraw

from ...models import PageFingerprint, PageRole

logger = logging.getLogger(__name__)

_ANCHOR_KEYWORDS = {
    "invoice_number": ("invoice", "inv", "number", "no", "ref"),
    "invoice_date": ("date", "issued"),
    "provider": ("seller", "vendor", "supplier", "clinic", "hospital", "company"),
    "customer": ("customer", "buyer", "bill", "ship", "client", "owner"),
    "summary": ("subtotal", "total", "tax", "vat", "amount", "balance", "due"),
    "footer": ("terms", "payment", "bank", "iban", "swift", "remit"),
}


class PaddleOcrUnavailableError(RuntimeError):
    pass


def _normalize_anchor_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _token_region(y: int, h: int, page_h: int) -> str:
    norm_cy = (y + (h / 2)) / max(page_h, 1)
    if norm_cy < 0.2:
        return "header"
    if norm_cy > 0.8:
        return "footer"
    if norm_cy > 0.6:
        return "summary_band"
    return "body"


def classify_page_role(
    page_index: int, ocr_results: List[Tuple[str, int, int, int, int]]
) -> PageRole:
    """
    Assign a lightweight structural role for family-level routing.
    """
    tokens = [_normalize_anchor_token(text) for text, *_ in ocr_results]
    joined = " ".join(token for token in tokens if token)
    if any(
        keyword in joined
        for keyword in ("subtotal", "total", "amount due", "balance due", "vat", "tax")
    ):
        return PageRole.summary_page
    if page_index == 0:
        return PageRole.header_page
    return PageRole.line_item_page


def extract_page_anchor_signature(
    page_index: int,
    ocr_results: List[Tuple[str, int, int, int, int]],
    page_w: int,
    page_h: int,
) -> Dict[str, Any]:
    keyword_hits: Dict[str, List[str]] = {name: [] for name in _ANCHOR_KEYWORDS}
    region_tokens: Dict[str, List[str]] = {
        "header": [],
        "body": [],
        "summary_band": [],
        "footer": [],
    }

    for text, _x, y, _w, h in ocr_results:
        normalized = _normalize_anchor_token(text)
        if not normalized:
            continue
        region_tokens[_token_region(y, h, page_h)].append(normalized)
        for label, keywords in _ANCHOR_KEYWORDS.items():
            if any(keyword in normalized.split() or keyword in normalized for keyword in keywords):
                keyword_hits[label].append(normalized)

    role = classify_page_role(page_index, ocr_results)
    summary_labels = sorted({token for token in keyword_hits["summary"]})[:10]
    header_tokens = sorted({token for token in region_tokens["header"] if not token.isdigit()})[:15]
    footer_tokens = sorted({token for token in region_tokens["footer"] if not token.isdigit()})[:10]
    return {
        "page_index": page_index,
        "page_role": role.value,
        "page_size": {"width": page_w, "height": page_h},
        "header_tokens": header_tokens,
        "summary_labels": summary_labels,
        "footer_tokens": footer_tokens,
        "keyword_hits": {
            name: sorted(set(values))[:10] for name, values in keyword_hits.items() if values
        },
    }


def extract_document_anchor_summary(
    ocr_results_per_page: List[List[Tuple[str, int, int, int, int]]],
    page_dimensions: List[Tuple[int, int]],
) -> Dict[str, Any]:
    page_signatures: List[Dict[str, Any]] = []
    aggregate_keywords: Dict[str, List[str]] = {}
    page_roles: List[str] = []

    for page_index, ocr_results in enumerate(ocr_results_per_page):
        page_w, page_h = page_dimensions[page_index]
        signature = extract_page_anchor_signature(page_index, ocr_results, page_w, page_h)
        page_signatures.append(signature)
        page_roles.append(signature["page_role"])
        for key, values in signature.get("keyword_hits", {}).items():
            aggregate_keywords.setdefault(key, [])
            aggregate_keywords[key].extend(values)

    return {
        "page_count": len(page_signatures),
        "page_roles": page_roles,
        "pages": page_signatures,
        "aggregate_keywords": {
            key: sorted(set(values))[:20] for key, values in aggregate_keywords.items()
        },
    }


@lru_cache(maxsize=1)
def get_paddle_ocr_status() -> Tuple[bool, Optional[str]]:
    if importlib.util.find_spec("paddleocr") is None:
        return False, "paddleocr package is not installed"
    if importlib.util.find_spec("paddle") is None:
        return False, "paddle runtime package is not installed"

    try:
        from paddleocr import PaddleOCR  # noqa: F401
    except Exception as exc:
        return False, f"paddleocr import failed: {exc}"
    return True, None


@lru_cache(maxsize=1)
def _init_paddle_ocr():
    available, reason = get_paddle_ocr_status()
    if not available:
        raise PaddleOcrUnavailableError(reason or "PaddleOCR is unavailable")

    from paddleocr import PaddleOCR

    try:
        return PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    except Exception as exc:
        if "show_log" not in str(exc):
            raise
        return PaddleOCR(use_angle_cls=False, lang="en")


def run_full_page_ocr(
    image: np.ndarray, engine: str = "tesseract"
) -> List[Tuple[str, int, int, int, int]]:
    """
    Run Tesseract or PaddleOCR on full page.
    Returns: [(text, x, y, w, h), ...]
    """
    results = []
    if engine == "tesseract":
        import pytesseract

        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        n_boxes = len(data["text"])
        for i in range(n_boxes):
            if int(data["conf"][i]) > -1:
                text = data["text"][i].strip()
                if text:
                    x = data["left"][i]
                    y = data["top"][i]
                    w = data["width"][i]
                    h = data["height"][i]
                    results.append((text, x, y, w, h))
    elif engine == "paddle":
        try:
            ocr = _init_paddle_ocr()
            try:
                res = ocr.ocr(image, cls=False)
            except TypeError as exc:
                if "unexpected keyword argument 'cls'" not in str(exc):
                    raise
                # Newer PaddleOCR releases no longer accept `cls` here.
                res = ocr.ocr(image)
            if res and isinstance(res, list) and isinstance(res[0], dict):
                page = res[0]
                polys = page.get("dt_polys") or []
                texts = page.get("rec_texts") or []
                for box, text in zip(polys, texts):
                    if hasattr(box, "tolist"):
                        box = box.tolist()
                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]
                    x = int(min(x_coords))
                    y = int(min(y_coords))
                    w = int(max(x_coords) - min(x_coords))
                    h = int(max(y_coords) - min(y_coords))
                    if str(text).strip():
                        results.append((str(text).strip(), x, y, w, h))
            elif res and res[0]:
                for line in res[0]:
                    box = line[0]
                    text = line[1][0]

                    # box is a list of 4 points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    x_coords = [p[0] for p in box]
                    y_coords = [p[1] for p in box]
                    x = int(min(x_coords))
                    y = int(min(y_coords))
                    w = int(max(x_coords) - min(x_coords))
                    h = int(max(y_coords) - min(y_coords))
                    if text.strip():
                        results.append((text.strip(), x, y, w, h))
        except (ModuleNotFoundError, PaddleOcrUnavailableError) as exc:
            raise PaddleOcrUnavailableError(f"PaddleOCR unavailable: {exc}") from exc
    else:
        raise ValueError(f"Unknown OCR engine: {engine}")

    return results


def compute_visual_hash(
    image: np.ndarray, ocr_results: List[Tuple[str, int, int, int, int]]
) -> Tuple[int, str]:
    """
    Mask all OCR text bounding boxes (white rectangles).
    Compute 64-bit pHash of masked image.
    Returns: (int_hash, hex_hash)
    """
    masked_img = Image.fromarray(image).copy()
    draw = ImageDraw.Draw(masked_img)
    for _, x, y, w, h in ocr_results:
        draw.rectangle((x, y, x + w, y + h), fill=(255, 255, 255))

    phash = imagehash.phash(masked_img, hash_size=8)

    # Convert imagehash object to integer and hex string
    hex_str = str(phash)
    # The hex_str is 16 chars (64 bits). Convert to int.
    int_val = int(hex_str, 16)

    return int_val, hex_str


def compute_page_fingerprint(
    page_index: int, image: np.ndarray, ocr_results: List[Tuple[str, int, int, int, int]]
) -> PageFingerprint:
    """
    Per-page fingerprint: {page_index, visual_hash, visual_hash_hex, role}
    """
    visual_hash, visual_hash_hex = compute_visual_hash(image, ocr_results)
    role = classify_page_role(page_index, ocr_results)

    return PageFingerprint(
        page_index=page_index,
        visual_hash=visual_hash,
        visual_hash_hex=visual_hash_hex,
        role=role,
        stable_anchor_signature=extract_page_anchor_signature(
            page_index, ocr_results, image.shape[1], image.shape[0]
        ),
    )


def compute_document_fingerprint(page_fingerprints: List[PageFingerprint]) -> str:
    """
    Multi-page fingerprint: fingerprint_hash = SHA256(visual_hash_hex_concat)
    """
    sorted_pages = sorted(page_fingerprints, key=lambda x: x.page_index)
    concat_hashes = "".join(p.visual_hash_hex for p in sorted_pages)

    return hashlib.sha256(concat_hashes.encode("utf-8")).hexdigest()
