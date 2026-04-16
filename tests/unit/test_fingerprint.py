import sys
import types

import cv2
import numpy as np
import pytest

from invoice_router.domain.templates.fingerprinting import (
    PaddleOcrUnavailableError,
    classify_page_role,
    compute_visual_hash,
    get_paddle_ocr_status,
    run_full_page_ocr,
)
from invoice_router.models import PageRole


def test_visual_hash_text_masking():
    # Create a 100x100 white image
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 255

    # Add a horizontal line and a box to both to simulate non-text structure
    cv2.line(img1, (10, 50), (90, 50), (0, 0, 0), 2)
    cv2.rectangle(img1, (10, 10), (90, 90), (0, 0, 0), 1)

    img2 = img1.copy()

    # Add fake text (that will be masked) to img1
    ocr_results1 = [("Invoice", 20, 20, 40, 10)]
    # Add different text to img2 at the same location
    ocr_results2 = [("Receipt", 20, 20, 40, 10)]

    hash1_int, hash1_hex = compute_visual_hash(img1, ocr_results1)
    hash2_int, hash2_hex = compute_visual_hash(img2, ocr_results2)

    # The hashes should be identical because the text areas are identically masked out
    assert hash1_hex == hash2_hex
    assert hash1_int == hash2_int


def test_page_role_classification():
    # By default, page 0 is a header page
    assert classify_page_role(0, []) == PageRole.header_page
    assert classify_page_role(1, []) == PageRole.line_item_page


def test_paddle_ocr_falls_back_when_show_log_is_unsupported(monkeypatch):
    class FakePaddleOCR:
        init_calls = []

        def __init__(self, **kwargs):
            FakePaddleOCR.init_calls.append(kwargs)
            if "show_log" in kwargs:
                raise TypeError("Unknown argument: show_log")

        def ocr(self, _image, cls=False):
            return [
                [
                    (
                        [[0, 0], [10, 0], [10, 10], [0, 10]],
                        ("Invoice", 0.99),
                    )
                ]
            ]

    fake_module = types.SimpleNamespace(PaddleOCR=FakePaddleOCR)
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)
    monkeypatch.setattr(
        "invoice_router.domain.templates.fingerprinting.get_paddle_ocr_status", lambda: (True, None)
    )

    results = run_full_page_ocr(np.ones((10, 10, 3), dtype=np.uint8), engine="paddle")

    assert len(FakePaddleOCR.init_calls) == 2
    assert FakePaddleOCR.init_calls[0] == {"use_angle_cls": False, "lang": "en", "show_log": False}
    assert FakePaddleOCR.init_calls[1] == {"use_angle_cls": False, "lang": "en"}
    assert results == [("Invoice", 0, 0, 10, 10)]


def test_paddle_ocr_raises_when_paddle_is_unavailable(monkeypatch):
    def fake_init_paddle():
        raise ModuleNotFoundError("No module named 'paddle'")

    monkeypatch.setattr(
        "invoice_router.domain.templates.fingerprinting._init_paddle_ocr", fake_init_paddle
    )

    with pytest.raises(PaddleOcrUnavailableError, match="PaddleOCR unavailable"):
        run_full_page_ocr(np.ones((10, 10, 3), dtype=np.uint8), engine="paddle")


def test_get_paddle_ocr_status_reports_missing_paddle_runtime(monkeypatch):
    monkeypatch.setattr(
        "invoice_router.domain.templates.fingerprinting.importlib.util.find_spec",
        lambda name: object() if name == "paddleocr" else None,
    )
    get_paddle_ocr_status.cache_clear()

    available, reason = get_paddle_ocr_status()

    assert available is False
    assert reason == "paddle runtime package is not installed"
    get_paddle_ocr_status.cache_clear()
