from types import SimpleNamespace

import numpy as np

from invoice_router.extraction.ocr import extract_with_ocr


def test_extract_with_ocr_applies_family_field_overrides(monkeypatch):
    page = np.ones((100, 100, 3), dtype=np.uint8)
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "Invoice Date": {
                        "region": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.1},
                        "field_type": "string",
                    }
                },
                "table": None,
            }
        ]
    }
    captured = {}

    def _capture_extract_field(_page_img, _region, field_type, buffer_px):
        captured["field_type"] = field_type
        captured["buffer_px"] = buffer_px
        return "2026-04-15"

    monkeypatch.setattr("invoice_router.extraction.ocr.extract_field", _capture_extract_field)

    extracted = extract_with_ocr(
        [page],
        template,
        10,
        config=None,
        family_profile={
            "ocr": {
                "region_buffer_multiplier": 1.1,
                "field_type_overrides": {"Invoice Date": "date"},
                "field_buffer_multipliers": {"date": 1.2},
            }
        },
    )

    assert extracted["Invoice Date"] == "2026-04-15"
    assert captured["field_type"] == "date"
    assert captured["buffer_px"] == 13


def test_extract_with_ocr_passes_family_profile_into_table_extraction(monkeypatch):
    page = np.ones((100, 100, 3), dtype=np.uint8)
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {},
                "table": {
                    "region": {"x": 0.1, "y": 0.2, "width": 0.7, "height": 0.4},
                    "columns": {"Amount": {"x_left": 0.6, "x_right": 0.8}},
                },
            }
        ]
    }
    captured = {}

    def _capture_extract_table(
        _page_img,
        _table_info,
        _buffer_px,
        ocr_engine="paddle",
        table_detection_config=None,
        family_profile=None,
    ):
        captured["ocr_engine"] = ocr_engine
        captured["row_gap_multiplier"] = getattr(table_detection_config, "row_gap_multiplier", None)
        captured["min_line_span_fraction"] = getattr(
            table_detection_config, "min_line_span_fraction", None
        )
        captured["family_profile"] = family_profile
        return [{"Amount": "10.00"}]

    monkeypatch.setattr("invoice_router.extraction.ocr.extract_table", _capture_extract_table)

    extracted = extract_with_ocr(
        [page],
        template,
        8,
        config=SimpleNamespace(
            ocr=SimpleNamespace(table_engine="tesseract"),
            table_detection=SimpleNamespace(row_gap_multiplier=1.5, min_line_span_fraction=0.4),
        ),
        family_profile={
            "table": {
                "enabled": True,
                "ocr_engine": "paddle",
                "row_gap_multiplier": 1.2,
                "min_line_span_fraction": 0.28,
            }
        },
    )

    assert extracted["line_items"] == [{"Amount": "10.00"}]
    assert captured["ocr_engine"] == "paddle"
    assert captured["row_gap_multiplier"] == 1.5
    assert captured["min_line_span_fraction"] == 0.4
    assert captured["family_profile"]["table"]["row_gap_multiplier"] == 1.2
