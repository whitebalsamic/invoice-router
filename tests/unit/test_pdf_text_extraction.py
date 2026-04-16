import sys
import types

from invoice_router.extraction.pdf_text import extract_from_text_pdf


def test_extract_from_text_pdf_parses_basic_fields(monkeypatch, tmp_path):
    pdf_path = tmp_path / "invoice.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    class FakePage:
        def __init__(self, text):
            self._text = text

        def get_text(self, _mode):
            return self._text

    class FakeDoc:
        def __init__(self, pages):
            self._pages = pages

        def __len__(self):
            return len(self._pages)

        def __getitem__(self, index):
            return self._pages[index]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_fitz = types.SimpleNamespace(
        open=lambda _path: FakeDoc(
            [
                FakePage("Invoice Number: INV-123\nDate: 2026-04-13\nTotal: $100.00"),
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    extracted = extract_from_text_pdf(str(pdf_path))

    assert extracted["page_count"] == 1
    assert extracted["Invoice Number"] == "INV-123"
    assert extracted["invoice_number"] == "INV-123"
    assert extracted["invoice_date"] == "2026-04-13"
    assert extracted["total"] == "$100.00"


def test_extract_from_text_pdf_parses_line_items(monkeypatch, tmp_path):
    pdf_path = tmp_path / "invoice-lines.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    class FakePage:
        def __init__(self, text):
            self._text = text

        def get_text(self, _mode):
            return self._text

    class FakeDoc:
        def __init__(self, pages):
            self._pages = pages

        def __len__(self):
            return len(self._pages)

        def __getitem__(self, index):
            return self._pages[index]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_fitz = types.SimpleNamespace(
        open=lambda _path: FakeDoc(
            [
                FakePage("Office Visit Consult 1 75.00\nLab Test 1 25.00\nTotal: $100.00"),
            ]
        )
    )
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    extracted = extract_from_text_pdf(str(pdf_path))

    assert extracted["line_items"] == [
        {"description": "Office Visit Consult", "quantity": "1", "amount": "75.00"},
        {"description": "Lab Test", "quantity": "1", "amount": "25.00"},
    ]


def test_extract_from_text_pdf_uses_supplied_page_texts_without_opening_pdf(monkeypatch, tmp_path):
    pdf_path = tmp_path / "invoice-preloaded.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    class _FailingFitzModule:
        @staticmethod
        def open(_path):
            raise AssertionError("fitz.open should not be called when page_texts are supplied")

    monkeypatch.setitem(sys.modules, "fitz", _FailingFitzModule())

    extracted = extract_from_text_pdf(
        str(pdf_path),
        page_texts=["Invoice Number: INV-777\nDate: 2026-04-16\nTotal: $42.00"],
    )

    assert extracted["invoice_number"] == "INV-777"
    assert extracted["invoice_date"] == "2026-04-16"
    assert extracted["total"] == "$42.00"
