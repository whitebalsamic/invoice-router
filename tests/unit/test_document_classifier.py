from invoice_router.domain.invoices.classification import build_document_context
from invoice_router.models import DocumentFamily, ExtractionStrategy, SourceFormat


def test_build_document_context_for_image_invoice(tmp_path):
    invoice_path = tmp_path / "invoice.png"
    invoice_path.write_bytes(b"fake")

    context = build_document_context(
        str(invoice_path),
        [[("Invoice", 0, 0, 10, 10), ("CAD", 0, 0, 10, 10), ("GST", 0, 0, 10, 10)]],
    )

    assert context.source_format == SourceFormat.image
    assert context.document_family == DocumentFamily.invoice
    assert context.currency_code == "CAD"
    assert context.country_code == "CA"
    assert context.extraction_strategy == ExtractionStrategy.ocr_structured


def test_build_document_context_infers_us_from_state_and_zip(tmp_path):
    invoice_path = tmp_path / "invoice.png"
    invoice_path.write_bytes(b"fake")

    context = build_document_context(
        str(invoice_path),
        [
            [
                ("Invoice", 0, 0, 10, 10),
                ("Austin", 0, 0, 10, 10),
                ("TX", 0, 0, 10, 10),
                ("78701", 0, 0, 10, 10),
            ]
        ],
    )

    assert context.country_code == "US"
    assert context.currency_code == "USD"


def test_build_document_context_uses_supplied_source_format(monkeypatch, tmp_path):
    invoice_path = tmp_path / "invoice.pdf"
    invoice_path.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(
        "invoice_router.domain.invoices.classification._detect_source_format",
        lambda _invoice_path: (_ for _ in ()).throw(AssertionError("should not detect format")),
    )

    context = build_document_context(
        str(invoice_path),
        [[("Invoice", 0, 0, 10, 10)]],
        source_format=SourceFormat.pdf_text,
    )

    assert context.source_format == SourceFormat.pdf_text
    assert context.extraction_strategy == ExtractionStrategy.native_pdf
