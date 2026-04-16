import fitz

from invoice_router.config import load_config
from invoice_router.infrastructure.persistence.storage import FingerprintDB
from invoice_router.pipeline import process_single_invoice


def _write_invoice_pdf(pdf_path):
    doc = fitz.open()
    page = doc.new_page()
    lines = [
        "Invoice",
        "Invoice Number: INV-2026-001",
        "Date: 2026-04-16",
        "Office Visit Consult 1 75.00",
        "Lab Test 1 25.00",
        "Total: $100.00",
    ]
    y = 72
    for line in lines:
        page.insert_text((72, y), line, fontsize=12)
        y += 18
    doc.save(pdf_path)
    doc.close()


def test_process_single_invoice_end_to_end_native_pdf(tmp_path, postgres_database_url, monkeypatch):
    invoice_path = tmp_path / "invoice.pdf"
    _write_invoice_pdf(invoice_path)

    monkeypatch.setenv("INVOICE_INPUT_DIR", str(tmp_path))
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("DATABASE_URL", postgres_database_url)
    monkeypatch.setenv("ANALYSIS_DATABASE_URL", postgres_database_url)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))

    settings, config = load_config()
    db = FingerprintDB(settings.database_url)

    result = process_single_invoice(str(invoice_path), settings, config, db)

    assert result.route_used.value == "APPLY"
    assert result.validation_passed is None
    assert result.provenance.source_format.value == "pdf_text"
    assert result.provenance.extraction_strategy.value == "native_pdf"
    assert result.extracted_data["invoice_number"] == "INV-2026-001"
    assert result.normalized_data["invoice_number"] == "INV-2026-001"
    assert result.normalized_data["total"] == 100.0
    assert result.extracted_data["line_items"] == [
        {"description": "Office Visit Consult", "quantity": "1", "amount": "75.00"},
        {"description": "Lab Test", "quantity": "1", "amount": "25.00"},
    ]

    stored = db.get_result(str(invoice_path))
    assert stored is not None
    assert stored.route_used.value == "APPLY"
    assert stored.provenance.source_format.value == "pdf_text"

    output_path = tmp_path / "output" / tmp_path.name / "invoice.json"
    assert output_path.exists()
