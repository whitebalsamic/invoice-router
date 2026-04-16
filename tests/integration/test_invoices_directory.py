import numpy as np
import pytest

from invoice_router.config import load_config
from invoice_router.infrastructure.filesystem.source import list_invoices
from invoice_router.infrastructure.persistence.storage import FingerprintDB
from invoice_router.pipeline import process_single_invoice


@pytest.fixture
def mock_llm(monkeypatch):
    mock_template = {
        "version": "v3",
        "inference_method": "heuristic_discovery",
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "invoiceNumber": {
                        "region": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.1},
                        "field_type": "string",
                        "anchor_direction": "right",
                    },
                },
                "table": None,
                "label_confirmation_set": [],
            }
        ],
    }
    monkeypatch.setattr(
        "invoice_router.pipeline.infer_template",
        lambda *_args, **_kwargs: (
            mock_template,
            0.99,
            {"invoiceNumber": "INV-1", "line_items": []},
            {"located_field_count": 1, "extracted_field_count": 1},
        ),
    )


def test_invoices_directory_processing(mock_llm, tmp_path, postgres_database_url, monkeypatch):
    dataset_root = tmp_path / "datasets"
    dataset_dir = dataset_root / "invoices-test"
    dataset_dir.mkdir(parents=True)
    for name in ("one.png", "two.png", "three.png"):
        (dataset_dir / name).write_bytes(b"fake-image")

    monkeypatch.setenv("INVOICE_INPUT_DIR", "invoices-test")
    monkeypatch.setenv("DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("DATABASE_URL", postgres_database_url)
    monkeypatch.setenv("ANALYSIS_DATABASE_URL", postgres_database_url)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))

    monkeypatch.setattr(
        "invoice_router.pipeline.normalize_page",
        lambda _path: [np.ones((100, 100, 3), dtype=np.uint8) * 255],
    )
    monkeypatch.setattr(
        "invoice_router.pipeline.run_full_page_ocr",
        lambda _img, _engine: [("Invoice No.", 10, 10, 20, 10)],
    )

    settings, config = load_config()
    db = FingerprintDB(settings.database_url)
    invoices = list_invoices("invoices-test")[:3]

    assert len(invoices) > 0, "No invoices found in invoices-test"

    success_count = 0
    for inv in invoices:
        try:
            res = process_single_invoice(inv, settings, config, db)
            assert res.invoice_path == inv
            success_count += 1
        except Exception as e:
            pytest.fail(f"Failed to process {inv}: {e}")

    assert success_count == len(invoices)
