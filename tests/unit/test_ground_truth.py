import json
from pathlib import Path

from invoice_router.domain.ground_truth import (
    canonicalize_ground_truth,
    is_ground_truth_discovery_ready,
)
from invoice_router.infrastructure.filesystem.source import load_ground_truth


def test_canonicalize_ground_truth_v2_flattens_present_fields():
    raw = {
        "schemaVersion": "gt-v2-draft",
        "document": {
            "invoiceNumber": {"status": "present", "value": "INV-100"},
            "invoiceDate": {"status": "present", "value": "2024-01-15"},
            "sellerName": {"status": "present", "value": "Acme Ltd"},
            "customerName": {"status": "unclear"},
            "currency": {"status": "present", "value": "CAD"},
            "country": {"status": "present", "value": "CA"},
        },
        "summary": {
            "subtotal": {"status": "present", "value": 100.0},
            "tax": {"status": "derived", "value": 8.0},
            "discount": {"status": "absent"},
            "shipping": {"status": "present", "value": 5.0},
            "totalAmount": {"status": "present", "value": 113.0},
        },
        "lineItems": [
            {
                "index": 1,
                "description": {"status": "present", "value": "Widget"},
                "quantity": {"status": "present", "value": 2},
                "unitPrice": {"status": "present", "value": 50.0},
                "amount": {"status": "present", "value": 100.0},
                "tax": {"status": "present", "value": 8.0},
                "taxRate": {"status": "present", "value": 8.0},
            }
        ],
    }

    canonical = canonicalize_ground_truth(raw)

    assert canonical["invoiceNumber"] == "INV-100"
    assert canonical["invoiceDate"] == "2024-01-15"
    assert canonical["sellerName"] == "Acme Ltd"
    assert canonical["customerName"] is None
    assert canonical["currency_code"] == "CAD"
    assert canonical["country_code"] == "CA"
    assert canonical["subtotal"] == 100.0
    assert canonical["tax"] is None
    assert canonical["discount"] is None
    assert canonical["shipping"] == 5.0
    assert canonical["totalAmount"] == 113.0
    assert canonical["lineItems"][0]["unit_price"] == 50.0
    assert canonical["lineItems"][0]["tax_amount"] == 8.0
    assert canonical["lineItems"][0]["tax_rate"] == 8.0


def test_load_ground_truth_canonicalizes_v2_json(tmp_path):
    gt_path = tmp_path / "invoice.json"
    gt_path.write_text(
        json.dumps(
            {
                "schemaVersion": "gt-v2",
                "document": {"invoiceNumber": {"status": "present", "value": "INV-9"}},
                "summary": {"totalAmount": {"status": "present", "value": 42.0}},
                "lineItems": [],
            }
        )
    )

    loaded = load_ground_truth(gt_path)

    assert loaded == {
        "invoiceNumber": "INV-9",
        "invoiceDate": None,
        "sellerName": None,
        "customerName": None,
        "currency_code": None,
        "country_code": None,
        "subtotal": None,
        "tax": None,
        "discount": None,
        "shipping": None,
        "totalAmount": 42.0,
        "lineItems": [],
    }


def test_load_ground_truth_upgrades_flat_legacy_json(tmp_path):
    gt_path = tmp_path / "invoice.json"
    gt_path.write_text(
        json.dumps(
            {
                "invoiceNumber": "INV-17",
                "invoiceDate": "2026-04-01",
                "sellerName": "Acme Ltd",
                "totalAmount": "42.50",
            }
        )
    )

    loaded = load_ground_truth(gt_path)

    assert loaded == {
        "invoiceNumber": "INV-17",
        "invoiceDate": "2026-04-01",
        "sellerName": "Acme Ltd",
        "customerName": None,
        "currency_code": None,
        "country_code": None,
        "subtotal": None,
        "tax": None,
        "discount": None,
        "shipping": None,
        "totalAmount": 42.5,
        "lineItems": [],
    }


def test_load_ground_truth_returns_none_for_unrecognized_json(tmp_path):
    gt_path = tmp_path / "invoice.json"
    gt_path.write_text(json.dumps({"foo": "bar"}))

    assert load_ground_truth(gt_path) is None


def test_ground_truth_discovery_readiness_requires_total_identifier_and_coverage():
    assert not is_ground_truth_discovery_ready(
        {
            "totalAmount": 99.0,
        }
    )
    assert not is_ground_truth_discovery_ready(
        {
            "invoiceNumber": "INV-100",
            "invoiceDate": "2026-04-01",
            "sellerName": "Acme Ltd",
        }
    )
    assert is_ground_truth_discovery_ready(
        {
            "invoiceNumber": "INV-100",
            "totalAmount": 99.0,
        }
    )


def test_ground_truth_schema_file_is_valid_json():
    schema_path = Path(__file__).resolve().parents[2] / "schemas" / "ground-truth-v2.schema.json"
    schema = json.loads(schema_path.read_text())

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "invoice-router Ground Truth v2"
