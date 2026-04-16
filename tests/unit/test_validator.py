from invoice_router.config import ValidationConfig
from invoice_router.domain.validation.validator import (
    _compare_values,
    normalize_amount,
    normalize_date,
    normalize_string,
    validate_invoice,
)


def test_normalize_amount():
    assert normalize_amount("1234.56") == 1234.56
    assert normalize_amount("$1,234.56") == 1234.56
    assert normalize_amount("€1 234,56") == 1234.56
    assert normalize_amount("1.234,56") == 1234.56
    assert normalize_amount("82.49") == 82.49
    assert normalize_amount("abc") == 0.0


def test_normalize_date():
    assert normalize_date("2026-01-03") == "2026-01-03"
    assert normalize_date("Mar 14, 2019") == "2019-03-14"
    assert normalize_date("01/03/2026", country_code="US") == "2026-01-03"
    assert normalize_date("01-03-2026", country_code="US") == "2026-01-03"
    assert normalize_date("01/03/2026", country_code="CA") == "2026-03-01"
    assert normalize_date("13/03/2026", country_code="US") == "2026-03-13"
    assert normalize_date("March 5, 2020", country_code="CA") == "2020-03-05"


def test_compare_values_date_uses_country_for_ambiguous_numeric_dates():
    config = ValidationConfig()

    assert (
        _compare_values("01/03/2026", "2026-01-03", "invoiceDate", config, country_code="US")
        is True
    )
    assert (
        _compare_values("01/03/2026", "2026-03-01", "invoiceDate", config, country_code="CA")
        is True
    )
    assert (
        _compare_values("01/03/2026", "2026-01-03", "invoiceDate", config, country_code="CA")
        is False
    )


def test_normalize_string():
    assert normalize_string("  LuLaRoe    Nicole Dress  ") == "LuLaRoe Nicole Dress"


def test_compare_values_amount_tolerance():
    config = ValidationConfig()
    # 82.489 vs 82.49 = match (within 0.01 tolerance)
    assert _compare_values(82.489, 82.49, "amount", config) is True
    # 82.49 vs 82.51 = mismatch
    assert _compare_values(82.49, 82.51, "amount", config) is False


def test_compare_values_treats_shipping_as_amount():
    config = ValidationConfig()

    assert _compare_values("and Handling 0.00", 0.0, "shipping", config) is True


def test_compare_values_cent_scale_equivalence_is_opt_in():
    default_config = ValidationConfig()
    assert _compare_values(65.0, 6500.0, "tax", default_config) is False

    scaled_config = ValidationConfig(allow_cent_scale_equivalence=True)
    assert _compare_values(65.0, 6500.0, "tax", scaled_config) is True


def test_compare_values_qty_three_decimals():
    config = ValidationConfig()
    assert _compare_values(3, 3.000, "qty", config) is True
    assert _compare_values(3, 3.001, "qty", config) is False


def test_compare_values_jaccard():
    config = ValidationConfig(jaccard_threshold=0.85)
    # 5 words, 4 match -> 4/6 = 0.66 < 0.85 -> False
    assert (
        _compare_values("Nike Air Max Size 10", "Adidas Ultraboost SZ 9", "description", config)
        is False
    )
    # Identical words minus case/spaces -> True
    assert (
        _compare_values("Nike Air Max Size 10", "nike air max size 10", "description", config)
        is True
    )


def test_validate_invoice_catches_line_item_arithmetic_mismatch():
    config = ValidationConfig()
    extracted = {
        "subtotal": "10.00",
        "tax": "1.00",
        "total": "11.00",
        "line_items": [
            {"description": "Consult", "quantity": "2", "unit_price": "4.00", "amount": "10.00"},
        ],
    }
    ground_truth = {
        "subtotal": "10.00",
        "tax": "1.00",
        "total": "11.00",
        "line_items": [
            {"description": "Consult"},
        ],
    }

    result = validate_invoice(extracted, ground_truth, config, {})

    assert result.passed is False
    assert any("Line item arithmetic mismatch" in error for error in result.errors)


def test_validate_invoice_skips_line_item_arithmetic_when_amount_was_resolved_from_summary_consistency():
    config = ValidationConfig()
    extracted = {
        "subtotal": "756.00",
        "tax": "60.24",
        "shipping": "50.00",
        "totalAmount": "866.24",
        "line_items": [
            {
                "description": "Item A",
                "quantity": 2,
                "unit_price": 334.00,
                "amount": 334.00,
                "_provenance": {
                    "amount": {
                        "source": "summary_consistency",
                    }
                },
            },
        ],
    }
    ground_truth = {
        "subtotal": "756.00",
        "tax": "60.24",
        "shipping": "50.00",
        "totalAmount": "866.24",
        "line_items": [
            {"description": "Item A"},
        ],
    }

    result = validate_invoice(extracted, ground_truth, config, {})

    assert not any("Line item arithmetic mismatch" in error for error in result.errors)


def test_validate_invoice_catches_summary_reconciliation_mismatch():
    config = ValidationConfig()
    extracted = {
        "subtotal": "90.00",
        "tax": "10.00",
        "total": "105.00",
        "line_items": [
            {"description": "Consult", "amount": "50.00"},
            {"description": "Lab", "amount": "40.00"},
        ],
    }
    ground_truth = {
        "subtotal": "90.00",
        "tax": "10.00",
        "total": "105.00",
        "line_items": [
            {"description": "Consult"},
            {"description": "Lab"},
        ],
    }

    result = validate_invoice(extracted, ground_truth, config, {})

    assert result.passed is False
    assert any("Total mismatch" in error for error in result.errors)


def test_validate_invoice_uses_reconciliation_summary_and_returns_error_counts():
    config = ValidationConfig()
    extracted = {
        "subtotal": "90.00",
        "tax": "10.00",
        "total": "105.00",
        "line_items": [
            {"description": "Consult", "amount": "50.00"},
            {"description": "Lab", "amount": "40.00"},
        ],
        "reconciliation_summary": {
            "status": "inconsistent",
            "issue_types": ["total_mismatch"],
            "issues": [
                {
                    "kind": "total_mismatch",
                    "message": "Total mismatch: expected subtotal+tax 100.00, got 105.00",
                }
            ],
            "derived_fields": [],
            "supports": {
                "line_items_present": True,
                "fully_priced_line_items": False,
                "row_tax_breakdown_reliable": False,
                "supports_invoice_gross_total": False,
                "summary_formula_complete": True,
            },
        },
    }
    ground_truth = {
        "subtotal": "90.00",
        "tax": "10.00",
        "total": "105.00",
        "line_items": [
            {"description": "Consult"},
            {"description": "Lab"},
        ],
    }

    result = validate_invoice(extracted, ground_truth, config, {})

    assert result.passed is False
    assert result.reconciliation_summary["status"] == "inconsistent"
    assert any("Total mismatch" in error for error in result.arithmetic_errors)
    assert result.error_counts["summary_arithmetic_mismatch"] == 1


def test_validate_invoice_summary_reconciliation_includes_shipping():
    config = ValidationConfig()
    extracted = {
        "subtotal": "756.00",
        "tax": "60.24",
        "shipping": "50.00",
        "totalAmount": "866.24",
        "line_items": [
            {"description": "Item A", "amount": "334.00"},
            {"description": "Item B", "amount": "214.00"},
            {"description": "Item C", "amount": "124.00"},
            {"description": "Item D", "amount": "84.00"},
        ],
    }
    ground_truth = {
        "subtotal": "756.00",
        "tax": "60.24",
        "shipping": "50.00",
        "totalAmount": "866.24",
        "line_items": [
            {"description": "Item A"},
            {"description": "Item B"},
            {"description": "Item C"},
            {"description": "Item D"},
        ],
    }

    result = validate_invoice(extracted, ground_truth, config, {})

    assert result.passed is True


def test_validate_invoice_skips_line_reconciliation_without_gt_line_items():
    config = ValidationConfig()
    extracted = {
        "subtotal": "90.00",
        "tax": "10.00",
        "total": "105.00",
        "line_items": [
            {"description": "Consult", "amount": "50.00"},
            {"description": "Lab", "amount": "40.00"},
        ],
    }
    ground_truth = {
        "subtotal": "90.00",
        "tax": "10.00",
        "total": "105.00",
        "line_items": [],
    }

    result = validate_invoice(extracted, ground_truth, config, {})

    assert result.passed is True
    assert not any(
        "Subtotal mismatch" in error or "Total mismatch" in error for error in result.errors
    )


def test_validate_invoice_skips_subtotal_mismatch_when_scalar_summary_already_reconciles():
    config = ValidationConfig()
    extracted = {
        "subtotal": "7106.00",
        "total": "7106.00",
        "line_items": [
            {"description": "Item A", "amount": "666.00"},
            {"description": "Item B", "amount": "19.00"},
            {"description": "Item C", "amount": "5340.00"},
        ],
    }
    ground_truth = {
        "subtotal": "7106.00",
        "total": "7106.00",
        "line_items": [
            {"description": "Item A"},
            {"description": "Item B"},
            {"description": "Item C"},
        ],
    }

    result = validate_invoice(extracted, ground_truth, config, {})

    assert not any("Subtotal mismatch" in error for error in result.errors)


def test_validate_invoice_catches_country_currency_mismatch():
    config = ValidationConfig()
    extracted = {
        "invoiceNumber": "INV-1",
        "country_code": "MX",
        "currency_code": "USD",
    }
    ground_truth = {
        "invoiceNumber": "INV-1",
    }

    result = validate_invoice(extracted, ground_truth, config, {})

    assert result.passed is False
    assert "Currency mismatch for country MX: expected MXN, got USD" in result.errors


def test_validate_invoice_accepts_gt_v2_structure():
    config = ValidationConfig()
    extracted = {
        "invoiceNumber": "INV-200",
        "invoiceDate": "2024-03-05",
        "sellerName": "Acme Ltd",
        "customerName": "Buyer Inc",
        "tax": "8.00",
        "totalAmount": "108.00",
        "line_items": [
            {"description": "Widget", "quantity": "2", "unit_price": "50.00", "amount": "100.00"},
        ],
    }
    ground_truth = {
        "schemaVersion": "gt-v2",
        "document": {
            "invoiceNumber": {"status": "present", "value": "INV-200"},
            "invoiceDate": {"status": "present", "value": "2024-03-05"},
            "sellerName": {"status": "present", "value": "Acme Ltd"},
            "customerName": {"status": "present", "value": "Buyer Inc"},
        },
        "summary": {
            "tax": {"status": "present", "value": 8.0},
            "totalAmount": {"status": "present", "value": 108.0},
        },
        "lineItems": [
            {
                "description": {"status": "present", "value": "Widget"},
                "quantity": {"status": "present", "value": 2},
                "unitPrice": {"status": "present", "value": 50.0},
                "amount": {"status": "present", "value": 100.0},
            }
        ],
    }

    result = validate_invoice(extracted, ground_truth, config, {})

    assert result.passed is True
