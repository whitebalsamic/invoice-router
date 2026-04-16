from invoice_router.domain.validation.field_resolver import _resolve_date_value
from invoice_router.domain.validation.normalizer import (
    merge_for_validation,
    normalize_extracted_invoice,
)
from invoice_router.models import DocumentContext, DocumentFamily, ExtractionStrategy, SourceFormat


def test_normalize_extracted_invoice_adds_canonical_fields_and_categories():
    context = DocumentContext(
        source_format=SourceFormat.pdf_text,
        document_family=DocumentFamily.invoice,
        country_code="US",
        currency_code="USD",
        extraction_strategy=ExtractionStrategy.native_pdf,
    )
    extracted = {
        "Invoice Number": "INV-42",
        "Date": "04/13/2026",
        "Total": "$100.00",
        "line_items": [
            {"description": "Office Visit Consult", "quantity": "1", "amount": "75.00"},
            {"description": "Lab Test", "quantity": "1", "amount": "25.00"},
        ],
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["invoice_number"] == "INV-42"
    assert normalized["invoice_date"] == "2026-04-13"
    assert normalized["total"] == 100.0
    assert normalized["country_code"] == "US"
    assert normalized["currency_code"] == "USD"
    assert normalized["line_item_count"] == 2
    assert normalized["line_items"][0]["category"] == "consultation"
    assert normalized["line_items"][1]["category"] == "diagnostics"
    assert normalized["category_totals"] == {"consultation": 75.0, "diagnostics": 25.0}
    assert normalized["has_line_items"] is True
    assert normalized["invoiceNumber"] == "INV-42"
    assert normalized["totalAmount"] == 100.0


def test_merge_for_validation_preserves_raw_and_normalized_fields():
    merged = merge_for_validation(
        {"Invoice No.": "1234", "Total": "$10.00"},
        {"invoice_number": "1234", "total": 10.0},
    )

    assert merged["Invoice No."] == "1234"
    assert merged["invoice_number"] == "1234"
    assert merged["total"] == 10.0


def test_resolve_date_value_does_not_degrade_existing_valid_date():
    resolved = _resolve_date_value("Oct. 10, 2023", ["Oct. 23, 2023"])

    assert resolved == "Oct. 10, 2023"


def test_normalize_extracted_invoice_prefers_total_amount_over_subtotal_like_keys():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="CA",
        currency_code="CAD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "subtotal": "$680.00",
        "totalAmount": "$740.08",
        "invoiceNumber": "5667",
        "invoiceDate": "Apr 22, 2023",
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["subtotal"] == 680.0
    assert normalized["total"] == 740.08
    assert normalized["invoice_number"] == "5667"


def test_normalize_extracted_invoice_uses_country_for_ambiguous_slash_dates():
    us_context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="US",
        currency_code="USD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    ca_context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="CA",
        currency_code="CAD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )

    extracted = {
        "invoiceDate": "03/05/2020",
    }

    assert normalize_extracted_invoice(extracted, us_context)["invoice_date"] == "2020-03-05"
    assert normalize_extracted_invoice(extracted, ca_context)["invoice_date"] == "2020-05-03"


def test_normalize_extracted_invoice_strips_label_prefixed_dates_and_invoice_numbers():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="US",
        currency_code="USD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "invoiceDate": "Date of issue: Nov. 23, 2023",
        "invoiceNumber": "INV-2023-0042 | Date 2023-11-23",
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["invoice_date"] == "2023-11-23"
    assert normalized["invoice_number"] == "INV-2023-0042"


def test_normalize_extracted_invoice_rejects_implausible_numeric_garbage():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="US",
        currency_code="USD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "totalAmount": "1e10",
        "subtotal": "12345678901234567890",
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["total"] is None
    assert normalized["subtotal"] is None


def test_normalize_extracted_invoice_trims_party_blocks_at_generic_boundaries():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="US",
        currency_code="USD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "sellerName": "Acme Corp Address 123 Main St",
        "customerName": "Alpha LLC Tax ID 12-345",
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["provider_name"] == "Acme Corp"
    assert normalized["customer_name"] == "Alpha LLC"


def test_normalize_extracted_invoice_prefers_fully_priced_line_sum_for_subtotal():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="CA",
        currency_code="CAD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "subtotal": "200",
        "totalAmount": "$217.48",
        "line_items": [
            {"description": "Item A", "quantity": "2", "price": "$10.00", "total": "$20.00"},
            {"description": "Item B", "quantity": "4", "price": "$16.00", "total": "$64.00"},
            {"description": "Item C", "quantity": "5", "price": "$7.00", "total": "$35.00"},
            {"description": "Item D", "quantity": "4", "price": "$4.00", "total": "$16.00"},
            {"description": "Item E", "quantity": "5", "price": "$12.00", "total": "$60.00"},
        ],
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["subtotal"] == 195.0


def test_normalize_extracted_invoice_records_reconciliation_summary():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="US",
        currency_code="USD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "subtotal": "$90.00",
        "totalAmount": "$100.00",
        "line_items": [
            {"description": "Consult", "quantity": "1", "price": "$50.00", "amount": "$50.00"},
            {"description": "Lab", "quantity": "1", "price": "$40.00", "amount": "$40.00"},
        ],
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["tax"] == 10.0
    assert normalized["reconciliation_summary"]["status"] == "reconciled"
    assert normalized["reconciliation_summary"]["issue_types"] == []
    assert "tax" in normalized["reconciliation_summary"]["derived_fields"]
    assert normalized["reconciliation_summary"]["supports"]["summary_formula_complete"] is True


def test_normalize_extracted_invoice_keeps_sparse_line_sum_out_of_subtotal_override():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="US",
        currency_code="USD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "subtotal": "$7106.00",
        "line_items": [
            {"description": "Item A", "quantity": "6", "amount": "$666.00"},
            {"description": "Item B", "quantity": "1", "amount": "$19.00"},
            {"description": "Item C", "quantity": "6", "amount": "$5340.00"},
        ],
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["subtotal"] == 7106.0


def test_normalize_extracted_invoice_keeps_scalar_subtotal_when_total_already_matches_it():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="US",
        currency_code="USD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "subtotal": "$7106.00",
        "totalAmount": "$7106.00",
        "line_items": [
            {"description": "Item A", "quantity": "6", "price": "$114.00", "amount": "$666.00"},
            {"description": "Item B", "quantity": "1", "price": "$19.00", "amount": "$19.00"},
            {"description": "Item C", "quantity": "6", "price": "$890.00", "amount": "$5340.00"},
        ],
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["subtotal"] == 7106.0
    assert normalized["tax"] is None
    assert normalized["total"] == 7106.0


def test_normalize_extracted_invoice_recovers_line_total_when_amount_duplicates_unit_price():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="US",
        currency_code="USD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "line_items": [
            {"DESCRIPTION": "Item A", "QUANTITY": "4", "PRICE, $": "247.00", "TOTAL, $": "247.00"},
            {"DESCRIPTION": "Item B", "QUANTITY": "1", "PRICE, $": "80.00", "TOTAL, $": "80.00"},
        ],
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["line_items"][0]["unit_price"] == 247.0
    assert normalized["line_items"][0]["amount"] == 988.0
    assert normalized["line_items"][1]["amount"] == 80.0


def test_normalize_extracted_invoice_aligns_noisy_unit_price_to_row_total():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="US",
        currency_code="USD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "line_items": [
            {"description": "Item A", "qty": "5", "price": "$150.00", "amount": "735"},
            {"description": "Item B", "qty": "4", "price": "$182.00", "amount": "$712"},
        ],
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["line_items"][0]["unit_price"] == 147.0
    assert normalized["line_items"][0]["amount"] == 735.0
    assert normalized["line_items"][1]["unit_price"] == 178.0
    assert normalized["line_items"][1]["amount"] == 712.0


def test_normalize_extracted_invoice_derives_net_amount_from_gross_when_amount_is_truncated():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="US",
        currency_code="USD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "line_items": [
            {
                "description": "Item A",
                "qty": "3,00 each",
                "price": "1",
                "amount": "049,85",
                "gross amount": "1154,83",
                "tax rate": "10%",
            },
            {
                "description": "Item B",
                "qty": "5,00 each",
                "price": "39",
                "amount": "000,00",
                "gross amount": "42900,00",
                "tax rate": "10%",
            },
        ],
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["line_items"][0]["amount"] == 1049.85
    assert normalized["line_items"][0]["unit_price"] == 349.95
    assert normalized["line_items"][1]["amount"] == 39000.0
    assert normalized["line_items"][1]["unit_price"] == 7800.0


def test_normalize_extracted_invoice_keeps_scalar_subtotal_when_row_sum_conflicts_with_total():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="CA",
        currency_code="CAD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "subtotal": "888.0",
        "totalAmount": "1007.3",
        "line_items": [
            {"QUANTITY": "2", "PRICE, $": "367.00", "TOTAL, $": "367.00"},
            {"QUANTITY": "4", "PRICE, $": "247.00", "TOTAL, $": "247.00"},
            {"QUANTITY": "6", "PRICE, $": "157.00", "TOTAL, $": "157.00"},
            {"QUANTITY": "8", "PRICE, $": "117.00", "TOTAL, $": "117.00"},
        ],
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["subtotal"] == 888.0
    assert [row["amount"] for row in normalized["line_items"]] == [734.0, 988.0, 942.0, 936.0]


def test_normalize_extracted_invoice_prefers_raw_line_amounts_when_summary_supports_them():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="US",
        currency_code="USD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "subtotal": "756.00",
        "tax": "60.24",
        "shipping": "50.00",
        "totalAmount": "866.24",
        "line_items": [
            {"description": "Item A", "qty": "2", "price": "334.00", "amount": "334.00"},
            {"description": "Item B", "qty": "4", "price": "214.00", "amount": "214.00"},
            {"description": "Item C", "qty": "6", "price": "124.00", "amount": "124.00"},
            {"description": "Item D", "qty": "8", "price": "84.00", "amount": "84.00"},
        ],
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert [row["amount"] for row in normalized["line_items"]] == [334.0, 214.0, 124.0, 84.0]
    assert normalized["subtotal"] == 756.0
    assert normalized["field_provenance"]["subtotal"]["parsed_value"] == 756.0
    assert normalized["line_items"][0]["_provenance"]["amount"]["source"] == "summary_consistency"


def test_normalize_extracted_invoice_derives_net_subtotal_tax_and_gross_total_from_table_columns():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="CA",
        currency_code="CAD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "invoiceNumber": "27405523",
        "line_items": [
            {
                "Description": "Item A",
                "Qty": "1.00",
                "Net worth": "4.47",
                "Gross worth": "4.92",
                "VAT (%)": "10%",
            },
            {
                "Description": "Item B",
                "Qty": "2.00",
                "Net worth": "20.00",
                "Gross worth": "22.00",
                "VAT (%)": "10%",
            },
            {
                "Description": "Item C",
                "Qty": "2.00",
                "Net worth": "29.98",
                "Gross worth": "32.98",
                "VAT (%)": "10%",
            },
        ],
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["subtotal"] == 54.45
    assert normalized["tax"] == 5.45
    assert normalized["total"] == 59.9
    assert normalized["line_items"][0]["tax_amount"] == 0.45
    assert normalized["field_provenance"]["subtotal"]["kind"] == "derived"
    assert normalized["field_provenance"]["tax"]["kind"] == "derived"
    assert normalized["field_provenance"]["total"]["kind"] == "derived"


def test_normalize_extracted_invoice_prefers_complete_row_tax_breakdown_over_noisy_scalar_tax():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="CA",
        currency_code="CAD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "tax": "3",
        "line_items": [
            {
                "Description": "Item A",
                "Qty": "4.00",
                "Net worth": "991.12",
                "Gross worth": "1090.23",
                "VAT (%)": "10%",
            },
            {
                "Description": "Item B",
                "Qty": "3.00",
                "Net worth": "714.00",
                "Gross worth": "785.40",
                "VAT (%)": "10%",
            },
            {
                "Description": "Item C",
                "Qty": "1.00",
                "Net worth": "1840.10",
                "Gross worth": "2024.11",
                "VAT (%)": "10%",
            },
        ],
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["tax"] == 354.52
    assert normalized["subtotal"] == 3545.22
    assert normalized["total"] == 3899.74


def test_normalize_extracted_invoice_derives_missing_tax_from_subtotal_and_total():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="US",
        currency_code="USD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "subtotal": "$555.68",
        "totalAmount": "$611.25",
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["tax"] == 55.57
    assert normalized["field_provenance"]["tax"]["kind"] == "derived"
    assert normalized["field_provenance"]["tax"]["supporting_fields"] == ["subtotal", "total"]


def test_normalize_extracted_invoice_does_not_treat_gross_only_rows_as_invoice_total():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="CA",
        currency_code="CAD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "subtotal": "413.38",
        "tax": "41.34",
        "line_items": [
            {"Description": "Item A", "Qty": "2", "Gross worth": "200.00"},
            {"Description": "Item B", "Qty": "1", "Gross worth": "254.72"},
        ],
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["subtotal"] == 413.38
    assert normalized["tax"] == 41.34
    assert normalized["total"] == 454.72


def test_normalize_extracted_invoice_keeps_scalar_tax_when_row_breakdown_contains_impossible_negative_tax():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="CA",
        currency_code="CAD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "tax": "1321.00",
        "line_items": [
            {
                "Description": "Item A",
                "Qty": "5.00",
                "Net worth": "2981.25",
                "Gross worth": "3279.37",
                "VAT (%)": "10%",
            },
            {
                "Description": "Item B",
                "Qty": "4.00",
                "Net worth": "10980.00",
                "Gross worth": "10978.00",
                "VAT (%)": "10%",
            },
            {
                "Description": "Item C",
                "Qty": "1.00",
                "Net worth": "248.78",
                "Gross worth": "273.66",
                "VAT (%)": "10%",
            },
        ],
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["tax"] == 1321.0


def test_normalize_extracted_invoice_keeps_scalar_tax_when_row_tax_breakdown_is_all_zero_despite_positive_vat():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="CA",
        currency_code="CAD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "tax": "40600.0",
        "line_items": [
            {
                "No.": "1",
                "Description": "Item A",
                "Qty": "100",
                "Net worth": "143000.00",
                "Gross worth": "143000.00",
                "VAT (%)": "10%",
            },
            {
                "No.": "2",
                "Description": "Item B",
                "Qty": "400",
                "Net worth": "303596.00",
                "Gross worth": "303596.00",
                "VAT (%)": "10%",
            },
        ],
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["tax"] == 40600.0


def test_normalize_extracted_invoice_records_numeric_provenance_for_scalar_fields():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="DE",
        currency_code="EUR",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "invoiceNumber": "INV-77",
        "subtotal": "1.023,40",
        "tax": "163,40",
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["subtotal"] == 1023.4
    assert normalized["tax"] == 163.4
    assert normalized["numeric_convention"]["decimal_separator"] == ","
    assert (
        normalized["field_provenance"]["subtotal"]["numeric_convention"]["decimal_separator"] == ","
    )
    assert normalized["field_provenance"]["subtotal"]["kind"] == "normalized"


def test_normalize_extracted_invoice_records_summary_reconciliation_provenance():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="CA",
        currency_code="CAD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "subtotal": "$930.00",
        "tax": "$60.08",
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["total"] == 990.08
    assert normalized["field_provenance"]["total"]["kind"] == "derived"
    assert normalized["field_provenance"]["total"]["source"] == "summary_reconciliation"


def test_normalize_extracted_invoice_normalizes_shipping_and_uses_it_in_total_derivation():
    context = DocumentContext(
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code="US",
        currency_code="USD",
        extraction_strategy=ExtractionStrategy.ocr_structured,
    )
    extracted = {
        "subtotal": "756.00",
        "tax": "60.24",
        "shipping": "and Handling 50.00",
    }

    normalized = normalize_extracted_invoice(extracted, context)

    assert normalized["shipping"] == 50.0
    assert normalized["total"] == 866.24
    assert normalized["totalAmount"] == 866.24
    assert normalized["field_provenance"]["total"]["source"] == "summary_reconciliation"
    assert normalized["field_provenance"]["total"]["supporting_fields"] == [
        "subtotal",
        "tax",
        "shipping",
    ]
