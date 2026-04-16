from invoice_router.domain.validation.field_resolver import refine_critical_fields


def test_refine_critical_fields_prefers_single_ocr_date_candidate():
    extracted = {
        "invoiceDate": "Date of issue: Nov. 23, 2023",
        "totalAmount": "$7106",
    }
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "invoiceDate": {
                        "label_text": "Invoice Date",
                        "region": {"x": 0.4, "y": 0.1, "width": 0.3, "height": 0.1},
                    }
                },
            }
        ]
    }
    ocr_results = [
        [
            ("Invoice", 40, 20, 50, 20),
            ("Date", 95, 20, 35, 20),
            ("Oct.", 210, 25, 35, 20),
            ("10,", 250, 25, 25, 20),
            ("2023", 280, 25, 45, 20),
        ]
    ]

    refined = refine_critical_fields(extracted, template, ocr_results, [(500, 200)])

    assert refined["invoiceDate"] == "Oct. 10, 2023"


def test_refine_critical_fields_strips_label_prefixed_date_text():
    extracted = {
        "invoiceDate": "Date of issue: Nov. 23, 2023",
    }
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "invoiceDate": {
                        "label_text": "Invoice Date",
                        "region": {"x": 0.3, "y": 0.08, "width": 0.5, "height": 0.12},
                    }
                },
            }
        ]
    }
    ocr_results = [
        [
            ("Date", 40, 18, 35, 18),
            ("of", 80, 18, 18, 18),
            ("issue:", 104, 18, 54, 18),
            ("Nov.", 210, 20, 38, 18),
            ("23,", 252, 20, 28, 18),
            ("2023", 286, 20, 42, 18),
        ]
    ]

    refined = refine_critical_fields(extracted, template, ocr_results, [(500, 200)])

    assert refined["invoiceDate"] == "Nov. 23, 2023"


def test_refine_critical_fields_rejects_implausible_numeric_garbage():
    extracted = {
        "totalAmount": "$123.45",
    }
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "totalAmount": {
                        "label_text": "Total",
                        "region": {"x": 0.55, "y": 0.7, "width": 0.25, "height": 0.08},
                    }
                },
            }
        ]
    }
    ocr_results = [
        [
            ("Total", 120, 140, 40, 18),
            ("12345678901234567890", 280, 142, 150, 18),
        ]
    ]

    refined = refine_critical_fields(extracted, template, ocr_results, [(500, 200)])

    assert refined["totalAmount"] == "$123.45"


def test_refine_critical_fields_trims_party_block_at_generic_boundary():
    extracted = {
        "sellerName": "",
    }
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "sellerName": {
                        "label_text": "Vendor",
                        "region": {"x": 0.05, "y": 0.08, "width": 0.5, "height": 0.18},
                    }
                },
            }
        ]
    }
    ocr_results = [
        [
            ("Vendor", 30, 18, 55, 18),
            ("Acme", 94, 18, 42, 18),
            ("Corp", 142, 18, 40, 18),
            ("Address", 190, 18, 68, 18),
            ("123", 268, 18, 30, 18),
            ("Main", 304, 18, 42, 18),
            ("St", 352, 18, 22, 18),
        ]
    ]

    refined = refine_critical_fields(extracted, template, ocr_results, [(500, 200)])

    assert refined["sellerName"] == "Acme Corp"


def test_refine_critical_fields_prefers_single_ocr_amount_candidate():
    extracted = {
        "subtotal": "$680.00",
        "totalAmount": "$680.00",
    }
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "totalAmount": {
                        "label_text": "Total",
                        "region": {"x": 0.55, "y": 0.7, "width": 0.2, "height": 0.08},
                    }
                },
            }
        ]
    }
    ocr_results = [
        [
            ("Total", 120, 140, 40, 18),
            ("$740.08", 300, 142, 70, 18),
        ]
    ]

    refined = refine_critical_fields(extracted, template, ocr_results, [(500, 200)])

    assert refined["totalAmount"] == "$740.08"


def test_refine_critical_fields_keeps_extracted_value_when_ocr_amount_is_ambiguous():
    extracted = {
        "totalAmount": "$217.48",
    }
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "totalAmount": {
                        "label_text": "Total",
                        "region": {"x": 0.55, "y": 0.7, "width": 0.3, "height": 0.12},
                    }
                },
            }
        ]
    }
    ocr_results = [
        [
            ("$195.00", 280, 142, 70, 18),
            ("$217.48", 360, 142, 70, 18),
        ]
    ]

    refined = refine_critical_fields(extracted, template, ocr_results, [(500, 200)])

    assert refined["totalAmount"] == "$217.48"


def test_refine_critical_fields_recovers_split_decimal_total_candidate():
    extracted = {
        "subtotal": "$930.00",
        "totalAmount": "8",
    }
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "totalAmount": {
                        "label_text": "Total",
                        "region": {"x": 0.55, "y": 0.7, "width": 0.3, "height": 0.12},
                    }
                },
            }
        ]
    }
    ocr_results = [
        [
            ("Total", 120, 140, 40, 18),
            ("990", 300, 142, 40, 18),
            (".08", 345, 142, 30, 18),
        ]
    ]

    refined = refine_critical_fields(extracted, template, ocr_results, [(500, 200)])

    assert refined["totalAmount"] == "990.08"


def test_refine_critical_fields_prefers_single_invoice_number_candidate():
    extracted = {
        "invoiceNumber": "INV-OLD-999",
    }
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "invoiceNumber": {
                        "label_text": "Invoice Number",
                        "region": {"x": 0.45, "y": 0.15, "width": 0.3, "height": 0.08},
                    }
                },
            }
        ]
    }
    ocr_results = [
        [
            ("Invoice", 40, 30, 50, 20),
            ("Number", 95, 30, 60, 20),
            ("INV-2023-0042", 230, 32, 120, 20),
        ]
    ]

    refined = refine_critical_fields(extracted, template, ocr_results, [(500, 200)])

    assert refined["invoiceNumber"] == "INV-2023-0042"


def test_refine_critical_fields_uses_reconciliation_to_pick_total_candidate():
    extracted = {
        "subtotal": "$195.00",
        "tax": "$12.48",
        "totalAmount": "$195.00",
    }
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "totalAmount": {
                        "label_text": "Total",
                        "region": {"x": 0.55, "y": 0.7, "width": 0.3, "height": 0.12},
                    }
                },
            }
        ]
    }
    ocr_results = [
        [
            ("Total", 120, 140, 40, 18),
            ("$195.00", 280, 142, 70, 18),
            ("$217.48", 360, 142, 70, 18),
        ]
    ]

    refined = refine_critical_fields(extracted, template, ocr_results, [(500, 200)])

    assert refined["totalAmount"] == "$217.48"


def test_refine_critical_fields_reconciles_total_when_current_value_is_below_subtotal():
    extracted = {
        "subtotal": "$4200.00",
        "tax": "$464.00",
        "totalAmount": "$1600.00",
    }
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "totalAmount": {
                        "label_text": "Total",
                        "region": {"x": 0.75, "y": 0.7, "width": 0.2, "height": 0.1},
                    }
                },
            }
        ]
    }
    ocr_results = [
        [
            ("$800.00", 380, 140, 70, 18),
            ("$1600.00", 460, 140, 80, 18),
        ]
    ]

    refined = refine_critical_fields(extracted, template, ocr_results, [(600, 200)])

    assert refined["totalAmount"] == "4664.00"


def test_refine_critical_fields_prefers_plausible_tax_candidate_over_tiny_scalar():
    extracted = {
        "subtotal": "$930.00",
        "tax": "8",
    }
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "tax": {
                        "label_text": "Tax",
                        "region": {"x": 0.55, "y": 0.62, "width": 0.3, "height": 0.12},
                    }
                },
            }
        ]
    }
    ocr_results = [
        [
            ("Tax", 120, 124, 30, 18),
            ("60", 300, 126, 20, 18),
            (".08", 325, 126, 25, 18),
        ]
    ]

    refined = refine_critical_fields(extracted, template, ocr_results, [(500, 200)])

    assert refined["tax"] == "60.08"


def test_refine_critical_fields_prefers_digit_heavy_invoice_candidate_over_year():
    extracted = {
        "invoiceNumber": "2022",
    }
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "invoiceNumber": {
                        "label_text": "Invoice Number",
                        "region": {"x": 0.45, "y": 0.15, "width": 0.4, "height": 0.08},
                    }
                },
            }
        ]
    }
    ocr_results = [
        [
            ("Invoice", 40, 30, 50, 20),
            ("Number", 95, 30, 60, 20),
            ("2022", 220, 32, 60, 20),
            ("4976", 310, 32, 60, 20),
        ]
    ]

    refined = refine_critical_fields(extracted, template, ocr_results, [(500, 200)])

    assert refined["invoiceNumber"] == "4976"


def test_refine_critical_fields_prefers_numeric_invoice_candidate_over_garbled_text():
    extracted = {
        "invoiceNumber": "SSO15",
    }
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "invoiceNumber": {
                        "label_text": "Invoice Number",
                        "region": {"x": 0.45, "y": 0.15, "width": 0.45, "height": 0.08},
                    }
                },
            }
        ]
    }
    ocr_results = [
        [
            ("Invoice", 40, 30, 50, 20),
            ("Number", 95, 30, 60, 20),
            ("SSO15", 210, 32, 70, 20),
            ("28608590", 320, 32, 90, 20),
        ]
    ]

    refined = refine_critical_fields(extracted, template, ocr_results, [(500, 200)])

    assert refined["invoiceNumber"] == "28608590"


def test_refine_critical_fields_accepts_short_numeric_invoice_candidate_over_alphanumeric_noise():
    extracted = {
        "invoiceNumber": "210Au",
    }
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "invoiceNumber": {
                        "label_text": "Invoice Number",
                        "region": {"x": 0.4, "y": 0.12, "width": 0.45, "height": 0.1},
                    }
                },
            }
        ]
    }
    ocr_results = [
        [
            ("210Au", 210, 32, 70, 20),
            ("Invoice", 40, 30, 50, 20),
            ("Number", 95, 30, 60, 20),
            ("131", 320, 32, 40, 20),
        ]
    ]

    refined = refine_critical_fields(extracted, template, ocr_results, [(500, 200)])

    assert refined["invoiceNumber"] == "131"


def test_refine_critical_fields_keeps_strong_current_invoice_number_when_only_ocr_garbage_is_found():
    extracted = {
        "invoiceNumber": "59482422",
    }
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "invoiceNumber": {
                        "label_text": "Invoice Number",
                        "region": {"x": 0.4, "y": 0.12, "width": 0.45, "height": 0.1},
                    }
                },
            }
        ]
    }
    ocr_results = [
        [
            ("Invoice", 40, 30, 50, 20),
            ("Number", 95, 30, 60, 20),
            ("e1es", 320, 32, 45, 20),
        ]
    ]

    refined = refine_critical_fields(extracted, template, ocr_results, [(500, 200)])

    assert refined["invoiceNumber"] == "59482422"


def test_refine_critical_fields_recovers_seller_name_from_labeled_row_when_current_value_is_noisy():
    extracted = {
        "sellerName": "SUMMARY Seller: ITEMS Date Invoice Tax IBAN: Lewisfurt, 244 Little No. PLC Tee Together SMALL",
    }
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "sellerName": {
                        "label_text": "Seller",
                        "region": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.1},
                    }
                },
            }
        ]
    }
    ocr_results = [
        [
            ("SUMMARY", 20, 20, 70, 20),
            ("Seller:", 100, 20, 60, 20),
            ("Lewisfurt,", 170, 20, 80, 20),
            ("244", 260, 20, 30, 20),
            ("Little", 300, 20, 55, 20),
            ("PLC", 365, 20, 35, 20),
            ("Date", 420, 20, 40, 20),
        ]
    ]

    refined = refine_critical_fields(extracted, template, ocr_results, [(600, 200)])

    assert refined["sellerName"] == "Little PLC"


def test_refine_critical_fields_recovers_customer_name_from_labeled_row_when_current_value_is_missing():
    extracted = {
        "customerName": "",
    }
    template = {
        "pages": [
            {
                "page_index": 0,
                "fields": {
                    "customerName": {
                        "label_text": "Bill To",
                        "region": {"x": 0.1, "y": 0.2, "width": 0.8, "height": 0.1},
                    }
                },
            }
        ]
    }
    ocr_results = [
        [
            ("Bill", 20, 40, 30, 20),
            ("To", 55, 40, 20, 20),
            ("Moreno,", 120, 40, 65, 20),
            ("Montoya", 195, 40, 70, 20),
            ("and", 275, 40, 30, 20),
            ("Washington", 315, 40, 100, 20),
            ("Invoice", 430, 40, 50, 20),
        ]
    ]

    refined = refine_critical_fields(extracted, template, ocr_results, [(600, 200)])

    assert refined["customerName"] == "Moreno, Montoya and Washington"
