import numpy as np

from invoice_router.config import (
    AppConfig,
    DiscoveryConfig,
    FingerprintingConfig,
    HeuristicDiscoveryConfig,
    OcrConfig,
    ProcessingConfig,
    QualityConfig,
    TableDetectionConfig,
    TemplateLifecycleConfig,
    ValidationConfig,
)
from invoice_router.extraction.heuristics.discovery import (
    _build_row_blocks,
    _clean_party_candidate,
    _cleanup_party_name_overlap,
    _collect_summary_candidates,
    _extract_amount_candidates_from_row,
    _extract_summary_row_amount,
    _fallback_field_value_and_bbox,
    _invoice_candidate_score,
    _merge_page_heuristic_diagnostics,
    _reconcile_party_names,
    _reconcile_summary_amounts,
    _summary_concepts_for_row,
    expected_scalar_fields,
    group_tokens_into_rows,
    infer_field_type,
    infer_template_heuristic,
    infer_value_region,
    normalize_ocr_text,
    score_label_candidate,
)
from invoice_router.extraction.heuristics.party import _extract_party_name_from_tokens
from invoice_router.extraction.heuristics.summary import line_item_amount_total


def _config() -> AppConfig:
    return AppConfig(
        validation=ValidationConfig(),
        fingerprinting=FingerprintingConfig(),
        template_lifecycle=TemplateLifecycleConfig(),
        quality=QualityConfig(),
        processing=ProcessingConfig(worker_concurrency=2),
        region_buffer_pixels=5,
        ocr=OcrConfig(),
        table_detection=TableDetectionConfig(),
        discovery=DiscoveryConfig(),
        heuristic_discovery=HeuristicDiscoveryConfig(
            min_label_score=0.65,
            max_region_width_fraction=0.4,
            address_region_height_multiplier=4.0,
            enable_table_detection=True,
            min_table_header_score=0.75,
        ),
        field_mapping={},
    )


def test_normalize_ocr_text_collapses_punctuation_and_case():
    assert normalize_ocr_text("Invoice No.:") == "invoice no"


def test_score_label_candidate_prefers_exact_alias_match():
    exact = score_label_candidate(
        "Invoice No.", "invoice no", has_trailing_colon=False, label_x=20, page_w=1000
    )
    fuzzy = score_label_candidate(
        "Inv N0", "invoice no", has_trailing_colon=False, label_x=20, page_w=1000
    )

    assert exact > fuzzy
    assert exact > 0.8


def test_score_label_candidate_handles_common_ocr_corruption():
    score = score_label_candidate(
        "1nvoi ce:",
        "invoice",
        has_trailing_colon=True,
        label_x=30,
        label_y=40,
        page_w=1000,
        page_h=1400,
        field_name="invoiceNumber",
    )

    assert score > 0.75


def test_score_label_candidate_penalizes_generic_false_positive():
    generic = score_label_candidate(
        "Company",
        "invoice number",
        label_x=30,
        label_y=40,
        page_w=1000,
        page_h=1400,
        field_name="invoiceNumber",
    )

    assert generic < 0.5


def test_score_label_candidate_requires_invoice_number_cues_for_invoice_number():
    plain_invoice = score_label_candidate(
        "INVOICE",
        "invoice #",
        label_x=30,
        label_y=40,
        page_w=1000,
        page_h=1400,
        field_name="invoiceNumber",
    )
    labeled_invoice = score_label_candidate(
        "Invoice#:",
        "invoice #",
        has_trailing_colon=True,
        label_x=30,
        label_y=40,
        page_w=1000,
        page_h=1400,
        field_name="invoiceNumber",
    )

    assert labeled_invoice > plain_invoice
    assert plain_invoice < 0.8


def test_score_label_candidate_penalizes_tax_id_for_tax_field():
    tax_label = score_label_candidate(
        "Tax:",
        "tax",
        has_trailing_colon=True,
        label_x=30,
        label_y=40,
        page_w=1000,
        page_h=1400,
        field_name="tax",
    )
    tax_id_label = score_label_candidate(
        "Tax Id:",
        "tax",
        has_trailing_colon=True,
        label_x=30,
        label_y=40,
        page_w=1000,
        page_h=1400,
        field_name="tax",
    )

    assert tax_label > tax_id_label
    assert tax_id_label == 0.0


def test_score_label_candidate_prefers_issue_date_over_due_date_for_invoice_date():
    issue_score = score_label_candidate(
        "Date of issue:",
        "date",
        has_trailing_colon=True,
        label_x=40,
        label_y=60,
        page_w=1000,
        page_h=1400,
        field_name="invoiceDate",
    )
    due_score = score_label_candidate(
        "Date due:",
        "date",
        has_trailing_colon=True,
        label_x=40,
        label_y=60,
        page_w=1000,
        page_h=1400,
        field_name="invoiceDate",
    )

    assert issue_score > due_score


def test_group_tokens_into_rows_clusters_by_vertical_position():
    rows = group_tokens_into_rows(
        [
            ("Invoice", 10, 10, 40, 12),
            ("No.", 60, 12, 20, 12),
            ("Total", 10, 80, 35, 12),
        ]
    )

    assert len(rows) == 2
    assert [token[0] for token in rows[0]] == ["Invoice", "No."]


def test_infer_value_region_prefers_right_of_label_for_scalar_fields():
    region = infer_value_region(
        (100, 50, 80, 20), 1000, 1400, "invoiceNumber", anchor_direction="right_of_label"
    )

    assert region["x"] > 0.18
    assert region["y"] <= 0.05
    assert region["width"] <= 0.45


def test_infer_value_region_expands_downward_for_address_like_fields():
    region = infer_value_region(
        (100, 100, 80, 20), 1000, 1400, "customerName", anchor_direction="below_label"
    )

    assert region["y"] > 0.08
    assert region["height"] > 0.04


def test_infer_field_type_uses_known_scalar_hints():
    assert infer_field_type("invoiceDate") == "date"
    assert infer_field_type("total") == "currency"
    assert infer_field_type("invoiceNumber") == "string"


def test_expected_scalar_fields_prefers_ground_truth_key_shape_when_available():
    fields = expected_scalar_fields({"Invoice No.": "INV-42", "Total": "100.00"})

    assert "Invoice No." in fields
    assert "Total" in fields


def test_infer_template_heuristic_emits_scalar_template_from_ocr_labels():
    page = np.zeros((1000, 800, 3), dtype=np.uint8)
    ocr = [
        [
            ("Invoice", 40, 40, 70, 20),
            ("No.", 120, 40, 35, 20),
            ("INV-42", 190, 40, 90, 20),
            ("Date", 40, 90, 45, 20),
            ("2026-04-14", 190, 90, 110, 20),
            ("Total", 500, 800, 60, 24),
            ("100.00", 610, 800, 80, 24),
        ]
    ]

    template, confidence, extracted, diagnostics = infer_template_heuristic(
        [page],
        {"Invoice No.": "INV-42", "Date": "2026-04-14", "Total": "100.00"},
        _config(),
        ocr_results_per_page=ocr,
    )

    assert template["inference_method"] == "heuristic_discovery"
    assert template["pages"][0]["table"] is None
    assert set(template["pages"][0]["fields"].keys()) >= {"Invoice No.", "Date", "Total"}
    assert extracted["Invoice No."] == "INV-42"
    assert extracted["Date"] == "2026-04-14"
    assert extracted["Total"] == "100.00"
    assert confidence > 0.5
    assert diagnostics["matched_label_count"] >= 3


def test_infer_template_heuristic_keeps_same_row_scalar_labels_from_bleeding_together():
    page = np.zeros((1200, 1200, 3), dtype=np.uint8)
    ocr = [
        [
            ("INVOICE", 130, 347, 90, 24),
            ("NUMBER", 235, 347, 105, 24),
            ("INVOICE", 470, 347, 90, 24),
            ("DATE", 575, 347, 60, 24),
            ("DUE", 760, 347, 50, 24),
            ("DATE", 820, 347, 60, 24),
            ("692367", 140, 398, 90, 24),
            ("MAY", 470, 398, 65, 24),
            ("30,", 545, 398, 40, 24),
            ("2021", 595, 398, 70, 24),
            ("JUN", 760, 398, 60, 24),
            ("30,", 830, 398, 40, 24),
            ("2021", 880, 398, 70, 24),
        ]
    ]

    _template, _confidence, extracted, _diagnostics = infer_template_heuristic(
        [page],
        {"invoiceNumber": "692367", "invoiceDate": "2021-05-30", "dueDate": "2021-06-30"},
        _config(),
        ocr_results_per_page=ocr,
    )

    assert extracted["invoiceNumber"] == "692367"
    assert extracted["invoiceDate"] == "MAY 30, 2021"


def test_infer_template_heuristic_detects_obvious_table_rows():
    page = np.zeros((1000, 800, 3), dtype=np.uint8)
    ocr = [
        [
            ("Description", 40, 300, 100, 20),
            ("Qty", 320, 300, 40, 20),
            ("Amount", 600, 300, 80, 20),
            ("Widget", 40, 340, 80, 20),
            ("2", 330, 340, 10, 20),
            ("10.00", 610, 340, 60, 20),
            ("Service", 40, 380, 90, 20),
            ("1", 330, 380, 10, 20),
            ("15.00", 610, 380, 60, 20),
            ("Total", 560, 850, 60, 24),
            ("25.00", 650, 850, 70, 24),
        ]
    ]

    template, _confidence, extracted, diagnostics = infer_template_heuristic(
        [page],
        {"Total": "25.00", "lineItems": [{"description": "Widget"}, {"description": "Service"}]},
        _config(),
        ocr_results_per_page=ocr,
    )

    assert template["pages"][0]["table"] is not None
    assert len(extracted["line_items"]) == 2
    assert extracted["line_items"][0]["qty"] == "2"
    assert extracted["line_items"][0]["amount"] == "10.00"
    assert diagnostics["table_detected"] is True


def test_infer_template_heuristic_merges_header_table_continuation_rows():
    page = np.zeros((1200, 1000, 3), dtype=np.uint8)
    ocr = [
        [
            ("Description", 40, 300, 100, 20),
            ("Qty", 320, 300, 40, 20),
            ("Amount", 760, 300, 80, 20),
            ("1.", 40, 340, 18, 20),
            ("Widget", 90, 340, 70, 20),
            ("Pro", 170, 340, 40, 20),
            ("2", 330, 340, 12, 20),
            ("10.00", 790, 340, 60, 20),
            ("Extended", 90, 374, 80, 20),
            ("support", 180, 374, 70, 20),
            ("2.", 40, 420, 18, 20),
            ("Service", 90, 420, 80, 20),
            ("Plan", 180, 420, 50, 20),
            ("1", 330, 420, 12, 20),
            ("15.00", 790, 420, 60, 20),
            ("Subtotal", 700, 900, 100, 20),
            ("25.00", 820, 900, 80, 20),
        ]
    ]

    _template, _confidence, extracted, _diagnostics = infer_template_heuristic(
        [page],
        {
            "subtotal": "25.0",
            "lineItems": [
                {"description": "Widget Pro Extended support"},
                {"description": "Service Plan"},
            ],
        },
        _config(),
        ocr_results_per_page=ocr,
    )

    assert len(extracted["line_items"]) == 2
    assert extracted["line_items"][0]["description"] == "1. Widget Pro Extended support"


def test_infer_template_heuristic_merges_numeric_model_continuation_row():
    page = np.zeros((1200, 1000, 3), dtype=np.uint8)
    ocr = [
        [
            ("Description", 40, 300, 100, 20),
            ("Qty", 320, 300, 40, 20),
            ("Amount", 760, 300, 80, 20),
            ("1", 40, 340, 18, 20),
            ("Dell", 90, 340, 60, 20),
            ("Optiplex", 160, 340, 90, 20),
            ("All", 220, 340, 30, 20),
            ("in", 255, 340, 20, 20),
            ("One", 280, 340, 40, 20),
            ("5,00", 340, 340, 40, 20),
            ("649,95", 790, 340, 60, 20),
            ("780", 90, 374, 40, 20),
            ("3.0GHz", 140, 374, 70, 20),
            ("4GB", 220, 374, 40, 20),
            ("250GB", 270, 374, 60, 20),
            ("19", 235, 404, 20, 20),
            ("LCD", 265, 404, 40, 20),
            ("Subtotal", 700, 900, 100, 20),
            ("649,95", 820, 900, 80, 20),
        ]
    ]

    _template, _confidence, extracted, _diagnostics = infer_template_heuristic(
        [page],
        {
            "subtotal": "649.95",
            "lineItems": [{"description": "Dell Optiplex All in One 780 3.0GHz 4GB 250GB 19 LCD"}],
        },
        _config(),
        ocr_results_per_page=ocr,
    )

    assert len(extracted["line_items"]) == 1
    assert (
        extracted["line_items"][0]["description"]
        == "1 Dell Optiplex All in One 780 3.0GHz 4GB 250GB 19 LCD"
    )


def test_infer_template_heuristic_prefers_net_amount_when_taxed_row_includes_gross():
    page = np.zeros((1600, 1400, 3), dtype=np.uint8)
    ocr = [
        [
            ("Description", 80, 320, 120, 24),
            ("Qty", 920, 320, 40, 24),
            ("Amount", 1160, 320, 90, 24),
            ("2.", 80, 380, 28, 24),
            ("Nintendo", 140, 380, 100, 24),
            ("64", 250, 380, 30, 24),
            ("Console", 290, 380, 90, 24),
            ("with", 390, 380, 40, 24),
            ("Box", 440, 380, 50, 24),
            ("4,00", 930, 380, 60, 24),
            ("each", 1000, 380, 50, 24),
            ("120,00", 1140, 380, 90, 24),
            ("480,00", 1190, 380, 90, 24),
            ("10%", 1290, 380, 50, 24),
            ("528,00", 1350, 380, 90, 24),
            ("Subtotal", 1100, 1200, 100, 24),
            ("480,00", 1230, 1200, 90, 24),
        ]
    ]

    _template, _confidence, extracted, _diagnostics = infer_template_heuristic(
        [page],
        {"subtotal": "480.00", "lineItems": [{"description": "Nintendo 64 Console with Box"}]},
        _config(),
        ocr_results_per_page=ocr,
    )

    assert len(extracted["line_items"]) == 1
    assert extracted["line_items"][0]["qty"] == "4,00 each"
    assert extracted["line_items"][0]["price"] == "120,00"
    assert extracted["line_items"][0]["amount"] == "480,00"
    assert extracted["line_items"][0]["gross amount"] == "528,00"
    assert extracted["line_items"][0]["tax rate"] == "10%"


def test_infer_template_heuristic_prefers_unit_times_qty_when_taxed_row_only_has_net_total():
    page = np.zeros((1400, 1200, 3), dtype=np.uint8)
    ocr = [
        [
            ("Description", 80, 320, 120, 24),
            ("Qty", 760, 320, 40, 24),
            ("Amount", 980, 320, 90, 24),
            ("1.", 80, 380, 28, 24),
            ("Nicole", 140, 380, 80, 24),
            ("Dress", 230, 380, 70, 24),
            ("3,00", 770, 380, 60, 24),
            ("each", 840, 380, 50, 24),
            ("1,99", 940, 380, 70, 24),
            ("5,97", 1060, 380, 70, 24),
            ("10%", 1140, 380, 50, 24),
            ("Subtotal", 900, 1100, 100, 24),
            ("5,97", 1020, 1100, 70, 24),
        ]
    ]

    _template, _confidence, extracted, _diagnostics = infer_template_heuristic(
        [page],
        {"subtotal": "5.97", "lineItems": [{"description": "Nicole Dress"}]},
        _config(),
        ocr_results_per_page=ocr,
    )

    assert len(extracted["line_items"]) == 1
    assert extracted["line_items"][0]["qty"] == "3,00 each"
    assert extracted["line_items"][0]["price"] == "1,99"
    assert extracted["line_items"][0]["amount"] == "5,97"
    assert extracted["line_items"][0].get("gross amount") is None
    assert extracted["line_items"][0]["tax rate"] == "10%"


def test_infer_template_heuristic_repairs_merged_total_token_on_orphan_measure_row():
    page = np.zeros((1500, 1200, 3), dtype=np.uint8)
    ocr = [
        [
            ("Description", 60, 280, 120, 24),
            ("Qty", 820, 280, 40, 24),
            ("Price", 900, 280, 70, 24),
            ("2", 60, 340, 18, 24),
            ("Permabond", 110, 340, 110, 24),
            ("105", 230, 340, 40, 24),
            ("Cyanoacrylate", 110, 392, 130, 24),
            ("1-oz", 250, 392, 50, 24),
            ("10", 310, 392, 25, 24),
            ("bottles/case", 346, 392, 120, 24),
            ("$150.00", 860, 392, 90, 24),
            ("9735", 1010, 392, 70, 24),
            ("5", 820, 392, 20, 24),
            ("Subtotal", 920, 1180, 100, 24),
            ("735", 1040, 1180, 60, 24),
        ]
    ]

    _template, _confidence, extracted, _diagnostics = infer_template_heuristic(
        [page],
        {
            "subtotal": "735.0",
            "lineItems": [{"description": "Permabond 105 Cyanoacrylate 1-oz 10 bottles/case"}],
        },
        _config(),
        ocr_results_per_page=ocr,
    )

    assert len(extracted["line_items"]) == 1
    assert (
        extracted["line_items"][0]["description"]
        == "2 Permabond 105 Cyanoacrylate 1-oz 10 bottles/case"
    )
    assert extracted["line_items"][0]["amount"] == "735"


def test_infer_template_heuristic_extracts_party_name_from_labeled_block():
    page = np.zeros((1000, 800, 3), dtype=np.uint8)
    ocr = [
        [
            ("Bill", 40, 120, 35, 18),
            ("To:", 80, 120, 25, 18),
            ("Acme", 40, 155, 55, 18),
            ("Labs", 100, 155, 45, 18),
            ("billing@acme.com", 40, 188, 120, 18),
            ("123", 40, 220, 25, 18),
            ("High", 70, 220, 35, 18),
            ("Street", 110, 220, 45, 18),
        ]
    ]

    template, _confidence, extracted, _diagnostics = infer_template_heuristic(
        [page],
        {"customerName": "Acme Labs"},
        _config(),
        ocr_results_per_page=ocr,
    )

    assert template["pages"][0]["fields"]["customerName"]["label_text"] == "Bill To:"
    assert extracted["customerName"] == "Acme Labs"


def test_infer_template_heuristic_splits_shared_seller_and_client_rows():
    page = np.zeros((1800, 1400, 3), dtype=np.uint8)
    ocr = [
        [
            ("Seller:", 120, 180, 90, 24),
            ("Client:", 760, 180, 90, 24),
            ("Little", 120, 230, 80, 24),
            ("PLC", 210, 230, 45, 24),
            ("Moreno,", 760, 230, 110, 24),
            ("Montoya", 880, 230, 120, 24),
            ("and", 1010, 230, 45, 24),
            ("Washington", 1060, 230, 150, 24),
        ]
    ]

    _template, _confidence, extracted, _diagnostics = infer_template_heuristic(
        [page],
        {"sellerName": "Little PLC", "customerName": "Moreno, Montoya and Washington"},
        _config(),
        ocr_results_per_page=ocr,
    )

    assert extracted["sellerName"] == "Little PLC"
    assert extracted["customerName"] == "Moreno, Montoya and Washington"


def test_infer_template_heuristic_splits_same_block_seller_and_client_with_inline_noise():
    page = np.zeros((1800, 1400, 3), dtype=np.uint8)
    ocr = [
        [
            ("Seller:", 120, 180, 90, 24),
            ("Client:", 760, 180, 90, 24),
            ("Little", 120, 230, 80, 24),
            ("PLC", 210, 230, 45, 24),
            ("Moreno,", 760, 230, 110, 24),
            ("Montoya", 880, 230, 120, 24),
            ("and", 1010, 230, 45, 24),
            ("Washington", 1060, 230, 150, 24),
            ("billing@little.example", 120, 280, 220, 24),
            ("123", 760, 280, 40, 24),
            ("Main", 810, 280, 70, 24),
            ("St", 890, 280, 30, 24),
        ]
    ]

    _template, _confidence, extracted, _diagnostics = infer_template_heuristic(
        [page],
        {"sellerName": "Little PLC", "customerName": "Moreno, Montoya and Washington"},
        _config(),
        ocr_results_per_page=ocr,
    )

    assert extracted["sellerName"] == "Little PLC"
    assert extracted["customerName"] == "Moreno, Montoya and Washington"


def test_extract_party_name_from_tokens_prefers_segment_over_noisy_full_row():
    ocr = [
        [
            ("BILL", 120, 180, 55, 24),
            ("TO", 180, 180, 35, 24),
            ("BILL", 760, 180, 55, 24),
            ("FROM", 820, 180, 65, 24),
        ],
        [
            ("1865", 120, 230, 60, 24),
            ("Birchmount", 190, 230, 140, 24),
            ("Road", 340, 230, 70, 24),
            ("ACTIVE", 280, 230, 95, 24),
            ("EXHAUST", 385, 230, 105, 24),
            ("acctpay@activexhaust.com", 760, 230, 290, 24),
            ("ACTIVE", 1080, 230, 95, 24),
            ("EXHAUST", 1185, 230, 105, 24),
        ],
    ]

    seller = _extract_party_name_from_tokens("sellerName", (760, 180, 120, 24), ocr, 1400, 1800)

    assert seller == "ACTIVE EXHAUST"


def test_extract_party_name_from_tokens_stops_before_tax_id_metadata():
    ocr = [
        [
            ("Seller:", 120, 180, 90, 24),
        ],
        [
            ("Little", 120, 230, 80, 24),
            ("PLC", 210, 230, 45, 24),
        ],
        [
            ("Tax", 120, 280, 45, 24),
            ("Id:", 170, 280, 30, 24),
            ("938-72-1087", 210, 280, 160, 24),
        ],
    ]

    seller = _extract_party_name_from_tokens("sellerName", (120, 180, 90, 24), ocr, 800, 1000)

    assert seller == "Little PLC"


def test_infer_template_heuristic_recovers_unlabeled_party_blocks_and_summary_total():
    page = np.zeros((2000, 1600, 3), dtype=np.uint8)
    ocr = [
        [
            ("INVOICE", 650, 190, 260, 35),
            ("SWIFT", 94, 296, 87, 23),
            ("CANOE", 194, 296, 104, 23),
            ("COMPANY", 310, 296, 153, 23),
            ("INC.", 476, 296, 56, 23),
            ("14", 548, 297, 32, 22),
            ("Howard", 594, 297, 107, 22),
            ("Invoice#:", 1258, 296, 122, 23),
            ("26703", 1394, 296, 92, 23),
            ("Invoice", 1107, 334, 92, 22),
            ("date:", 1212, 333, 70, 23),
            ("Feb", 1295, 333, 48, 23),
            ("23,", 1357, 333, 44, 28),
            ("2022", 1414, 333, 72, 23),
            ("Subtotal", 1105, 1021, 112, 40),
            ("$4200.00", 1353, 1024, 138, 29),
            ("Sales", 1027, 1116, 71, 23),
            ("Tax", 1110, 1117, 50, 22),
            ("8%", 1173, 1116, 45, 24),
            ("$464.00", 1371, 1113, 120, 29),
            ("Lisa", 97, 1904, 61, 28),
            ("|", 160, 1900, 14, 37),
            ("DATA", 178, 1906, 74, 22),
            ("CIRCUITS", 265, 1906, 126, 23),
            ("Ltd.", 741, 1906, 47, 22),
        ]
    ]

    _template, _confidence, extracted, diagnostics = infer_template_heuristic(
        [page],
        {
            "invoiceNumber": "26703",
            "invoiceDate": "Feb 23, 2022",
            "sellerName": "SWIFT CANOE COMPANY INC.",
            "customerName": "Lisa",
            "subtotal": "4200.0",
            "tax": "464.0",
            "totalAmount": "4664.0",
        },
        _config(),
        ocr_results_per_page=ocr,
    )

    assert extracted["sellerName"] == "SWIFT CANOE COMPANY INC."
    assert extracted["customerName"].startswith("Lisa")
    assert extracted["tax"] == "$464.00"
    assert extracted["totalAmount"] == "4664.00"
    assert diagnostics["party_block_diagnostics"]


def test_infer_template_heuristic_trims_footer_noise_from_short_customer_name():
    page = np.zeros((2000, 1600, 3), dtype=np.uint8)
    ocr = [
        [
            ("INVOICE", 650, 190, 260, 35),
            ("SWIFT", 94, 296, 87, 23),
            ("CANOE", 194, 296, 104, 23),
            ("COMPANY", 310, 296, 153, 23),
            ("INC.", 476, 296, 56, 23),
            ("Invoice#:", 1258, 296, 122, 23),
            ("26703", 1394, 296, 92, 23),
            ("Subtotal", 1105, 1021, 112, 40),
            ("$4200.00", 1353, 1024, 138, 29),
            ("Lisa", 97, 1904, 61, 28),
            ("|", 160, 1900, 14, 37),
            ("DATA", 178, 1906, 74, 22),
            ("CIRCUITS", 265, 1906, 126, 23),
            ("Ltd.", 741, 1906, 47, 22),
        ]
    ]

    _template, _confidence, extracted, _diagnostics = infer_template_heuristic(
        [page],
        {
            "invoiceNumber": "26703",
            "sellerName": "SWIFT CANOE COMPANY INC.",
            "customerName": "Lisa",
            "subtotal": "4200.0",
            "totalAmount": "4664.0",
        },
        _config(),
        ocr_results_per_page=ocr,
    )

    assert extracted["customerName"] == "Lisa"


def test_clean_party_candidate_keeps_multi_word_company_name_when_not_pipe_contaminated():
    assert (
        _clean_party_candidate(
            "CONSUMERS PACKAGING GROUP INC.",
            prefer_short_prefix=True,
        )
        == "CONSUMERS PACKAGING GROUP INC."
    )


def test_infer_template_heuristic_ignores_giant_mixed_content_row_for_party_and_summary():
    page = np.zeros((2200, 1600, 3), dtype=np.uint8)
    ocr = [
        [
            ("INVOICE", 650, 190, 260, 35),
            ("SWIFT", 94, 296, 87, 23),
            ("CANOE", 194, 296, 104, 23),
            ("COMPANY", 310, 296, 153, 23),
            ("INC.", 476, 296, 56, 23),
            ("Invoice#:", 1258, 296, 122, 23),
            ("26703", 1394, 296, 92, 23),
            ("Lisa", 97, 1960, 61, 28),
            ("|", 160, 1956, 14, 37),
            ("DATA", 178, 1962, 74, 22),
            ("CIRCUITS", 265, 1962, 126, 23),
            ("Ltd.", 400, 1962, 47, 22),
            ("Subtotal", 480, 1960, 112, 24),
            ("$4200.00", 610, 1960, 138, 24),
            ("Sales", 770, 1960, 71, 23),
            ("Tax", 855, 1961, 50, 22),
            ("8%", 917, 1960, 45, 24),
            ("Invoice", 980, 1960, 92, 22),
            ("date:", 1080, 1960, 70, 23),
            ("Feb", 1160, 1960, 48, 23),
            ("23,", 1220, 1960, 44, 28),
            ("2022", 1274, 1960, 72, 23),
        ]
    ]

    _template, _confidence, extracted, diagnostics = infer_template_heuristic(
        [page],
        {
            "invoiceNumber": "26703",
            "customerName": "Lisa",
        },
        _config(),
        ocr_results_per_page=ocr,
    )

    assert extracted["invoiceNumber"] == "26703"
    assert extracted.get("customerName") is None
    assert diagnostics["summary_candidate_diagnostics"] == {}


def test_infer_template_heuristic_uses_row_fallback_for_headerless_line_items():
    page = np.zeros((1400, 1000, 3), dtype=np.uint8)
    ocr = [
        [
            ("1", 80, 420, 12, 20),
            ("Widget", 140, 420, 80, 20),
            ("A", 230, 420, 15, 20),
            ("2", 560, 420, 15, 20),
            ("$10.00", 820, 420, 90, 20),
            ("2", 80, 470, 12, 20),
            ("Service", 140, 470, 90, 20),
            ("Plan", 240, 470, 50, 20),
            ("1", 560, 470, 12, 20),
            ("$15.00", 820, 470, 90, 20),
            ("Subtotal", 720, 900, 100, 20),
            ("$25.00", 860, 900, 90, 20),
        ]
    ]

    template, _confidence, extracted, diagnostics = infer_template_heuristic(
        [page],
        {
            "lineItems": [{"description": "Widget A"}, {"description": "Service Plan"}],
            "subtotal": "25.0",
        },
        _config(),
        ocr_results_per_page=ocr,
    )

    assert template["pages"][0]["table"] is not None
    assert diagnostics["line_item_source"] == "row_fallback"
    assert len(extracted["line_items"]) == 2


def test_build_row_blocks_splits_party_table_and_summary_sections():
    rows = [
        [("Seller:", 140, 447, 90, 25), ("Client:", 700, 447, 90, 25)],
        [("Little", 144, 510, 80, 27), ("PLC", 230, 510, 50, 27), ("Moreno,", 800, 510, 110, 27)],
        [("244", 143, 542, 40, 35), ("Judith", 190, 542, 90, 35), ("Suite", 820, 542, 80, 35)],
        [("Id:", 141, 653, 35, 21), ("938-72-1087", 190, 653, 160, 21), ("Tax", 940, 653, 40, 21)],
        [("ITEMS", 136, 763, 110, 26)],
        [
            ("Description", 162, 849, 180, 44),
            ("Qty", 980, 849, 60, 44),
            ("price", 1080, 849, 80, 44),
            ("Gross", 1260, 849, 90, 44),
        ],
        [
            ("1.", 172, 936, 30, 44),
            ("Dress", 260, 936, 80, 44),
            ("2,00", 1000, 936, 70, 44),
            ("26,33", 1330, 936, 70, 44),
        ],
        [("SUMMARY", 135, 1428, 190, 26)],
        [
            ("Total", 503, 1632, 90, 20),
            ("246,04", 1180, 1632, 90, 20),
            ("24,60", 1300, 1632, 80, 20),
            ("270,64", 1420, 1632, 90, 20),
        ],
    ]

    blocks = _build_row_blocks(rows, page_h=1800)
    metadata_block_text = normalize_ocr_text(" ".join(token[0] for token in blocks[2]["rows"][0]))

    assert len(blocks) >= 6
    assert (
        normalize_ocr_text(" ".join(token[0] for token in blocks[0]["rows"][0])) == "seller client"
    )
    assert metadata_block_text.startswith("id")
    assert metadata_block_text.endswith("tax")
    assert normalize_ocr_text(" ".join(token[0] for token in blocks[3]["rows"][0])) == "items"
    assert normalize_ocr_text(" ".join(token[0] for token in blocks[-2]["rows"][0])) == "summary"


def test_collect_summary_candidates_ignores_table_total_header_and_prefers_total_due_row():
    rows = [
        [
            ("DESCRIPTION", 160, 1417, 220, 61),
            ("QUANTITY", 760, 1417, 180, 61),
            ("PRICE", 1260, 1417, 120, 61),
            ("TOTAL,", 1820, 1417, 160, 61),
        ],
        [("Subtotal", 1795, 2022, 190, 57), ("756", 2060, 2022, 90, 57)],
        [
            ("Sales", 1708, 2145, 110, 38),
            ("8%", 1835, 2145, 40, 38),
            ("Tax", 1890, 2145, 80, 38),
            ("60.24", 2040, 2145, 130, 38),
        ],
        [
            ("Total", 1760, 2383, 120, 51),
            ("Due", 1898, 2383, 90, 51),
            ("866.24", 2070, 2383, 150, 51),
        ],
    ]

    candidates = _collect_summary_candidates(rows, page_w=2400, page_h=2600)

    assert "invoice_total" in candidates
    assert len(candidates["invoice_total"]) == 1
    assert candidates["invoice_total"][0]["value"] == "866.24"
    assert candidates["subtotal"][0]["value"] == "756"
    assert candidates["tax"][0]["value"] == "60.24"


def test_collect_summary_candidates_recovers_tax_from_repeated_summary_triplet():
    rows = [
        [
            ("SUMMARY", 118, 1101, 160, 24),
            ("[%]", 310, 1101, 40, 24),
            ("worth", 390, 1101, 80, 24),
            ("Gross", 515, 1101, 80, 24),
            ("worth", 620, 1101, 80, 24),
            ("VAT", 735, 1101, 55, 24),
            ("Net", 820, 1101, 60, 24),
            ("VAT", 910, 1101, 55, 24),
            ("10%", 1005, 1101, 45, 24),
            ("374,95", 1095, 1101, 90, 24),
            ("37,49", 1210, 1101, 80, 24),
            ("412,44", 1310, 1101, 90, 24),
            ("Total", 118, 1160, 80, 24),
            ("$", 230, 1160, 20, 24),
            ("$", 270, 1160, 20, 24),
            ("$", 310, 1160, 20, 24),
            ("374,95", 1095, 1160, 90, 24),
            ("37,49", 1210, 1160, 80, 24),
            ("412,44", 1310, 1160, 90, 24),
        ]
    ]

    candidates = _collect_summary_candidates(rows, page_w=1600, page_h=2000)

    assert candidates["subtotal"][0]["value"] == "374,95"
    assert candidates["tax"][0]["value"] == "37,49"
    assert candidates["invoice_total"][0]["value"] == "412,44"


def test_infer_template_heuristic_prefers_invoice_total_over_amount_due_for_total_amount():
    page = np.zeros((1600, 1200, 3), dtype=np.uint8)
    ocr = [
        [
            ("Invoice", 80, 90, 80, 24),
            ("number", 170, 90, 100, 24),
            ("713867", 320, 90, 120, 24),
            ("SubTotal:", 820, 1180, 130, 30),
            ("$7106", 980, 1180, 100, 30),
            ("Total:", 820, 1245, 100, 30),
            ("$7106", 980, 1245, 100, 30),
            ("Amount", 780, 1310, 120, 30),
            ("due:", 910, 1310, 80, 30),
            ("$5015", 1000, 1310, 100, 30),
        ]
    ]

    _template, _confidence, extracted, _diagnostics = infer_template_heuristic(
        [page],
        {"invoiceNumber": "713867", "subtotal": "7106.0", "totalAmount": "7106.0"},
        _config(),
        ocr_results_per_page=ocr,
    )

    assert extracted["totalAmount"] == "$7106"


def test_reconcile_summary_amounts_uses_line_item_sum_when_total_matches_subtotal_only():
    extracted = {
        "subtotal": "581.95",
        "tax": "58.20",
        "totalAmount": "640.15",
        "line_items": [
            {"amount": "16.49"},
            {"amount": "528.00"},
            {"amount": "52.76"},
            {"amount": "42.90"},
        ],
    }

    from invoice_router.extraction.heuristics.discovery import _reconcile_summary_amounts

    _reconcile_summary_amounts(extracted)

    assert extracted["totalAmount"] == "698.35"


def test_line_item_amount_total_sums_numeric_amounts_and_ignores_missing_values():
    extracted = {
        "line_items": [
            {"amount": "16.49"},
            {"amount": None},
            {"amount": "528.00"},
            {"description": "ignore me"},
        ],
    }

    assert line_item_amount_total(extracted) == 544.49


def test_collect_summary_candidates_derives_tax_from_total_row_without_explicit_tax_label():
    rows = [
        [
            ("Total", 920, 1200, 90, 28),
            ("$", 1030, 1200, 20, 28),
            ("374,95", 1060, 1200, 100, 28),
            ("$", 1200, 1200, 20, 28),
            ("37,49", 1230, 1200, 90, 28),
            ("$", 1350, 1200, 20, 28),
            ("412,44", 1380, 1200, 100, 28),
        ]
    ]

    candidates = _collect_summary_candidates(rows, page_w=1800, page_h=1600)

    assert candidates["tax"]
    assert candidates["tax"][0]["value"].endswith("37,49")


def test_extract_amount_candidates_from_row_merges_fragmented_thousands_tokens():
    row = [
        ("Total", 900, 1200, 90, 28),
        ("$", 1030, 1200, 20, 28),
        ("28", 1060, 1200, 40, 28),
        ("929,32", 1105, 1200, 100, 28),
        ("$", 1230, 1200, 20, 28),
        ("2", 1260, 1200, 20, 28),
        ("892,93", 1285, 1200, 100, 28),
        ("$", 1410, 1200, 20, 28),
        ("31", 1440, 1200, 40, 28),
        ("822,25", 1485, 1200, 100, 28),
    ]

    candidates = _extract_amount_candidates_from_row(row)

    assert "28929,32" in candidates
    assert "2892,93" in candidates
    assert "31822,25" in candidates


def test_extract_summary_row_amount_ignores_hyphenated_id_like_numbers():
    row = [
        ("Tax", 980, 653, 40, 21),
        ("938-72-1087", 1035, 653, 160, 21),
        ("24.60", 1360, 653, 80, 21),
    ]

    candidate = _extract_summary_row_amount(row, page_w=1600, page_h=2000, field_hint="tax")

    assert candidate is not None
    assert candidate["value"] == "24.60"


def test_cleanup_party_name_overlap_removes_shared_business_word_from_customer():
    assert _cleanup_party_name_overlap("Pierce Group PLC", "Baldwin Group") == "Pierce PLC"


def test_cleanup_party_name_overlap_keeps_internal_and_when_it_belongs_to_name():
    assert (
        _cleanup_party_name_overlap(
            "Mills, Nelson and Taylor",
            "Pitts, Anderson and Mitchell",
        )
        == "Mills, Nelson and Taylor"
    )


def test_reconcile_party_names_trims_edge_words_that_leak_from_other_party():
    extracted = {
        "sellerName": "Mccormick, Hendricks and Powers",
        "customerName": "Hendricks and Poole, Hood and Landry Powers",
    }

    _reconcile_party_names(extracted)

    assert extracted["customerName"] == "Poole, Hood and Landry"


def test_clean_party_candidate_rejects_country_word():
    assert _clean_party_candidate("Canada") == ""


def test_reconcile_party_names_trims_overlap_and_leading_noise():
    extracted = {
        "sellerName": "GOODRICH CORPORATION",
        "customerName": "CORPORATION GOODRICH LANDING GEAR OE -",
    }

    _reconcile_party_names(extracted)

    assert extracted["customerName"] == "GOODRICH LANDING GEAR - OE"


def test_merge_page_heuristic_diagnostics_accumulates_page_level_context():
    party_block_diagnostics = [{"page_index": 0, "kind": "party"}]
    summary_candidate_diagnostics = {"subtotal": [{"value": "10.00"}]}
    page_diagnostics = {
        "party_block_diagnostics": [{"page_index": 1, "kind": "party"}],
        "summary_candidate_diagnostics": {
            "subtotal": [{"value": "12.00"}],
            "tax": [{"value": "2.00"}],
        },
        "line_item_source": "header_table",
    }

    line_item_source = _merge_page_heuristic_diagnostics(
        page_diagnostics,
        party_block_diagnostics,
        summary_candidate_diagnostics,
    )

    assert line_item_source == "header_table"
    assert party_block_diagnostics == [
        {"page_index": 0, "kind": "party"},
        {"page_index": 1, "kind": "party"},
    ]
    assert summary_candidate_diagnostics == {
        "subtotal": [{"value": "10.00"}, {"value": "12.00"}],
        "tax": [{"value": "2.00"}],
    }


def test_extract_party_name_from_tokens_uses_label_window_and_fallback_text():
    ocr = [
        [
            ("Bill", 40, 120, 35, 18),
            ("To:", 80, 120, 25, 18),
        ],
        [
            ("Acme", 40, 155, 55, 18),
            ("Labs", 100, 155, 45, 18),
        ],
        [
            ("billing@acme.com", 40, 188, 120, 18),
        ],
    ]

    result = _extract_party_name_from_tokens("customerName", (40, 120, 60, 18), ocr, 800, 1000)

    assert result == "Acme Labs"


def test_cleanup_party_name_overlap_reorders_trailing_acronym_after_dash():
    extracted = {
        "sellerName": "GOODRICH CORPORATION",
        "customerName": "CORPORATION GOODRICH LANDING GEAR OE -",
    }

    _reconcile_party_names(extracted)

    assert extracted["customerName"] == "GOODRICH LANDING GEAR - OE"


def test_fallback_summary_value_only_uses_rows_with_matching_summary_concept():
    rows = [
        [
            ("Invoice", 80, 120, 80, 20),
            ("No.", 170, 120, 40, 20),
            ("938-72-1087", 240, 120, 150, 20),
        ],
        [("Sales", 980, 653, 50, 21), ("Tax", 1035, 653, 40, 21), ("24.60", 1360, 653, 80, 21)],
    ]
    summary_candidates = _collect_summary_candidates(rows, page_w=1600, page_h=2000)

    value, _bbox = _fallback_field_value_and_bbox(
        "tax",
        rows,
        page_w=1600,
        page_h=2000,
        structural_context={"summary_candidates": summary_candidates},
    )

    assert value == "24.60"


def test_fallback_total_keeps_invoice_total_when_shipping_is_separate():
    rows = [
        [("Subtotal", 1795, 2022, 190, 57), ("756", 2060, 2022, 90, 57)],
        [
            ("Sales", 1708, 2145, 110, 38),
            ("8%", 1835, 2145, 40, 38),
            ("Tax", 1890, 2145, 80, 38),
            ("60.24", 2040, 2145, 130, 38),
        ],
        [("S&H", 1760, 2260, 80, 38), ("50", 2040, 2260, 90, 38)],
        [
            ("Total", 1760, 2383, 120, 51),
            ("Due", 1898, 2383, 90, 51),
            ("866.24", 2070, 2383, 150, 51),
        ],
    ]
    summary_candidates = _collect_summary_candidates(rows, page_w=2400, page_h=2600)

    value, _bbox = _fallback_field_value_and_bbox(
        "totalAmount",
        rows,
        page_w=2400,
        page_h=2600,
        structural_context={"summary_candidates": summary_candidates},
    )

    assert value == "866.24"


def test_fallback_invoice_number_accepts_short_value_near_invoice_label():
    rows = [
        [
            ("Invoice", 1180, 296, 90, 23),
            ("No.", 1280, 296, 60, 23),
            ("105", 1394, 296, 92, 23),
        ]
    ]

    value, _bbox = _fallback_field_value_and_bbox(
        "invoiceNumber",
        rows,
        page_w=1600,
        page_h=2000,
        structural_context={},
    )

    assert value == "105"


def test_fallback_invoice_number_prefers_short_digit_id_over_street_ordinal():
    rows = [
        [
            ("Invoice", 1180, 296, 90, 23),
            ("No.", 1280, 296, 60, 23),
            ("105", 1394, 296, 92, 23),
            ("167th", 1498, 296, 92, 23),
        ]
    ]

    value, _bbox = _fallback_field_value_and_bbox(
        "invoiceNumber",
        rows,
        page_w=1600,
        page_h=2000,
        structural_context={},
    )

    assert value == "105"


def test_invoice_candidate_score_penalizes_postal_code_shape():
    postal_score = _invoice_candidate_score("44131-250", (1394, 296, 92, 23), 1600, 2000)
    invoice_score = _invoice_candidate_score("105", (1394, 296, 92, 23), 1600, 2000)

    assert invoice_score > postal_score


def test_invoice_candidate_score_penalizes_street_ordinal_shape():
    ordinal_score = _invoice_candidate_score("167th", (1394, 296, 92, 23), 1600, 2000)
    invoice_score = _invoice_candidate_score("105", (1394, 296, 92, 23), 1600, 2000)

    assert invoice_score > ordinal_score


def test_infer_template_heuristic_uses_fallback_when_label_match_has_no_value():
    page = np.zeros((1000, 800, 3), dtype=np.uint8)
    ocr = [
        [
            ("Invoice", 40, 40, 70, 20),
            ("No.", 120, 40, 35, 20),
            ("Date", 190, 40, 45, 20),
            ("105", 280, 40, 40, 20),
        ]
    ]

    _template, _confidence, extracted, _diagnostics = infer_template_heuristic(
        [page],
        {"invoiceNumber": "105"},
        _config(),
        ocr_results_per_page=ocr,
    )

    assert extracted["invoiceNumber"] == "105"


def test_reconcile_summary_amounts_derives_missing_tax_from_total_and_subtotal():
    extracted = {
        "subtotal": "4200.00",
        "totalAmount": "4664.00",
    }

    _reconcile_summary_amounts(extracted)

    assert extracted["tax"] == "464.00"


def test_summary_concepts_do_not_treat_subtotal_row_as_invoice_total():
    assert _summary_concepts_for_row("Sub Total 2337 :") == ["subtotal"]


def test_collect_summary_candidates_derives_tax_from_total_triplet_row():
    rows = [
        [
            ("Total", 1180, 1632, 90, 20),
            ("80,96", 1320, 1632, 90, 20),
            ("8,10", 1420, 1632, 80, 20),
            ("89,06", 1520, 1632, 90, 20),
        ]
    ]

    candidates = _collect_summary_candidates(rows, page_w=1800, page_h=2000)

    assert candidates["subtotal"][0]["value"] == "80,96"
    assert candidates["invoice_total"][0]["value"] == "89,06"
    assert candidates["tax"][0]["value"] == "8,10"
