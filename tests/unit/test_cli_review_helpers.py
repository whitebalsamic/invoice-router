from invoice_router.cli.main import (
    _display_status_for_result,
    _format_failed_result_line,
    _invoice_date_review_signal,
    _review_signals_for_result,
)
from invoice_router.models import (
    DocumentFamily,
    ExtractionStrategy,
    ProcessingDiagnostics,
    ProcessingResult,
    Provenance,
    Route,
    SourceFormat,
)


def _build_result(
    *,
    validation_passed=None,
    route=Route.APPLY,
    raw_date=None,
    normalized_date=None,
    country_code="CA",
    validation_errors=None,
):
    extracted_data = {}
    if raw_date is not None:
        extracted_data["invoiceDate"] = raw_date

    normalized_data = {}
    if normalized_date is not None:
        normalized_data["invoice_date"] = normalized_date

    diagnostics = ProcessingDiagnostics(
        attempted_route=route,
        extraction_strategy=ExtractionStrategy.ocr_structured,
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code=country_code,
        currency_code="CAD" if country_code == "CA" else "USD",
        validation_score=0.82 if validation_errors else 1.0 if validation_passed else None,
        validation_errors=validation_errors or [],
    )
    provenance = Provenance(
        request_id="req-1",
        route=route,
        fingerprint_hash=None,
        extraction_strategy=ExtractionStrategy.ocr_structured,
        source_format=SourceFormat.image,
        document_family=DocumentFamily.invoice,
        country_code=country_code,
        currency_code="CAD" if country_code == "CA" else "USD",
        provider_name="Demo Supplier",
        provider_confidence=0.9,
        template_status_at_time=None,
        template_confidence_at_time=None,
        ocr_engine="tesseract",
        inference_method="heuristic",
        image_quality_score=[0.9],
        quality_flag=False,
        input_document_hash="hash",
        per_page_hashes=[],
        extraction_output_hash=None,
        started_at="2026-01-01T00:00:00Z",
        completed_at="2026-01-01T00:00:01Z",
        latency_ms=1000,
    )
    return ProcessingResult(
        invoice_path="/tmp/invoice.png",
        fingerprint_hash=None,
        extracted_data=extracted_data,
        normalized_data=normalized_data,
        ground_truth=None,
        provenance=provenance,
        validation_passed=validation_passed,
        route_used=route,
        attempted_route=route,
        diagnostics=diagnostics,
        image_quality_score=0.9,
        template_status_at_time=None,
        processed_at="2026-01-01T00:00:01Z",
    )


def test_invoice_date_review_signal_reports_ambiguous_numeric_dates():
    result = _build_result(raw_date="03/05/2020", normalized_date="2020-05-03", country_code="CA")

    assert _invoice_date_review_signal(result) == "ambiguous date 03/05/2020 -> 2020-05-03 (CA)"


def test_review_signals_include_simplified_validation_errors():
    result = _build_result(
        validation_passed=False,
        route=Route.REJECTED,
        validation_errors=["Total mismatch: expected subtotal+tax 100.00, got 120.00"],
    )

    signals = _review_signals_for_result(result)

    assert "total does not match subtotal + tax" in signals


def test_display_status_marks_ambiguous_no_gt_result_as_check():
    result = _build_result(
        validation_passed=None,
        raw_date="03/05/2020",
        normalized_date="2020-05-03",
        country_code="CA",
    )

    assert _display_status_for_result(result) == "CHECK"


def test_format_failed_result_line_surfaces_route_score_and_reason():
    result = _build_result(
        validation_passed=False,
        route=Route.REJECTED,
        raw_date="03/05/2020",
        normalized_date="2020-05-03",
        country_code="CA",
        validation_errors=["Missing extracted field: totalAmount"],
    )

    line = _format_failed_result_line(result)

    assert "invoice.png" in line
    assert "route=REJECTED" in line
    assert "country=CA" in line
    assert "why=ambiguous date 03/05/2020 -> 2020-05-03 (CA); missing totalAmount" in line
