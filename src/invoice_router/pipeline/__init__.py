import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import AppConfig, Settings
from ..domain.ground_truth import is_ground_truth_discovery_ready
from ..domain.invoices.classification import build_document_context
from ..domain.invoices.family_profiles import (
    build_family_extraction_profile,
    merge_family_extraction_profiles,
)
from ..domain.invoices.provider_resolution import resolve_provider
from ..domain.templates.fingerprinting import (
    compute_document_fingerprint,
    compute_page_fingerprint,
    extract_document_anchor_summary,
    run_full_page_ocr,
)
from ..domain.templates.lifecycle import (
    record_rejection,
    record_template_family_rejection,
    resolve_gt_healing_authority,
    update_template_confidence,
    update_template_family_confidence,
)
from ..domain.templates.routing import (
    determine_extraction_strategy,
    lookup_fingerprint,
    lookup_template_family,
    route_invoice,
)
from ..domain.validation.field_resolver import refine_critical_fields
from ..domain.validation.normalizer import merge_for_validation, normalize_extracted_invoice
from ..domain.validation.validator import should_reject, validate_invoice
from ..extraction.ocr import extract_with_ocr
from ..extraction.pdf_text import extract_from_text_pdf, load_pdf_page_texts
from ..extraction.template_inference import infer_template
from ..infrastructure.filesystem.source import pair_ground_truth
from ..infrastructure.persistence.redis import get_redis_client
from ..infrastructure.persistence.storage import FingerprintDB
from ..models import (
    DocumentContext,
    ExtractionStrategy,
    FingerprintRecord,
    PageFingerprint,
    ProcessingDiagnostics,
    ProcessingResult,
    Provenance,
    Route,
    SourceFormat,
    TemplateFamilyExample,
    TemplateFamilyRecord,
    TemplateStatus,
)
from ..provenance import compute_hash, compute_string_hash, generate_request_id, get_utc_now
from .preprocessing import normalize_page
from .quality import apply_quality_gate, assess_quality

logger = logging.getLogger(__name__)


class ProcessingStageError(Exception):
    def __init__(self, stage: str, attempted_route: Optional[Route], original: Exception):
        super().__init__(str(original))
        self.stage = stage
        self.attempted_route = attempted_route
        self.original = original


@dataclass
class PreparedInvoiceInputs:
    input_document_hash: str
    pages_img: List[Any]
    quality_metrics: List[Any]
    quality_flag: bool
    adjusted_buffer: int
    per_page_hashes: List[str]
    ocr_results: List[List[Tuple[str, int, int, int, int]]]
    page_fingerprints: List[PageFingerprint]
    document_hash: str
    page_dimensions: List[Tuple[int, int]]
    anchor_summary: Dict[str, Any]
    min_quality: float
    source_format: SourceFormat
    native_pdf_page_texts: Optional[List[str]]

    def release_extraction_buffers(self) -> None:
        self.pages_img = []
        self.ocr_results = []
        self.native_pdf_page_texts = None


@dataclass
class RoutingDecision:
    document_context: DocumentContext
    fingerprint_match: Optional[FingerprintRecord]
    family_match: Optional[TemplateFamilyRecord]
    family_representative: Optional[FingerprintRecord]
    family_match_score: float
    family_match_candidate_count: int
    family_match_attempted: bool
    apply_match: Optional[FingerprintRecord]
    route: Route
    current_family_id: Optional[str]
    current_family_record: Optional[TemplateFamilyRecord]


@dataclass
class DiscoveryStageResult:
    template_to_use: Optional[Dict[str, Any]]
    discovered_extracted: Optional[Dict[str, Any]]
    discovery_meta: Dict[str, Any]
    rejection_result: Optional[ProcessingResult] = None


@dataclass
class ExtractionStageResult:
    extracted: Dict[str, Any]
    normalized: Dict[str, Any]
    validation_payload: Dict[str, Any]


@dataclass
class ValidationStageResult:
    validation_passed: Optional[bool]
    validation_result: Optional[Any]
    current_family_id: Optional[str]
    current_family_record: Optional[TemplateFamilyRecord]
    final_result: Optional[ProcessingResult] = None


@dataclass
class HealingAssessment:
    route: Route
    current_family_id: Optional[str]
    trigger_family_id: Optional[str]
    trigger_fingerprint_hash: Optional[str]
    trigger_match_type: Optional[str]
    trigger_score: Optional[float]
    gt_trust_scope: Optional[str]
    gt_trusted: bool
    gt_apply_count: int
    gt_reject_count: int
    gt_confidence: float


def _build_output_artifact_path(output_dir: str, invoice_path: str) -> Path:
    invoice = Path(invoice_path)
    try:
        relative_invoice = invoice.relative_to(Path.cwd())
    except ValueError:
        if invoice.parent.name:
            relative_invoice = Path(invoice.parent.name) / invoice.name
        else:
            relative_invoice = Path(invoice.name)
    return Path(output_dir) / relative_invoice.with_suffix(".json")


def _count_line_items(data: Optional[Dict[str, Any]]) -> Optional[int]:
    if not data:
        return None
    line_items = data.get("lineItems")
    if line_items is None:
        line_items = data.get("line_items")
    if isinstance(line_items, list):
        return len(line_items)
    return 0 if line_items is not None else None


def _count_scalar_fields(data: Optional[Dict[str, Any]]) -> Optional[int]:
    if data is None:
        return None
    return sum(1 for key in data.keys() if key not in ("line_items", "lineItems"))


def _count_template_fields(template: Optional[Dict[str, Any]]) -> Optional[int]:
    if not template:
        return None
    return sum(len(page.get("fields", {})) for page in template.get("pages", []))


def _template_has_table(template: Optional[Dict[str, Any]]) -> bool:
    if not template:
        return False
    return any(page.get("table") for page in template.get("pages", []))


def _count_label_confirmations(template: Optional[Dict[str, Any]]) -> Optional[int]:
    if not template:
        return None
    return sum(len(page.get("label_confirmation_set", [])) for page in template.get("pages", []))


def _build_family_id(
    provider_name: Optional[str], anchor_summary: Dict[str, Any], document_hash: str
) -> str:
    provider_slug = (provider_name or "unknown").lower().replace(" ", "-")
    anchor_seed = json.dumps(
        {
            "provider": provider_name,
            "page_roles": anchor_summary.get("page_roles"),
            "aggregate_keywords": anchor_summary.get("aggregate_keywords"),
        },
        sort_keys=True,
    )
    return f"family-{provider_slug[:24]}-{compute_string_hash(anchor_seed + document_hash)[:12]}"


def _build_diagnostics(
    attempted_route: Optional[Route],
    template_family_id: Optional[str] = None,
    family_match_attempted: Optional[bool] = None,
    family_match_outcome: Optional[str] = None,
    family_match_score: Optional[float] = None,
    family_match_method: Optional[str] = None,
    family_match_candidate_count: Optional[int] = None,
    document_context: Optional[DocumentContext] = None,
    extracted: Optional[Dict[str, Any]] = None,
    normalized_data: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[Dict[str, Any]] = None,
    validation_result: Optional[Any] = None,
    template: Optional[Dict[str, Any]] = None,
    discovery_mode: Optional[str] = None,
    discovery_stage_status: Optional[str] = None,
    locate_error_category: Optional[str] = None,
    extract_error_category: Optional[str] = None,
    locate_failure_detail: Optional[str] = None,
    extract_failure_detail: Optional[str] = None,
    locate_raw_response: Optional[str] = None,
    extract_raw_response: Optional[str] = None,
    located_field_count: Optional[int] = None,
    heuristic_confidence: Optional[float] = None,
    matched_label_count: Optional[int] = None,
    critical_field_count: Optional[int] = None,
    table_detected: Optional[bool] = None,
    party_block_diagnostics: Optional[List[Dict[str, Any]]] = None,
    summary_candidate_diagnostics: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    line_item_source: Optional[str] = None,
) -> ProcessingDiagnostics:
    validation_errors = list(validation_result.errors) if validation_result else []
    missing_count = sum(
        1 for err in validation_errors if err.startswith("Missing extracted field:")
    )
    mismatch_count = sum(1 for err in validation_errors if err.startswith("Mismatch on "))
    validation_error_counts = dict(validation_result.error_counts) if validation_result else {}
    reconciliation_summary = None
    if validation_result and validation_result.reconciliation_summary:
        reconciliation_summary = validation_result.reconciliation_summary
    elif normalized_data and isinstance(normalized_data.get("reconciliation_summary"), dict):
        reconciliation_summary = normalized_data.get("reconciliation_summary")
    reconciliation_issue_types = (
        list(reconciliation_summary.get("issue_types") or []) if reconciliation_summary else []
    )
    reconciliation_derived_fields = (
        list(reconciliation_summary.get("derived_fields") or []) if reconciliation_summary else []
    )
    reconciliation_status = reconciliation_summary.get("status") if reconciliation_summary else None
    reconciliation_failure = bool(
        (
            reconciliation_status == "inconsistent"
            or validation_error_counts.get("summary_arithmetic_mismatch")
            or validation_error_counts.get("line_item_arithmetic_mismatch")
        )
        if validation_result or reconciliation_summary
        else False
    )
    wrong_template_suspected = (
        bool(validation_errors)
        and mismatch_count >= max(2, missing_count + 1)
        and attempted_route == Route.APPLY
    )

    return ProcessingDiagnostics(
        attempted_route=attempted_route,
        template_family_id=template_family_id,
        family_match_attempted=family_match_attempted,
        family_match_outcome=family_match_outcome,
        family_match_score=family_match_score,
        family_match_method=family_match_method,
        family_match_candidate_count=family_match_candidate_count,
        wrong_template_suspected=wrong_template_suspected,
        normalization_failure=False if extracted is not None else None,
        reconciliation_failure=reconciliation_failure
        if (validation_result or reconciliation_summary)
        else None,
        reconciliation_status=reconciliation_status,
        reconciliation_issue_types=reconciliation_issue_types,
        reconciliation_derived_fields=reconciliation_derived_fields,
        validator_policy_failure=(validation_result.score < 1.0) if validation_result else None,
        extraction_strategy=document_context.extraction_strategy if document_context else None,
        source_format=document_context.source_format if document_context else None,
        document_family=document_context.document_family if document_context else None,
        provider_name=document_context.provider_match.provider_name
        if document_context and document_context.provider_match
        else None,
        provider_confidence=document_context.provider_match.confidence
        if document_context and document_context.provider_match
        else None,
        country_code=document_context.country_code if document_context else None,
        currency_code=document_context.currency_code if document_context else None,
        discovery_mode=discovery_mode,
        discovery_stage_status=discovery_stage_status,
        locate_error_category=locate_error_category,
        extract_error_category=extract_error_category,
        locate_failure_detail=locate_failure_detail,
        extract_failure_detail=extract_failure_detail,
        locate_raw_response=locate_raw_response,
        extract_raw_response=extract_raw_response,
        validation_score=validation_result.score if validation_result else None,
        validation_errors=validation_errors,
        validation_error_counts=validation_error_counts,
        extracted_field_count=_count_scalar_fields(extracted),
        gt_line_item_count=_count_line_items(ground_truth),
        extracted_line_item_count=_count_line_items(extracted),
        scalar_field_missing_count=missing_count if validation_result else None,
        scalar_field_mismatch_count=mismatch_count if validation_result else None,
        discovered_field_count=_count_template_fields(template),
        located_field_count=located_field_count
        if located_field_count is not None
        else _count_template_fields(template),
        label_confirmation_count=_count_label_confirmations(template),
        heuristic_confidence=heuristic_confidence,
        matched_label_count=matched_label_count,
        critical_field_count=critical_field_count,
        table_detected=table_detected
        if table_detected is not None
        else _template_has_table(template),
        party_block_diagnostics=party_block_diagnostics or [],
        summary_candidate_diagnostics=summary_candidate_diagnostics or {},
        line_item_source=line_item_source,
    )


def _prepare_invoice_inputs(invoice_path: str, config: AppConfig) -> PreparedInvoiceInputs:
    fname = Path(invoice_path).name

    with open(invoice_path, "rb") as f:
        raw_bytes = f.read()
    input_document_hash = compute_hash(raw_bytes)

    native_pdf_page_texts: Optional[List[str]] = None
    invoice_suffix = Path(invoice_path).suffix.lower()
    if invoice_suffix == ".pdf":
        try:
            candidate_texts = load_pdf_page_texts(invoice_path)
        except Exception:
            source_format = SourceFormat.pdf_scanned
        else:
            if any(text.strip() for text in candidate_texts):
                source_format = SourceFormat.pdf_text
                native_pdf_page_texts = candidate_texts
            else:
                source_format = SourceFormat.pdf_scanned
    elif invoice_suffix in {".png", ".jpg", ".jpeg"}:
        source_format = SourceFormat.image
    else:
        source_format = SourceFormat.unknown

    t_normalize = time.time()
    pages_img = normalize_page(invoice_path)
    logger.debug(
        f"[{fname}] Pages after normalization: {len(pages_img)} (took {int((time.time() - t_normalize) * 1000)}ms)"
    )
    for i, page in enumerate(pages_img):
        logger.debug(f"[{fname}] Page {i} dimensions: {page.shape[1]}x{page.shape[0]}px")

    quality_metrics = [assess_quality(page, config.quality) for page in pages_img]
    quality_flag, adjusted_buffer = apply_quality_gate(
        quality_metrics,
        config.region_buffer_pixels,
        config.quality.quality_region_buffer_multiplier,
    )
    for i, metric in enumerate(quality_metrics):
        logger.debug(
            f"[{fname}] Page {i} quality — blur: {metric.blur_score:.1f}, "
            f"contrast: {metric.contrast_score:.3f}, score: {metric.quality_score:.3f}, "
            f"flagged: {metric.quality_flag}"
        )
    logger.debug(f"[{fname}] Overall quality_flag: {quality_flag}, adj_buffer: {adjusted_buffer}")

    per_page_hashes = [compute_hash(page.tobytes()) for page in pages_img]
    t_ocr = time.time()
    ocr_results = [run_full_page_ocr(page, config.ocr.single_field_engine) for page in pages_img]
    logger.debug(
        f"[{fname}] OCR complete (engine={config.ocr.single_field_engine}, took {int((time.time() - t_ocr) * 1000)}ms)"
    )
    for i, page_ocr in enumerate(ocr_results):
        ocr_texts = [text for text, *_ in page_ocr]
        logger.debug(f"[{fname}] Page {i} OCR tokens: {len(page_ocr)} | sample: {ocr_texts[:5]}")

    t_fp = time.time()
    page_fingerprints = [
        compute_page_fingerprint(index, page, ocr_results[index])
        for index, page in enumerate(pages_img)
    ]
    logger.debug(f"[{fname}] Fingerprinting complete (took {int((time.time() - t_fp) * 1000)}ms)")
    for i, fingerprint in enumerate(page_fingerprints):
        logger.debug(
            f"[{fname}] Page {i} visual hash: {fingerprint.visual_hash_hex[:16]}... role={fingerprint.role}"
        )

    document_hash = compute_document_fingerprint(page_fingerprints)
    logger.debug(f"[{fname}] Document hash: {document_hash[:12]}...")

    page_dimensions = [(page.shape[1], page.shape[0]) for page in pages_img]
    anchor_summary = extract_document_anchor_summary(ocr_results, page_dimensions)
    min_quality = (
        min(metric.quality_score for metric in quality_metrics) if quality_metrics else 1.0
    )

    return PreparedInvoiceInputs(
        input_document_hash=input_document_hash,
        pages_img=pages_img,
        quality_metrics=quality_metrics,
        quality_flag=quality_flag,
        adjusted_buffer=adjusted_buffer,
        per_page_hashes=per_page_hashes,
        ocr_results=ocr_results,
        page_fingerprints=page_fingerprints,
        document_hash=document_hash,
        page_dimensions=page_dimensions,
        anchor_summary=anchor_summary,
        min_quality=min_quality,
        source_format=source_format,
        native_pdf_page_texts=native_pdf_page_texts,
    )


def _resolve_routing_decision(
    invoice_path: str,
    config: AppConfig,
    db: FingerprintDB,
    prepared: PreparedInvoiceInputs,
    *,
    gt_present: bool,
    gt_discovery_ready: bool,
    active_fingerprints_override: Optional[List[FingerprintRecord]] = None,
) -> RoutingDecision:
    fname = Path(invoice_path).name
    active_fingerprints = (
        active_fingerprints_override
        if active_fingerprints_override is not None
        else db.get_all_active_fingerprints()
    )
    active_families = db.get_active_template_families()
    active_family_map = {family.template_family_id: family for family in active_families}
    logger.debug(f"[{fname}] Active fingerprints in DB: {len(active_fingerprints)}")

    document_context = build_document_context(
        invoice_path,
        prepared.ocr_results,
        source_format=prepared.source_format,
    )
    document_context.provider_match = resolve_provider(prepared.ocr_results, config)
    if document_context.provider_match and not document_context.country_code:
        document_context.country_code = document_context.provider_match.country_code

    fingerprint_match, confirmation_ratio = lookup_fingerprint(
        prepared.page_fingerprints,
        prepared.ocr_results,
        prepared.page_dimensions,
        active_fingerprints,
        config,
    )
    family_match = None
    family_representative = None
    family_match_score = 0.0
    family_match_candidate_count = 0
    family_match_attempted = bool(active_families)
    if active_families:
        family_match, family_representative, family_match_score, family_match_candidate_count = (
            lookup_template_family(
                prepared.page_fingerprints,
                active_fingerprints,
                active_families,
                document_context,
                config,
            )
        )

    apply_match = fingerprint_match
    if (
        apply_match is None
        and family_match is not None
        and family_representative is not None
        and family_match_score >= config.discovery.family_apply_threshold
    ):
        apply_match = family_representative
        confirmation_ratio = family_match_score

    document_context.extraction_strategy = determine_extraction_strategy(
        document_context,
        fingerprint_match,
        family_representative=family_representative,
        family_match_score=family_match_score,
        family_apply_threshold=config.discovery.family_apply_threshold,
    )
    route = route_invoice(
        document_context,
        fingerprint_match,
        gt_present,
        gt_discovery_ready=gt_discovery_ready,
        family_representative=family_representative,
        family_match_score=family_match_score,
        family_apply_threshold=config.discovery.family_apply_threshold,
    )
    if fingerprint_match:
        logger.info(
            f"[{fname}] Fingerprint match: {fingerprint_match.hash[:12]}... "
            f"conf_ratio={confirmation_ratio:.3f} status={fingerprint_match.status.value}"
        )
    else:
        logger.info(f"[{fname}] No fingerprint match (conf_ratio={confirmation_ratio:.3f})")
    if family_match is not None:
        logger.info(
            f"[{fname}] Family match: {family_match.template_family_id} "
            f"score={family_match_score:.3f} candidates={family_match_candidate_count}"
        )
    logger.info(
        f"[{fname}] Route: {route.value} | strategy={document_context.extraction_strategy.value} "
        f"| format={document_context.source_format.value} | family={document_context.document_family.value}"
    )

    current_family_record = family_match
    if current_family_record is None and apply_match and apply_match.template_family_id:
        current_family_record = active_family_map.get(apply_match.template_family_id)
    current_family_id = (apply_match.template_family_id if apply_match else None) or (
        family_match.template_family_id if family_match else None
    )

    return RoutingDecision(
        document_context=document_context,
        fingerprint_match=fingerprint_match,
        family_match=family_match,
        family_representative=family_representative,
        family_match_score=family_match_score,
        family_match_candidate_count=family_match_candidate_count,
        family_match_attempted=family_match_attempted,
        apply_match=apply_match,
        route=route,
        current_family_id=current_family_id,
        current_family_record=current_family_record,
    )


def _build_provenance(
    request_id: str,
    started_at: str,
    config: AppConfig,
    prepared: PreparedInvoiceInputs,
    routing: RoutingDecision,
) -> Provenance:
    document_context = routing.document_context
    apply_match = routing.apply_match
    return Provenance(
        request_id=request_id,
        route=routing.route,
        fingerprint_hash=prepared.document_hash if not apply_match else apply_match.hash,
        template_family_id=routing.current_family_id,
        extraction_strategy=document_context.extraction_strategy,
        source_format=document_context.source_format,
        document_family=document_context.document_family,
        country_code=document_context.country_code,
        currency_code=document_context.currency_code,
        provider_name=document_context.provider_match.provider_name
        if document_context.provider_match
        else None,
        provider_confidence=document_context.provider_match.confidence
        if document_context.provider_match
        else None,
        template_status_at_time=apply_match.status if apply_match else None,
        template_confidence_at_time=apply_match.confidence if apply_match else None,
        ocr_engine=config.ocr.single_field_engine,
        inference_method=None,
        image_quality_score=[metric.quality_score for metric in prepared.quality_metrics],
        quality_flag=prepared.quality_flag,
        input_document_hash=prepared.input_document_hash,
        per_page_hashes=prepared.per_page_hashes,
        extraction_output_hash=None,
        started_at=started_at,
        completed_at="",
        latency_ms=0,
    )


def _extract_with_family_profile(
    pages_img: List[Any],
    template: Dict[str, Any],
    buffer_px: int,
    config: AppConfig,
    family_record: Optional[TemplateFamilyRecord],
) -> Dict[str, Any]:
    return extract_with_ocr(
        pages_img,
        template,
        buffer_px,
        config,
        family_profile=family_record.extraction_profile if family_record else None,
    )


def _family_match_outcome(routing: RoutingDecision) -> Optional[str]:
    if not routing.family_match_attempted:
        return None
    return "matched" if routing.family_match else "no_match"


def _resolve_discovery_mode(_config: AppConfig, discovery_meta: Dict[str, Any]) -> str:
    return discovery_meta.get("discovery_mode", "heuristic")


def _build_processing_diagnostics(
    routing: RoutingDecision,
    diagnostic_template_family_id: Optional[str],
    *,
    attempted_route: Route,
    extracted: Optional[Dict[str, Any]] = None,
    normalized_data: Optional[Dict[str, Any]] = None,
    ground_truth: Optional[Dict[str, Any]] = None,
    validation_result: Optional[Any] = None,
    template: Optional[Dict[str, Any]] = None,
    discovery_mode: Optional[str] = None,
    discovery_stage_status: Optional[str] = None,
    locate_error_category: Optional[str] = None,
    extract_error_category: Optional[str] = None,
    locate_failure_detail: Optional[str] = None,
    extract_failure_detail: Optional[str] = None,
    locate_raw_response: Optional[str] = None,
    extract_raw_response: Optional[str] = None,
    located_field_count: Optional[int] = None,
    heuristic_confidence: Optional[float] = None,
    matched_label_count: Optional[int] = None,
    critical_field_count: Optional[int] = None,
    table_detected: Optional[bool] = None,
    party_block_diagnostics: Optional[List[Dict[str, Any]]] = None,
    summary_candidate_diagnostics: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    line_item_source: Optional[str] = None,
    family_match_outcome: Optional[str] = None,
) -> ProcessingDiagnostics:
    return _build_diagnostics(
        attempted_route,
        template_family_id=diagnostic_template_family_id,
        family_match_attempted=routing.family_match_attempted,
        family_match_outcome=family_match_outcome
        if family_match_outcome is not None
        else _family_match_outcome(routing),
        family_match_score=routing.family_match_score if routing.family_match_attempted else None,
        family_match_method="anchor_summary" if routing.family_match_attempted else None,
        family_match_candidate_count=routing.family_match_candidate_count
        if routing.family_match_attempted
        else None,
        document_context=routing.document_context,
        extracted=extracted,
        normalized_data=normalized_data,
        ground_truth=ground_truth,
        validation_result=validation_result,
        template=template,
        discovery_mode=discovery_mode,
        discovery_stage_status=discovery_stage_status,
        locate_error_category=locate_error_category,
        extract_error_category=extract_error_category,
        locate_failure_detail=locate_failure_detail,
        extract_failure_detail=extract_failure_detail,
        locate_raw_response=locate_raw_response,
        extract_raw_response=extract_raw_response,
        located_field_count=located_field_count,
        heuristic_confidence=heuristic_confidence,
        matched_label_count=matched_label_count,
        critical_field_count=critical_field_count,
        table_detected=table_detected,
        party_block_diagnostics=party_block_diagnostics,
        summary_candidate_diagnostics=summary_candidate_diagnostics,
        line_item_source=line_item_source,
    )


def _build_processing_result(
    invoice_path: str,
    *,
    fingerprint_hash: str,
    result_template_family_id: Optional[str],
    provenance: Provenance,
    validation_passed: Optional[bool],
    route_used: Route,
    image_quality_score: float,
    attempted_route: Route,
    diagnostics: ProcessingDiagnostics,
    ground_truth: Optional[Dict[str, Any]] = None,
    extracted_data: Optional[Dict[str, Any]] = None,
    normalized_data: Optional[Dict[str, Any]] = None,
    template_status_at_time: Optional[TemplateStatus] = None,
) -> ProcessingResult:
    return ProcessingResult(
        invoice_path=invoice_path,
        fingerprint_hash=fingerprint_hash,
        template_family_id=result_template_family_id,
        extracted_data=extracted_data,
        normalized_data=normalized_data,
        ground_truth=ground_truth,
        provenance=provenance,
        validation_passed=validation_passed,
        route_used=route_used,
        image_quality_score=image_quality_score,
        attempted_route=attempted_route,
        diagnostics=diagnostics,
        template_status_at_time=template_status_at_time,
        processed_at="",
    )


def _apply_retry_context(
    result: ProcessingResult,
    *,
    gt_present: bool,
    retry_context: Optional[Dict[str, Any]],
) -> ProcessingResult:
    provenance = result.provenance
    if provenance is None:
        return result

    determination_sources: List[str] = []
    if gt_present:
        determination_sources.append("gt_backed")
    if retry_context:
        determination_sources.append("healing")
    if not determination_sources:
        determination_sources.append("operational")

    provenance.determination_sources = determination_sources
    provenance.gt_backed = bool(gt_present)
    provenance.healing_reprocessed = retry_context is not None

    if retry_context:
        previous_result = retry_context.get("previous_result") or {}
        provenance.healing_origin = retry_context.get("healing_origin")
        provenance.healing_attempt_count = int(retry_context.get("healing_attempt_count") or 1)
        provenance.previous_route = (
            Route(previous_result["route_used"]) if previous_result.get("route_used") else None
        )
        provenance.previous_validation_passed = previous_result.get("validation_passed")
        provenance.healing_trigger_family_id = retry_context.get("trigger_family_id")
        provenance.healing_trigger_fingerprint_hash = retry_context.get("trigger_fingerprint_hash")
        provenance.healing_trigger_match_type = retry_context.get("trigger_match_type")
        provenance.healing_trigger_score = retry_context.get("trigger_score")
        provenance.prior_result_provenance = previous_result.get("provenance")
        if result.diagnostics is not None:
            result.diagnostics.retry_context = {
                "origin": retry_context.get("healing_origin"),
                "attempt_count": provenance.healing_attempt_count,
                "previous_route": previous_result.get("route_used"),
                "previous_validation_passed": previous_result.get("validation_passed"),
                "trigger_family_id": retry_context.get("trigger_family_id"),
                "trigger_fingerprint_hash": retry_context.get("trigger_fingerprint_hash"),
                "trigger_match_type": retry_context.get("trigger_match_type"),
                "trigger_score": retry_context.get("trigger_score"),
            }
    return result


def _complete_processing_result(
    result: ProcessingResult,
    *,
    start_time: float,
    settings: Settings,
    invoice_path: str,
    db: FingerprintDB,
    do_store: bool = True,
) -> ProcessingResult:
    result.provenance.completed_at = get_utc_now()
    result.provenance.latency_ms = int((time.time() - start_time) * 1000)
    result.processed_at = result.provenance.completed_at

    out_path = _build_output_artifact_path(settings.output_dir, invoice_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result.model_dump(), f, indent=2)

    if do_store:
        result.id = db.store_result(result)
    return result


def _run_discovery_stage(
    invoice_path: str,
    settings: Settings,
    config: AppConfig,
    gt_data: Optional[Dict[str, Any]],
    prepared: PreparedInvoiceInputs,
    routing: RoutingDecision,
    provenance: Provenance,
    min_quality: float,
) -> DiscoveryStageResult:
    template_to_use = routing.apply_match.layout_template if routing.apply_match else None
    discovered_extracted = None
    discovery_meta: Dict[str, Any] = {}
    if routing.route != Route.DISCOVERY:
        return DiscoveryStageResult(
            template_to_use=template_to_use,
            discovered_extracted=discovered_extracted,
            discovery_meta=discovery_meta,
        )

    fname = Path(invoice_path).name
    try:
        discovery_mode = "heuristic"
        logger.info(f"[{fname}] Running discovery inference ({discovery_mode})")
        template_to_use, discovery_conf, discovered_extracted, discovery_meta = infer_template(
            prepared.pages_img,
            gt_data,
            config,
            ocr_results_per_page=prepared.ocr_results,
        )
        provenance.inference_method = f"heuristic:{discovery_mode}"
        logger.info(f"[{fname}] Discovery inference complete — confidence: {discovery_conf:.3f}")
        return DiscoveryStageResult(
            template_to_use=template_to_use,
            discovered_extracted=discovered_extracted,
            discovery_meta=discovery_meta,
        )
    except Exception as exc:
        logger.error(f"[{fname}] DISCOVERY failed — inference error: {exc}", exc_info=True)
        provenance.route = Route.REJECTED
        if hasattr(exc, "partial_template"):
            template_to_use = getattr(exc, "partial_template")
        if hasattr(exc, "discovery_meta"):
            discovery_meta = getattr(exc, "discovery_meta") or discovery_meta
        locate_error_category = None
        extract_error_category = None
        locate_failure_detail = None
        extract_failure_detail = None
        stage_status = "locate_parse_failed"
        if hasattr(exc, "reason"):
            reason = getattr(exc, "reason")
            if "extract" in reason:
                stage_status = (
                    "extract_validation_failed"
                    if "validation" in reason
                    else "extract_parse_failed"
                )
                extract_error_category = reason
                extract_failure_detail = getattr(exc, "failure_detail", None)
            else:
                stage_status = (
                    "locate_validation_failed" if "validation" in reason else "locate_parse_failed"
                )
                locate_error_category = reason
                locate_failure_detail = getattr(exc, "failure_detail", None)
        rejection_result = _build_processing_result(
            invoice_path,
            fingerprint_hash=prepared.document_hash,
            result_template_family_id=None,
            provenance=provenance,
            validation_passed=False,
            route_used=Route.REJECTED,
            image_quality_score=min_quality,
            attempted_route=routing.route,
            diagnostics=_build_processing_diagnostics(
                routing,
                routing.current_family_id,
                attempted_route=routing.route,
                ground_truth=gt_data,
                template=template_to_use,
                discovery_mode=_resolve_discovery_mode(config, discovery_meta),
                discovery_stage_status=stage_status,
                locate_error_category=locate_error_category,
                extract_error_category=extract_error_category,
                locate_failure_detail=locate_failure_detail,
                extract_failure_detail=extract_failure_detail,
                locate_raw_response=discovery_meta.get("locate_raw_response")
                or getattr(exc, "raw_response", None),
                extract_raw_response=discovery_meta.get("extract_raw_response"),
                located_field_count=discovery_meta.get("located_field_count"),
                heuristic_confidence=discovery_meta.get("heuristic_confidence"),
                matched_label_count=discovery_meta.get("matched_label_count"),
                critical_field_count=discovery_meta.get("critical_field_count"),
                table_detected=discovery_meta.get("table_detected"),
                party_block_diagnostics=discovery_meta.get("party_block_diagnostics"),
                summary_candidate_diagnostics=discovery_meta.get("summary_candidate_diagnostics"),
                line_item_source=discovery_meta.get("line_item_source"),
            ),
            ground_truth=gt_data,
            template_status_at_time=None,
        )
        return DiscoveryStageResult(
            template_to_use=template_to_use,
            discovered_extracted=None,
            discovery_meta=discovery_meta,
            rejection_result=rejection_result,
        )


def _run_extraction_stage(
    invoice_path: str,
    config: AppConfig,
    prepared: PreparedInvoiceInputs,
    routing: RoutingDecision,
    current_family_record: Optional[TemplateFamilyRecord],
    discovery_result: DiscoveryStageResult,
) -> ExtractionStageResult:
    fname = Path(invoice_path).name
    template_to_use = discovery_result.template_to_use
    discovered_extracted = discovery_result.discovered_extracted
    if discovered_extracted is not None and template_to_use:
        extracted = refine_critical_fields(
            discovered_extracted,
            template_to_use,
            prepared.ocr_results,
            prepared.page_dimensions,
        )
        extracted_line_items = extracted.get("line_items")
        if _template_has_table(template_to_use) and not extracted_line_items:
            try:
                ocr_fallback = _extract_with_family_profile(
                    prepared.pages_img,
                    template_to_use,
                    prepared.adjusted_buffer,
                    config,
                    current_family_record,
                )
            except Exception:
                ocr_fallback = {}
            fallback_line_items = (
                ocr_fallback.get("line_items") if isinstance(ocr_fallback, dict) else None
            )
            if fallback_line_items:
                extracted["line_items"] = fallback_line_items
        logger.debug(f"[{fname}] Using discovery-extracted values")
    elif (
        routing.route == Route.APPLY
        and template_to_use is None
        and routing.document_context.extraction_strategy == ExtractionStrategy.native_pdf
    ):
        logger.info(f"[{fname}] APPLY: extracting via native PDF text path")
        try:
            extracted = extract_from_text_pdf(
                invoice_path,
                page_texts=prepared.native_pdf_page_texts,
            )
        except Exception as exc:
            raise ProcessingStageError("native_pdf_extraction", routing.route, exc) from exc
    elif routing.route == Route.APPLY and template_to_use is not None:
        logger.info(f"[{fname}] APPLY: extracting via OCR using stored template regions")
        template_fields = [
            name
            for page in template_to_use.get("pages", [])
            for name in page.get("fields", {}).keys()
        ]
        logger.debug(f"[{fname}] Template fields to extract: {template_fields}")
        try:
            extracted = _extract_with_family_profile(
                prepared.pages_img,
                template_to_use,
                prepared.adjusted_buffer,
                config,
                current_family_record,
            )
        except Exception as exc:
            raise ProcessingStageError("apply_extraction", routing.route, exc) from exc
    else:
        try:
            extracted = _extract_with_family_profile(
                prepared.pages_img,
                template_to_use,
                prepared.adjusted_buffer,
                config,
                current_family_record,
            )
        except Exception as exc:
            raise ProcessingStageError("ocr_extraction", routing.route, exc) from exc

    normalized = normalize_extracted_invoice(extracted, routing.document_context)
    validation_payload = merge_for_validation(extracted, normalized)
    return ExtractionStageResult(
        extracted=extracted,
        normalized=normalized,
        validation_payload=validation_payload,
    )


def _record_apply_success(
    db: FingerprintDB,
    redis_client: Any,
    match: Optional[FingerprintRecord],
    current_family_id: Optional[str],
    score: float,
    config: AppConfig,
    *,
    reason: str,
    gt_backed: bool,
) -> None:
    if match is None:
        return
    update_template_confidence(
        db,
        redis_client,
        match.hash,
        score,
        config.template_lifecycle,
        gt_backed=gt_backed,
    )
    if current_family_id:
        update_template_family_confidence(
            db,
            current_family_id,
            score,
            config.template_lifecycle,
            reason=reason,
            gt_backed=gt_backed,
        )


def _record_discovery_success(
    invoice_path: str,
    config: AppConfig,
    db: FingerprintDB,
    redis_client: Any,
    prepared: PreparedInvoiceInputs,
    routing: RoutingDecision,
    provenance: Provenance,
    template_to_use: Dict[str, Any],
    validation_result: Any,
    *,
    gt_backed: bool,
) -> tuple[Optional[str], Optional[TemplateFamilyRecord]]:
    document_context = routing.document_context
    family_id = routing.current_family_id
    family_record = routing.family_match if routing.family_match is not None else None
    if family_record is None:
        family_id = _build_family_id(
            document_context.provider_match.provider_name
            if document_context.provider_match
            else None,
            prepared.anchor_summary,
            prepared.document_hash,
        )
        family_record = db.get_template_family(family_id)
    else:
        family_id = family_record.template_family_id

    provenance.template_family_id = family_id
    new_record = FingerprintRecord(
        hash=prepared.document_hash,
        layout_template=template_to_use,
        template_family_id=family_id,
        provider_name=document_context.provider_match.provider_name
        if document_context.provider_match
        else None,
        country_code=document_context.country_code,
        page_fingerprints=prepared.page_fingerprints,
        confidence=0.0,
        apply_count=0,
        reject_count=0,
        gt_apply_count=0,
        gt_reject_count=0,
        gt_confidence=0.0,
        status=TemplateStatus.provisional,
        version="v3",
        created_at=get_utc_now(),
        last_used=get_utc_now(),
    )
    db.store_fingerprint(
        new_record,
        [page.visual_hash_hex for page in prepared.page_fingerprints],
        [page.model_dump() for page in prepared.page_fingerprints],
    )
    db.link_fingerprint_to_family(prepared.document_hash, family_id)
    page_roles = [page.role for page in prepared.page_fingerprints if page.role is not None]
    stored_family = TemplateFamilyRecord(
        template_family_id=family_id,
        provider_name=document_context.provider_match.provider_name
        if document_context.provider_match
        else (family_record.provider_name if family_record else None),
        country_code=document_context.country_code
        or (family_record.country_code if family_record else None),
        document_family=document_context.document_family,
        stable_anchor_regions={
            "tokens": sorted(
                {
                    token
                    for values in prepared.anchor_summary.get("aggregate_keywords", {}).values()
                    for token in values
                }
            )[:30],
        },
        anchor_summary=prepared.anchor_summary,
        page_role_expectations=page_roles,
        summary_area_anchors={
            "summary_labels": prepared.anchor_summary.get("aggregate_keywords", {}).get(
                "summary", []
            )
        },
        variable_region_masks=[{"kind": "line_item_body", "page_role": "line_item_page"}],
        extraction_profile=merge_family_extraction_profiles(
            family_record.extraction_profile if family_record else None,
            build_family_extraction_profile(
                template_to_use,
                document_context,
                default_table_engine=config.ocr.table_engine,
            ),
        ),
        confidence=family_record.confidence if family_record else 0.0,
        apply_count=family_record.apply_count if family_record else 0,
        reject_count=family_record.reject_count if family_record else 0,
        gt_apply_count=family_record.gt_apply_count if family_record else 0,
        gt_reject_count=family_record.gt_reject_count if family_record else 0,
        gt_confidence=family_record.gt_confidence if family_record else 0.0,
        status=family_record.status if family_record else TemplateStatus.provisional,
        created_at=family_record.created_at if family_record else get_utc_now(),
        updated_at=get_utc_now(),
    )
    db.store_template_family(stored_family)
    db.add_template_family_example(
        TemplateFamilyExample(
            template_family_id=family_id,
            fingerprint_hash=prepared.document_hash,
            invoice_path=invoice_path,
            example_metadata={
                "validation_score": validation_result.score,
                "route": routing.route.value,
                "provider_name": stored_family.provider_name,
            },
            created_at=get_utc_now(),
        )
    )
    provenance.route = Route.DISCOVERY
    update_template_confidence(
        db,
        redis_client,
        prepared.document_hash,
        validation_result.score,
        config.template_lifecycle,
        gt_backed=gt_backed,
    )
    update_template_family_confidence(
        db,
        family_id,
        validation_result.score,
        config.template_lifecycle,
        reason="discovery_success",
        gt_backed=gt_backed,
    )
    return family_id, stored_family


def _run_validation_stage(
    invoice_path: str,
    config: AppConfig,
    db: FingerprintDB,
    gt_data: Optional[Dict[str, Any]],
    prepared: PreparedInvoiceInputs,
    routing: RoutingDecision,
    provenance: Provenance,
    redis_client: Any,
    current_family_id: Optional[str],
    current_family_record: Optional[TemplateFamilyRecord],
    discovery_result: DiscoveryStageResult,
    extraction_result: ExtractionStageResult,
    *,
    lifecycle_updates_enabled: bool,
) -> ValidationStageResult:
    fname = Path(invoice_path).name
    validation_passed = None
    validation_result = None
    if gt_data:
        validation_result = validate_invoice(
            extraction_result.validation_payload,
            gt_data,
            config.validation,
            config.field_mapping,
        )
        threshold = (
            config.validation.discovery_threshold
            if routing.route == Route.DISCOVERY
            else config.validation.apply_threshold
        )
        logger.info(
            f"[{fname}] Validation — score: {validation_result.score:.3f} "
            f"(threshold: {threshold}), matched: {validation_result.matched_fields}, "
            f"mismatched: {validation_result.mismatched_fields}"
        )
        if validation_result.errors:
            for err in validation_result.errors:
                logger.info(f"[{fname}]   Validation error: {err}")

        if should_reject(validation_result, routing.route.value, config.validation):
            logger.warning(
                f"[{fname}] REJECTED — validation score {validation_result.score:.3f} below threshold {threshold}"
            )
            provenance.route = Route.REJECTED
            if lifecycle_updates_enabled:
                if routing.fingerprint_match:
                    record_rejection(
                        db,
                        redis_client,
                        routing.fingerprint_match.hash,
                        config.template_lifecycle,
                        gt_backed=True,
                    )
                if current_family_id:
                    record_template_family_rejection(
                        db,
                        current_family_id,
                        config.template_lifecycle,
                        gt_backed=True,
                    )
            rejection_result = _build_processing_result(
                invoice_path,
                fingerprint_hash=prepared.document_hash,
                result_template_family_id=current_family_id,
                provenance=provenance,
                validation_passed=False,
                route_used=Route.REJECTED,
                image_quality_score=prepared.min_quality,
                attempted_route=routing.route,
                diagnostics=_build_processing_diagnostics(
                    routing,
                    current_family_id,
                    attempted_route=routing.route,
                    extracted=extraction_result.extracted,
                    normalized_data=extraction_result.normalized,
                    ground_truth=gt_data,
                    validation_result=validation_result,
                    template=discovery_result.template_to_use,
                    discovery_mode=_resolve_discovery_mode(config, discovery_result.discovery_meta)
                    if routing.route == Route.DISCOVERY
                    else None,
                    discovery_stage_status="validation_failed_after_extract"
                    if routing.route == Route.DISCOVERY
                    else None,
                    locate_raw_response=discovery_result.discovery_meta.get("locate_raw_response"),
                    extract_raw_response=discovery_result.discovery_meta.get(
                        "extract_raw_response"
                    ),
                    located_field_count=discovery_result.discovery_meta.get("located_field_count"),
                    heuristic_confidence=discovery_result.discovery_meta.get(
                        "heuristic_confidence"
                    ),
                    matched_label_count=discovery_result.discovery_meta.get("matched_label_count"),
                    critical_field_count=discovery_result.discovery_meta.get(
                        "critical_field_count"
                    ),
                    table_detected=discovery_result.discovery_meta.get("table_detected"),
                    party_block_diagnostics=discovery_result.discovery_meta.get(
                        "party_block_diagnostics"
                    ),
                    summary_candidate_diagnostics=discovery_result.discovery_meta.get(
                        "summary_candidate_diagnostics"
                    ),
                    line_item_source=discovery_result.discovery_meta.get("line_item_source"),
                ),
                ground_truth=gt_data,
                extracted_data=extraction_result.extracted,
                normalized_data=extraction_result.normalized,
                template_status_at_time=routing.fingerprint_match.status
                if routing.fingerprint_match
                else None,
            )
            return ValidationStageResult(
                validation_passed=False,
                validation_result=validation_result,
                current_family_id=current_family_id,
                current_family_record=current_family_record,
                final_result=rejection_result,
            )

        validation_passed = True
        if (
            routing.route == Route.DISCOVERY
            and discovery_result.template_to_use is not None
            and lifecycle_updates_enabled
        ):
            current_family_id, current_family_record = _record_discovery_success(
                invoice_path,
                config,
                db,
                redis_client,
                prepared,
                routing,
                provenance,
                discovery_result.template_to_use,
                validation_result,
                gt_backed=True,
            )
        elif routing.fingerprint_match and routing.route == Route.APPLY:
            if lifecycle_updates_enabled:
                _record_apply_success(
                    db,
                    redis_client,
                    routing.fingerprint_match,
                    current_family_id,
                    validation_result.score,
                    config,
                    reason="apply_success",
                    gt_backed=True,
                )
    elif routing.fingerprint_match and routing.route == Route.APPLY:
        if lifecycle_updates_enabled:
            _record_apply_success(
                db,
                redis_client,
                routing.fingerprint_match,
                current_family_id,
                1.0,
                config,
                reason="apply_without_gt",
                gt_backed=False,
            )

    return ValidationStageResult(
        validation_passed=validation_passed,
        validation_result=validation_result,
        current_family_id=current_family_id,
        current_family_record=current_family_record,
    )


def process_single_invoice(
    invoice_path: str,
    settings: Settings,
    config: AppConfig,
    db: FingerprintDB,
    active_fingerprints_override: Optional[List[FingerprintRecord]] = None,
    force_reprocess: bool = False,
    retry_context: Optional[Dict[str, Any]] = None,
) -> ProcessingResult:
    start_time = time.time()
    started_at = get_utc_now()
    req_id = generate_request_id()

    fname = Path(invoice_path).name
    logger.info(f"[{fname}] Starting processing")

    gt_data = pair_ground_truth(invoice_path)
    gt_present = gt_data is not None
    gt_discovery_ready = is_ground_truth_discovery_ready(gt_data) if gt_data else False
    logger.debug(
        f"[{fname}] Ground truth present: {gt_present}, discovery_ready: {gt_discovery_ready}"
        + (f", keys: {list(gt_data.keys())}" if gt_data else "")
    )

    existing = db.get_result(invoice_path)
    if existing and existing.validation_passed is not None and not force_reprocess:
        logger.info(
            f"[{fname}] Idempotent skip — returning cached result: route={existing.route_used}, validation_passed={existing.validation_passed}"
        )
        return existing

    prepared = _prepare_invoice_inputs(invoice_path, config)
    routing = _resolve_routing_decision(
        invoice_path,
        config,
        db,
        prepared,
        gt_present=gt_present,
        gt_discovery_ready=gt_discovery_ready,
        active_fingerprints_override=active_fingerprints_override,
    )
    doc_hash = prepared.document_hash
    min_qual = prepared.min_quality

    match = routing.fingerprint_match
    route = routing.route
    current_family_id = routing.current_family_id
    current_family_record = routing.current_family_record

    provenance = _build_provenance(req_id, started_at, config, prepared, routing)
    redis_client = get_redis_client(settings)
    lifecycle_updates_enabled = retry_context is None
    if route == Route.FAIL:
        fail_result = _build_processing_result(
            invoice_path,
            fingerprint_hash=doc_hash,
            result_template_family_id=None,
            provenance=provenance,
            validation_passed=False,
            route_used=route,
            image_quality_score=min_qual,
            attempted_route=route,
            diagnostics=_build_processing_diagnostics(
                routing,
                current_family_id,
                attempted_route=route,
                ground_truth=gt_data,
            ),
            ground_truth=gt_data,
            template_status_at_time=None,
        )
        fail_result = _apply_retry_context(
            fail_result, gt_present=gt_present, retry_context=retry_context
        )
        return _complete_processing_result(
            fail_result,
            start_time=start_time,
            settings=settings,
            invoice_path=invoice_path,
            db=db,
        )

    discovery_result = _run_discovery_stage(
        invoice_path,
        settings,
        config,
        gt_data,
        prepared,
        routing,
        provenance,
        min_qual,
    )
    if discovery_result.rejection_result is not None:
        discovery_result.rejection_result = _apply_retry_context(
            discovery_result.rejection_result,
            gt_present=gt_present,
            retry_context=retry_context,
        )
        return _complete_processing_result(
            discovery_result.rejection_result,
            start_time=start_time,
            settings=settings,
            invoice_path=invoice_path,
            db=db,
        )

    extraction_result = _run_extraction_stage(
        invoice_path,
        config,
        prepared,
        routing,
        current_family_record,
        discovery_result,
    )
    provenance.extraction_output_hash = compute_string_hash(
        json.dumps(extraction_result.validation_payload, sort_keys=True)
    )
    logger.debug(
        f"[{fname}] Extracted fields: {list(extraction_result.extracted.keys()) if extraction_result.extracted else []}"
    )
    logger.debug(f"[{fname}] Extracted values: {extraction_result.extracted}")
    logger.debug(f"[{fname}] Normalized values: {extraction_result.normalized}")
    logger.debug(
        f"[{fname}] Total pipeline latency so far: {int((time.time() - start_time) * 1000)}ms"
    )
    prepared.release_extraction_buffers()

    validation_stage = _run_validation_stage(
        invoice_path,
        config,
        db,
        gt_data,
        prepared,
        routing,
        provenance,
        redis_client,
        current_family_id,
        current_family_record,
        discovery_result,
        extraction_result,
        lifecycle_updates_enabled=lifecycle_updates_enabled,
    )
    if validation_stage.final_result is not None:
        validation_stage.final_result = _apply_retry_context(
            validation_stage.final_result,
            gt_present=gt_present,
            retry_context=retry_context,
        )
        return _complete_processing_result(
            validation_stage.final_result,
            start_time=start_time,
            settings=settings,
            invoice_path=invoice_path,
            db=db,
        )

    current_family_id = validation_stage.current_family_id
    final_result = _build_processing_result(
        invoice_path,
        fingerprint_hash=doc_hash if route == Route.DISCOVERY or match is None else match.hash,
        result_template_family_id=current_family_id,
        provenance=provenance,
        validation_passed=validation_stage.validation_passed,
        route_used=provenance.route,
        image_quality_score=min_qual,
        attempted_route=route,
        diagnostics=_build_processing_diagnostics(
            routing,
            current_family_id,
            attempted_route=route,
            extracted=extraction_result.extracted,
            normalized_data=extraction_result.normalized,
            ground_truth=gt_data,
            validation_result=validation_stage.validation_result,
            template=discovery_result.template_to_use,
            discovery_mode=_resolve_discovery_mode(config, discovery_result.discovery_meta)
            if route == Route.DISCOVERY
            else None,
            discovery_stage_status="passed" if route == Route.DISCOVERY else None,
            locate_raw_response=discovery_result.discovery_meta.get("locate_raw_response"),
            extract_raw_response=discovery_result.discovery_meta.get("extract_raw_response"),
            located_field_count=discovery_result.discovery_meta.get("located_field_count"),
            heuristic_confidence=discovery_result.discovery_meta.get("heuristic_confidence"),
            matched_label_count=discovery_result.discovery_meta.get("matched_label_count"),
            critical_field_count=discovery_result.discovery_meta.get("critical_field_count"),
            table_detected=discovery_result.discovery_meta.get("table_detected"),
            party_block_diagnostics=discovery_result.discovery_meta.get("party_block_diagnostics"),
            summary_candidate_diagnostics=discovery_result.discovery_meta.get(
                "summary_candidate_diagnostics"
            ),
            line_item_source=discovery_result.discovery_meta.get("line_item_source"),
        ),
        ground_truth=gt_data,
        extracted_data=extraction_result.extracted,
        normalized_data=extraction_result.normalized,
        template_status_at_time=match.status if match else None,
    )
    final_result = _apply_retry_context(
        final_result, gt_present=gt_present, retry_context=retry_context
    )
    return _complete_processing_result(
        final_result,
        start_time=start_time,
        settings=settings,
        invoice_path=invoice_path,
        db=db,
    )


def assess_healing_candidate(
    invoice_path: str,
    config: AppConfig,
    db: FingerprintDB,
) -> HealingAssessment:
    gt_data = pair_ground_truth(invoice_path)
    gt_present = gt_data is not None
    gt_discovery_ready = is_ground_truth_discovery_ready(gt_data) if gt_data else False
    prepared = _prepare_invoice_inputs(invoice_path, config)
    routing = _resolve_routing_decision(
        invoice_path,
        config,
        db,
        prepared,
        gt_present=gt_present,
        gt_discovery_ready=gt_discovery_ready,
    )
    authority = resolve_gt_healing_authority(
        routing.current_family_record,
        routing.apply_match,
        config.template_lifecycle,
    )
    trigger_match_type = None
    trigger_score = None
    if routing.route == Route.APPLY and routing.fingerprint_match is not None:
        trigger_match_type = "fingerprint"
        trigger_score = 1.0
    elif routing.route == Route.APPLY and routing.family_representative is not None:
        trigger_match_type = "family"
        trigger_score = routing.family_match_score

    return HealingAssessment(
        route=routing.route,
        current_family_id=routing.current_family_id,
        trigger_family_id=authority["template_family_id"],
        trigger_fingerprint_hash=authority["fingerprint_hash"],
        trigger_match_type=trigger_match_type,
        trigger_score=trigger_score,
        gt_trust_scope=authority["scope"],
        gt_trusted=bool(authority["trusted"]),
        gt_apply_count=int(authority["gt_apply_count"]),
        gt_reject_count=int(authority["gt_reject_count"]),
        gt_confidence=float(authority["gt_confidence"]),
    )
