from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .enums import DocumentFamily, ExtractionStrategy, Route, SourceFormat, TemplateStatus


class Provenance(BaseModel):
    request_id: str
    route: Optional[Route]
    fingerprint_hash: Optional[str]
    determination_sources: List[str] = Field(default_factory=list)
    gt_backed: bool = False
    healing_reprocessed: bool = False
    healing_origin: Optional[str] = None
    healing_attempt_count: int = 0
    previous_route: Optional[Route] = None
    previous_validation_passed: Optional[bool] = None
    healing_trigger_family_id: Optional[str] = None
    healing_trigger_fingerprint_hash: Optional[str] = None
    healing_trigger_match_type: Optional[str] = None
    healing_trigger_score: Optional[float] = None
    prior_result_provenance: Optional[Dict[str, Any]] = None
    template_family_id: Optional[str] = None
    extraction_strategy: Optional[ExtractionStrategy] = None
    source_format: Optional[SourceFormat] = None
    document_family: Optional[DocumentFamily] = None
    country_code: Optional[str] = None
    currency_code: Optional[str] = None
    provider_name: Optional[str] = None
    provider_confidence: Optional[float] = None
    template_status_at_time: Optional[TemplateStatus]
    template_confidence_at_time: Optional[float]
    ocr_engine: Optional[str]
    inference_method: Optional[str]
    image_quality_score: Optional[List[float]]
    quality_flag: bool
    input_document_hash: str
    per_page_hashes: List[str]
    extraction_output_hash: Optional[str]
    started_at: str
    completed_at: str
    latency_ms: int


class QualityMetrics(BaseModel):
    blur_score: float
    contrast_score: float
    quality_score: float
    quality_flag: bool


class ValidationResult(BaseModel):
    passed: bool
    score: float
    matched_fields: int
    mismatched_fields: int
    errors: List[str]
    arithmetic_errors: List[str] = Field(default_factory=list)
    error_counts: Dict[str, int] = Field(default_factory=dict)
    reconciliation_summary: Optional[Dict[str, Any]] = None


class ProcessingDiagnostics(BaseModel):
    attempted_route: Optional[Route] = None
    template_family_id: Optional[str] = None
    family_match_attempted: Optional[bool] = None
    family_match_outcome: Optional[str] = None
    family_match_score: Optional[float] = None
    family_match_method: Optional[str] = None
    family_match_candidate_count: Optional[int] = None
    wrong_template_suspected: Optional[bool] = None
    normalization_failure: Optional[bool] = None
    reconciliation_failure: Optional[bool] = None
    reconciliation_status: Optional[str] = None
    reconciliation_issue_types: List[str] = Field(default_factory=list)
    reconciliation_derived_fields: List[str] = Field(default_factory=list)
    validator_policy_failure: Optional[bool] = None
    extraction_strategy: Optional[ExtractionStrategy] = None
    source_format: Optional[SourceFormat] = None
    document_family: Optional[DocumentFamily] = None
    provider_name: Optional[str] = None
    provider_confidence: Optional[float] = None
    country_code: Optional[str] = None
    currency_code: Optional[str] = None
    discovery_mode: Optional[str] = None
    discovery_stage_status: Optional[str] = None
    locate_error_category: Optional[str] = None
    extract_error_category: Optional[str] = None
    locate_failure_detail: Optional[str] = None
    extract_failure_detail: Optional[str] = None
    locate_raw_response: Optional[str] = None
    extract_raw_response: Optional[str] = None
    validation_score: Optional[float] = None
    validation_errors: List[str] = Field(default_factory=list)
    validation_error_counts: Dict[str, int] = Field(default_factory=dict)
    extracted_field_count: Optional[int] = None
    gt_line_item_count: Optional[int] = None
    extracted_line_item_count: Optional[int] = None
    scalar_field_missing_count: Optional[int] = None
    scalar_field_mismatch_count: Optional[int] = None
    discovered_field_count: Optional[int] = None
    located_field_count: Optional[int] = None
    label_confirmation_count: Optional[int] = None
    heuristic_confidence: Optional[float] = None
    matched_label_count: Optional[int] = None
    critical_field_count: Optional[int] = None
    table_detected: Optional[bool] = None
    party_block_diagnostics: List[Dict[str, Any]] = Field(default_factory=list)
    summary_candidate_diagnostics: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    line_item_source: Optional[str] = None
    retry_context: Dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    extracted_data: Dict[str, Any]
    ground_truth: Optional[Dict[str, Any]]
    provenance: Provenance
    route: Route
    validation_passed: Optional[bool]
    match_percentage: Optional[float]


class ProcessingResult(BaseModel):
    id: Optional[int] = None
    invoice_path: str
    fingerprint_hash: Optional[str]
    template_family_id: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]]
    normalized_data: Optional[Dict[str, Any]] = None
    ground_truth: Optional[Dict[str, Any]]
    provenance: Optional[Provenance]
    validation_passed: Optional[bool]
    route_used: Optional[Route]
    attempted_route: Optional[Route] = None
    diagnostics: Optional[ProcessingDiagnostics] = None
    image_quality_score: Optional[float]
    template_status_at_time: Optional[TemplateStatus]
    processed_at: str
