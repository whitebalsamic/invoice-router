from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ValidationConfig(BaseModel):
    apply_threshold: float = 0.90
    discovery_threshold: float = 0.95
    jaccard_threshold: float = 0.85
    allow_cent_scale_equivalence: bool = False


class FingerprintingConfig(BaseModel):
    visual_hash_hamming_threshold: float = 10


class TemplateLifecycleConfig(BaseModel):
    establish_min_count: int = 5
    establish_min_confidence: float = 0.95
    degradation_threshold: float = 0.85
    degradation_window: int = 10
    rediscovery_attempts: int = 3


class QualityConfig(BaseModel):
    blur_threshold: float = 100.0
    contrast_threshold: float = 0.3
    quality_threshold: float = 0.5
    quality_region_buffer_multiplier: int = 2


class ProcessingProfileConfig(BaseModel):
    batch_size: Optional[int] = None
    worker_concurrency: Optional[int] = None


class ProcessingMachineProfile(BaseModel):
    profile: str
    system: Optional[str] = None
    architecture: Optional[str] = None
    model_name_patterns: List[str] = Field(default_factory=list)
    model_identifier_patterns: List[str] = Field(default_factory=list)
    min_memory_gb: Optional[float] = None
    max_memory_gb: Optional[float] = None
    min_cpu_count: Optional[int] = None
    max_cpu_count: Optional[int] = None


class ProcessingConfig(BaseModel):
    batch_size: int = 50
    worker_concurrency: int = 8
    auto_profile: bool = True
    default_profile: Optional[str] = "default"
    profiles: Dict[str, ProcessingProfileConfig] = Field(default_factory=dict)
    machine_profiles: List[ProcessingMachineProfile] = Field(default_factory=list)
    applied_profile: Optional[str] = None
    detected_system: Optional[str] = None
    detected_architecture: Optional[str] = None
    detected_model_name: Optional[str] = None
    detected_model_identifier: Optional[str] = None
    detected_memory_gb: Optional[float] = None
    detected_cpu_count: Optional[int] = None


class OcrConfig(BaseModel):
    single_field_engine: str = "tesseract"
    table_engine: str = "paddle"


class TableDetectionConfig(BaseModel):
    min_line_span_fraction: float = 0.40
    column_gap_px: int = 20
    row_gap_multiplier: float = 1.5


class DiscoveryConfig(BaseModel):
    inference_confidence_threshold: float = 0.60
    label_confirmation_threshold: float = 0.70
    label_position_tolerance: float = 0.05
    family_anchor_threshold: float = 0.55
    family_apply_threshold: float = 0.70
    family_page_count_tolerance: int = 1


class HeuristicDiscoveryConfig(BaseModel):
    enabled: bool = True
    min_label_score: float = 0.72
    min_page_confidence: float = 0.35
    prefer_right_of_label: bool = True
    max_region_width_fraction: float = 0.45
    address_region_height_multiplier: float = 4.0
    enable_table_detection: bool = False
    min_table_header_score: float = 0.75


class ProviderCatalogEntry(BaseModel):
    aliases: List[str] = Field(default_factory=list)
    country_code: Optional[str] = None


class ProviderResolutionConfig(BaseModel):
    minimum_confidence: float = 0.75
    providers: Dict[str, ProviderCatalogEntry] = Field(default_factory=dict)


class AppConfig(BaseModel):
    validation: ValidationConfig
    fingerprinting: FingerprintingConfig
    template_lifecycle: TemplateLifecycleConfig
    quality: QualityConfig
    processing: ProcessingConfig
    region_buffer_pixels: int = 5
    ocr: OcrConfig
    table_detection: TableDetectionConfig
    discovery: DiscoveryConfig
    heuristic_discovery: HeuristicDiscoveryConfig = Field(default_factory=HeuristicDiscoveryConfig)
    provider_resolution: ProviderResolutionConfig = Field(default_factory=ProviderResolutionConfig)
    field_mapping: Dict[str, str]
