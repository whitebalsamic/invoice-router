from ..domain.templates.fingerprinting import get_paddle_ocr_status
from .loader import load_config, resolve_config_path
from .machine_profiles import MachineInfo, _apply_processing_profile
from .schema import (
    AppConfig,
    DiscoveryConfig,
    FingerprintingConfig,
    HeuristicDiscoveryConfig,
    OcrConfig,
    ProcessingConfig,
    ProcessingMachineProfile,
    ProcessingProfileConfig,
    ProviderCatalogEntry,
    ProviderResolutionConfig,
    QualityConfig,
    TableDetectionConfig,
    TemplateLifecycleConfig,
    ValidationConfig,
)
from .settings import Settings

__all__ = [
    "AppConfig",
    "DiscoveryConfig",
    "FingerprintingConfig",
    "get_paddle_ocr_status",
    "HeuristicDiscoveryConfig",
    "load_config",
    "resolve_config_path",
    "MachineInfo",
    "OcrConfig",
    "ProcessingConfig",
    "ProcessingMachineProfile",
    "ProcessingProfileConfig",
    "ProviderCatalogEntry",
    "ProviderResolutionConfig",
    "QualityConfig",
    "Settings",
    "TableDetectionConfig",
    "TemplateLifecycleConfig",
    "ValidationConfig",
    "_apply_processing_profile",
]
