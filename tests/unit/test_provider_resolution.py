from invoice_router.config import AppConfig
from invoice_router.domain.invoices.provider_resolution import resolve_provider


def test_resolve_provider_from_alias_match():
    config = AppConfig(
        **{
            "validation": {
                "apply_threshold": 0.9,
                "discovery_threshold": 0.95,
                "jaccard_threshold": 0.85,
            },
            "fingerprinting": {"visual_hash_hamming_threshold": 10},
            "template_lifecycle": {
                "establish_min_count": 5,
                "establish_min_confidence": 0.95,
                "degradation_threshold": 0.85,
                "degradation_window": 10,
                "rediscovery_attempts": 3,
            },
            "quality": {
                "blur_threshold": 100.0,
                "contrast_threshold": 0.3,
                "quality_threshold": 0.5,
                "quality_region_buffer_multiplier": 2,
            },
            "processing": {"batch_size": 50, "worker_concurrency": 4},
            "region_buffer_pixels": 5,
            "ocr": {"single_field_engine": "tesseract", "table_engine": "paddle"},
            "table_detection": {
                "min_line_span_fraction": 0.4,
                "column_gap_px": 20,
                "row_gap_multiplier": 1.5,
            },
            "discovery": {
                "inference_confidence_threshold": 0.6,
                "label_confirmation_threshold": 0.7,
                "label_position_tolerance": 0.05,
            },
            "provider_resolution": {
                "minimum_confidence": 0.75,
                "providers": {
                    "Vet Clinic Group": {
                        "aliases": ["VCG", "Vet Clinic"],
                        "country_code": "US",
                    }
                },
            },
            "field_mapping": {},
        }
    )

    match = resolve_provider([[("Welcome", 0, 0, 1, 1), ("VCG", 0, 0, 1, 1)]], config)

    assert match is not None
    assert match.provider_name == "Vet Clinic Group"
    assert match.country_code == "US"
    assert match.confidence == 1.0
