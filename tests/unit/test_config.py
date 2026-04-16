from pathlib import Path

import pytest

from invoice_router.config import AppConfig, MachineInfo, _apply_processing_profile, load_config
from invoice_router.infrastructure.filesystem.paths import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEMP_DIR,
)


def _minimal_app_config(processing):
    return AppConfig(
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
            "processing": processing,
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
            "provider_resolution": {"minimum_confidence": 0.75, "providers": {}},
            "field_mapping": {},
        }
    )


def test_apply_processing_profile_matches_machine_specific_profile():
    app_config = _minimal_app_config(
        {
            "worker_concurrency": 2,
            "default_profile": "default",
            "profiles": {
                "default": {"worker_concurrency": 4},
                "apple_silicon_workhorse": {"worker_concurrency": 8},
            },
            "machine_profiles": [
                {
                    "profile": "apple_silicon_workhorse",
                    "system": "Darwin",
                    "architecture": "arm64",
                    "model_name_patterns": ["MacBook Pro*"],
                }
            ],
        }
    )

    _apply_processing_profile(
        app_config,
        MachineInfo(
            system="Darwin",
            architecture="arm64",
            cpu_count=12,
            memory_gb=36.0,
            model_name="MacBook Pro",
            model_identifier="Mac15,6",
        ),
    )

    assert app_config.processing.worker_concurrency == 8
    assert app_config.processing.applied_profile == "apple_silicon_workhorse"
    assert app_config.processing.detected_model_name == "MacBook Pro"


def test_apply_processing_profile_falls_back_to_default_when_no_machine_match():
    app_config = _minimal_app_config(
        {
            "worker_concurrency": 2,
            "default_profile": "default",
            "profiles": {
                "default": {"worker_concurrency": 4},
                "apple_silicon_air": {"worker_concurrency": 3},
            },
            "machine_profiles": [
                {
                    "profile": "apple_silicon_air",
                    "system": "Darwin",
                    "architecture": "arm64",
                    "model_name_patterns": ["MacBook Air*"],
                }
            ],
        }
    )

    _apply_processing_profile(
        app_config,
        MachineInfo(
            system="Linux",
            architecture="x86_64",
            cpu_count=8,
            memory_gb=32.0,
            model_name=None,
            model_identifier=None,
        ),
    )

    assert app_config.processing.worker_concurrency == 4
    assert app_config.processing.applied_profile == "default"
    assert app_config.processing.detected_system == "Linux"


def test_load_config_fails_when_paddle_table_engine_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("INVOICE_INPUT_DIR", "/tmp/invoices")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://invoice_router:invoice_router@localhost:5432/invoice_router"
    )
    monkeypatch.setattr(
        "invoice_router.config.get_paddle_ocr_status",
        lambda: (False, "paddle runtime package is not installed"),
    )
    config_template = (Path(__file__).resolve().parents[2] / "config.yaml").read_text(
        encoding="utf-8"
    )
    config_path = tmp_path / "config-paddle.yaml"
    config_path.write_text(
        config_template.replace("table_engine: tesseract", "table_engine: paddle"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Configured OCR table engine 'paddle' is unavailable"):
        load_config(str(config_path))


def test_load_config_uses_external_data_defaults(monkeypatch):
    monkeypatch.setenv("INVOICE_INPUT_DIR", "/tmp/invoices")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://invoice_router:invoice_router@localhost:5432/invoice_router"
    )
    monkeypatch.setattr("invoice_router.config.get_paddle_ocr_status", lambda: (True, "ok"))

    settings, _ = load_config(str(Path(__file__).resolve().parents[2] / "config.yaml"))

    assert settings.dataset_root == str(DEFAULT_DATASET_ROOT)
    assert settings.output_dir == str(DEFAULT_OUTPUT_DIR)
    assert settings.temp_dir == str(DEFAULT_TEMP_DIR)
