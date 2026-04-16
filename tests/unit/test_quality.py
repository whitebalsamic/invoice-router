import numpy as np

from invoice_router.config import QualityConfig
from invoice_router.models import QualityMetrics
from invoice_router.pipeline.quality import apply_quality_gate, assess_quality


def _checkerboard(size: int = 64) -> np.ndarray:
    pattern = ((np.indices((size, size)).sum(axis=0) % 2) * 255).astype(np.uint8)
    return np.stack([pattern, pattern, pattern], axis=-1)


def test_assess_quality_flags_flat_low_contrast_image():
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    config = QualityConfig(
        blur_threshold=100.0,
        contrast_threshold=0.3,
        quality_threshold=0.05,
        quality_region_buffer_multiplier=2,
    )

    metrics = assess_quality(image, config)

    assert metrics.blur_score == 0.0
    assert metrics.contrast_score == 0.0
    assert metrics.quality_score == 0.0
    assert metrics.quality_flag is True


def test_assess_quality_accepts_high_detail_high_contrast_image():
    image = _checkerboard()
    config = QualityConfig(
        blur_threshold=10.0,
        contrast_threshold=0.3,
        quality_threshold=0.3,
        quality_region_buffer_multiplier=2,
    )

    metrics = assess_quality(image, config)

    assert metrics.blur_score > 0.0
    assert metrics.contrast_score > 0.9
    assert metrics.quality_score > 0.3
    assert metrics.quality_flag is False


def test_apply_quality_gate_expands_buffer_when_any_page_is_flagged():
    metrics_list = [
        QualityMetrics(blur_score=150.0, contrast_score=0.8, quality_score=0.8, quality_flag=False),
        QualityMetrics(blur_score=10.0, contrast_score=0.1, quality_score=0.01, quality_flag=True),
    ]

    quality_flag, adjusted_buffer = apply_quality_gate(metrics_list, base_buffer=5, multiplier=3)

    assert quality_flag is True
    assert adjusted_buffer == 15
