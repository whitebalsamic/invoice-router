import cv2
import numpy as np

from ..config import QualityConfig
from ..models import QualityMetrics


def assess_quality(image: np.ndarray, config: QualityConfig) -> QualityMetrics:
    """
    Compute QualityMetrics for a preprocessed page image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Blur score: Laplacian variance
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Contrast score: Spread of the image histogram (98th percentile - 2nd percentile, normalized to [0,1])
    percentile_2 = np.percentile(gray, 2)
    percentile_98 = np.percentile(gray, 98)
    contrast_score = (percentile_98 - percentile_2) / 255.0

    # Quality score: min(blur_score / blur_threshold, 1.0) * contrast_score
    quality_score = min(blur_score / config.blur_threshold, 1.0) * contrast_score

    quality_flag = quality_score < config.quality_threshold

    return QualityMetrics(
        blur_score=float(blur_score),
        contrast_score=float(contrast_score),
        quality_score=float(quality_score),
        quality_flag=quality_flag,
    )


def apply_quality_gate(
    metrics_list: list[QualityMetrics], base_buffer: int, multiplier: int
) -> tuple[bool, int]:
    """
    Evaluates quality metrics across all pages and determines if the quality flag should be set,
    adjusting the region buffer pixels if necessary.

    Returns: (quality_flag, adjusted_region_buffer_pixels)
    """
    quality_flag = any(m.quality_flag for m in metrics_list)
    adjusted_buffer = base_buffer * multiplier if quality_flag else base_buffer
    return quality_flag, adjusted_buffer
