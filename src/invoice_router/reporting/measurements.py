"""Lightweight benchmark-time runtime and memory measurements."""

from __future__ import annotations

import sys
import time
from typing import Callable, Optional, TypeVar

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on some platforms
    resource = None  # type: ignore[assignment]

T = TypeVar("T")


def read_peak_rss_mb() -> Optional[float]:
    """Return peak RSS for the current process in MB when supported."""
    if resource is None:
        return None

    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError, ValueError):
        return None

    try:
        raw_value = float(usage)
    except (TypeError, ValueError):
        return None

    if raw_value < 0:
        return None

    bytes_per_unit = 1.0 if sys.platform == "darwin" else 1024.0
    return round((raw_value * bytes_per_unit) / (1024.0 * 1024.0), 3)


def measure_operation(operation: Callable[[], T]) -> tuple[T, int, Optional[float]]:
    """Run an operation and capture elapsed wall time plus peak process RSS."""
    started = time.time()
    result = operation()
    elapsed_ms = int((time.time() - started) * 1000)
    return result, elapsed_ms, read_peak_rss_mb()
