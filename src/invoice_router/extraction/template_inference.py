from typing import Any, Dict, List, Tuple

import numpy as np

from .heuristics.discovery import infer_template_heuristic


def infer_template(
    pages: List[np.ndarray],
    gt_json: Dict[str, Any],
    config,
    ocr_results_per_page: List[List[tuple]] = None,
) -> Tuple[Dict[str, Any], float, Dict[str, Any], Dict[str, Any]]:
    return infer_template_heuristic(
        pages,
        gt_json,
        config,
        ocr_results_per_page=ocr_results_per_page,
    )
