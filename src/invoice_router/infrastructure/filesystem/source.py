import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...domain.ground_truth import normalize_ground_truth
from .paths import configured_dataset_root, resolve_dataset_dir

SUPPORTED_INVOICE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf"}
_SUBSET_DATASET_NAMES = {"invoices-small", "invoices-test"}


def list_invoices(input_dir: str) -> List[str]:
    """List invoice image/pdf paths in the input directory."""
    invoices = []
    try:
        path = resolve_dataset_dir(input_dir)
    except (FileNotFoundError, NotADirectoryError):
        return invoices
    if not path.exists() or not path.is_dir():
        return invoices

    for file in path.iterdir():
        if file.is_file() and file.suffix.lower() in SUPPORTED_INVOICE_EXTENSIONS:
            invoices.append(str(file))
    return sorted(invoices)


def resolve_ground_truth_source_dir(
    input_dir: str, *, dataset_root: Optional[str | Path] = None
) -> Path:
    dataset_dir = resolve_dataset_dir(input_dir, dataset_root=dataset_root)
    if dataset_dir.name in _SUBSET_DATASET_NAMES:
        source_dir = dataset_dir.parent / "invoices-all"
        if source_dir.exists() and source_dir.is_dir():
            return source_dir
        external_root = configured_dataset_root(dataset_root)
        external_source_dir = external_root / "invoices-all"
        if external_source_dir.exists() and external_source_dir.is_dir():
            return external_source_dir.resolve()
    return dataset_dir


def summarize_ground_truth_sync(
    input_dir: str,
    *,
    source_dir: Optional[str | Path] = None,
    dataset_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    dataset_dir = resolve_dataset_dir(input_dir, dataset_root=dataset_root)
    authoritative_dir = (
        resolve_dataset_dir(source_dir, dataset_root=dataset_root)
        if source_dir is not None
        else resolve_ground_truth_source_dir(input_dir, dataset_root=dataset_root)
    )

    invoice_stems = sorted(
        file.stem
        for file in dataset_dir.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_INVOICE_EXTENSIONS
    )

    matching = 0
    mismatched = 0
    missing_local = 0
    missing_authoritative = 0
    checked = 0

    for stem in invoice_stems:
        local_gt = dataset_dir / f"{stem}.json"
        authoritative_gt = authoritative_dir / f"{stem}.json"
        local_exists = local_gt.exists()
        authoritative_exists = authoritative_gt.exists()

        if authoritative_exists:
            checked += 1
        if authoritative_exists and not local_exists:
            missing_local += 1
            continue
        if local_exists and not authoritative_exists:
            missing_authoritative += 1
            continue
        if not local_exists and not authoritative_exists:
            continue
        if local_gt.read_bytes() == authoritative_gt.read_bytes():
            matching += 1
        else:
            mismatched += 1

    if dataset_dir == authoritative_dir:
        status = "self_contained"
    elif mismatched:
        status = "out_of_sync"
    elif missing_local:
        status = "missing_local_gt"
    elif missing_authoritative:
        status = "missing_source_gt"
    else:
        status = "in_sync"

    return {
        "dataset_dir": str(dataset_dir),
        "source_of_truth_dir": str(authoritative_dir),
        "uses_external_source_of_truth": dataset_dir != authoritative_dir,
        "status": status,
        "invoice_count": len(invoice_stems),
        "checked_invoice_count": checked,
        "matching_count": matching,
        "mismatched_count": mismatched,
        "missing_local_count": missing_local,
        "missing_source_count": missing_authoritative,
    }


def sync_ground_truth_from_source(
    input_dir: str,
    *,
    source_dir: Optional[str | Path] = None,
    dataset_root: Optional[str | Path] = None,
) -> Dict[str, Any]:
    dataset_dir = resolve_dataset_dir(input_dir, dataset_root=dataset_root)
    authoritative_dir = (
        resolve_dataset_dir(source_dir, dataset_root=dataset_root)
        if source_dir is not None
        else resolve_ground_truth_source_dir(input_dir, dataset_root=dataset_root)
    )

    summary = summarize_ground_truth_sync(
        input_dir, source_dir=authoritative_dir, dataset_root=dataset_root
    )
    if dataset_dir == authoritative_dir:
        return {
            **summary,
            "copied_count": 0,
            "updated_count": 0,
            "unchanged_count": summary["matching_count"],
        }

    copied_count = 0
    updated_count = 0
    unchanged_count = 0
    for stem in sorted(
        file.stem
        for file in dataset_dir.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_INVOICE_EXTENSIONS
    ):
        local_gt = dataset_dir / f"{stem}.json"
        authoritative_gt = authoritative_dir / f"{stem}.json"
        if not authoritative_gt.exists():
            continue
        local_exists = local_gt.exists()
        if local_exists and local_gt.read_bytes() == authoritative_gt.read_bytes():
            unchanged_count += 1
            continue
        shutil.copy2(authoritative_gt, local_gt)
        if local_exists:
            updated_count += 1
        else:
            copied_count += 1

    refreshed = summarize_ground_truth_sync(
        input_dir, source_dir=authoritative_dir, dataset_root=dataset_root
    )
    return {
        **refreshed,
        "copied_count": copied_count,
        "updated_count": updated_count,
        "unchanged_count": unchanged_count,
    }


def load_ground_truth(gt_path: Path) -> Optional[Dict[str, Any]]:
    """Load GT JSON from a given path."""
    if gt_path.exists():
        try:
            with open(gt_path, "r") as f:
                return normalize_ground_truth(json.load(f))
        except json.JSONDecodeError:
            pass
    return None


def pair_ground_truth(invoice_path: str) -> Optional[Dict[str, Any]]:
    """Pair an invoice path with its corresponding GT JSON file."""
    path = Path(invoice_path)
    gt_path = path.with_suffix(".json")
    return load_ground_truth(gt_path)
