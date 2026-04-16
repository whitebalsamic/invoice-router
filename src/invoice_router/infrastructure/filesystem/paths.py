import os
from pathlib import Path
from typing import Optional

DEFAULT_EXTERNAL_DATA_ROOT = Path.home() / "Desktop" / "data"
DEFAULT_DATASET_ROOT = DEFAULT_EXTERNAL_DATA_ROOT / "datasets"
DEFAULT_OUTPUT_DIR = DEFAULT_EXTERNAL_DATA_ROOT / "output"
DEFAULT_TEMP_DIR = DEFAULT_EXTERNAL_DATA_ROOT / "temp"
DEFAULT_BACKUP_DIR = DEFAULT_EXTERNAL_DATA_ROOT / "backups"
DEFAULT_ANALYSIS_SURFACE_REPORT = DEFAULT_OUTPUT_DIR / "analysis_surface_report.json"


def configured_dataset_root(dataset_root: Optional[str | Path] = None) -> Path:
    if dataset_root is not None:
        return Path(dataset_root).expanduser()
    return Path(os.environ.get("DATASET_ROOT", DEFAULT_DATASET_ROOT)).expanduser()


def resolve_dataset_dir(
    input_dir: str | Path,
    *,
    dataset_root: Optional[str | Path] = None,
    must_exist: bool = True,
) -> Path:
    raw = Path(input_dir).expanduser()
    looks_like_path = raw.is_absolute() or len(raw.parts) > 1 or str(input_dir).startswith(".")
    if looks_like_path:
        if raw.exists():
            if raw.is_dir():
                return raw.resolve()
            raise NotADirectoryError(f"Dataset path is not a directory: {raw}")
        if must_exist:
            raise FileNotFoundError(f"Dataset path does not exist: {raw}")
        return raw

    candidate = configured_dataset_root(dataset_root) / raw
    if candidate.exists():
        if candidate.is_dir():
            return candidate.resolve()
        raise NotADirectoryError(f"Resolved dataset path is not a directory: {candidate}")
    if raw.exists():
        if raw.is_dir():
            return raw.resolve()
        raise NotADirectoryError(f"Dataset path is not a directory: {raw}")
    if must_exist:
        raise FileNotFoundError(
            f"Dataset '{input_dir}' was not found. Checked '{candidate}'. "
            "Pass an absolute path or configure DATASET_ROOT."
        )
    return candidate


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory
