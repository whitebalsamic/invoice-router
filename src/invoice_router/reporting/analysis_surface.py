import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

DEFAULT_EXCLUDED_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    ".venv",
    ".venv312",
    "__pycache__",
    "node_modules",
    "invoices-all",
    "invoices-test",
    "invoices-small",
    "invoices-all-quarantine",
    "data",
    "dist",
    "build",
}
DEFAULT_EXCLUDED_RELATIVE_PATHS = {
    "docs/assets/node_modules",
    "docs/assets/package-lock.json",
    "docs/assets/package.json",
    "dump.rdb",
}


def _estimate_tokens(char_count: int) -> int:
    return max(0, char_count // 4)


def _should_skip_dir(
    relative_dir: str,
    dir_name: str,
    excluded_dir_names: set[str],
    excluded_relative_paths: set[str],
) -> bool:
    if dir_name in excluded_dir_names:
        return True
    normalized = relative_dir.replace("\\", "/")
    return any(
        normalized == excluded or normalized.startswith(excluded + "/")
        for excluded in excluded_relative_paths
    )


def build_analysis_surface_report(
    workspace_root: str | Path,
    *,
    baseline_report: Optional[Dict[str, Any]] = None,
    excluded_dir_names: Optional[Iterable[str]] = None,
    excluded_relative_paths: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    root = Path(workspace_root).resolve()
    dir_name_exclusions = set(excluded_dir_names or DEFAULT_EXCLUDED_DIR_NAMES)
    relative_path_exclusions = set(excluded_relative_paths or DEFAULT_EXCLUDED_RELATIVE_PATHS)

    total_files = sum(1 for path in root.rglob("*") if path.is_file())
    included_files = 0
    included_lines = 0
    included_chars = 0
    included_by_top_level: Counter[str] = Counter()
    largest_included_paths: list[dict[str, Any]] = []
    excluded_path_matches: Counter[str] = Counter()

    for current_root, dirs, files in os.walk(root):
        current_path = Path(current_root)
        relative_dir = "." if current_path == root else str(current_path.relative_to(root))
        kept_dirs = []
        for dir_name in dirs:
            child_relative = dir_name if relative_dir == "." else f"{relative_dir}/{dir_name}"
            if _should_skip_dir(
                child_relative, dir_name, dir_name_exclusions, relative_path_exclusions
            ):
                excluded_path_matches[child_relative] += 1
                continue
            kept_dirs.append(dir_name)
        dirs[:] = kept_dirs

        for file_name in files:
            file_path = current_path / file_name
            relative_file = str(file_path.relative_to(root)).replace("\\", "/")
            if relative_file in relative_path_exclusions:
                excluded_path_matches[relative_file] += 1
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            included_files += 1
            line_count = text.count("\n") + (0 if not text else 1)
            char_count = len(text)
            included_lines += line_count
            included_chars += char_count
            parts = Path(relative_file).parts
            top_level = parts[0] if parts else "."
            included_by_top_level[top_level] += line_count
            largest_included_paths.append(
                {
                    "path": relative_file,
                    "lines": line_count,
                    "chars": char_count,
                    "estimated_tokens": _estimate_tokens(char_count),
                }
            )

    estimated_tokens = _estimate_tokens(included_chars)
    largest_included_paths.sort(
        key=lambda item: (item["estimated_tokens"], item["lines"], item["path"]),
        reverse=True,
    )

    report: Dict[str, Any] = {
        "workspace_root": str(root),
        "total_files": total_files,
        "included_files": included_files,
        "excluded_files": max(total_files - included_files, 0),
        "included_lines": included_lines,
        "estimated_tokens": estimated_tokens,
        "included_by_top_level": dict(included_by_top_level.most_common()),
        "largest_included_paths": largest_included_paths[:25],
        "excluded_path_matches": dict(excluded_path_matches.most_common()),
    }
    if baseline_report:
        report["delta_from_baseline"] = {
            "total_files": total_files - int(baseline_report.get("total_files", 0)),
            "included_files": included_files - int(baseline_report.get("included_files", 0)),
            "included_lines": included_lines - int(baseline_report.get("included_lines", 0)),
            "estimated_tokens": estimated_tokens - int(baseline_report.get("estimated_tokens", 0)),
        }
    return report


def write_analysis_surface_report(report: Dict[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return destination
