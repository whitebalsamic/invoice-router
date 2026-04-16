#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from invoice_router.config import load_config
from invoice_router.infrastructure.filesystem.paths import (
    DEFAULT_BACKUP_DIR,
    DEFAULT_OUTPUT_DIR,
    ensure_directory,
    resolve_dataset_dir,
)
from invoice_router.infrastructure.persistence.postgres import (
    benchmark_postgres_dsn,
    recreate_postgres_database,
)
from invoice_router.infrastructure.persistence.storage import FingerprintDB
from invoice_router.pipeline import process_single_invoice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate GT v2 locally without API or Ollama.")
    parser.add_argument("input_dir", help="Directory containing invoice images and GT JSON files.")
    parser.add_argument(
        "--limit", type=int, default=100, help="Number of unresolved files to process."
    )
    return parser.parse_args()


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _is_gt_v2(data: Optional[Dict[str, Any]]) -> bool:
    return isinstance(data, dict) and str(data.get("schemaVersion") or "").startswith("gt-v2")


def _find_image_path(root: Path, stem: str) -> Optional[Path]:
    for suffix in (".png", ".jpg", ".jpeg", ".pdf"):
        path = root / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def _field_status_from_provenance(parsed_value: Any, provenance: Optional[Dict[str, Any]]) -> str:
    if parsed_value is None:
        return "absent"
    kind = (provenance or {}).get("kind")
    if kind in {"derived", "resolved"}:
        return "derived"
    return "present"


def _scalar_field(
    value: Any, provenance: Optional[Dict[str, Any]], default_evidence: Optional[str] = None
) -> Dict[str, Any]:
    status = _field_status_from_provenance(value, provenance)
    field: Dict[str, Any] = {
        "status": status,
        "confidence": 0.55 if status == "derived" else 0.8 if status == "present" else 0.95,
    }
    raw_value = (provenance or {}).get("raw_value")
    evidence_items = list((provenance or {}).get("evidence") or [])
    if not evidence_items and raw_value not in (None, ""):
        evidence_items = [str(raw_value)]
    if not evidence_items and default_evidence:
        evidence_items = [default_evidence]
    if raw_value not in (None, ""):
        field["raw"] = raw_value
    if evidence_items:
        field["evidence"] = "; ".join(str(item) for item in evidence_items)
    if status != "absent":
        field["value"] = value
    return field


def _context_field(value: Any, label: str) -> Dict[str, Any]:
    if value in (None, ""):
        return {
            "status": "absent",
            "confidence": 0.95,
            "evidence": f"No {label} inferred locally",
        }
    return {
        "status": "present",
        "value": value,
        "raw": value,
        "evidence": f"Local document context inferred {label}",
        "confidence": 0.6,
    }


def _line_item_field(
    value: Any, provenance: Optional[Dict[str, Any]], default_evidence: Optional[str] = None
) -> Dict[str, Any]:
    return _scalar_field(value, provenance, default_evidence=default_evidence)


def _build_line_item(item: Dict[str, Any], index: int) -> Dict[str, Any]:
    raw_row = item.get("raw") or {}
    prov = item.get("_provenance") or {}

    explicit_tax_keys = [
        key for key in raw_row.keys() if "tax" in key.lower() or "vat" in key.lower()
    ]
    tax_status = "absent"
    tax_value = None
    tax_evidence = "Per-line tax not explicitly extracted"
    if explicit_tax_keys and item.get("tax_amount") is not None:
        tax_status = "present"
        tax_value = item.get("tax_amount")
        tax_evidence = f"Explicit tax columns present: {', '.join(explicit_tax_keys)}"
    elif item.get("tax_amount") is not None:
        tax_status = "derived"
        tax_value = item.get("tax_amount")
        tax_evidence = "Derived from local line-item normalization"

    tax_field: Dict[str, Any] = {
        "status": tax_status,
        "confidence": 0.55 if tax_status == "derived" else 0.8 if tax_status == "present" else 0.95,
        "evidence": tax_evidence,
    }
    if tax_value is not None:
        tax_field["value"] = tax_value
    if explicit_tax_keys:
        first_key = explicit_tax_keys[0]
        if raw_row.get(first_key) not in (None, ""):
            tax_field["raw"] = raw_row.get(first_key)

    description_value = item.get("description")
    description_field = {
        "status": "present" if description_value else "absent",
        "confidence": 0.82 if description_value else 0.95,
        "evidence": "Local OCR line item description"
        if description_value
        else "No description extracted",
    }
    if description_value:
        description_field["value"] = description_value
        description_field["raw"] = raw_row.get("description", description_value)

    return {
        "index": index,
        "description": description_field,
        "quantity": _line_item_field(item.get("quantity"), prov.get("quantity")),
        "unitPrice": _line_item_field(item.get("unit_price"), prov.get("unit_price")),
        "amount": _line_item_field(item.get("amount"), prov.get("amount")),
        "tax": tax_field,
        "taxRate": _line_item_field(item.get("tax_rate"), prov.get("tax_rate")),
        "sku": {
            "status": "absent",
            "confidence": 0.95,
            "evidence": "No SKU extracted locally",
        },
        "itemCode": {
            "status": "absent",
            "confidence": 0.95,
            "evidence": "No item code extracted locally",
        },
    }


def _build_gt_v2(image_path: Path, normalized: Dict[str, Any], diagnostics: Any) -> Dict[str, Any]:
    field_prov = normalized.get("field_provenance") or {}
    line_items = normalized.get("line_items") or []

    notes: List[str] = ["Generated locally from heuristic OCR pipeline without API/Ollama."]
    if diagnostics and getattr(diagnostics, "validation_errors", None):
        notes.append(
            f"Local extraction had {len(diagnostics.validation_errors)} validation warnings against legacy GT."
        )

    payload = {
        "schemaVersion": "gt-v2",
        "sourceImage": image_path.name,
        "annotator": {
            "type": "local_pipeline",
            "model": "heuristic-ocr",
            "reviewed": False,
        },
        "document": {
            "invoiceNumber": _scalar_field(
                normalized.get("invoiceNumber"), field_prov.get("invoice_number")
            ),
            "invoiceDate": _scalar_field(
                normalized.get("invoiceDate"), field_prov.get("invoice_date")
            ),
            "sellerName": _scalar_field(
                normalized.get("sellerName"), field_prov.get("provider_name")
            ),
            "customerName": _scalar_field(
                normalized.get("customerName"), field_prov.get("customer_name")
            ),
            "currency": _context_field(normalized.get("currency_code"), "currency"),
            "country": _context_field(normalized.get("country_code"), "country"),
        },
        "summary": {
            "subtotal": _scalar_field(normalized.get("subtotal"), field_prov.get("subtotal")),
            "tax": _scalar_field(normalized.get("tax"), field_prov.get("tax")),
            "discount": _scalar_field(normalized.get("discount"), field_prov.get("discount")),
            "shipping": {
                "status": "absent",
                "confidence": 0.95,
                "evidence": "No shipping field extracted locally",
            },
            "totalAmount": _scalar_field(normalized.get("totalAmount"), field_prov.get("total")),
        },
        "lineItems": [_build_line_item(item, index + 1) for index, item in enumerate(line_items)],
        "notes": notes,
    }
    return payload


def _candidate_is_reasonable(payload: Dict[str, Any]) -> bool:
    document = payload.get("document") or {}
    summary = payload.get("summary") or {}
    line_items = payload.get("lineItems") or []
    present_document = sum(
        1
        for value in document.values()
        if isinstance(value, dict) and value.get("status") in {"present", "derived"}
    )
    present_summary = sum(
        1
        for value in summary.values()
        if isinstance(value, dict) and value.get("status") in {"present", "derived"}
    )
    return present_document >= 2 or present_summary >= 2 or bool(line_items)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n")


def main() -> int:
    args = parse_args()
    root = resolve_dataset_dir(args.input_dir)
    unresolved: List[str] = []
    for json_path in sorted(root.glob("*.json")):
        data = _load_json(json_path)
        if not _is_gt_v2(data):
            unresolved.append(json_path.stem)
    targets = unresolved[: args.limit]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = ensure_directory(DEFAULT_BACKUP_DIR) / f"gt-local-{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    settings, config = load_config(str(REPO_ROOT / "config.yaml"))
    settings.output_dir = str(ensure_directory(DEFAULT_OUTPUT_DIR / "local-gt-output"))
    settings.database_url = benchmark_postgres_dsn(settings.database_url, f"local_gt_{timestamp}")
    settings.analysis_database_url = None
    settings.benchmark_database_url = None
    config.discovery.backend = "heuristic"
    config.ocr.single_field_engine = "tesseract"
    config.ocr.table_engine = "tesseract"
    recreate_postgres_database(settings.database_url)

    db = FingerprintDB(settings.database_url)
    db.clear_processing_results()
    db.clear_fingerprints()

    failures: List[Dict[str, str]] = []
    processed = 0
    success = 0

    print(f"Processing {len(targets)} unresolved invoices locally into GT v2...")
    for stem in targets:
        processed += 1
        json_path = root / f"{stem}.json"
        image_path = _find_image_path(root, stem)
        if image_path is None:
            failures.append({"file": stem, "error": "image file not found"})
            print(
                f"[{processed}/{len(targets)}] ok={success} failed={len(failures)} current={stem} image missing"
            )
            continue

        shutil.copy2(json_path, backup_dir / json_path.name)

        try:
            result = process_single_invoice(str(image_path), settings, config, db)
            normalized = result.normalized_data or {}
            candidate = _build_gt_v2(image_path, normalized, result.diagnostics)
            if not _candidate_is_reasonable(candidate):
                raise ValueError("candidate GT too sparse")
            _write_json(json_path, candidate)
            success += 1
            print(
                f"[{processed}/{len(targets)}] ok={success} failed={len(failures)} current={stem}"
            )
        except Exception as exc:
            failures.append({"file": stem, "error": str(exc)})
            print(
                f"[{processed}/{len(targets)}] ok={success} failed={len(failures)} current={stem} error={exc}"
            )

    if failures:
        report_path = backup_dir / "failures.json"
        _write_json(report_path, {"failures": failures})
        print(f"Wrote failures to {report_path}")

    print(
        f"Completed local GT regeneration for {len(targets)} files. ok={success} failed={len(failures)} backup={backup_dir}"
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
