import json
import logging
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import click

from .. import __version__
from ..config import load_config
from ..demo import create_demo_dataset
from ..domain.templates.lifecycle import (
    manually_update_template_family,
    merge_template_families,
    retire_template_family,
    split_template_family,
)
from ..infrastructure.filesystem.paths import resolve_dataset_dir
from ..infrastructure.filesystem.source import (
    list_invoices,
    summarize_ground_truth_sync,
    sync_ground_truth_from_source,
)
from ..infrastructure.persistence.postgres import (
    benchmark_postgres_dsn,
    is_postgres_target,
    recreate_postgres_database,
)
from ..infrastructure.persistence.storage import FingerprintDB
from ..models import DocumentFamily, Route, TemplateStatus
from ..pipeline import assess_healing_candidate, process_single_invoice
from ..reporting import (
    build_issue_ledger,
    build_process_run_summary,
    compare_analysis_runs,
    compare_benchmark_summaries,
    describe_template_family,
    record_analysis_run,
    suggest_template_family_merges,
    suggest_template_family_retirements,
    suggest_template_family_splits,
    suggest_template_family_updates,
    summarize_failure_modes,
    summarize_family_benchmark_comparison,
    summarize_template_families,
    write_run_summary,
)
from ..reporting.measurements import measure_operation

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLES_ROOT = REPO_ROOT / "samples"
DEFAULT_STRUCTURAL_SUBSET_FILE = (
    SAMPLES_ROOT / "benchmarks" / "invoices_small_structural_subset.json"
)
logger = logging.getLogger(__name__)


def _primary_database_target(settings):
    target = getattr(settings, "primary_database_target", getattr(settings, "database_url", None))
    if not is_postgres_target(target):
        raise click.ClickException(f"DATABASE_URL must be a Postgres DSN, got: {target}")
    return target


def _analysis_database_target(settings):
    target = getattr(
        settings,
        "analysis_database_target",
        getattr(settings, "analysis_database_url", None) or getattr(settings, "database_url", None),
    )
    if not is_postgres_target(target):
        raise click.ClickException(f"ANALYSIS_DATABASE_URL must be a Postgres DSN, got: {target}")
    return target


def _benchmark_database_target(settings, dataset_name: str, variant: str):
    explicit_target = getattr(settings, "benchmark_database_url", None)
    if explicit_target:
        if not is_postgres_target(explicit_target):
            raise click.ClickException(
                f"BENCHMARK_DATABASE_URL must be a Postgres DSN, got: {explicit_target}"
            )
        recreate_postgres_database(explicit_target)
        return explicit_target

    primary_target = _primary_database_target(settings)
    benchmark_target = benchmark_postgres_dsn(primary_target, f"benchmark_{dataset_name}_{variant}")
    recreate_postgres_database(benchmark_target)
    return benchmark_target


def _make_thread_local_benchmark_db_getter(db_target: str):
    state = threading.local()

    def _get_db():
        db = getattr(state, "db", None)
        if db is None:
            db = FingerprintDB(db_target)
            state.db = db
        return db

    return _get_db


def _resolve_input_dir(input_dir, settings) -> str:
    try:
        return str(
            resolve_dataset_dir(input_dir, dataset_root=getattr(settings, "dataset_root", None))
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise click.ClickException(str(exc)) from exc


def _normalize_metric_key(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _find_value_by_alias(data, aliases):
    if not isinstance(data, dict):
        return None
    normalized = {_normalize_metric_key(key): val for key, val in data.items()}
    for alias in aliases:
        key = _normalize_metric_key(alias)
        if key in normalized:
            return normalized[key]
    return None


_AMBIGUOUS_SLASH_DATE_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{2,4})\s*$")


def _format_validation_score(score):
    if score is None:
        return "n/a"
    return f"{score * 100:.1f}%"


def _emit_family_priority_block(summary, *, heading: str = "Highest-value unstable families"):
    family_metrics = summary.get("family_metrics") or {}
    unstable_families = family_metrics.get("highest_value_unstable_families") or []
    if not unstable_families:
        return
    click.echo(heading + ":")
    for family in unstable_families[:5]:
        pass_rate = family.get("pass_rate")
        pass_rate_text = f"{pass_rate * 100:.1f}%" if isinstance(pass_rate, (int, float)) else "n/a"
        click.echo(
            "  "
            f"{family.get('template_family_id')}: "
            f"unresolved={family.get('unresolved_count', 0)} "
            f"| invoices={family.get('invoice_count', 0)} "
            f"| pass_rate={pass_rate_text} "
            f"| impact={family.get('impact_score', 0)}"
        )


def _simplify_validation_error(error):
    text = str(error or "").strip()
    if text.startswith("Missing extracted field:"):
        return f"missing {text.split(':', 1)[1].strip()}"
    if text.startswith("Mismatch on "):
        field_name = text[len("Mismatch on ") :].split(":", 1)[0].strip()
        return f"mismatch on {field_name}"
    if text.startswith("Line item count mismatch:"):
        return "line-item count mismatch"
    if text.startswith("Line item arithmetic mismatch"):
        return "line-item arithmetic mismatch"
    if text.startswith("Subtotal mismatch:"):
        return "subtotal does not match line items"
    if text.startswith("Total mismatch:"):
        return "total does not match subtotal + tax"
    if text.startswith("Currency mismatch for country"):
        return "country/currency mismatch"
    return text


def _invoice_date_review_signal(result):
    extracted = result.extracted_data or {}
    normalized = result.normalized_data or {}
    raw_date = _find_value_by_alias(
        extracted, ["invoiceDate", "invoice date", "date", "issue date", "billing date"]
    )
    normalized_date = normalized.get("invoice_date") or normalized.get("invoiceDate")
    country_code = (result.diagnostics.country_code if result.diagnostics else None) or (
        result.provenance.country_code if result.provenance else None
    )
    if not raw_date or not normalized_date or not country_code:
        return None
    match = _AMBIGUOUS_SLASH_DATE_RE.match(str(raw_date))
    if not match:
        return None
    first, second, _year = match.groups()
    if int(first) > 12 or int(second) > 12:
        return None
    return f"ambiguous date {raw_date} -> {normalized_date} ({country_code})"


def _review_signals_for_result(result):
    if result is None:
        return []

    diagnostics = result.diagnostics
    provenance = result.provenance
    signals = []

    date_signal = _invoice_date_review_signal(result)
    if date_signal:
        signals.append(date_signal)

    if provenance and provenance.quality_flag:
        signals.append("low image quality")

    if (
        diagnostics
        and diagnostics.provider_name
        and diagnostics.provider_confidence is not None
        and diagnostics.provider_confidence < 0.85
    ):
        signals.append(
            f"weak provider match {diagnostics.provider_name} ({diagnostics.provider_confidence:.2f})"
        )

    if diagnostics and diagnostics.line_item_source == "row_fallback":
        signals.append("line items came from row fallback")

    if diagnostics and diagnostics.table_detected is False:
        signals.append("no table detected")

    if diagnostics:
        simplified_errors = []
        for error in diagnostics.validation_errors:
            simplified = _simplify_validation_error(error)
            if simplified not in simplified_errors:
                simplified_errors.append(simplified)
            if len(simplified_errors) == 2:
                break
        signals.extend(simplified_errors)

    return signals[:3]


def _display_status_for_result(result):
    if result is None:
        return "ERROR"
    if result.route_used and result.route_used.value == "REJECTED":
        return "REVIEW"
    if result.validation_passed is False:
        return "REVIEW"
    if result.validation_passed is True:
        return "ACCEPT"
    if _review_signals_for_result(result):
        return "CHECK"
    return "EXTRACT"


def _format_result_summary_line(invoice_name, result, elapsed_ms):
    status = _display_status_for_result(result)
    route = result.route_used.value if result and result.route_used else "None"
    diagnostics = result.diagnostics if result else None
    provenance = result.provenance if result else None
    score = _format_validation_score(diagnostics.validation_score if diagnostics else None)
    context_bits = []
    if diagnostics and diagnostics.country_code:
        context_bits.append(diagnostics.country_code)
    if diagnostics and diagnostics.currency_code:
        context_bits.append(diagnostics.currency_code)
    if diagnostics and diagnostics.provider_name:
        context_bits.append(diagnostics.provider_name)
    elif provenance and provenance.provider_name:
        context_bits.append(provenance.provider_name)

    line = f"{invoice_name}: {status} | route={route} | {elapsed_ms}ms"
    if diagnostics and diagnostics.validation_score is not None:
        line += f" | score={score}"
    if context_bits:
        line += f" | {' / '.join(context_bits[:2])}"

    signals = _review_signals_for_result(result)
    if signals:
        line += f" | {'; '.join(signals)}"
    return line


def _serialize_result_provenance(result):
    if result is None or result.provenance is None:
        return None
    return result.provenance.model_dump(mode="json")


def _retry_context_from_result(previous_result, assessment, *, healing_origin: str):
    previous_provenance = _serialize_result_provenance(previous_result)
    prior_attempts = 0
    if previous_result and previous_result.provenance:
        prior_attempts = int(previous_result.provenance.healing_attempt_count or 0)
    return {
        "healing_origin": healing_origin,
        "healing_attempt_count": prior_attempts + 1,
        "previous_result": {
            "invoice_path": previous_result.invoice_path if previous_result else None,
            "route_used": previous_result.route_used.value
            if previous_result and previous_result.route_used
            else None,
            "validation_passed": previous_result.validation_passed if previous_result else None,
            "provenance": previous_provenance,
        },
        "trigger_family_id": assessment.trigger_family_id,
        "trigger_fingerprint_hash": assessment.trigger_fingerprint_hash,
        "trigger_match_type": assessment.trigger_match_type,
        "trigger_score": assessment.trigger_score,
    }


def _heal_rejections_pass(
    input_dir,
    settings,
    config,
    db,
    *,
    healing_origin: str,
    template_family_id=None,
    limit=None,
    show_per_invoice=False,
    force=False,
):
    candidates = db.get_failed_results_for_input_dir(
        input_dir,
        template_family_id=template_family_id,
        limit=limit,
    )
    summary = {
        "candidates": len(candidates),
        "attempted": 0,
        "recovered": 0,
        "still_failing": 0,
        "skipped_as_already_attempted": 0,
        "skipped_not_apply": 0,
        "skipped_not_gt_trusted": 0,
        "entries": [],
    }

    for previous_result in candidates:
        prior_attempts = 0
        if previous_result.provenance:
            prior_attempts = int(previous_result.provenance.healing_attempt_count or 0)
        if prior_attempts > 0 and not force:
            summary["skipped_as_already_attempted"] += 1
            continue

        assessment = assess_healing_candidate(previous_result.invoice_path, config, db)
        if assessment.route != Route.APPLY:
            summary["skipped_not_apply"] += 1
            continue
        if not assessment.gt_trusted:
            summary["skipped_not_gt_trusted"] += 1
            continue

        retry_context = _retry_context_from_result(
            previous_result,
            assessment,
            healing_origin=healing_origin,
        )
        healed_result = process_single_invoice(
            previous_result.invoice_path,
            settings,
            config,
            db,
            force_reprocess=True,
            retry_context=retry_context,
        )
        summary["attempted"] += 1
        recovered = (
            healed_result.validation_passed is True or healed_result.route_used == Route.APPLY
        )
        if recovered:
            summary["recovered"] += 1
        else:
            summary["still_failing"] += 1
        entry = {
            "invoice_path": previous_result.invoice_path,
            "invoice_name": Path(previous_result.invoice_path).name,
            "recovered": recovered,
            "route": healed_result.route_used.value if healed_result.route_used else None,
            "validation_passed": healed_result.validation_passed,
            "trigger_match_type": assessment.trigger_match_type,
            "trigger_score": assessment.trigger_score,
            "trigger_family_id": assessment.trigger_family_id,
            "trigger_fingerprint_hash": assessment.trigger_fingerprint_hash,
            "gt_trust_scope": assessment.gt_trust_scope,
            "gt_apply_count": assessment.gt_apply_count,
            "gt_reject_count": assessment.gt_reject_count,
            "gt_confidence": assessment.gt_confidence,
        }
        summary["entries"].append(entry)
        if show_per_invoice:
            status = "RECOVERED" if recovered else "STILL_FAILING"
            trigger = assessment.trigger_match_type or "unknown"
            score = (
                f"{assessment.trigger_score:.2f}" if assessment.trigger_score is not None else "n/a"
            )
            click.echo(
                f"{entry['invoice_name']}: {status} | route={entry['route'] or 'n/a'} "
                f"| trigger={trigger} | score={score}"
            )

    return summary


def _attention_items(results, limit=8):
    items = []
    for item in results:
        result = item.get("result")
        if item.get("error") is not None:
            items.append(
                {
                    "name": item["name"],
                    "status": "ERROR",
                    "route": item.get("route"),
                    "reason": str(item["error"]),
                }
            )
            continue
        if result is None:
            continue
        status = _display_status_for_result(result)
        if status not in {"REVIEW", "CHECK"}:
            continue
        diagnostics = result.diagnostics
        items.append(
            {
                "name": item["name"],
                "status": status,
                "route": result.route_used.value if result.route_used else None,
                "score": diagnostics.validation_score if diagnostics else None,
                "reason": "; ".join(_review_signals_for_result(result)) or "needs review",
            }
        )
    items.sort(
        key=lambda entry: (
            0 if entry["status"] == "ERROR" else 1,
            entry.get("score") is not None,
            entry.get("score") or 1.0,
            entry["name"],
        )
    )
    return items[:limit]


def _format_failed_result_line(result):
    diagnostics = result.diagnostics
    route = result.route_used.value if result.route_used else "None"
    attempted = (
        result.attempted_route.value
        if result.attempted_route
        else (
            diagnostics.attempted_route.value
            if diagnostics and diagnostics.attempted_route
            else None
        )
    )
    score = _format_validation_score(diagnostics.validation_score if diagnostics else None)
    pieces = [f"{Path(result.invoice_path).name}", f"route={route}"]
    if attempted:
        pieces.append(f"attempted={attempted}")
    if diagnostics and diagnostics.validation_score is not None:
        pieces.append(f"score={score}")
    if diagnostics and diagnostics.country_code:
        pieces.append(f"country={diagnostics.country_code}")
    signals = _review_signals_for_result(result)
    if signals:
        pieces.append(f"why={'; '.join(signals)}")
    return " | ".join(pieces)


def _resolved_worker_count(config, override=None):
    configured = (
        override if override is not None else getattr(config.processing, "worker_concurrency", 1)
    )
    return max(1, int(configured or 1))


def _collect_family_review_invoices(db, template_family_id, *, limit):
    selected = []
    seen = set()
    recent_results = db.get_recent_results_for_family(template_family_id, limit=limit)
    for result in recent_results:
        invoice_path = result.invoice_path
        if invoice_path and invoice_path not in seen and Path(invoice_path).exists():
            selected.append(invoice_path)
            seen.add(invoice_path)
        if len(selected) >= limit:
            return selected

    examples = list(reversed(db.get_template_family_examples(template_family_id)))
    for example in examples:
        invoice_path = example.invoice_path
        if invoice_path and invoice_path not in seen and Path(invoice_path).exists():
            selected.append(invoice_path)
            seen.add(invoice_path)
        if len(selected) >= limit:
            break
    return selected


def _seed_family_review_db(source_db, target_db, template_family_id):
    family = source_db.get_template_family(template_family_id)
    if family is None:
        raise click.ClickException(f"Template family not found: {template_family_id}")
    target_db.store_template_family(family)
    for record in source_db.get_fingerprints_for_family(template_family_id):
        target_db.store_fingerprint(
            record,
            visual_hashes=[page.visual_hash_hex for page in record.page_fingerprints],
            page_fingerprints=[page.model_dump() for page in record.page_fingerprints],
        )


def _progress_bucket_for_result(result):
    if result is None:
        return "error"
    if result.validation_passed is True:
        return "passed"
    if result.validation_passed is False or (
        result.route_used and result.route_used.value == "REJECTED"
    ):
        return "rejected"
    return "other"


def _format_eta(seconds):
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s"


class _ProgressReporter:
    def __init__(self, total, *, label="Progress", emit_interval_s=1.0):
        self.total = total
        self.label = label
        self.emit_interval_s = emit_interval_s
        self.started_at = time.time()
        self.last_emit_at = 0.0
        self.processed = 0
        self.passed = 0
        self.rejected = 0
        self.errors = 0
        self._last_line = None

    def record(self, outcome):
        self.processed += 1
        if outcome == "passed":
            self.passed += 1
        elif outcome == "rejected":
            self.rejected += 1
        elif outcome == "error":
            self.errors += 1

    def _line(self):
        elapsed = max(time.time() - self.started_at, 1e-6)
        rate = self.processed / elapsed
        remaining = max(self.total - self.processed, 0)
        eta = (remaining / rate) if rate > 0 and remaining else 0.0 if remaining == 0 else None
        line = (
            f"{self.label}: processed={self.processed}/{self.total} "
            f"passed={self.passed} rejected={self.rejected} "
            f"remaining={remaining} rate={rate:.2f}/s ETA={_format_eta(eta)}"
        )
        if self.errors:
            line += f" errors={self.errors}"
        return line

    def maybe_emit(self, *, force=False):
        now = time.time()
        if not force and (now - self.last_emit_at) < self.emit_interval_s:
            return
        line = self._line()
        if line == self._last_line:
            return
        click.echo(line)
        self.last_emit_at = now
        self._last_line = line


def _run_invoice_executor(
    invoices,
    *,
    max_workers,
    run_one,
    on_success,
    on_error,
    progress_label,
    show_progress=True,
):
    results = []
    progress = (
        _ProgressReporter(len(invoices), label=progress_label)
        if show_progress and invoices
        else None
    )

    def _record_success(inv, res, elapsed_ms):
        payload = on_success(inv, res, elapsed_ms)
        results.append(payload["entry"])
        if payload.get("message"):
            click.echo(payload["message"])
        if progress is not None:
            progress.record(payload.get("progress_outcome", _progress_bucket_for_result(res)))
            progress.maybe_emit(force=bool(payload.get("force_progress_emit")))

    def _record_error(inv, elapsed_ms, exc):
        payload = on_error(inv, elapsed_ms, exc)
        results.append(payload["entry"])
        if payload.get("message"):
            click.echo(payload["message"])
        if progress is not None:
            progress.record(payload.get("progress_outcome", "error"))
            progress.maybe_emit(force=True)

    if max_workers == 1:
        for inv in invoices:
            try:
                inv, res, elapsed_ms = run_one(inv)
                _record_success(inv, res, elapsed_ms)
            except Exception as exc:
                _record_error(inv, 0, exc)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(run_one, inv): inv for inv in invoices}
            for future in as_completed(futures):
                inv = futures[future]
                try:
                    inv, res, elapsed_ms = future.result()
                    _record_success(inv, res, elapsed_ms)
                except Exception as exc:
                    _record_error(inv, 0, exc)

    if progress is not None:
        progress.maybe_emit(force=True)
    return results


def _current_config_path():
    ctx = click.get_current_context(silent=True)
    if ctx is None or not getattr(ctx, "obj", None):
        return None
    return ctx.obj.get("config_path")


def _load_runtime(*required_settings):
    config_path = _current_config_path()
    kwargs = {}
    if required_settings:
        kwargs["required_settings"] = required_settings

    try:
        if config_path is None:
            return load_config(**kwargs)
        return load_config(config_path, **kwargs)
    except TypeError as exc:
        # Unit tests often monkeypatch load_config with a zero-argument lambda.
        if "unexpected keyword argument" in str(exc) or "positional argument" in str(exc):
            return load_config()
        raise


def _current_command():
    program = Path(sys.argv[0]).name or "invoice-router"
    if program in {"python", "python3", "__main__.py"}:
        return "invoice-router"
    return program


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    envvar="INVOICE_ROUTER_CONFIG",
    help="Path to the YAML config file. Defaults to ./config.yaml or packaged defaults.",
)
@click.version_option(version=__version__, prog_name="invoice-router")
@click.pass_context
def cli(ctx, config_path):
    """Deterministic invoice routing and extraction for invoice datasets."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = str(config_path) if config_path else None


def _to_jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, SimpleNamespace):
        return {key: _to_jsonable(val) for key, val in vars(value).items()}
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _load_json_file(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError as exc:
        raise click.ClickException(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON in {path}: {exc}") from exc


def _load_structural_subset_manifest(path: Path):
    raw = _load_json_file(path)
    if not isinstance(raw, list):
        raise click.ClickException(f"Structural subset manifest must be a JSON list: {path}")

    manifest = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise click.ClickException(f"Subset entry #{index} in {path} must be an object")
        invoice = item.get("invoice")
        if not invoice:
            raise click.ClickException(
                f"Subset entry #{index} in {path} is missing an invoice name"
            )
        manifest.append(
            {
                "invoice": str(invoice),
                "bucket": str(item.get("bucket") or "uncategorized"),
                "failure_mode": item.get("failure_mode"),
                "priority": item.get("priority", index),
                "notes": item.get("notes"),
            }
        )
    if not manifest:
        raise click.ClickException(f"Structural subset manifest is empty: {path}")
    return manifest


def _select_invoices_for_subset(all_invoices, subset_manifest):
    invoice_lookup = {Path(inv).name: inv for inv in all_invoices}
    selected = []
    missing = []
    for entry in subset_manifest:
        invoice_name = entry["invoice"]
        invoice_path = invoice_lookup.get(invoice_name)
        if invoice_path is None:
            missing.append(invoice_name)
            continue
        if invoice_path not in selected:
            selected.append(invoice_path)
    if missing:
        raise click.ClickException(
            "Subset invoices were not found in the input directory: " + ", ".join(sorted(missing))
        )
    return selected


@cli.command()
@click.argument("input_dir", type=str)
@click.option("--batch-size", type=int, default=50, help="Batch size for processing.")
@click.option("--limit", type=int, default=None, help="Limit number of invoices to process.")
@click.option("--no-validate", is_flag=True, default=False, help="Disable GT validation.")
@click.option("--report-accuracy", is_flag=True, default=False, help="Report accuracy summary.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging.")
@click.option(
    "--show-per-invoice",
    is_flag=True,
    default=False,
    help="Print one line per invoice during processing.",
)
@click.option(
    "--reset", is_flag=True, default=False, help="Clear cached processing results before running."
)
@click.option(
    "--reset-templates",
    is_flag=True,
    default=False,
    help="Clear all stored fingerprint templates before running. Implies --reset.",
)
@click.option(
    "--discovery-only",
    is_flag=True,
    default=False,
    help="Evaluation mode: do not reuse templates created earlier in the same batch.",
)
def process(
    input_dir,
    batch_size,
    limit,
    no_validate,
    report_accuracy,
    verbose,
    show_per_invoice,
    reset,
    reset_templates,
    discovery_only,
):
    """Process invoices from a given input directory."""
    settings, config = _load_runtime("database_url", "redis_url")
    input_dir = _resolve_input_dir(input_dir, settings)
    if reset_templates and not reset:
        reset = True

    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")
    db = FingerprintDB(_primary_database_target(settings))

    if reset_templates:
        db.clear_fingerprints()
        # Also flush Redis counters so apply_count/reject_count restart from 0
        try:
            from ..infrastructure.persistence.redis import get_redis_client

            rc = get_redis_client(settings)
            keys = rc.keys("fingerprint_stats:*")
            if keys:
                rc.delete(*keys)
        except Exception:
            pass  # Redis may be unavailable; non-fatal for a reset
        click.echo("Cleared all fingerprint templates.")
        click.echo("Resetting cached processing results as part of --reset-templates.")

    if reset:
        invoices_to_reset = [str(Path(input_dir) / Path(p).name) for p in list_invoices(input_dir)]
        db.clear_processing_results(invoices_to_reset)
        click.echo(f"Cleared cached results for {len(invoices_to_reset)} invoices.")

    invoices = list_invoices(input_dir)
    if limit:
        invoices = invoices[:limit]

    import time as _time

    click.echo(f"Processing {len(invoices)} invoices...")
    click.echo("Discovery strategy: heuristic")
    if getattr(config.processing, "applied_profile", None):
        click.echo(
            f"Runner profile: {config.processing.applied_profile} ({config.processing.worker_concurrency} workers)"
        )
    if not show_per_invoice and not verbose:
        click.echo(
            "Per-invoice output is suppressed by default; use --show-per-invoice or query results later with validate/analyze-failures."
        )
    if discovery_only:
        click.echo(
            "Discovery-only mode active: batch will use only the fingerprint snapshot present at batch start."
        )

    batch_start = _time.time()
    frozen_fingerprints = db.get_all_active_fingerprints() if discovery_only else None

    def _process_one(inv):
        inv_start = _time.time()
        res = process_single_invoice(
            inv, settings, config, db, active_fingerprints_override=frozen_fingerprints
        )
        inv_elapsed = int((_time.time() - inv_start) * 1000)
        return inv, res, inv_elapsed

    def _process_success(inv, res, inv_elapsed):
        route_val = res.route_used.value if res.route_used else "None"
        status = _display_status_for_result(res)
        message = None
        if show_per_invoice or verbose or status in {"REVIEW", "ERROR"}:
            message = _format_result_summary_line(Path(inv).name, res, inv_elapsed)
        return {
            "entry": {
                "name": Path(inv).name,
                "route": route_val,
                "status": status,
                "elapsed_ms": inv_elapsed,
                "result": res,
                "error": None,
            },
            "message": message,
            "progress_outcome": _progress_bucket_for_result(res),
            "force_progress_emit": status in {"REVIEW", "ERROR"},
        }

    def _process_error(inv, elapsed_ms, exc):
        return {
            "entry": {
                "name": Path(inv).name,
                "route": "ERROR",
                "status": str(exc),
                "elapsed_ms": elapsed_ms,
                "result": None,
                "error": exc,
            },
            "message": f"{Path(inv).name}: ERROR | route=ERROR | {exc}",
            "progress_outcome": "error",
            "force_progress_emit": True,
        }

    results = _run_invoice_executor(
        invoices,
        max_workers=_resolved_worker_count(config),
        run_one=_process_one,
        on_success=_process_success,
        on_error=_process_error,
        progress_label="Process progress",
        show_progress=not show_per_invoice and not verbose,
    )

    invoice_order = {Path(inv).name: idx for idx, inv in enumerate(invoices)}
    results.sort(key=lambda item: invoice_order.get(item["name"], len(invoice_order)))

    total_ms = int((_time.time() - batch_start) * 1000)
    healing_summary = _heal_rejections_pass(
        input_dir,
        settings,
        config,
        db,
        healing_origin="auto_process",
    )
    failure_summary = summarize_failure_modes(
        db,
        dataset_filter=Path(input_dir).name,
        discovery_threshold=config.validation.discovery_threshold,
    )
    run_summary = build_process_run_summary(
        input_dir=input_dir,
        results=results,
        total_ms=total_ms,
        failure_summary=failure_summary,
        discovery_only=discovery_only,
        discovery_mode="heuristic",
        healing_summary=healing_summary,
    )
    run_summary_path = write_run_summary(run_summary, settings.output_dir)
    command = " ".join(shlex.quote(arg) for arg in [_current_command(), *sys.argv[1:]]).strip()
    git_branch = None
    git_commit = None
    try:
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            cwd=REPO_ROOT,
        ).strip()
    except Exception:
        pass
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=REPO_ROOT,
        ).strip()
    except Exception:
        pass
    analysis_db_path = None
    try:
        analysis_db_path = record_analysis_run(
            run_summary,
            results,
            settings.output_dir,
            analysis_db_target=_analysis_database_target(settings),
            command=command,
            git_branch=git_branch,
            git_commit=git_commit,
            config_snapshot=_to_jsonable(config),
        )
    except Exception as exc:
        logger.warning("Skipping analysis history recording: %s", exc)
    click.echo(f"Run summary written to {run_summary_path}")
    if analysis_db_path is not None:
        click.echo(f"Analysis history recorded in {analysis_db_path}")
    else:
        click.echo("Analysis history recording skipped.")
    click.echo(
        "Healing summary: "
        f"candidates={healing_summary['candidates']} "
        f"| attempted={healing_summary['attempted']} "
        f"| recovered={healing_summary['recovered']} "
        f"| still_failing={healing_summary['still_failing']} "
        f"| skipped_as_already_attempted={healing_summary['skipped_as_already_attempted']}"
    )

    attention_items = _attention_items(results)
    if attention_items:
        click.echo("\nAttention queue:")
        for item in attention_items:
            suffix = (
                f" | score={_format_validation_score(item['score'])}"
                if item.get("score") is not None
                else ""
            )
            click.echo(
                f"  {item['name']}: {item['status']} | route={item.get('route')}{suffix} | {item['reason']}"
            )

    if verbose and results:
        click.echo("\n" + "=" * 60)
        click.echo("VERBOSE ANALYSIS SUMMARY")
        click.echo("=" * 60)

        route_counts = run_summary["route_counts"]
        status_counts = run_summary["status_counts"]

        click.echo(f"Total invoices:  {run_summary['total_invoices']}")
        click.echo(
            f"Total time:      {run_summary['total_time_ms']}ms  (avg {run_summary['avg_time_ms']}ms/invoice)"
        )
        click.echo("\nRoute breakdown:")
        for route, count in sorted(route_counts.items()):
            click.echo(f"  {route:<20} {count}")
        click.echo("\nValidation breakdown:")
        for status, count in sorted(status_counts.items()):
            click.echo(f"  {status:<20} {count}")

        click.echo("\nDiscovery outcomes:")
        click.echo(f"  Mode                 {run_summary['discovery_mode']}")
        click.echo(f"  Passed               {run_summary['discovery_passed']}")
        click.echo(f"  Rejected             {run_summary['discovery_rejected']}")
        click.echo(f"  Runtime errors       {run_summary['discovery_runtime_errors']}")
        if run_summary["discovery_stage_status_counts"]:
            click.echo("  Stage status counts:")
            for status, count in sorted(run_summary["discovery_stage_status_counts"].items()):
                click.echo(f"    {status:<20} {count}")
        click.echo("\nApply outcomes:")
        click.echo(f"  Rejected             {run_summary['apply_rejected']}")
        click.echo(f"  Runtime errors       {run_summary['apply_runtime_errors']}")

        click.echo("\nPer-invoice timings:")
        for r in results:
            click.echo(f"  {r['name']:<40} {r['elapsed_ms']:>6}ms  route={r['route']}")

        quality_scores = [
            r["result"].image_quality_score
            for r in results
            if r["result"] and r["result"].image_quality_score is not None
        ]
        if quality_scores:
            avg_q = sum(quality_scores) / len(quality_scores)
            click.echo(
                f"\nImage quality scores: min={min(quality_scores):.3f}  avg={avg_q:.3f}  max={max(quality_scores):.3f}"
            )

        if run_summary["runtime_error_categories"]:
            click.echo("\nRuntime error categories:")
            for category, count in sorted(run_summary["runtime_error_categories"].items()):
                click.echo(f"  {category:<20} {count}")

        click.echo("=" * 60)


@cli.command()
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path(".demo"),
    show_default=True,
    help="Workspace directory where the demo invoice and outputs will be written.",
)
def demo(workspace):
    """Generate a synthetic invoice and run the public quickstart flow."""
    settings, config = _load_runtime("database_url", "redis_url")
    workspace = Path(workspace).expanduser()
    dataset_dir, invoice_path = create_demo_dataset(workspace / "datasets")
    output_dir = workspace / "output"
    temp_dir = workspace / "temp"
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    runtime_settings = settings.model_copy(
        update={
            "invoice_input_dir": str(dataset_dir),
            "output_dir": str(output_dir),
            "temp_dir": str(temp_dir),
        }
    )

    db = FingerprintDB(_primary_database_target(runtime_settings))
    result = process_single_invoice(str(invoice_path), runtime_settings, config, db)
    output_path = output_dir / dataset_dir.name / "demo_invoice.json"

    click.echo(f"Demo workspace: {workspace}")
    click.echo(f"Invoice:        {invoice_path}")
    click.echo(f"Route:          {result.route_used.value if result.route_used else 'UNKNOWN'}")
    click.echo(f"Output JSON:    {output_path}")
    if result.normalized_data:
        click.echo(f"Invoice number: {result.normalized_data.get('invoice_number')}")
        click.echo(f"Total:          {result.normalized_data.get('total')}")


@cli.command()
@click.option("--limit", type=int, default=50, help="Limit output list.")
def fingerprints(limit):
    """List active known fingerprint templates and their status."""
    settings, config = _load_runtime("database_url")
    db = FingerprintDB(_primary_database_target(settings))
    active = db.get_all_active_fingerprints()
    click.echo(f"Active Fingerprints: {len(active)}")
    for r in active[:limit]:
        click.echo(
            f"Hash: {r.hash[:8]}... | Status: {r.status.value} | Conf: {r.confidence:.2f} | Apply/Rej: {r.apply_count}/{r.reject_count}"
        )


@cli.command(name="families")
@click.option("--limit", type=int, default=50, help="Limit output list.")
@click.option(
    "--status",
    type=click.Choice(["provisional", "established", "degraded", "retired"]),
    default=None,
    help="Filter by family status.",
)
def families(limit, status):
    """List template families and their current lifecycle state."""
    settings, _ = _load_runtime("database_url")
    db = FingerprintDB(_primary_database_target(settings))
    summary = summarize_template_families(db)
    click.echo(f"Template Families: {summary['total_families']}")
    if summary["status_counts"]:
        click.echo("Status counts:")
        for family_status, count in sorted(summary["status_counts"].items()):
            click.echo(f"  {family_status:<12} {count}")
    if summary["review_queue"]:
        click.echo(f"Review queue: {', '.join(summary['review_queue'][:10])}")

    rows = summary["families"]
    if status is not None:
        rows = [row for row in rows if row["status"] == status]
    for row in rows[:limit]:
        profile = row.get("extraction_profile") or {}
        profile_summary = f"profile={profile.get('preferred_strategy') or 'none'}"
        if profile.get("table_enabled"):
            profile_summary += f"/table:{profile.get('table_engine') or 'default'}"
        if profile.get("field_override_count"):
            profile_summary += f"/fields:{profile['field_override_count']}"
        triage_summary = f"triage={row.get('triage_class', 'unknown')}"
        click.echo(
            f"{row['template_family_id']} | status={row['status']} | conf={row['confidence']:.2f} "
            f"| apply/reject={row['apply_count']}/{row['reject_count']} "
            f"| {triage_summary} "
            f"| {profile_summary} "
            f"| rep={((row['representative_fingerprint_hash'] or '')[:8] + '...') if row['representative_fingerprint_hash'] else 'none'}"
        )


@cli.command(name="family-show")
@click.argument("template_family_id", type=str)
@click.option("--examples", type=int, default=5, help="Number of recent family examples to show.")
@click.option("--versions", type=int, default=5, help="Number of recent family versions to show.")
@click.option("--recent", type=int, default=5, help="Number of recent processing results to show.")
def family_show(template_family_id, examples, versions, recent):
    """Show one template family in detail."""
    settings, _ = _load_runtime("database_url")
    db = FingerprintDB(_primary_database_target(settings))
    detail = describe_template_family(
        db,
        template_family_id,
        example_limit=examples,
        version_limit=versions,
        recent_limit=recent,
    )
    if detail is None:
        raise click.ClickException(f"Template family not found: {template_family_id}")

    click.echo(f"Template Family: {detail['template_family_id']}")
    click.echo(
        f"Status: {detail['status']} | Confidence: {detail['confidence']:.2f} "
        f"| Apply/Reject: {detail['apply_count']}/{detail['reject_count']}"
    )
    click.echo(
        f"GT trust: qualified={'yes' if detail.get('gt_trust_qualified') else 'no'} "
        f"| gt_apply/reject={detail.get('gt_apply_count', 0)}/{detail.get('gt_reject_count', 0)} "
        f"| gt_confidence={float(detail.get('gt_confidence', 0.0)):.2f}"
    )
    click.echo(
        f"Context: provider={detail['provider_name'] or 'unknown'} "
        f"| country={detail['country_code'] or 'unknown'} "
        f"| document_family={detail['document_family']}"
    )
    representative = detail["representative"]
    click.echo(
        f"Representative: {((representative['fingerprint_hash'] or '')[:8] + '...') if representative['fingerprint_hash'] else 'none'} "
        f"| status={representative['status'] or 'n/a'} "
        f"| apply/reject={representative['apply_count'] if representative['apply_count'] is not None else 'n/a'}/"
        f"{representative['reject_count'] if representative['reject_count'] is not None else 'n/a'}"
    )
    click.echo(
        f"Representative GT trust: qualified={'yes' if representative.get('gt_trust_qualified') else 'no'} "
        f"| gt_apply/reject={representative.get('gt_apply_count') if representative.get('gt_apply_count') is not None else 'n/a'}/"
        f"{representative.get('gt_reject_count') if representative.get('gt_reject_count') is not None else 'n/a'} "
        f"| gt_confidence={representative.get('gt_confidence') if representative.get('gt_confidence') is not None else 'n/a'}"
    )
    profile = detail["extraction_profile_summary"]
    click.echo(
        f"Profile: strategy={profile.get('preferred_strategy') or 'none'} "
        f"| table={profile.get('table_engine') if profile.get('table_enabled') else 'off'} "
        f"| field_overrides={profile.get('field_override_count', 0)}"
    )
    click.echo(
        f"Triage: {detail.get('triage_class', 'unknown')} | {detail.get('triage_explanation', 'n/a')}"
    )
    click.echo(
        f"History: examples={detail['example_count']} | versions={detail['version_count']} "
        f"| created={detail['created_at']} | updated={detail['updated_at'] or 'n/a'}"
    )
    click.echo(
        f"Healing: attempts={detail.get('healed_attempt_count', 0)} | recoveries={detail.get('healed_recovery_count', 0)}"
    )
    if detail["review_signals"]:
        click.echo(f"Review signals: {'; '.join(detail['review_signals'])}")

    if detail["versions"]:
        click.echo("Recent versions:")
        for version in detail["versions"]:
            click.echo(
                f"  v{version['version']}: {version['change_reason'] or 'no reason'} ({version['created_at']})"
            )

    if detail["examples"]:
        click.echo("Recent examples:")
        for example in detail["examples"]:
            click.echo(
                f"  {Path(example['invoice_path']).name if example['invoice_path'] else 'unknown'} "
                f"| fp={(example['fingerprint_hash'] or '')[:8] + '...' if example['fingerprint_hash'] else 'none'}"
            )

    if detail["recent_results"]:
        click.echo("Recent outcomes:")
        for row in detail["recent_results"]:
            score = (
                f"{row['validation_score'] * 100:.1f}%"
                if row["validation_score"] is not None
                else "n/a"
            )
            signals = f" | {'; '.join(row['review_signals'])}" if row["review_signals"] else ""
            click.echo(
                f"  {row['invoice_name']} | {row['outcome']} | route={row['route'] or 'n/a'} "
                f"| score={score}"
                f"{' | ' + '/'.join(row.get('determination_sources', [])) if row.get('determination_sources') else ''}"
                f"{signals}"
            )

    if detail.get("suggestions"):
        click.echo("Suggested updates:")
        for suggestion in detail["suggestions"]:
            click.echo(
                f"  {suggestion['kind']} | support={suggestion['support_count']} "
                f"| {suggestion['title']}"
            )
            click.echo(f"    {suggestion['reason']}")

    if detail.get("split_suggestions"):
        click.echo("Split candidates:")
        for suggestion in detail["split_suggestions"][:3]:
            click.echo(
                f"  {suggestion['cluster_id']} | fingerprints={suggestion['fingerprint_count']} "
                f"| review={suggestion['status_counts'].get('review', 0)} "
                f"| similarity={suggestion['similarity_to_primary']:.2f}"
            )
            click.echo(f"    {suggestion['reason']}")
            if suggestion["example_invoices"]:
                click.echo(f"    examples: {', '.join(suggestion['example_invoices'])}")

    if detail.get("merge_suggestions"):
        click.echo("Merge candidates:")
        for suggestion in detail["merge_suggestions"][:3]:
            click.echo(
                f"  {suggestion['source_family_id']} -> {suggestion['target_family_id']} "
                f"| similarity={suggestion['similarity']:.2f}"
            )
            click.echo(f"    {suggestion['reason']}")

    if detail.get("retirement_suggestions"):
        click.echo("Retirement candidates:")
        for suggestion in detail["retirement_suggestions"][:3]:
            click.echo(
                f"  {suggestion['kind']} | fingerprints={suggestion['active_fingerprint_count']} "
                f"| rejects={suggestion['reject_count']}"
            )
            click.echo(f"    {suggestion['reason']}")


@cli.command(name="family-suggestions")
@click.option("--limit", type=int, default=20, help="Limit output list.")
@click.option("--family-id", type=str, default=None, help="Restrict suggestions to one family.")
@click.option(
    "--recent",
    type=int,
    default=10,
    show_default=True,
    help="Recent results to inspect per family.",
)
@click.option(
    "--min-support",
    type=int,
    default=2,
    show_default=True,
    help="Minimum recent supporting invoices for a suggestion.",
)
def family_suggestions(limit, family_id, recent, min_support):
    """List suggested family profile changes based on repeated recent failures."""
    settings, _ = _load_runtime("database_url")
    db = FingerprintDB(_primary_database_target(settings))
    queue = suggest_template_family_updates(
        db,
        template_family_id=family_id,
        recent_limit=recent,
        min_support=min_support,
    )
    click.echo(f"Family suggestions: {queue['queue_count']}")
    if queue["family_ids"]:
        click.echo(f"Families in queue: {', '.join(queue['family_ids'][:10])}")
    for suggestion in queue["suggestions"][:limit]:
        click.echo(
            f"{suggestion['template_family_id']} | kind={suggestion['kind']} "
            f"| support={suggestion['support_count']} | priority={suggestion['priority']:.1f}"
        )
        click.echo(f"  {suggestion['title']}: {suggestion['reason']}")
        if suggestion["example_invoices"]:
            click.echo(f"  examples: {', '.join(suggestion['example_invoices'])}")
        click.echo(f"  patch: {json.dumps(suggestion['profile_patch'], sort_keys=True)}")


@cli.command(name="family-apply-suggestion")
@click.argument("template_family_id", type=str)
@click.argument("kind", type=str)
@click.option(
    "--recent",
    type=int,
    default=10,
    show_default=True,
    help="Recent results to inspect per family.",
)
@click.option(
    "--min-support",
    type=int,
    default=2,
    show_default=True,
    help="Minimum recent supporting invoices for a suggestion.",
)
@click.option(
    "--reason",
    type=str,
    default=None,
    help="Reason recorded in version history. Defaults to apply_suggestion:<kind>.",
)
def family_apply_suggestion(template_family_id, kind, recent, min_support, reason):
    """Apply one suggested family profile change after review."""
    settings, _ = _load_runtime("database_url")
    db = FingerprintDB(_primary_database_target(settings))
    queue = suggest_template_family_updates(
        db,
        template_family_id=template_family_id,
        recent_limit=recent,
        min_support=min_support,
    )
    suggestion = next((item for item in queue["suggestions"] if item["kind"] == kind), None)
    if suggestion is None:
        raise click.ClickException(f"Suggestion not found for {template_family_id}: {kind}")

    updated = manually_update_template_family(
        db,
        template_family_id,
        extraction_profile_updates=suggestion["profile_patch"],
        replace_extraction_profile=False,
        reason=reason or f"apply_suggestion:{kind}",
    )
    if updated is None:
        raise click.ClickException(f"Template family not found: {template_family_id}")

    click.echo(f"Applied suggestion: {kind}")
    click.echo(f"Template family: {updated.template_family_id}")
    click.echo(f"Version note: {reason or f'apply_suggestion:{kind}'}")


@cli.command(name="family-splits")
@click.option("--limit", type=int, default=20, help="Limit output list.")
@click.option(
    "--family-id", type=str, default=None, help="Restrict split suggestions to one family."
)
@click.option(
    "--min-cluster-size",
    type=int,
    default=2,
    show_default=True,
    help="Minimum fingerprints or supporting examples/results in a split candidate.",
)
@click.option(
    "--similarity-threshold",
    type=float,
    default=0.72,
    show_default=True,
    help="Minimum structural similarity needed to keep fingerprints in the same cluster.",
)
def family_splits(limit, family_id, min_cluster_size, similarity_threshold):
    """List families that appear structurally mixed and may need splitting."""
    settings, _ = _load_runtime("database_url")
    db = FingerprintDB(_primary_database_target(settings))
    queue = suggest_template_family_splits(
        db,
        template_family_id=family_id,
        min_cluster_size=min_cluster_size,
        similarity_threshold=similarity_threshold,
    )
    click.echo(f"Family split suggestions: {queue['queue_count']}")
    if queue["family_ids"]:
        click.echo(f"Families in queue: {', '.join(queue['family_ids'][:10])}")
    for suggestion in queue["suggestions"][:limit]:
        click.echo(
            f"{suggestion['template_family_id']} | {suggestion['cluster_id']} "
            f"| fingerprints={suggestion['fingerprint_count']} "
            f"| review={suggestion['status_counts'].get('review', 0)} "
            f"| similarity={suggestion['similarity_to_primary']:.2f}"
        )
        click.echo(f"  {suggestion['title']}: {suggestion['reason']}")
        if suggestion["example_invoices"]:
            click.echo(f"  examples: {', '.join(suggestion['example_invoices'])}")
        click.echo(
            f"  new family: {suggestion['proposed_family_id']} "
            f"| fingerprints: {', '.join(suggestion['fingerprint_hashes'][:4])}"
        )


@cli.command(name="family-split")
@click.argument("template_family_id", type=str)
@click.option(
    "--cluster",
    "cluster_id",
    type=str,
    default=None,
    help="Suggested cluster ID from family-splits or family-show.",
)
@click.option(
    "--fingerprint",
    "fingerprint_hashes",
    type=str,
    multiple=True,
    help="Explicit fingerprint hash to move. Repeat as needed.",
)
@click.option(
    "--new-family-id", type=str, default=None, help="Override the generated new family ID."
)
@click.option(
    "--reason",
    type=str,
    default="manual_split",
    show_default=True,
    help="Reason recorded in family version history.",
)
@click.option(
    "--min-cluster-size",
    type=int,
    default=2,
    show_default=True,
    help="Minimum support used when resolving a suggested cluster.",
)
@click.option(
    "--similarity-threshold",
    type=float,
    default=0.72,
    show_default=True,
    help="Structural similarity threshold used when resolving a suggested cluster.",
)
def family_split(
    template_family_id,
    cluster_id,
    fingerprint_hashes,
    new_family_id,
    reason,
    min_cluster_size,
    similarity_threshold,
):
    """Split one structurally mixed cluster out into a new family."""
    settings, config = _load_runtime("database_url")
    db = FingerprintDB(_primary_database_target(settings))

    selected_hashes = set(fingerprint_hashes)
    if cluster_id:
        queue = suggest_template_family_splits(
            db,
            template_family_id=template_family_id,
            min_cluster_size=min_cluster_size,
            similarity_threshold=similarity_threshold,
        )
        suggestion = next(
            (item for item in queue["suggestions"] if item["cluster_id"] == cluster_id), None
        )
        if suggestion is None:
            raise click.ClickException(
                f"Split suggestion not found for {template_family_id}: {cluster_id}"
            )
        selected_hashes.update(suggestion["fingerprint_hashes"])
        if new_family_id is None:
            new_family_id = suggestion["proposed_family_id"]

    if not selected_hashes:
        raise click.ClickException(
            "Provide --cluster or at least one --fingerprint to define the split."
        )

    try:
        split_result = split_template_family(
            db,
            template_family_id,
            fingerprint_hashes=sorted(selected_hashes),
            new_template_family_id=new_family_id,
            reason=reason,
            config=config.template_lifecycle,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if split_result is None:
        raise click.ClickException(f"Template family not found: {template_family_id}")

    source_family = split_result["source_family"]
    new_family = split_result["new_family"]
    click.echo(f"Split template family: {template_family_id} -> {split_result['new_family_id']}")
    click.echo(
        f"Moved fingerprints/examples/results: "
        f"{split_result['moved_fingerprint_count']}/"
        f"{split_result['moved_example_count']}/"
        f"{split_result['moved_result_count']}"
    )
    click.echo(
        f"Source family: status={source_family.status.value} "
        f"| apply/reject={source_family.apply_count}/{source_family.reject_count}"
    )
    click.echo(
        f"New family: status={new_family.status.value} "
        f"| apply/reject={new_family.apply_count}/{new_family.reject_count}"
    )


@cli.command(name="family-merges")
@click.option("--limit", type=int, default=20, help="Limit output list.")
@click.option(
    "--family-id", type=str, default=None, help="Restrict merge suggestions to one family."
)
@click.option(
    "--similarity-threshold",
    type=float,
    default=0.84,
    show_default=True,
    help="Minimum similarity required before two families are suggested for merge.",
)
def family_merges(limit, family_id, similarity_threshold):
    """List families that look like duplicates and should be merged."""
    settings, _ = _load_runtime("database_url")
    db = FingerprintDB(_primary_database_target(settings))
    queue = suggest_template_family_merges(
        db,
        template_family_id=family_id,
        similarity_threshold=similarity_threshold,
    )
    click.echo(f"Family merge suggestions: {queue['queue_count']}")
    if queue["family_ids"]:
        click.echo(f"Families in queue: {', '.join(queue['family_ids'][:10])}")
    for suggestion in queue["suggestions"][:limit]:
        click.echo(
            f"{suggestion['source_family_id']} -> {suggestion['target_family_id']} "
            f"| similarity={suggestion['similarity']:.2f} "
            f"| source={suggestion['source_status']}:{suggestion['source_apply_count']} "
            f"| target={suggestion['target_status']}:{suggestion['target_apply_count']}"
        )
        click.echo(f"  {suggestion['title']}: {suggestion['reason']}")
        signals = suggestion.get("shared_signals") or {}
        if signals.get("anchor_tokens"):
            click.echo(f"  shared anchors: {', '.join(signals['anchor_tokens'])}")


@cli.command(name="family-merge")
@click.argument("source_family_id", type=str)
@click.argument("target_family_id", type=str)
@click.option(
    "--reason",
    type=str,
    default="manual_merge",
    show_default=True,
    help="Reason recorded in version history.",
)
def family_merge(source_family_id, target_family_id, reason):
    """Merge one family into another and retire the source family."""
    settings, config = _load_runtime("database_url")
    db = FingerprintDB(_primary_database_target(settings))
    try:
        result = merge_template_families(
            db,
            target_family_id,
            source_family_id,
            reason=reason,
            config=config.template_lifecycle,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if result is None:
        raise click.ClickException(
            f"Template family not found: {source_family_id} or {target_family_id}"
        )

    click.echo(f"Merged template family: {source_family_id} -> {target_family_id}")
    click.echo(
        f"Moved fingerprints/examples/results: "
        f"{result['moved_fingerprint_count']}/"
        f"{result['moved_example_count']}/"
        f"{result['moved_result_count']}"
    )
    click.echo(
        f"Target family: status={result['target_family'].status.value} "
        f"| apply/reject={result['target_family'].apply_count}/{result['target_family'].reject_count}"
    )
    click.echo(
        f"Source family: status={result['source_family'].status.value} "
        f"| apply/reject={result['source_family'].apply_count}/{result['source_family'].reject_count}"
    )


@cli.command(name="family-retirements")
@click.option("--limit", type=int, default=20, help="Limit output list.")
@click.option(
    "--family-id", type=str, default=None, help="Restrict retirement suggestions to one family."
)
@click.option(
    "--min-reject-count",
    type=int,
    default=3,
    show_default=True,
    help="Minimum reject count before weak families are suggested for retirement.",
)
def family_retirements(limit, family_id, min_reject_count):
    """List weak or dead families that can be retired."""
    settings, _ = _load_runtime("database_url")
    db = FingerprintDB(_primary_database_target(settings))
    queue = suggest_template_family_retirements(
        db,
        template_family_id=family_id,
        min_reject_count=min_reject_count,
    )
    click.echo(f"Family retirement suggestions: {queue['queue_count']}")
    if queue["family_ids"]:
        click.echo(f"Families in queue: {', '.join(queue['family_ids'][:10])}")
    for suggestion in queue["suggestions"][:limit]:
        click.echo(
            f"{suggestion['template_family_id']} | kind={suggestion['kind']} "
            f"| fingerprints={suggestion['active_fingerprint_count']} "
            f"| apply/reject={suggestion['apply_count']}/{suggestion['reject_count']}"
        )
        click.echo(f"  {suggestion['title']}: {suggestion['reason']}")
        if suggestion["example_invoices"]:
            click.echo(f"  examples: {', '.join(suggestion['example_invoices'])}")


@cli.command(name="family-retire")
@click.argument("template_family_id", type=str)
@click.option(
    "--reason",
    type=str,
    default="manual_retire",
    show_default=True,
    help="Reason recorded in version history.",
)
def family_retire(template_family_id, reason):
    """Retire one family and its active fingerprints."""
    settings, _ = _load_runtime("database_url")
    db = FingerprintDB(_primary_database_target(settings))
    result = retire_template_family(
        db,
        template_family_id,
        reason=reason,
        retire_fingerprints=True,
    )
    if result is None:
        raise click.ClickException(f"Template family not found: {template_family_id}")

    click.echo(f"Retired template family: {template_family_id}")
    click.echo(f"Retired fingerprints: {result['retired_fingerprint_count']}")
    click.echo(
        f"Family status: {result['family'].status.value} "
        f"| apply/reject={result['family'].apply_count}/{result['family'].reject_count}"
    )


@cli.command(name="family-benchmark")
@click.argument("template_family_id", type=str)
@click.option(
    "--limit", type=int, default=10, show_default=True, help="Maximum family invoices to rerun."
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Invoice worker count. Defaults to configured concurrency.",
)
@click.option(
    "--output-json",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path for the family benchmark summary JSON.",
)
def family_benchmark(template_family_id, limit, workers, output_json):
    """Rerun a small review benchmark for one family and compare before vs after."""
    settings, config = _load_runtime("database_url", "redis_url")
    source_db = FingerprintDB(_primary_database_target(settings))
    invoice_paths = _collect_family_review_invoices(source_db, template_family_id, limit=limit)
    if not invoice_paths:
        raise click.ClickException(
            f"No accessible invoice files found for template family: {template_family_id}"
        )

    baseline_results = {
        invoice_path: source_db.get_result(invoice_path) for invoice_path in invoice_paths
    }
    benchmark_target = _benchmark_database_target(settings, template_family_id, "family_review")
    benchmark_workers = _resolved_worker_count(config, workers)
    config.processing.worker_concurrency = benchmark_workers
    benchmark_db = FingerprintDB(benchmark_target)
    _seed_family_review_db(source_db, benchmark_db, template_family_id)
    benchmark_worker_db = _make_thread_local_benchmark_db_getter(benchmark_target)

    click.echo(f"Family benchmark: {template_family_id}")
    click.echo(f"Selected invoices: {len(invoice_paths)}")
    click.echo(f"Isolated review DB: {benchmark_target}")
    click.echo(f"Workers: {benchmark_workers}")

    def _run_one(inv):
        started = time.time()
        local_db = benchmark_worker_db()
        res = process_single_invoice(inv, settings, config, local_db)
        elapsed_ms = int((time.time() - started) * 1000)
        return inv, res, elapsed_ms

    def _record_success(inv, res, elapsed_ms):
        message = None
        if res.validation_passed is not True:
            message = _format_result_summary_line(Path(inv).name, res, elapsed_ms)
        return {
            "entry": {
                "invoice_path": inv,
                "result": res,
                "error": None,
                "elapsed_ms": elapsed_ms,
            },
            "message": message,
            "progress_outcome": _progress_bucket_for_result(res),
            "force_progress_emit": res.validation_passed is not True,
        }

    def _record_error(inv, elapsed_ms, exc):
        return {
            "entry": {
                "invoice_path": inv,
                "result": None,
                "error": exc,
                "elapsed_ms": elapsed_ms,
            },
            "message": f"{Path(inv).name}: ERROR | {exc}",
            "progress_outcome": "error",
            "force_progress_emit": True,
        }

    candidate_entries = _run_invoice_executor(
        invoice_paths,
        max_workers=benchmark_workers,
        run_one=_run_one,
        on_success=_record_success,
        on_error=_record_error,
        progress_label="Family benchmark progress",
        show_progress=True,
    )
    candidate_entries.sort(
        key=lambda item: (
            invoice_paths.index(item["invoice_path"])
            if item["invoice_path"] in invoice_paths
            else len(invoice_paths)
        )
    )

    comparison = summarize_family_benchmark_comparison(
        template_family_id,
        invoice_paths,
        baseline_results,
        candidate_entries,
    )
    suggestion_queue = suggest_template_family_updates(
        benchmark_db,
        template_family_id=template_family_id,
    )
    comparison["suggestion_queue"] = {
        "queue_count": suggestion_queue["queue_count"],
        "suggestions": suggestion_queue["suggestions"],
    }
    comparison["benchmark_db_path"] = benchmark_target

    click.echo(
        f"Baseline accepted/review/error: "
        f"{comparison['baseline']['status_counts'].get('accepted', 0)}/"
        f"{comparison['baseline']['status_counts'].get('review', 0)}/"
        f"{comparison['baseline']['status_counts'].get('error', 0)}"
    )
    click.echo(
        f"Candidate accepted/review/error: "
        f"{comparison['candidate']['status_counts'].get('accepted', 0)}/"
        f"{comparison['candidate']['status_counts'].get('review', 0)}/"
        f"{comparison['candidate']['status_counts'].get('error', 0)}"
    )
    click.echo(
        f"Improvements/regressions/unchanged: "
        f"{comparison['progress']['improvements']}/"
        f"{comparison['progress']['regressions']}/"
        f"{comparison['progress']['unchanged']}"
    )
    if comparison["progress"]["avg_validation_score_delta"] is not None:
        click.echo(
            f"Average validation score: "
            f"{comparison['baseline']['avg_validation_score']:.3f} -> "
            f"{comparison['candidate']['avg_validation_score']:.3f} "
            f"({comparison['progress']['avg_validation_score_delta']:+.3f})"
        )
    if comparison["suggestion_queue"]["queue_count"]:
        click.echo(
            f"Remaining suggestions after rerun: {comparison['suggestion_queue']['queue_count']}"
        )

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(comparison, indent=2))
        click.echo(f"Family benchmark summary written to {output_json}")


@cli.command(name="family-update")
@click.argument("template_family_id", type=str)
@click.option(
    "--status",
    type=click.Choice(["provisional", "established", "degraded", "retired"]),
    default=None,
    help="Override the family lifecycle status.",
)
@click.option("--provider-name", type=str, default=None, help="Override provider name.")
@click.option("--country-code", type=str, default=None, help="Override country code.")
@click.option(
    "--document-family",
    type=click.Choice(["invoice", "estimate", "statement", "attachment", "unknown"]),
    default=None,
    help="Override document family.",
)
@click.option(
    "--profile-json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSON file with extraction profile updates.",
)
@click.option(
    "--replace-profile",
    is_flag=True,
    default=False,
    help="Replace the stored profile instead of merging it.",
)
@click.option(
    "--reason",
    type=str,
    default="manual_update",
    show_default=True,
    help="Reason recorded in family version history.",
)
def family_update(
    template_family_id,
    status,
    provider_name,
    country_code,
    document_family,
    profile_json,
    replace_profile,
    reason,
):
    """Manually update a template family and record a versioned change."""
    if not any(
        [status, provider_name is not None, country_code is not None, document_family, profile_json]
    ):
        raise click.ClickException("No family changes provided.")

    profile_updates = _load_json_file(profile_json) if profile_json else None
    if profile_updates is not None and not isinstance(profile_updates, dict):
        raise click.ClickException(f"Family profile update must be a JSON object: {profile_json}")

    settings, _ = _load_runtime("database_url")
    db = FingerprintDB(_primary_database_target(settings))
    updated = manually_update_template_family(
        db,
        template_family_id,
        status=TemplateStatus(status) if status else None,
        provider_name=provider_name,
        country_code=country_code,
        document_family=DocumentFamily(document_family) if document_family else None,
        extraction_profile_updates=profile_updates,
        replace_extraction_profile=replace_profile,
        reason=reason,
    )
    if updated is None:
        raise click.ClickException(f"Template family not found: {template_family_id}")

    click.echo(f"Updated template family: {updated.template_family_id}")
    click.echo(
        f"Status: {updated.status.value} | provider={updated.provider_name or 'unknown'} "
        f"| country={updated.country_code or 'unknown'} | document_family={updated.document_family.value}"
    )
    click.echo(f"Version note: {reason}")


@cli.command(name="heal-rejections")
@click.argument("input_dir", type=str)
@click.option(
    "--family-id", type=str, default=None, help="Restrict healing to one stored template family id."
)
@click.option("--limit", type=int, default=None, help="Maximum failed rows to inspect.")
@click.option(
    "--show-per-invoice",
    is_flag=True,
    default=False,
    help="Print one line per attempted healing retry.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Retry rows even if they were already auto-healed once.",
)
def heal_rejections(input_dir, family_id, limit, show_per_invoice, force):
    """Retry historical failed rows when GT-backed trust now authorizes reuse."""
    settings, config = _load_runtime("database_url", "redis_url")
    input_dir = _resolve_input_dir(input_dir, settings)
    db = FingerprintDB(_primary_database_target(settings))
    summary = _heal_rejections_pass(
        input_dir,
        settings,
        config,
        db,
        healing_origin="manual_cli",
        template_family_id=family_id,
        limit=limit,
        show_per_invoice=show_per_invoice,
        force=force,
    )
    click.echo(
        "Healing summary: "
        f"candidates={summary['candidates']} "
        f"| attempted={summary['attempted']} "
        f"| recovered={summary['recovered']} "
        f"| still_failing={summary['still_failing']} "
        f"| skipped_as_already_attempted={summary['skipped_as_already_attempted']}"
    )
    click.echo(
        f"Skipped not APPLY={summary['skipped_not_apply']} "
        f"| skipped not GT-trusted={summary['skipped_not_gt_trusted']}"
    )


@cli.command()
@click.option("--limit", type=int, default=50, help="Limit output list.")
@click.option(
    "--dataset", type=str, default=None, help="Dataset folder name filter, e.g. invoices-small."
)
def validate(limit, dataset):
    """List failed validation results with review context."""
    settings, config = _load_runtime("database_url")
    db = FingerprintDB(_primary_database_target(settings))
    failed_results = [
        result
        for result in db.get_failed_results()
        if not dataset or dataset in Path(result.invoice_path).parts
    ]
    click.echo(f"Failed validations: {len(failed_results)}")
    for result in failed_results[:limit]:
        click.echo(_format_failed_result_line(result))
        diagnostics = result.diagnostics
        if diagnostics and diagnostics.validation_errors:
            for error in diagnostics.validation_errors[:3]:
                click.echo(f"  - {_simplify_validation_error(error)}")


@cli.command(name="analyze-failures")
@click.option(
    "--dataset", type=str, default=None, help="Dataset folder name filter, e.g. invoices-small."
)
def analyze_failures(dataset):
    """Summarize rejected rows and discovery failure causes."""
    settings, config = _load_runtime("database_url")
    db = FingerprintDB(_primary_database_target(settings))
    summary = summarize_failure_modes(
        db, dataset_filter=dataset, discovery_threshold=config.validation.discovery_threshold
    )

    label = dataset or "all datasets"
    click.echo(f"Failure analysis for {label}:")
    click.echo(f"  Failed total: {summary['failed_total']}")
    click.echo(f"  Failed discovery: {summary['failed_discovery']}")
    click.echo(f"  Failed apply: {summary['failed_apply']}")
    click.echo("  Attempted routes:")
    for route, count in sorted(summary["attempted_route_counts"].items()):
        click.echo(f"    {route}: {count}")

    click.echo("  Discovery error categories:")
    for category, count in sorted(summary["discovery_error_category_counts"].items()):
        click.echo(f"    {category}: {count}")

    click.echo("  Discovery scalar missing/mismatch buckets:")
    for bucket, count in sorted(summary["discovery_scalar_count_buckets"].items()):
        click.echo(f"    {bucket}: {count}")

    click.echo(
        f"  Discovery line-item mismatch rows: {summary['discovery_line_item_mismatch_count']}"
    )

    click.echo("  Discovery score bands:")
    for band, count in sorted(summary["discovery_score_band_counts"].items()):
        click.echo(f"    {band}: {count}")


@cli.command(name="benchmark-heuristic")
@click.argument("input_dir", type=str)
@click.option("--limit", type=int, default=None, help="Limit number of invoices to process.")
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Invoice worker count. Defaults to configured concurrency on Postgres.",
)
@click.option(
    "--output-json",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path for the benchmark summary JSON.",
)
def benchmark_heuristic(input_dir, limit, workers, output_json):
    """Run a sequential heuristic-only benchmark without fingerprint reuse."""
    settings, config = _load_runtime("database_url", "redis_url")
    input_dir = _resolve_input_dir(input_dir, settings)

    invoices = list_invoices(input_dir)
    if limit:
        invoices = invoices[:limit]

    dataset_name = Path(input_dir).name
    benchmark_db_target = _benchmark_database_target(settings, dataset_name, "heuristic")
    benchmark_workers = _resolved_worker_count(config, workers)
    config.processing.worker_concurrency = benchmark_workers
    db = FingerprintDB(benchmark_db_target)

    critical_aliases = {
        "invoiceNumber": [
            "invoiceNumber",
            "invoice number",
            "invoice no",
            "invoice no.",
            "inv no",
            "inv #",
        ],
        "invoiceDate": ["invoiceDate", "invoice date", "date", "issue date", "billing date"],
        "total": [
            "total",
            "totalAmount",
            "invoice total",
            "amount due",
            "balance due",
            "total due",
        ],
    }

    results = []
    per_invoice = []
    critical_hits = 0
    critical_total = 0
    matched_labels = 0
    located_fields = 0
    elapsed_times = []
    peak_rss_samples = {}
    top_failure_categories: Counter[str] = Counter()
    benchmark_worker_db = _make_thread_local_benchmark_db_getter(benchmark_db_target)

    click.echo(f"Benchmarking {len(invoices)} invoices with heuristic discovery only...")
    click.echo("Discovery strategy: heuristic")
    if getattr(config.processing, "applied_profile", None):
        click.echo(f"Runner profile: {config.processing.applied_profile}")
    click.echo(f"Isolated benchmark DB: {benchmark_db_target}")
    click.echo(f"Workers: {benchmark_workers}")

    def _process_one(inv):
        local_db = benchmark_worker_db()
        res, elapsed_ms, peak_rss_mb = measure_operation(
            lambda: process_single_invoice(
                inv, settings, config, local_db, active_fingerprints_override=[]
            )
        )
        peak_rss_samples[inv] = peak_rss_mb
        return inv, res, elapsed_ms

    def _record_success(inv, res, elapsed_ms):
        nonlocal critical_hits, critical_total, matched_labels, located_fields
        elapsed_times.append(elapsed_ms)
        peak_rss_mb = peak_rss_samples.get(inv)
        entry = {
            "name": Path(inv).name,
            "route": res.route_used.value if res.route_used else "None",
            "status": "Passed"
            if res.validation_passed
            else ("Failed/Rejected" if res.validation_passed is False else "No GT/Skip"),
            "elapsed_ms": elapsed_ms,
            "peak_rss_mb": peak_rss_mb,
            "result": res,
            "error": None,
        }

        diagnostics = res.diagnostics
        matched_labels += (diagnostics.matched_label_count or 0) if diagnostics else 0
        located_fields += (diagnostics.located_field_count or 0) if diagnostics else 0
        for aliases in critical_aliases.values():
            if _find_value_by_alias(res.ground_truth or {}, aliases) is not None:
                critical_total += 1
                if _find_value_by_alias(res.extracted_data or {}, aliases) not in (None, ""):
                    critical_hits += 1
        if diagnostics:
            for error in diagnostics.validation_errors:
                if error.startswith("Missing extracted field:"):
                    top_failure_categories["missing_field"] += 1
                elif error.startswith("Mismatch on "):
                    top_failure_categories["field_mismatch"] += 1
                elif error.startswith("Line item count mismatch:"):
                    top_failure_categories["line_item_count_mismatch"] += 1
                elif error.startswith("Total mismatch:"):
                    top_failure_categories["total_mismatch"] += 1
                elif error.startswith("Subtotal mismatch:"):
                    top_failure_categories["subtotal_mismatch"] += 1
                else:
                    top_failure_categories["other"] += 1

        per_invoice.append(
            {
                "invoice_name": Path(inv).name,
                "invoice_path": inv,
                "elapsed_ms": elapsed_ms,
                "peak_rss_mb": peak_rss_mb,
                "attempted_route": res.attempted_route.value if res.attempted_route else None,
                "validation_passed": res.validation_passed,
                "validation_score": diagnostics.validation_score if diagnostics else None,
                "matched_label_count": diagnostics.matched_label_count if diagnostics else None,
                "located_field_count": diagnostics.located_field_count if diagnostics else None,
                "critical_field_count": diagnostics.critical_field_count if diagnostics else None,
                "table_detected": diagnostics.table_detected if diagnostics else None,
                "party_block_diagnostics": diagnostics.party_block_diagnostics
                if diagnostics
                else [],
                "summary_candidate_diagnostics": diagnostics.summary_candidate_diagnostics
                if diagnostics
                else {},
                "line_item_source": diagnostics.line_item_source if diagnostics else None,
                "validation_errors": diagnostics.validation_errors if diagnostics else [],
            }
        )
        message = None
        if entry["status"] != "Passed":
            message = f"{Path(inv).name}: {entry['status']} in {elapsed_ms}ms"
        return {
            "entry": entry,
            "message": message,
            "progress_outcome": _progress_bucket_for_result(res),
            "force_progress_emit": entry["status"] != "Passed",
        }

    def _record_error(inv, elapsed_ms, exc):
        elapsed_times.append(elapsed_ms)
        peak_rss_mb = peak_rss_samples.get(inv)
        entry = {
            "name": Path(inv).name,
            "route": "ERROR",
            "status": str(exc),
            "elapsed_ms": elapsed_ms,
            "peak_rss_mb": peak_rss_mb,
            "result": None,
            "error": exc,
        }
        per_invoice.append(
            {
                "invoice_name": Path(inv).name,
                "invoice_path": inv,
                "elapsed_ms": elapsed_ms,
                "peak_rss_mb": peak_rss_mb,
                "attempted_route": None,
                "validation_passed": False,
                "validation_score": None,
                "matched_label_count": None,
                "located_field_count": None,
                "critical_field_count": None,
                "table_detected": None,
                "party_block_diagnostics": [],
                "summary_candidate_diagnostics": {},
                "line_item_source": None,
                "validation_errors": [str(exc)],
            }
        )
        top_failure_categories["runtime_error"] += 1
        return {
            "entry": entry,
            "message": f"{Path(inv).name}: Exception in {elapsed_ms}ms - {exc}",
            "progress_outcome": "error",
            "force_progress_emit": True,
        }

    results = _run_invoice_executor(
        invoices,
        max_workers=benchmark_workers,
        run_one=_process_one,
        on_success=_record_success,
        on_error=_record_error,
        progress_label="Benchmark progress",
        show_progress=True,
    )

    invoice_order = {Path(inv).name: idx for idx, inv in enumerate(invoices)}
    results.sort(key=lambda item: invoice_order.get(item["name"], len(invoice_order)))
    per_invoice.sort(key=lambda item: invoice_order.get(item["invoice_name"], len(invoice_order)))

    failure_summary = summarize_failure_modes(
        db,
        dataset_filter=dataset_name,
        discovery_threshold=config.validation.discovery_threshold,
    )
    total_ms = sum(elapsed_times)
    summary = build_process_run_summary(
        input_dir=input_dir,
        results=results,
        total_ms=total_ms,
        failure_summary=failure_summary,
        discovery_only=True,
        discovery_mode="heuristic",
    )
    summary["heuristic_backend"] = "heuristic"
    summary["critical_field_hit_rate"] = (
        round((critical_hits / critical_total), 4) if critical_total else None
    )
    summary["label_match_rate"] = (
        round((matched_labels / located_fields), 4) if located_fields else None
    )
    summary["top_failure_categories"] = dict(top_failure_categories.most_common(10))
    summary["per_invoice"] = per_invoice
    summary["benchmark_db_path"] = str(benchmark_db_target)
    summary["ground_truth_sync"] = summarize_ground_truth_sync(input_dir)
    issue_ledger = build_issue_ledger(summary)
    summary["issue_ledger"] = issue_ledger
    summary["issue_count"] = issue_ledger["issue_count"]
    summary["weighted_issue_count"] = issue_ledger["weighted_issue_count"]
    summary["work_buckets"] = issue_ledger["work_buckets"]

    output_path = output_json or (
        Path(settings.output_dir) / f"heuristic_benchmark_{dataset_name}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))

    click.echo("")
    click.echo(f"Pass count:           {summary['discovery_passed']}")
    click.echo(f"Reject count:         {summary['discovery_rejected']}")
    click.echo(f"Runtime errors:       {summary['discovery_runtime_errors']}")
    click.echo(f"Average runtime:      {summary['avg_time_ms']}ms/invoice")
    click.echo(f"Critical-field hits:  {summary['critical_field_hit_rate']}")
    click.echo(f"Label-match rate:     {summary['label_match_rate']}")
    click.echo("Top failure categories:")
    for category, count in summary["top_failure_categories"].items():
        click.echo(f"  {category}: {count}")
    _emit_family_priority_block(summary)
    click.echo(f"Benchmark summary written to {output_path}")


@cli.command(name="benchmark-structural")
@click.argument("input_dir", type=str)
@click.option(
    "--subset-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_STRUCTURAL_SUBSET_FILE,
    show_default=True,
    help="Curated invoice subset manifest.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit number of invoices to process after subset selection.",
)
@click.option(
    "--output-json",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path for the benchmark summary JSON.",
)
def benchmark_structural(input_dir, subset_file, limit, output_json):
    """Run the curated structural benchmark subset."""
    settings, config = _load_runtime("database_url", "redis_url")
    input_dir = _resolve_input_dir(input_dir, settings)

    subset_manifest = _load_structural_subset_manifest(subset_file)
    invoices = _select_invoices_for_subset(list_invoices(input_dir), subset_manifest)
    if limit:
        invoices = invoices[:limit]

    dataset_name = Path(input_dir).name
    benchmark_db_target = _benchmark_database_target(settings, dataset_name, "structural")
    db = FingerprintDB(benchmark_db_target)

    click.echo(
        f"Benchmarking {len(invoices)} curated structural invoices with heuristic discovery only..."
    )
    click.echo(f"Subset manifest: {subset_file}")
    click.echo("Discovery strategy: heuristic")
    if getattr(config.processing, "applied_profile", None):
        click.echo(f"Runner profile: {config.processing.applied_profile}")
    click.echo(f"Isolated benchmark DB: {benchmark_db_target}")

    results = []
    per_invoice = []
    critical_aliases = {
        "invoiceNumber": [
            "invoiceNumber",
            "invoice number",
            "invoice no",
            "invoice no.",
            "inv no",
            "inv #",
        ],
        "invoiceDate": ["invoiceDate", "invoice date", "date", "issue date", "billing date"],
        "total": [
            "total",
            "totalAmount",
            "invoice total",
            "amount due",
            "balance due",
            "total due",
        ],
    }
    critical_hits = 0
    critical_total = 0
    matched_labels = 0
    located_fields = 0
    elapsed_times = []
    top_failure_categories: Counter[str] = Counter()

    for inv in invoices:
        try:
            res, elapsed_ms, peak_rss_mb = measure_operation(
                lambda: process_single_invoice(
                    inv, settings, config, db, active_fingerprints_override=[]
                )
            )
            elapsed_times.append(elapsed_ms)
            results.append(
                {
                    "name": Path(inv).name,
                    "route": res.route_used.value if res.route_used else "None",
                    "status": "Passed"
                    if res.validation_passed
                    else ("Failed/Rejected" if res.validation_passed is False else "No GT/Skip"),
                    "elapsed_ms": elapsed_ms,
                    "peak_rss_mb": peak_rss_mb,
                    "result": res,
                    "error": None,
                }
            )

            diagnostics = res.diagnostics
            matched_labels += (diagnostics.matched_label_count or 0) if diagnostics else 0
            located_fields += (diagnostics.located_field_count or 0) if diagnostics else 0
            for aliases in critical_aliases.values():
                if _find_value_by_alias(res.ground_truth or {}, aliases) is not None:
                    critical_total += 1
                    if _find_value_by_alias(res.extracted_data or {}, aliases) not in (None, ""):
                        critical_hits += 1
            if diagnostics:
                for error in diagnostics.validation_errors:
                    if error.startswith("Missing extracted field:"):
                        top_failure_categories["missing_field"] += 1
                    elif error.startswith("Mismatch on "):
                        top_failure_categories["field_mismatch"] += 1
                    elif error.startswith("Line item count mismatch:"):
                        top_failure_categories["line_item_count_mismatch"] += 1
                    elif error.startswith("Total mismatch:"):
                        top_failure_categories["total_mismatch"] += 1
                    elif error.startswith("Subtotal mismatch:"):
                        top_failure_categories["subtotal_mismatch"] += 1
                    else:
                        top_failure_categories["other"] += 1

            per_invoice.append(
                {
                    "invoice_name": Path(inv).name,
                    "invoice_path": inv,
                    "elapsed_ms": elapsed_ms,
                    "peak_rss_mb": peak_rss_mb,
                    "attempted_route": res.attempted_route.value if res.attempted_route else None,
                    "validation_passed": res.validation_passed,
                    "validation_score": diagnostics.validation_score if diagnostics else None,
                    "matched_label_count": diagnostics.matched_label_count if diagnostics else None,
                    "located_field_count": diagnostics.located_field_count if diagnostics else None,
                    "critical_field_count": diagnostics.critical_field_count
                    if diagnostics
                    else None,
                    "table_detected": diagnostics.table_detected if diagnostics else None,
                    "party_block_diagnostics": diagnostics.party_block_diagnostics
                    if diagnostics
                    else [],
                    "summary_candidate_diagnostics": diagnostics.summary_candidate_diagnostics
                    if diagnostics
                    else {},
                    "line_item_source": diagnostics.line_item_source if diagnostics else None,
                    "validation_errors": diagnostics.validation_errors if diagnostics else [],
                }
            )
            click.echo(f"{Path(inv).name}: {results[-1]['status']} in {elapsed_ms}ms")
        except Exception as exc:
            elapsed_ms = 0
            peak_rss_mb = None
            elapsed_times.append(elapsed_ms)
            results.append(
                {
                    "name": Path(inv).name,
                    "route": "ERROR",
                    "status": str(exc),
                    "elapsed_ms": elapsed_ms,
                    "peak_rss_mb": peak_rss_mb,
                    "result": None,
                    "error": exc,
                }
            )
            per_invoice.append(
                {
                    "invoice_name": Path(inv).name,
                    "invoice_path": inv,
                    "elapsed_ms": elapsed_ms,
                    "peak_rss_mb": peak_rss_mb,
                    "attempted_route": None,
                    "validation_passed": False,
                    "validation_score": None,
                    "matched_label_count": None,
                    "located_field_count": None,
                    "critical_field_count": None,
                    "table_detected": None,
                    "party_block_diagnostics": [],
                    "summary_candidate_diagnostics": {},
                    "line_item_source": None,
                    "validation_errors": [str(exc)],
                }
            )
            top_failure_categories["runtime_error"] += 1
            click.echo(f"{Path(inv).name}: Exception in {elapsed_ms}ms - {exc}")

    failure_summary = summarize_failure_modes(
        db,
        dataset_filter=dataset_name,
        discovery_threshold=config.validation.discovery_threshold,
    )
    total_ms = sum(elapsed_times)
    summary = build_process_run_summary(
        input_dir=input_dir,
        results=results,
        total_ms=total_ms,
        failure_summary=failure_summary,
        discovery_only=True,
        discovery_mode="heuristic",
    )
    summary["heuristic_backend"] = "heuristic"
    summary["benchmark_variant"] = "structural"
    summary["critical_field_hit_rate"] = (
        round((critical_hits / critical_total), 4) if critical_total else None
    )
    summary["label_match_rate"] = (
        round((matched_labels / located_fields), 4) if located_fields else None
    )
    summary["top_failure_categories"] = dict(top_failure_categories.most_common(10))
    summary["per_invoice"] = per_invoice
    summary["benchmark_db_path"] = str(benchmark_db_target)
    summary["ground_truth_sync"] = summarize_ground_truth_sync(input_dir)
    summary["issue_ledger"] = build_issue_ledger(summary)
    summary["structural_subset"] = {
        "source": str(subset_file),
        "case_count": len(subset_manifest),
        "buckets": dict(
            Counter(entry.get("bucket") or "uncategorized" for entry in subset_manifest)
        ),
        "bucket_order": list(
            dict.fromkeys(entry.get("bucket") or "uncategorized" for entry in subset_manifest)
        ),
        "cases": subset_manifest,
    }

    output_path = output_json or (
        Path(settings.output_dir) / f"heuristic_benchmark_{dataset_name}_structural.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))

    click.echo("")
    click.echo(f"Pass count:           {summary['discovery_passed']}")
    click.echo(f"Reject count:         {summary['discovery_rejected']}")
    click.echo(f"Runtime errors:       {summary['discovery_runtime_errors']}")
    click.echo(f"Average runtime:      {summary['avg_time_ms']}ms/invoice")
    click.echo(f"Critical-field hits:  {summary['critical_field_hit_rate']}")
    click.echo(f"Label-match rate:     {summary['label_match_rate']}")
    click.echo("Top failure categories:")
    for category, count in summary["top_failure_categories"].items():
        click.echo(f"  {category}: {count}")
    _emit_family_priority_block(summary)
    click.echo(f"Benchmark summary written to {output_path}")


@cli.command(name="sync-ground-truth")
@click.argument("input_dir", type=str)
@click.option(
    "--source-dir", type=str, default=None, help="Override the GT source-of-truth directory."
)
@click.option(
    "--check", is_flag=True, default=False, help="Report GT sync status without copying files."
)
def sync_ground_truth(input_dir, source_dir, check):
    """Audit or sync ground-truth JSONs from the source-of-truth dataset."""
    settings, _ = _load_runtime()
    input_dir = _resolve_input_dir(input_dir, settings)
    if check:
        summary = summarize_ground_truth_sync(
            input_dir, source_dir=source_dir, dataset_root=settings.dataset_root
        )
        click.echo(f"Dataset: {summary['dataset_dir']}")
        click.echo(f"Source of truth: {summary['source_of_truth_dir']}")
        click.echo(f"Status: {summary['status']}")
        click.echo(f"Checked invoices: {summary['checked_invoice_count']}")
        click.echo(f"Matching: {summary['matching_count']}")
        click.echo(f"Mismatched: {summary['mismatched_count']}")
        click.echo(f"Missing local: {summary['missing_local_count']}")
        click.echo(f"Missing source: {summary['missing_source_count']}")
        return

    summary = sync_ground_truth_from_source(
        input_dir, source_dir=source_dir, dataset_root=settings.dataset_root
    )
    click.echo(f"Dataset: {summary['dataset_dir']}")
    click.echo(f"Source of truth: {summary['source_of_truth_dir']}")
    click.echo(f"Status: {summary['status']}")
    click.echo(f"Copied: {summary['copied_count']}")
    click.echo(f"Updated: {summary['updated_count']}")
    click.echo(f"Unchanged: {summary['unchanged_count']}")


@cli.command(name="compare-runs")
@click.option("--dataset", type=str, required=True, help="Dataset folder name, e.g. invoices-test.")
@click.option(
    "--baseline-run",
    type=int,
    default=None,
    help="Baseline run id. Defaults to the previous run for the dataset.",
)
@click.option(
    "--candidate-run",
    type=int,
    default=None,
    help="Candidate run id. Defaults to the latest run for the dataset.",
)
@click.option(
    "--limit", type=int, default=20, help="Maximum number of regressions/improvements to print."
)
def compare_runs(dataset, baseline_run, candidate_run, limit):
    """Compare per-invoice outcomes between two recorded analysis runs."""
    settings, _ = _load_runtime()
    analysis_db = _analysis_database_target(settings)
    comparison = compare_analysis_runs(
        str(analysis_db),
        dataset,
        baseline_run_id=baseline_run,
        candidate_run_id=candidate_run,
    )

    click.echo(f"Dataset: {comparison['dataset']}")
    click.echo(
        f"Baseline run: {comparison['baseline_run']['id']} "
        f"({comparison['baseline_run']['run_at']}) "
        f"passed/rejected/errors="
        f"{comparison['baseline_run']['discovery_passed']}/"
        f"{comparison['baseline_run']['discovery_rejected']}/"
        f"{comparison['baseline_run']['runtime_errors']}"
    )
    click.echo(
        f"Candidate run: {comparison['candidate_run']['id']} "
        f"({comparison['candidate_run']['run_at']}) "
        f"passed/rejected/errors="
        f"{comparison['candidate_run']['discovery_passed']}/"
        f"{comparison['candidate_run']['discovery_rejected']}/"
        f"{comparison['candidate_run']['runtime_errors']}"
    )
    click.echo(f"Regressions: {len(comparison['regressions'])}")
    for item in comparison["regressions"][:limit]:
        click.echo(
            f"  REGRESSION {item['invoice_name']}: "
            f"{item['baseline_status']} ({item['baseline_score']}) -> "
            f"{item['candidate_status']} ({item['candidate_score']})"
        )
    click.echo(f"Improvements: {len(comparison['improvements'])}")
    for item in comparison["improvements"][:limit]:
        click.echo(
            f"  IMPROVEMENT {item['invoice_name']}: "
            f"{item['baseline_status']} ({item['baseline_score']}) -> "
            f"{item['candidate_status']} ({item['candidate_score']})"
        )
    click.echo(f"Unchanged: {comparison['unchanged']}")


@cli.command(name="compare-benchmark-summaries")
@click.argument(
    "baseline_summary", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "candidate_summary",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output-json",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the comparison report to this path.",
)
def compare_benchmark_summaries_cmd(baseline_summary, candidate_summary, output_json):
    """Compare two benchmark summary JSON files and emit a machine-readable issue report."""
    baseline = json.loads(baseline_summary.read_text())
    candidate = json.loads(candidate_summary.read_text())
    report = compare_benchmark_summaries(baseline, candidate)
    report["baseline_summary_path"] = str(baseline_summary)
    report["candidate_summary_path"] = str(candidate_summary)

    payload = json.dumps(report, indent=2)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(payload)
    click.echo(payload)


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
