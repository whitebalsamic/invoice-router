import logging
import re
from collections import Counter
from datetime import date
from typing import Any, Dict, List, Optional

from dateutil.parser import parse as parse_date

from ...config import ValidationConfig
from ...models import ValidationResult
from ..ground_truth import canonicalize_ground_truth
from ..invoices.country_rules import validate_country_currency

logger = logging.getLogger(__name__)

_NUMERIC_SEPARATED_DATE_RE = re.compile(r"^\s*(\d{1,4})[/-](\d{1,2})[/-](\d{1,4})\s*$")


def _normalize_year_fragment(value: str) -> int:
    if len(value) == 2:
        return int(f"20{value}") if int(value) < 70 else int(f"19{value}")
    return int(value)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute character-level edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def normalize_string(s: str) -> str:
    """Strip leading/trailing whitespace and collapse internal whitespace."""
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_amount(s: str) -> float:
    """Strip currency symbols, spaces, and parse float."""
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s)
    s = re.sub(r"[^\d.,-]", "", s)

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        if len(s) - s.rfind(",") == 4:
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


def normalize_date(s: str, country_code: Optional[str] = None) -> str:
    """Parse date to ISO8601 format, using country for ambiguous slash dates."""
    text = str(s).strip()
    match = _NUMERIC_SEPARATED_DATE_RE.match(text)
    if match:
        first, second, third = match.groups()
        normalized_country = (country_code or "").upper()

        try:
            if len(first) == 4:
                year = _normalize_year_fragment(first)
                month = int(second)
                day = int(third)
                return date(year, month, day).strftime("%Y-%m-%d")

            first_int = int(first)
            second_int = int(second)
            year = _normalize_year_fragment(third)

            if first_int > 12 and second_int <= 12:
                day, month = first_int, second_int
            elif second_int > 12 and first_int <= 12:
                month, day = first_int, second_int
            elif first_int <= 12 and second_int <= 12:
                if normalized_country == "US":
                    month, day = first_int, second_int
                else:
                    day, month = first_int, second_int
            else:
                return text
            return date(year, month, day).strftime("%Y-%m-%d")
        except Exception:
            return text

    try:
        dt = parse_date(text, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        # Fallback date swap logic could go here
        return text


def _compare_values(
    val1: Any,
    val2: Any,
    field_name: str,
    config: ValidationConfig,
    country_code: Optional[str] = None,
) -> bool:
    """Compare two values based on field heuristics."""
    if val1 is None or val2 is None:
        return False

    name_lower = field_name.lower()

    if any(
        token in name_lower for token in ("price", "amount", "total", "tax", "shipping", "discount")
    ):
        v1 = normalize_amount(val1)
        v2 = normalize_amount(val2)
        result = abs(v1 - v2) < 0.01
        if not result and config.allow_cent_scale_equivalence:
            result = abs(v1 * 100 - v2) < 0.01 or abs(v1 - v2 * 100) < 0.01
        logger.debug(
            f"  [{field_name}] amount compare: {v1} vs {v2} → {'MATCH' if result else 'MISMATCH'}"
        )
        return result

    if "qty" in name_lower or "quantity" in name_lower:
        v1 = normalize_amount(val1)
        v2 = normalize_amount(val2)
        result = round(v1, 3) == round(v2, 3)
        logger.debug(
            f"  [{field_name}] qty compare: {v1} vs {v2} → {'MATCH' if result else 'MISMATCH'}"
        )
        return result

    if "date" in name_lower:
        d1 = normalize_date(val1, country_code=country_code)
        d2 = normalize_date(val2, country_code=country_code)
        result = d1 == d2
        logger.debug(
            f"  [{field_name}] date compare: '{d1}' vs '{d2}' → {'MATCH' if result else 'MISMATCH'}"
        )
        return result

    s1 = normalize_string(val1).lower()
    s2 = normalize_string(val2).lower()

    # Use edit-distance for name fields; word-set Jaccard for everything else
    if "name" in name_lower:
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            result = True
        else:
            dist = levenshtein_distance(s1, s2)
            similarity = 1.0 - (dist / max_len)
            result = similarity >= config.jaccard_threshold
        logger.debug(
            f"  [{field_name}] edit-distance compare: '{s1}' vs '{s2}' → similarity={result:.3f} (threshold={config.jaccard_threshold}) → {'MATCH' if result else 'MISMATCH'}"
        )
        return result

    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        result = s1 == s2
        logger.debug(
            f"  [{field_name}] exact compare: '{s1}' vs '{s2}' → {'MATCH' if result else 'MISMATCH'}"
        )
        return result
    jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
    result = jaccard >= config.jaccard_threshold
    logger.debug(
        f"  [{field_name}] jaccard compare: '{s1}' vs '{s2}' → jaccard={jaccard:.3f} (threshold={config.jaccard_threshold}) → {'MATCH' if result else 'MISMATCH'}"
    )
    return result


def apply_field_mapping(data: Dict[str, Any], field_mapping: Dict[str, str]) -> Dict[str, Any]:
    """Map extracted field names to canonical schema."""
    mapped_data = {}
    for k, v in data.items():
        if k == "line_items":
            mapped_lines = []
            for item in v:
                mapped_item = {field_mapping.get(ik, ik): iv for ik, iv in item.items()}
                mapped_lines.append(mapped_item)
            mapped_data[k] = mapped_lines
        else:
            mapped_data[field_mapping.get(k, k)] = v
    return mapped_data


def _approx_equal(val1: Optional[float], val2: Optional[float], tolerance: float = 0.01) -> bool:
    if val1 is None or val2 is None:
        return False
    return abs(val1 - val2) < tolerance


def _validate_line_item_arithmetic(ex_lines: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    for idx, line in enumerate(ex_lines):
        if not isinstance(line, dict):
            continue
        amount_provenance = (
            line.get("_provenance", {}).get("amount", {})
            if isinstance(line.get("_provenance"), dict)
            else {}
        )
        if amount_provenance.get("source") == "summary_consistency":
            continue
        qty = line.get("quantity") if "quantity" in line else line.get("qty")
        unit_price = line.get("unit_price") if "unit_price" in line else line.get("price")
        amount = line.get("amount") if "amount" in line else line.get("total")

        if qty is None or unit_price is None or amount is None:
            continue

        qty_val = normalize_amount(qty)
        unit_price_val = normalize_amount(unit_price)
        amount_val = normalize_amount(amount)
        expected = round(qty_val * unit_price_val, 2)
        if not _approx_equal(expected, amount_val, tolerance=0.02):
            errors.append(
                f"Line item arithmetic mismatch at row {idx + 1}: expected {expected:.2f}, got {amount_val:.2f}"
            )
    return errors


def _validate_summary_arithmetic(mapped_extracted: Dict[str, Any]) -> List[str]:
    reconciliation_summary = mapped_extracted.get("reconciliation_summary")
    if isinstance(reconciliation_summary, dict):
        summary_errors: List[str] = []
        for issue in reconciliation_summary.get("issues") or []:
            if not isinstance(issue, dict):
                continue
            kind = str(issue.get("kind") or "")
            message = normalize_string(issue.get("message") or "")
            if kind.endswith("_mismatch") and message:
                summary_errors.append(message)
        return summary_errors

    errors: List[str] = []

    subtotal = mapped_extracted.get("subtotal")
    tax = mapped_extracted.get("tax")
    shipping = mapped_extracted.get("shipping")
    total = mapped_extracted.get("total")
    if total is None:
        total = mapped_extracted.get("totalAmount")
    ex_lines = mapped_extracted.get("line_items", [])

    subtotal_val = normalize_amount(subtotal) if subtotal is not None else None
    tax_val = normalize_amount(tax) if tax is not None else None
    shipping_val = normalize_amount(shipping) if shipping is not None else 0.0
    total_val = normalize_amount(total) if total is not None else None

    line_amounts = []
    for line in ex_lines:
        if not isinstance(line, dict):
            continue
        amount = line.get("amount") if "amount" in line else line.get("total")
        if amount is None:
            continue
        line_amounts.append(normalize_amount(amount))

    if subtotal_val is not None and line_amounts:
        line_sum = round(sum(line_amounts), 2)
        scalar_summary_reconciled = total_val is not None and _approx_equal(
            total_val, subtotal_val + (tax_val or 0.0) + shipping_val, tolerance=0.02
        )
        likely_incomplete_line_items = scalar_summary_reconciled and subtotal_val > line_sum + 0.02
        if (
            not _approx_equal(subtotal_val, line_sum, tolerance=0.02)
            and not likely_incomplete_line_items
        ):
            errors.append(
                f"Subtotal mismatch: expected line sum {line_sum:.2f}, got {subtotal_val:.2f}"
            )

    if subtotal_val is not None and tax_val is not None and total_val is not None:
        expected_total = round(subtotal_val + tax_val + shipping_val, 2)
        if not _approx_equal(expected_total, total_val, tolerance=0.02):
            if shipping is not None:
                errors.append(
                    f"Total mismatch: expected subtotal+tax+shipping {expected_total:.2f}, got {total_val:.2f}"
                )
            else:
                errors.append(
                    f"Total mismatch: expected subtotal+tax {expected_total:.2f}, got {total_val:.2f}"
                )

    return errors


def _categorize_validation_errors(errors: List[str]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for error in errors:
        lowered = str(error).strip().lower()
        if lowered.startswith("missing extracted field:"):
            counts["missing_field"] += 1
        elif lowered.startswith("mismatch on "):
            counts["field_mismatch"] += 1
        elif lowered.startswith("line item count mismatch"):
            counts["line_item_count_mismatch"] += 1
        elif lowered.startswith("line item arithmetic mismatch"):
            counts["line_item_arithmetic_mismatch"] += 1
        elif lowered.startswith(
            ("subtotal mismatch", "tax mismatch", "total mismatch", "gross total mismatch")
        ):
            counts["summary_arithmetic_mismatch"] += 1
        elif lowered.startswith("currency mismatch"):
            counts["country_currency_mismatch"] += 1
        else:
            counts["other"] += 1
    return dict(counts)


def validate_invoice(
    extracted_data: Dict[str, Any],
    ground_truth: Dict[str, Any],
    config: ValidationConfig,
    field_mapping: Dict[str, str],
) -> ValidationResult:
    """
    Validate extracted values against ground truth.
    Returns ValidationResult with pass/fail and score.
    """
    matched = 0
    mismatched = 0
    errors = []

    ground_truth = canonicalize_ground_truth(ground_truth)
    mapped_extracted = apply_field_mapping(extracted_data, field_mapping)
    country_code = mapped_extracted.get("country_code")
    reconciliation_summary = (
        mapped_extracted.get("reconciliation_summary")
        if isinstance(mapped_extracted.get("reconciliation_summary"), dict)
        else None
    )

    # 0. Null in GT is not verified; line items handled separately
    gt_keys = [
        k for k, v in ground_truth.items() if v is not None and k not in ("line_items", "lineItems")
    ]

    for key in gt_keys:
        gt_val = ground_truth[key]
        ex_val = mapped_extracted.get(key)

        if ex_val is None:
            mismatched += 1
            errors.append(f"Missing extracted field: {key}")
            continue

        if _compare_values(ex_val, gt_val, key, config, country_code=country_code):
            matched += 1
        else:
            mismatched += 1
            errors.append(f"Mismatch on {key}: expected '{gt_val}', got '{ex_val}'")

    # Line items (GT may use camelCase "lineItems" or snake_case "line_items")
    gt_lines = ground_truth.get("lineItems") or ground_truth.get("line_items")
    ex_lines = mapped_extracted.get("line_items", [])

    if gt_lines:
        gt_count = len(gt_lines)
        ex_count = len(ex_lines)
        # Allow ±1 row or ±20% tolerance because OCR/table parsing can split or merge rows.
        tolerance = max(1, round(gt_count * 0.20))
        if abs(gt_count - ex_count) > tolerance:
            mismatched += 1
            errors.append(f"Line item count mismatch: expected {gt_count}, got {ex_count}")
        else:
            # Count is within tolerance — treat as matched
            matched += gt_count

    arithmetic_errors: List[str] = []
    if gt_lines:
        arithmetic_errors.extend(_validate_line_item_arithmetic(ex_lines))
        arithmetic_errors.extend(_validate_summary_arithmetic(mapped_extracted))
    arithmetic_errors.extend(
        validate_country_currency(
            mapped_extracted.get("country_code"),
            mapped_extracted.get("currency_code"),
        )
    )
    for error in arithmetic_errors:
        mismatched += 1
        errors.append(error)

    total_gt = matched + mismatched
    score = (matched / total_gt) if total_gt > 0 else 1.0
    passed = abs(score - 1.0) < 0.001
    error_counts = _categorize_validation_errors(errors)

    return ValidationResult(
        passed=passed,
        score=float(score),
        matched_fields=matched,
        mismatched_fields=mismatched,
        errors=errors,
        arithmetic_errors=arithmetic_errors,
        error_counts=error_counts,
        reconciliation_summary=reconciliation_summary,
    )


def should_reject(result: ValidationResult, route: str, config: ValidationConfig) -> bool:
    if route == "APPLY":
        return result.score < config.apply_threshold
    elif route == "DISCOVERY":
        return result.score < config.discovery_threshold
    return False
