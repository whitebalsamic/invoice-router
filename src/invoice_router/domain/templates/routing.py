import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from ...config import AppConfig
from ...models import (
    DocumentContext,
    ExtractionStrategy,
    FingerprintRecord,
    PageFingerprint,
    PageRole,
    Route,
    TemplateFamilyRecord,
    TemplateStatus,
)

logger = logging.getLogger(__name__)


def _normalize_anchor_token(value: Any) -> str:
    return str(value).strip().lower()


def _stable_anchor_tokens_from_page(page: PageFingerprint) -> set[str]:
    signature = page.stable_anchor_signature or {}
    tokens: set[str] = set()
    for key in ("header_tokens", "summary_labels", "footer_tokens"):
        values = signature.get(key) or []
        tokens.update(_normalize_anchor_token(value) for value in values if value)
    keyword_hits = signature.get("keyword_hits") or {}
    for values in keyword_hits.values():
        tokens.update(_normalize_anchor_token(value) for value in values if value)
    return tokens


def _jaccard_similarity(values1: set[str], values2: set[str]) -> float:
    if not values1 or not values2:
        return 0.0
    union = values1 | values2
    if not union:
        return 0.0
    return len(values1 & values2) / len(union)


def _page_role_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    return value.value if hasattr(value, "value") else str(value)


def _page_weight(page: PageFingerprint) -> float:
    role = _page_role_value(page.role)
    if role in {PageRole.header_page.value, PageRole.summary_page.value}:
        return 1.25
    if role == PageRole.line_item_page.value:
        return 0.55
    return 1.0


def _page_signature_regions(signature: Dict[str, Any]) -> Dict[str, set[str]]:
    keyword_hits = signature.get("keyword_hits") or {}
    header = {
        _normalize_anchor_token(value) for value in (signature.get("header_tokens") or []) if value
    }
    summary = {
        _normalize_anchor_token(value) for value in (signature.get("summary_labels") or []) if value
    }
    summary.update(
        _normalize_anchor_token(value) for value in (keyword_hits.get("summary") or []) if value
    )
    footer = {
        _normalize_anchor_token(value) for value in (signature.get("footer_tokens") or []) if value
    }
    footer.update(
        _normalize_anchor_token(value) for value in (keyword_hits.get("footer") or []) if value
    )
    identity: set[str] = set()
    for key in ("invoice_number", "invoice_date", "provider", "customer"):
        identity.update(
            _normalize_anchor_token(value) for value in (keyword_hits.get(key) or []) if value
        )
    generic = set()
    generic.update(header)
    generic.update(summary)
    generic.update(footer)
    generic.update(identity)
    for values in keyword_hits.values():
        generic.update(_normalize_anchor_token(value) for value in values if value)
    return {
        "header": {token for token in header if token},
        "summary": {token for token in summary if token},
        "identity": {token for token in identity if token},
        "footer": {token for token in footer if token},
        "generic": {token for token in generic if token},
    }


def _aggregate_page_regions(pages: List[PageFingerprint]) -> Dict[str, set[str]]:
    regions = {key: set() for key in ("header", "summary", "identity", "footer", "generic")}
    for page in pages:
        page_regions = _page_signature_regions(page.stable_anchor_signature or {})
        for key, values in page_regions.items():
            regions[key].update(values)
    return regions


def _family_anchor_regions(family: TemplateFamilyRecord) -> Dict[str, set[str]]:
    pages = family.anchor_summary.get("pages") or []
    if pages:
        regions = {key: set() for key in ("header", "summary", "identity", "footer", "generic")}
        for page_signature in pages:
            page_regions = _page_signature_regions(page_signature or {})
            for key, values in page_regions.items():
                regions[key].update(values)
        return regions

    aggregate_keywords = family.anchor_summary.get("aggregate_keywords") or {}
    identity: set[str] = set()
    for key in ("invoice_number", "invoice_date", "provider", "customer"):
        identity.update(
            _normalize_anchor_token(value) for value in (aggregate_keywords.get(key) or []) if value
        )
    summary = {
        _normalize_anchor_token(value)
        for value in (aggregate_keywords.get("summary") or [])
        if value
    }
    footer = {
        _normalize_anchor_token(value)
        for value in (aggregate_keywords.get("footer") or [])
        if value
    }
    generic = {
        _normalize_anchor_token(value)
        for value in (family.stable_anchor_regions.get("tokens") or [])
        if value
    }
    generic.update(identity)
    generic.update(summary)
    generic.update(footer)
    header = generic - summary - footer
    return {
        "header": {token for token in header if token},
        "summary": {token for token in summary if token},
        "identity": {token for token in identity if token},
        "footer": {token for token in footer if token},
        "generic": {token for token in generic if token},
    }


def _weighted_region_similarity(left: Dict[str, set[str]], right: Dict[str, set[str]]) -> float:
    weights = {
        "header": 0.22,
        "identity": 0.28,
        "summary": 0.32,
        "footer": 0.12,
        "generic": 0.06,
    }
    weighted_total = 0.0
    weight_sum = 0.0
    for key, weight in weights.items():
        left_values = left.get(key) or set()
        right_values = right.get(key) or set()
        if not left_values and not right_values:
            continue
        weighted_total += _jaccard_similarity(left_values, right_values) * weight
        weight_sum += weight
    if weight_sum == 0.0:
        return 0.0
    return weighted_total / weight_sum


def _page_signature_similarity(left: PageFingerprint, right: PageFingerprint) -> float:
    left_regions = _page_signature_regions(left.stable_anchor_signature or {})
    right_regions = _page_signature_regions(right.stable_anchor_signature or {})
    score = _weighted_region_similarity(left_regions, right_regions)
    if (
        _page_role_value(left.role) == _page_role_value(right.role)
        and _page_role_value(left.role) is not None
    ):
        score += 0.08
    return min(score, 1.0)


def _directional_page_alignment(
    source_pages: List[PageFingerprint], target_pages: List[PageFingerprint]
) -> float:
    if not source_pages or not target_pages:
        return 0.0
    total = 0.0
    total_weight = 0.0
    for source_page in source_pages:
        weight = _page_weight(source_page)
        best_score = max(
            _page_signature_similarity(source_page, target_page) for target_page in target_pages
        )
        total += best_score * weight
        total_weight += weight
    if total_weight == 0.0:
        return 0.0
    return total / total_weight


def _weighted_page_alignment(
    left_pages: List[PageFingerprint], right_pages: List[PageFingerprint]
) -> float:
    if not left_pages or not right_pages:
        return 0.0
    return (
        _directional_page_alignment(left_pages, right_pages)
        + _directional_page_alignment(right_pages, left_pages)
    ) / 2.0


def _page_role_deltas(expected_roles: List[str], observed_roles: List[str]) -> Tuple[int, int]:
    expected_counts = Counter(role for role in expected_roles if role)
    observed_counts = Counter(role for role in observed_roles if role)
    strong_roles = {PageRole.header_page.value, PageRole.summary_page.value}
    variable_roles = {PageRole.line_item_page.value}
    strong_delta = sum(
        abs(expected_counts.get(role, 0) - observed_counts.get(role, 0)) for role in strong_roles
    )
    variable_delta = sum(
        abs(expected_counts.get(role, 0) - observed_counts.get(role, 0)) for role in variable_roles
    )
    return strong_delta, variable_delta


def _fingerprint_field_names(record: FingerprintRecord) -> set[str]:
    field_names: set[str] = set()
    for page in record.layout_template.get("pages", []):
        field_names.update(
            str(name).strip().lower() for name in (page.get("fields") or {}).keys() if name
        )
    return {name for name in field_names if name}


def _fingerprint_signature(record: FingerprintRecord) -> Dict[str, tuple[str, ...]]:
    return {
        "page_roles": tuple(
            _page_role_value(page.role)
            for page in record.page_fingerprints
            if _page_role_value(page.role)
        ),
        "field_names": tuple(sorted(_fingerprint_field_names(record))),
        "anchor_tokens": tuple(
            sorted(
                {
                    token
                    for page in record.page_fingerprints
                    for token in _stable_anchor_tokens_from_page(page)
                }
            )
        ),
    }


def _signature_similarity(
    left: Dict[str, tuple[str, ...]], right: Dict[str, tuple[str, ...]]
) -> float:
    role_overlap = _jaccard_similarity(set(left["page_roles"]), set(right["page_roles"]))
    field_overlap = _jaccard_similarity(set(left["field_names"]), set(right["field_names"]))
    anchor_overlap = _jaccard_similarity(set(left["anchor_tokens"]), set(right["anchor_tokens"]))
    return round((role_overlap * 0.35) + (field_overlap * 0.35) + (anchor_overlap * 0.30), 4)


def _family_reuse_multiplier(family: TemplateFamilyRecord) -> float:
    apply_count = max(int(family.apply_count), 0)
    reject_count = max(int(family.reject_count), 0)
    if apply_count == 0 and reject_count == 0 and family.status != TemplateStatus.degraded:
        return 1.0
    total_outcomes = max(apply_count + reject_count, 1)
    reject_pressure = reject_count / total_outcomes
    confidence = max(0.0, min(1.0, float(family.confidence)))
    status_multiplier = {
        TemplateStatus.established: 1.0,
        TemplateStatus.provisional: 0.94,
        TemplateStatus.degraded: 0.78,
        TemplateStatus.retired: 0.0,
    }.get(family.status, 0.9)
    support_multiplier = 0.94 + min(apply_count, 20) * 0.003
    confidence_multiplier = 0.9 + confidence * 0.1
    reject_multiplier = 1.0 - min(reject_pressure * 0.28, 0.22)
    multiplier = status_multiplier * support_multiplier * confidence_multiplier * reject_multiplier
    return max(0.6, min(1.0, multiplier))


def _family_split_pressure(
    family: TemplateFamilyRecord,
    family_records: List[FingerprintRecord],
    representative: Optional[FingerprintRecord],
) -> float:
    split_signals = family.anchor_summary.get("split_signals") or {}
    pressure = 0.0
    stored_split_pressure = split_signals.get("split_pressure")
    if stored_split_pressure is not None:
        pressure = max(pressure, min(max(float(stored_split_pressure), 0.0), 1.0))
    dominant_ratio = split_signals.get("dominant_signature_ratio")
    unique_signature_count = split_signals.get("unique_signature_count")
    if dominant_ratio is not None:
        pressure = max(pressure, max(0.0, 1.0 - float(dominant_ratio)))
    average_signature_similarity = split_signals.get("average_signature_similarity")
    if average_signature_similarity is not None:
        pressure = max(pressure, max(0.0, 1.0 - float(average_signature_similarity)))
    if unique_signature_count is not None and family_records:
        pressure = max(
            pressure,
            max(0.0, (int(unique_signature_count) - 1) / max(len(family_records), 1)),
        )
    if len(family_records) > 1:
        base_signature = _fingerprint_signature(representative or family_records[0])
        similarities = []
        for record in family_records:
            candidate_signature = _fingerprint_signature(record)
            similarities.append(_signature_similarity(base_signature, candidate_signature))
        if similarities:
            pressure = max(pressure, 1.0 - (sum(similarities) / len(similarities)))
    return min(max(pressure, 0.0), 1.0)


def hamming_distance(hex1: str, hex2: str) -> int:
    """Calculate the Hamming distance between two 64-bit hex strings."""
    int1 = int(hex1, 16)
    int2 = int(hex2, 16)
    return bin(int1 ^ int2).count("1")


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def normalize_coordinates(
    x: int, y: int, w: int, h: int, page_w: int, page_h: int
) -> Tuple[float, float]:
    """Convert absolute bounding box to normalized center coordinates."""
    cx = (x + w / 2) / page_w
    cy = (y + h / 2) / page_h
    return cx, cy


def lookup_fingerprint(
    new_pages: List[PageFingerprint],
    ocr_results_per_page: List[List[Tuple[str, int, int, int, int]]],
    page_dimensions: List[Tuple[int, int]],
    active_fingerprints: List[FingerprintRecord],
    config: AppConfig,
) -> Tuple[Optional[FingerprintRecord], float]:
    """
    1. Filter by page count and version.
    2. Filter by Hamming distance threshold (per-page).
    3. Sort by total Hamming distance.
    4. Run secondary label confirmation.
    Returns: (Matched Record or None, confirmation_ratio)
    """
    candidates = []
    threshold = config.fingerprinting.visual_hash_hamming_threshold

    skipped_version = sum(1 for r in active_fingerprints if r.version != "v3")
    skipped_page_count = 0
    for record in active_fingerprints:
        if record.version != "v3":
            continue

        stored_pages = record.page_fingerprints
        if len(new_pages) != len(stored_pages):
            logger.debug(
                f"Skipping {record.hash[:12]}: page count mismatch ({len(new_pages)} vs {len(stored_pages)})"
            )
            skipped_page_count += 1
            continue

        total_hamming = 0
        all_pages_pass = True

        sorted_new = sorted(new_pages, key=lambda p: p.page_index)
        sorted_stored = sorted(stored_pages, key=lambda p: p.page_index)

        for new_p, stored_p in zip(sorted_new, sorted_stored):
            dist = hamming_distance(new_p.visual_hash_hex, stored_p.visual_hash_hex)
            logger.debug(f"Hamming distance to {record.hash[:12]}: {dist} (threshold={threshold})")
            if dist > threshold:
                all_pages_pass = False
                break
            total_hamming += dist

        if all_pages_pass:
            candidates.append((total_hamming, record))

    logger.debug(
        f"Fingerprint filter: {len(active_fingerprints)} total, skipped {skipped_version} (version), {skipped_page_count} (page count), {len(candidates)} passed Hamming threshold"
    )
    if not candidates:
        return None, 0.0

    # Sort candidates by lowest total Hamming distance
    candidates.sort(key=lambda x: x[0])

    # Secondary label confirmation
    for total_hamming, record in candidates:
        stored_pages_template = record.layout_template.get("pages", [])

        total_stored_labels = 0
        confirmed_labels = 0

        for page_idx, page_template in enumerate(stored_pages_template):
            pidx = page_template.get("page_index", page_idx)
            if pidx >= len(ocr_results_per_page):
                continue

            ocr_results = ocr_results_per_page[pidx]
            page_w, page_h = page_dimensions[pidx]

            label_confirmation_set = page_template.get("label_confirmation_set", [])
            total_stored_labels += len(label_confirmation_set)

            for label_entry in label_confirmation_set:
                target_text = label_entry.get("label_text")
                target_cx = label_entry.get("norm_cx")
                target_cy = label_entry.get("norm_cy")

                # Exclude if no label string or coordinates are missing
                if not target_text or target_cx is None or target_cy is None:
                    continue

                fuzzy_dist_threshold = label_entry.get("fuzzy_distance_threshold", 2)

                matched = False
                for text, x, y, w, h in ocr_results:
                    cx, cy = normalize_coordinates(x, y, w, h, page_w, page_h)
                    if (
                        abs(cx - target_cx) <= config.discovery.label_position_tolerance
                        and abs(cy - target_cy) <= config.discovery.label_position_tolerance
                    ):
                        if levenshtein_distance(text, target_text) <= fuzzy_dist_threshold:
                            matched = True
                            break
                if matched:
                    confirmed_labels += 1
                else:
                    logger.debug(
                        f"  Label '{target_text}' not confirmed at ({target_cx:.3f},{target_cy:.3f})"
                    )

        if total_stored_labels == 0:
            # Without anchored labels, only accept an exact visual match.
            # Near-matches on masked page hashes are too permissive for similarly
            # structured invoices and can misroute into APPLY with the wrong template.
            if total_hamming == 0:
                return record, 1.0
            logger.debug(
                "Rejecting %s as template match: no label confirmations and non-zero hamming distance (%s)",
                record.hash[:12],
                total_hamming,
            )
            continue

        confirmation_ratio = confirmed_labels / total_stored_labels
        logger.debug(
            f"Label confirmation for {record.hash[:12]}: {confirmed_labels}/{total_stored_labels} = {confirmation_ratio:.3f} (threshold={config.discovery.label_confirmation_threshold})"
        )
        if confirmation_ratio >= config.discovery.label_confirmation_threshold:
            return record, confirmation_ratio

    return None, 0.0


def lookup_template_family(
    new_pages: List[PageFingerprint],
    active_fingerprints: List[FingerprintRecord],
    active_families: List[TemplateFamilyRecord],
    document_context: Optional[DocumentContext],
    config: AppConfig,
) -> Tuple[Optional[TemplateFamilyRecord], Optional[FingerprintRecord], float, int]:
    if not active_families:
        return None, None, 0.0, 0

    pages_by_family: Dict[str, List[FingerprintRecord]] = {}
    for record in active_fingerprints:
        if record.template_family_id:
            pages_by_family.setdefault(record.template_family_id, []).append(record)

    new_regions = _aggregate_page_regions(new_pages)
    new_tokens: set[str] = set()
    new_page_roles = [
        _page_role_value(page.role) for page in new_pages if _page_role_value(page.role)
    ]
    for page in new_pages:
        new_tokens.update(_stable_anchor_tokens_from_page(page))

    def _representative_score(record: FingerprintRecord) -> float:
        record_roles = [
            _page_role_value(page.role)
            for page in record.page_fingerprints
            if _page_role_value(page.role)
        ]
        strong_role_delta, variable_role_delta = _page_role_deltas(record_roles, new_page_roles)
        page_alignment = _weighted_page_alignment(new_pages, record.page_fingerprints)
        score = record.confidence * 0.32
        score += min(record.apply_count, 10) * 0.03
        score += page_alignment * 0.55
        score -= 0.06 * strong_role_delta
        score -= 0.015 * variable_role_delta
        record_tokens: set[str] = set()
        for record_page in record.page_fingerprints:
            record_tokens.update(_stable_anchor_tokens_from_page(record_page))
        if record_tokens and new_tokens:
            score += _jaccard_similarity(record_tokens, new_tokens) * 0.1
        return score

    family_candidates: List[Tuple[float, TemplateFamilyRecord, Optional[FingerprintRecord]]] = []
    for family in active_families:
        family_regions = _family_anchor_regions(family)
        aggregate_score = _weighted_region_similarity(new_regions, family_regions)
        family_tokens = set(
            str(value)
            for values in (family.anchor_summary.get("aggregate_keywords") or {}).values()
            for value in values
        )
        if not family_tokens:
            family_tokens = set(
                str(value) for value in family.stable_anchor_regions.get("tokens", [])
            )

        expected_roles = [
            _page_role_value(role)
            for role in family.page_role_expectations
            if _page_role_value(role)
        ]
        if not expected_roles:
            expected_roles = [
                str(role) for role in family.anchor_summary.get("page_roles", []) if role
            ]
        strong_role_delta, variable_role_delta = _page_role_deltas(expected_roles, new_page_roles)
        if strong_role_delta > max(config.discovery.family_page_count_tolerance, 1):
            continue

        role_overlap = 0.0
        if expected_roles and new_page_roles:
            role_overlap = len(set(expected_roles) & set(new_page_roles)) / max(
                len(set(expected_roles) | set(new_page_roles)), 1
            )

        provider_name = (
            document_context.provider_match.provider_name
            if document_context and document_context.provider_match
            else None
        )
        provider_bonus = (
            0.08
            if provider_name
            and family.provider_name
            and provider_name.lower() == family.provider_name.lower()
            else 0.0
        )

        representative = None
        family_records = sorted(
            pages_by_family.get(family.template_family_id, []),
            key=lambda record: (
                -_representative_score(record),
                -record.confidence,
                -record.apply_count,
                record.created_at,
            ),
        )
        if family_records:
            representative = family_records[0]
        page_alignment = (
            _weighted_page_alignment(new_pages, representative.page_fingerprints)
            if representative
            else 0.0
        )
        family_health_multiplier = _family_reuse_multiplier(family)
        split_pressure = _family_split_pressure(family, family_records, representative)

        page_count = family.anchor_summary.get("page_count")
        page_delta_penalty = 0.0
        if page_count is not None:
            page_delta = abs(int(page_count) - len(new_pages))
            page_delta_penalty = min(page_delta * 0.015, 0.06)

        score = (aggregate_score * 0.52) + (page_alignment * 0.24) + (role_overlap * 0.16)
        if family_tokens and new_tokens:
            score += (
                _jaccard_similarity({str(token).lower() for token in family_tokens}, new_tokens)
                * 0.05
            )
        score += provider_bonus
        score -= (strong_role_delta * 0.08) + (variable_role_delta * 0.02) + page_delta_penalty
        score = (score * family_health_multiplier) - (split_pressure * 0.12)
        family_candidates.append((score, family, representative))

    family_candidates = [
        candidate
        for candidate in family_candidates
        if candidate[0] >= config.discovery.family_anchor_threshold
    ]
    if not family_candidates:
        return None, None, 0.0, 0

    family_candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_family, representative = family_candidates[0]
    return best_family, representative, round(best_score, 3), len(family_candidates)


def route_invoice(
    document_context: DocumentContext,
    fingerprint_match: Optional[FingerprintRecord],
    gt_present: bool,
    *,
    gt_discovery_ready: Optional[bool] = None,
    family_representative: Optional[FingerprintRecord] = None,
    family_match_score: float = 0.0,
    family_apply_threshold: float = 1.0,
) -> Route:
    """
    Determine routing based on document context, fingerprint match, and GT readiness.
    """
    gt_ready = gt_present if gt_discovery_ready is None else gt_discovery_ready

    if (
        fingerprint_match is not None
        and fingerprint_match.status == TemplateStatus.provisional
        and gt_ready
    ):
        return Route.DISCOVERY

    if (
        fingerprint_match is None
        and family_representative is not None
        and family_match_score >= family_apply_threshold
    ):
        if family_representative.status == TemplateStatus.provisional and gt_ready:
            return Route.DISCOVERY
        if family_representative.status == TemplateStatus.degraded:
            return Route.DISCOVERY if gt_ready else Route.APPLY
        return Route.APPLY

    if (
        fingerprint_match is not None
        and document_context.extraction_strategy == ExtractionStrategy.provider_template
    ):
        if fingerprint_match.status == TemplateStatus.degraded:
            return Route.DISCOVERY if gt_ready else Route.APPLY
        return Route.APPLY

    if fingerprint_match is not None:
        if fingerprint_match.status == TemplateStatus.degraded:
            return Route.DISCOVERY if gt_ready else Route.APPLY
        return Route.APPLY

    if document_context.extraction_strategy == ExtractionStrategy.native_pdf and not gt_ready:
        return Route.APPLY

    if gt_ready:
        return Route.DISCOVERY
    return Route.FAIL


def determine_extraction_strategy(
    document_context: DocumentContext,
    fingerprint_match: Optional[FingerprintRecord],
    *,
    family_representative: Optional[FingerprintRecord] = None,
    family_match_score: float = 0.0,
    family_apply_threshold: float = 1.0,
) -> ExtractionStrategy:
    if fingerprint_match is not None:
        return ExtractionStrategy.provider_template
    if family_representative is not None and family_match_score >= family_apply_threshold:
        return ExtractionStrategy.provider_template
    return document_context.extraction_strategy
