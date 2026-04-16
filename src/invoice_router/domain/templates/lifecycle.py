import json
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import redis

from ...config import TemplateLifecycleConfig
from ...infrastructure.persistence.storage import FingerprintDB
from ...models import (
    DocumentFamily,
    FingerprintRecord,
    PageRole,
    ProcessingResult,
    Route,
    TemplateFamilyRecord,
    TemplateFamilyVersion,
    TemplateStatus,
)
from ..invoices.family_profiles import merge_family_extraction_profiles

logger = logging.getLogger(__name__)


def _record_family_version(db: FingerprintDB, template_family_id: str, reason: str):
    family = db.get_template_family(template_family_id)
    if family is None:
        return
    db.add_template_family_version(
        TemplateFamilyVersion(
            template_family_id=template_family_id,
            version=len(db.get_template_family_versions(template_family_id)) + 1,
            family_snapshot=family.model_dump(),
            change_reason=reason,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    )


def record_template_family_version(db: FingerprintDB, template_family_id: str, reason: str) -> None:
    _record_family_version(db, template_family_id, reason)


def _representative_fingerprint(records: List[FingerprintRecord]) -> Optional[FingerprintRecord]:
    if not records:
        return None
    return sorted(
        records,
        key=lambda record: (-record.confidence, -record.apply_count, record.created_at),
    )[0]


def _fingerprint_field_names(record: FingerprintRecord) -> set[str]:
    field_names: set[str] = set()
    for page in record.layout_template.get("pages", []):
        field_names.update(
            str(name).strip().lower() for name in (page.get("fields") or {}).keys() if name
        )
    return {name for name in field_names if name}


def _fingerprint_anchor_tokens(record: FingerprintRecord) -> set[str]:
    tokens: set[str] = set()
    for page in record.page_fingerprints:
        signature = page.stable_anchor_signature or {}
        for key in ("header_tokens", "summary_labels", "footer_tokens"):
            tokens.update(
                str(value).strip().lower() for value in (signature.get(key) or []) if value
            )
        for values in (signature.get("keyword_hits") or {}).values():
            tokens.update(str(value).strip().lower() for value in values if value)
    return {token for token in tokens if token}


def _fingerprint_signature(record: FingerprintRecord) -> Dict[str, tuple[str, ...]]:
    page_roles = tuple(
        page.role.value for page in record.page_fingerprints if page.role is not None
    )
    return {
        "page_roles": page_roles,
        "field_names": tuple(sorted(_fingerprint_field_names(record))),
        "anchor_tokens": tuple(sorted(_fingerprint_anchor_tokens(record))),
    }


def _signature_similarity(
    left: Dict[str, tuple[str, ...]], right: Dict[str, tuple[str, ...]]
) -> float:
    role_overlap = len(set(left["page_roles"]) & set(right["page_roles"])) / max(
        len(set(left["page_roles"]) | set(right["page_roles"])), 1
    )
    field_overlap = len(set(left["field_names"]) & set(right["field_names"])) / max(
        len(set(left["field_names"]) | set(right["field_names"])), 1
    )
    anchor_overlap = len(set(left["anchor_tokens"]) & set(right["anchor_tokens"])) / max(
        len(set(left["anchor_tokens"]) | set(right["anchor_tokens"])), 1
    )
    return round((role_overlap * 0.35) + (field_overlap * 0.35) + (anchor_overlap * 0.30), 4)


def _family_split_signals(records: List[FingerprintRecord]) -> Dict[str, Any]:
    signatures = [_fingerprint_signature(record) for record in records]
    signature_counts = Counter(json.dumps(signature, sort_keys=True) for signature in signatures)
    page_role_counts = Counter(role for signature in signatures for role in signature["page_roles"])
    field_name_counts = Counter(
        name for signature in signatures for name in signature["field_names"]
    )
    anchor_token_counts = Counter(
        token for signature in signatures for token in signature["anchor_tokens"]
    )

    dominant_signature_ratio = 1.0
    dominant_signature_count = 0
    average_signature_similarity = 1.0
    if signature_counts:
        dominant_signature_payload, dominant_signature_count = signature_counts.most_common(1)[0]
        dominant_signature_ratio = dominant_signature_count / max(len(records), 1)
        dominant_signature = json.loads(dominant_signature_payload)
        dominant_signature = {key: tuple(values) for key, values in dominant_signature.items()}
        if signatures:
            similarities = [
                _signature_similarity(dominant_signature, signature) for signature in signatures
            ]
            if similarities:
                average_signature_similarity = sum(similarities) / len(similarities)

    return {
        "member_count": len(records),
        "unique_signature_count": len(signature_counts),
        "dominant_signature_count": dominant_signature_count,
        "dominant_signature_ratio": round(dominant_signature_ratio, 4),
        "average_signature_similarity": round(average_signature_similarity, 4),
        "split_pressure": round(max(0.0, 1.0 - average_signature_similarity), 4),
        "page_role_variants": [
            {"value": role, "count": count} for role, count in page_role_counts.most_common(6)
        ],
        "field_name_variants": [
            {"value": name, "count": count} for name, count in field_name_counts.most_common(8)
        ],
        "anchor_token_variants": [
            {"value": token, "count": count} for token, count in anchor_token_counts.most_common(8)
        ],
    }


def _rebuild_family_anchor_summary(records: List[FingerprintRecord]) -> Dict[str, Any]:
    representative = _representative_fingerprint(records)
    aggregate_keywords: Dict[str, set[str]] = {}
    for record in records:
        for page in record.page_fingerprints:
            signature = page.stable_anchor_signature or {}
            for key, values in (signature.get("keyword_hits") or {}).items():
                aggregate_keywords.setdefault(str(key), set()).update(
                    str(value) for value in values if value
                )

    page_roles: List[str] = []
    if representative is not None:
        for page in representative.page_fingerprints:
            if page.role is not None:
                page_roles.append(page.role.value)

    split_signals = _family_split_signals(records) if records else {}
    return {
        "page_count": len(representative.page_fingerprints) if representative else 0,
        "page_roles": page_roles,
        "pages": [page.model_dump(mode="json") for page in representative.page_fingerprints]
        if representative
        else [],
        "aggregate_keywords": {
            key: sorted(values)[:20] for key, values in aggregate_keywords.items() if values
        },
        "split_signals": split_signals,
    }


def _derive_family_rollup(
    records: List[FingerprintRecord],
    results: List[ProcessingResult],
    config: TemplateLifecycleConfig,
) -> Dict[str, Any]:
    accepted_results = 0
    rejected_results = 0
    validation_scores: List[float] = []
    for result in results:
        if result.validation_passed is True:
            accepted_results += 1
        elif result.validation_passed is False or result.route_used == Route.REJECTED:
            rejected_results += 1
        if result.diagnostics and result.diagnostics.validation_score is not None:
            validation_scores.append(float(result.diagnostics.validation_score))

    fingerprint_apply_count = sum(max(record.apply_count, 0) for record in records)
    fingerprint_reject_count = sum(max(record.reject_count, 0) for record in records)
    apply_count = max(accepted_results, fingerprint_apply_count)
    reject_count = max(rejected_results, fingerprint_reject_count)

    confidence = 0.0
    if validation_scores:
        confidence = sum(validation_scores) / len(validation_scores)
    elif records:
        total_weight = sum(max(record.apply_count + record.reject_count, 1) for record in records)
        confidence = sum(
            float(record.confidence) * max(record.apply_count + record.reject_count, 1)
            for record in records
        ) / max(total_weight, 1)

    if reject_count > apply_count and reject_count > 0:
        status = TemplateStatus.degraded
    elif (
        apply_count >= config.establish_min_count and confidence >= config.establish_min_confidence
    ):
        status = TemplateStatus.established
    elif reject_count >= config.rediscovery_attempts and confidence < config.degradation_threshold:
        status = TemplateStatus.degraded
    else:
        status = TemplateStatus.provisional

    return {
        "confidence": round(confidence, 4),
        "apply_count": int(apply_count),
        "reject_count": int(reject_count),
        "status": status,
    }


def is_gt_trusted(apply_count: int, confidence: float, config: TemplateLifecycleConfig) -> bool:
    return (
        apply_count >= config.establish_min_count and confidence >= config.establish_min_confidence
    )


def fingerprint_has_gt_trust(
    record: Optional[FingerprintRecord], config: TemplateLifecycleConfig
) -> bool:
    if record is None:
        return False
    return is_gt_trusted(record.gt_apply_count, float(record.gt_confidence), config)


def family_has_gt_trust(
    record: Optional[TemplateFamilyRecord], config: TemplateLifecycleConfig
) -> bool:
    if record is None:
        return False
    return is_gt_trusted(record.gt_apply_count, float(record.gt_confidence), config)


def resolve_gt_healing_authority(
    family_record: Optional[TemplateFamilyRecord],
    fingerprint_record: Optional[FingerprintRecord],
    config: TemplateLifecycleConfig,
) -> Dict[str, Any]:
    if family_record is not None:
        return {
            "scope": "family",
            "trusted": family_has_gt_trust(family_record, config),
            "template_family_id": family_record.template_family_id,
            "fingerprint_hash": fingerprint_record.hash if fingerprint_record else None,
            "gt_apply_count": int(family_record.gt_apply_count),
            "gt_reject_count": int(family_record.gt_reject_count),
            "gt_confidence": round(float(family_record.gt_confidence), 4),
        }
    if fingerprint_record is not None:
        return {
            "scope": "fingerprint",
            "trusted": fingerprint_has_gt_trust(fingerprint_record, config),
            "template_family_id": fingerprint_record.template_family_id,
            "fingerprint_hash": fingerprint_record.hash,
            "gt_apply_count": int(fingerprint_record.gt_apply_count),
            "gt_reject_count": int(fingerprint_record.gt_reject_count),
            "gt_confidence": round(float(fingerprint_record.gt_confidence), 4),
        }
    return {
        "scope": None,
        "trusted": False,
        "template_family_id": None,
        "fingerprint_hash": None,
        "gt_apply_count": 0,
        "gt_reject_count": 0,
        "gt_confidence": 0.0,
    }


def _rebuilt_family_record(
    source_family: TemplateFamilyRecord,
    member_records: List[FingerprintRecord],
    results: List[ProcessingResult],
    *,
    template_family_id: str,
    created_at: str,
    updated_at: str,
    config: TemplateLifecycleConfig,
) -> TemplateFamilyRecord:
    anchor_summary = _rebuild_family_anchor_summary(member_records)
    representative = _representative_fingerprint(member_records)
    page_roles: List[PageRole] = []
    if representative is not None:
        page_roles = [
            page.role for page in representative.page_fingerprints if page.role is not None
        ]
    stable_tokens = sorted(
        {
            token
            for values in (anchor_summary.get("aggregate_keywords") or {}).values()
            for token in values
            if token
        }
    )[:30]
    rollup = _derive_family_rollup(member_records, results, config)
    return source_family.model_copy(
        update={
            "template_family_id": template_family_id,
            "stable_anchor_regions": {"tokens": stable_tokens},
            "anchor_summary": anchor_summary,
            "page_role_expectations": page_roles,
            "confidence": rollup["confidence"],
            "apply_count": rollup["apply_count"],
            "reject_count": rollup["reject_count"],
            "status": rollup["status"],
            "created_at": created_at,
            "updated_at": updated_at,
        }
    )


def _next_split_family_id(db: FingerprintDB, source_family_id: str) -> str:
    suffix = 1
    while True:
        candidate = f"{source_family_id}-split-{suffix}"
        if db.get_template_family(candidate) is None:
            return candidate
        suffix += 1


def _dedupe_jsonable_list(items: List[Any]) -> List[Any]:
    seen: set[str] = set()
    unique: List[Any] = []
    for item in items:
        marker = json.dumps(item, sort_keys=True)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(item)
    return unique


def _merged_family_base(
    target_family: TemplateFamilyRecord, source_family: TemplateFamilyRecord
) -> TemplateFamilyRecord:
    document_family = target_family.document_family
    if (
        document_family == DocumentFamily.unknown
        and source_family.document_family != DocumentFamily.unknown
    ):
        document_family = source_family.document_family

    return target_family.model_copy(
        update={
            "provider_name": target_family.provider_name or source_family.provider_name,
            "country_code": target_family.country_code or source_family.country_code,
            "document_family": document_family,
            "summary_area_anchors": {
                **(source_family.summary_area_anchors or {}),
                **(target_family.summary_area_anchors or {}),
            },
            "variable_region_masks": _dedupe_jsonable_list(
                list(target_family.variable_region_masks or [])
                + list(source_family.variable_region_masks or [])
            ),
            "extraction_profile": merge_family_extraction_profiles(
                source_family.extraction_profile,
                target_family.extraction_profile,
            ),
        }
    )


def update_template_confidence(
    db: FingerprintDB,
    redis_client: redis.Redis,
    hash_val: str,
    quality_score: float,
    config: TemplateLifecycleConfig,
    *,
    gt_backed: bool = False,
):
    """
    Update confidence and counts after APPLY result; transition status if criteria met.
    Uses Redis for atomic increments.
    """
    redis_key = f"fingerprint_stats:{hash_val}"

    # 1. Update in Redis atomically
    # We store apply_count and reject_count
    # Wait, confidence needs to be updated. Since it's a moving average, we can't easily do it purely in Redis without lua.
    # For now, we will fetch, calculate, then store in DB. The counts can be incremented atomically.

    # Actually, we can increment apply_count in Redis
    apply_count = redis_client.hincrby(redis_key, "apply_count", 1)

    # Fetch current record
    records = db.get_all_active_fingerprints()
    record = next((r for r in records if r.hash == hash_val), None)

    if not record:
        logger.warning(f"Fingerprint {hash_val} not found during confidence update.")
        return

    old_apply_count = record.apply_count
    # Confidence formula: confidence = (confidence * apply_count + quality_score) / (apply_count + 1)
    # Using old_apply_count ensures we weight it correctly.
    new_confidence = (record.confidence * old_apply_count + quality_score) / (old_apply_count + 1)

    new_status = record.status
    gt_apply_count = record.gt_apply_count
    gt_reject_count = record.gt_reject_count
    gt_confidence = record.gt_confidence

    if gt_backed:
        gt_old_apply_count = record.gt_apply_count
        gt_apply_count = gt_old_apply_count + 1
        gt_confidence = (record.gt_confidence * gt_old_apply_count + quality_score) / max(
            gt_apply_count, 1
        )

    if new_status == TemplateStatus.provisional:
        if (
            apply_count >= config.establish_min_count
            and new_confidence >= config.establish_min_confidence
        ):
            new_status = TemplateStatus.established
        elif record.reject_count > apply_count:
            new_status = TemplateStatus.degraded

    db.update_fingerprint_confidence(
        hash_val=hash_val,
        confidence=new_confidence,
        apply_count=apply_count,
        reject_count=record.reject_count,
        gt_apply_count=gt_apply_count if gt_backed else None,
        gt_reject_count=gt_reject_count if gt_backed else None,
        gt_confidence=gt_confidence if gt_backed else None,
        status=new_status,
    )


def record_rejection(
    db: FingerprintDB,
    redis_client: redis.Redis,
    hash_val: str,
    config: TemplateLifecycleConfig,
    *,
    gt_backed: bool = False,
):
    redis_key = f"fingerprint_stats:{hash_val}"
    reject_count = redis_client.hincrby(redis_key, "reject_count", 1)

    records = db.get_all_active_fingerprints()
    record = next((r for r in records if r.hash == hash_val), None)

    if not record:
        return

    new_status = record.status
    gt_reject_count = record.gt_reject_count
    gt_apply_count = record.gt_apply_count
    gt_confidence = record.gt_confidence
    if new_status == TemplateStatus.provisional:
        if reject_count > record.apply_count:
            new_status = TemplateStatus.degraded

    if gt_backed:
        gt_reject_count = record.gt_reject_count + 1

    db.update_fingerprint_confidence(
        hash_val=hash_val,
        confidence=record.confidence,
        apply_count=record.apply_count,
        reject_count=reject_count,
        gt_apply_count=gt_apply_count if gt_backed else None,
        gt_reject_count=gt_reject_count if gt_backed else None,
        gt_confidence=gt_confidence if gt_backed else None,
        status=new_status,
    )


def update_template_family_confidence(
    db: FingerprintDB,
    template_family_id: str,
    quality_score: float,
    config: TemplateLifecycleConfig,
    *,
    reason: str = "family_apply_success",
    gt_backed: bool = False,
):
    family = db.get_template_family(template_family_id)
    if family is None:
        logger.warning(f"Template family {template_family_id} not found during confidence update.")
        return

    old_apply_count = family.apply_count
    apply_count = old_apply_count + 1
    new_confidence = (family.confidence * old_apply_count + quality_score) / max(apply_count, 1)
    new_status = family.status
    gt_apply_count = family.gt_apply_count
    gt_reject_count = family.gt_reject_count
    gt_confidence = family.gt_confidence

    if gt_backed:
        gt_old_apply_count = family.gt_apply_count
        gt_apply_count = gt_old_apply_count + 1
        gt_confidence = (family.gt_confidence * gt_old_apply_count + quality_score) / max(
            gt_apply_count, 1
        )

    if new_status in {TemplateStatus.provisional, TemplateStatus.degraded}:
        if (
            apply_count >= config.establish_min_count
            and new_confidence >= config.establish_min_confidence
        ):
            new_status = TemplateStatus.established
    elif (
        new_status == TemplateStatus.established
        and new_confidence < config.degradation_threshold
        and family.reject_count >= config.rediscovery_attempts
    ):
        new_status = TemplateStatus.degraded

    db.update_template_family_lifecycle(
        template_family_id,
        confidence=new_confidence,
        apply_count=apply_count,
        reject_count=family.reject_count,
        gt_apply_count=gt_apply_count if gt_backed else None,
        gt_reject_count=gt_reject_count if gt_backed else None,
        gt_confidence=gt_confidence if gt_backed else None,
        status=new_status,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    _record_family_version(db, template_family_id, reason)


def record_template_family_rejection(
    db: FingerprintDB,
    template_family_id: str,
    config: TemplateLifecycleConfig,
    *,
    reason: str = "family_apply_rejection",
    gt_backed: bool = False,
):
    family = db.get_template_family(template_family_id)
    if family is None:
        return

    reject_count = family.reject_count + 1
    new_status = family.status
    gt_apply_count = family.gt_apply_count
    gt_reject_count = family.gt_reject_count
    gt_confidence = family.gt_confidence
    if new_status == TemplateStatus.provisional and reject_count > family.apply_count:
        new_status = TemplateStatus.degraded
    elif (
        new_status == TemplateStatus.established
        and reject_count >= config.rediscovery_attempts
        and family.confidence < config.degradation_threshold
    ):
        new_status = TemplateStatus.degraded

    if gt_backed:
        gt_reject_count = family.gt_reject_count + 1

    db.update_template_family_lifecycle(
        template_family_id,
        confidence=family.confidence,
        apply_count=family.apply_count,
        reject_count=reject_count,
        gt_apply_count=gt_apply_count if gt_backed else None,
        gt_reject_count=gt_reject_count if gt_backed else None,
        gt_confidence=gt_confidence if gt_backed else None,
        status=new_status,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    _record_family_version(db, template_family_id, reason)


def manually_update_template_family(
    db: FingerprintDB,
    template_family_id: str,
    *,
    status: Optional[TemplateStatus] = None,
    provider_name: Optional[str] = None,
    country_code: Optional[str] = None,
    document_family: Optional[DocumentFamily] = None,
    extraction_profile_updates: Optional[Dict[str, Any]] = None,
    replace_extraction_profile: bool = False,
    reason: str = "manual_update",
) -> Optional[TemplateFamilyRecord]:
    family = db.get_template_family(template_family_id)
    if family is None:
        return None

    extraction_profile = family.extraction_profile
    if extraction_profile_updates is not None:
        if replace_extraction_profile:
            extraction_profile = extraction_profile_updates
        else:
            extraction_profile = merge_family_extraction_profiles(
                family.extraction_profile,
                extraction_profile_updates,
            )

    updated = family.model_copy(
        update={
            "status": status or family.status,
            "provider_name": provider_name if provider_name is not None else family.provider_name,
            "country_code": country_code if country_code is not None else family.country_code,
            "document_family": document_family or family.document_family,
            "extraction_profile": extraction_profile,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    db.store_template_family(updated)
    _record_family_version(db, template_family_id, reason)
    return db.get_template_family(template_family_id)


def split_template_family(
    db: FingerprintDB,
    template_family_id: str,
    *,
    fingerprint_hashes: List[str],
    new_template_family_id: Optional[str] = None,
    reason: str = "manual_split",
    config: Optional[TemplateLifecycleConfig] = None,
) -> Optional[Dict[str, Any]]:
    source_family = db.get_template_family(template_family_id)
    if source_family is None:
        return None

    fingerprints = db.get_fingerprints_for_family(template_family_id)
    fingerprint_map = {record.hash: record for record in fingerprints}
    moved_hashes = [hash_val for hash_val in fingerprint_hashes if hash_val in fingerprint_map]
    if not moved_hashes:
        raise ValueError(f"No matching fingerprints found in template family {template_family_id}")
    if len(moved_hashes) == len(fingerprints):
        raise ValueError("Cannot split a family by moving every fingerprint out of it")

    moved_records = [fingerprint_map[hash_val] for hash_val in moved_hashes]
    remaining_records = [record for record in fingerprints if record.hash not in set(moved_hashes)]
    if not remaining_records:
        raise ValueError("Cannot leave the source family empty after a split")

    lifecycle_config = config or TemplateLifecycleConfig()
    new_family_id = new_template_family_id or _next_split_family_id(db, template_family_id)
    if db.get_template_family(new_family_id) is not None:
        raise ValueError(f"Template family already exists: {new_family_id}")

    moved_results = db.get_results_for_fingerprints(moved_hashes)
    remaining_results = db.get_results_for_fingerprints(
        [record.hash for record in remaining_records]
    )
    moved_invoice_paths = sorted(
        {result.invoice_path for result in moved_results if result.invoice_path}
    )
    now = datetime.now(timezone.utc).isoformat()

    _record_family_version(db, template_family_id, f"{reason}:before_split_to:{new_family_id}")

    new_family = _rebuilt_family_record(
        source_family,
        moved_records,
        moved_results,
        template_family_id=new_family_id,
        created_at=now,
        updated_at=now,
        config=lifecycle_config,
    )
    db.store_template_family(new_family)
    _record_family_version(db, new_family_id, f"{reason}:split_from:{template_family_id}")

    moved_fingerprint_count = db.relink_fingerprints_to_family(moved_hashes, new_family_id)
    moved_example_count = db.relink_template_family_examples(
        template_family_id,
        new_family_id,
        fingerprint_hashes=moved_hashes,
        invoice_paths=moved_invoice_paths,
    )
    moved_result_count = db.relink_processing_results_to_family(
        template_family_id,
        new_family_id,
        fingerprint_hashes=moved_hashes,
    )

    updated_source_family = _rebuilt_family_record(
        source_family,
        remaining_records,
        remaining_results,
        template_family_id=template_family_id,
        created_at=source_family.created_at,
        updated_at=now,
        config=lifecycle_config,
    )
    db.store_template_family(updated_source_family)
    _record_family_version(db, template_family_id, f"{reason}:split_out:{new_family_id}")

    return {
        "source_family": db.get_template_family(template_family_id),
        "new_family": db.get_template_family(new_family_id),
        "source_family_id": template_family_id,
        "new_family_id": new_family_id,
        "moved_fingerprint_hashes": moved_hashes,
        "moved_fingerprint_count": moved_fingerprint_count,
        "moved_example_count": moved_example_count,
        "moved_result_count": moved_result_count,
    }


def merge_template_families(
    db: FingerprintDB,
    target_family_id: str,
    source_family_id: str,
    *,
    reason: str = "manual_merge",
    config: Optional[TemplateLifecycleConfig] = None,
) -> Optional[Dict[str, Any]]:
    if target_family_id == source_family_id:
        raise ValueError("Target and source family must be different")

    target_family = db.get_template_family(target_family_id)
    source_family = db.get_template_family(source_family_id)
    if target_family is None or source_family is None:
        return None
    if target_family.status == TemplateStatus.retired:
        raise ValueError(f"Cannot merge into retired template family: {target_family_id}")
    if source_family.status == TemplateStatus.retired:
        raise ValueError(f"Template family is already retired: {source_family_id}")

    lifecycle_config = config or TemplateLifecycleConfig()
    target_records = db.get_fingerprints_for_family(target_family_id)
    source_records = db.get_fingerprints_for_family(source_family_id)
    if not source_records:
        raise ValueError(f"No active fingerprints found for template family {source_family_id}")

    source_hashes = [record.hash for record in source_records]
    source_results = db.get_results_for_fingerprints(source_hashes)
    source_examples = db.get_template_family_examples(source_family_id)
    source_invoice_paths = sorted(
        {example.invoice_path for example in source_examples if example.invoice_path}
    )
    source_invoice_paths.extend(
        sorted(
            {
                result.invoice_path
                for result in source_results
                if result.invoice_path and result.invoice_path not in source_invoice_paths
            }
        )
    )
    combined_records = target_records + source_records
    combined_results = db.get_results_for_fingerprints([record.hash for record in combined_records])
    now = datetime.now(timezone.utc).isoformat()

    _record_family_version(db, target_family_id, f"{reason}:before_merge_from:{source_family_id}")
    _record_family_version(db, source_family_id, f"{reason}:before_merge_into:{target_family_id}")

    moved_fingerprint_count = db.relink_fingerprints_to_family(source_hashes, target_family_id)
    moved_example_count = db.relink_template_family_examples(
        source_family_id,
        target_family_id,
        fingerprint_hashes=source_hashes,
        invoice_paths=source_invoice_paths,
    )
    moved_result_count = db.relink_processing_results_to_family(
        source_family_id,
        target_family_id,
        fingerprint_hashes=source_hashes,
        invoice_paths=source_invoice_paths,
    )

    merged_target_family = _rebuilt_family_record(
        _merged_family_base(target_family, source_family),
        combined_records,
        combined_results,
        template_family_id=target_family_id,
        created_at=target_family.created_at,
        updated_at=now,
        config=lifecycle_config,
    )
    db.store_template_family(merged_target_family)
    _record_family_version(db, target_family_id, f"{reason}:merge_from:{source_family_id}")

    retired_source_family = source_family.model_copy(
        update={
            "status": TemplateStatus.retired,
            "updated_at": now,
        }
    )
    db.store_template_family(retired_source_family)
    _record_family_version(db, source_family_id, f"{reason}:merged_into:{target_family_id}")

    return {
        "target_family": db.get_template_family(target_family_id),
        "source_family": db.get_template_family(source_family_id),
        "target_family_id": target_family_id,
        "source_family_id": source_family_id,
        "moved_fingerprint_count": moved_fingerprint_count,
        "moved_example_count": moved_example_count,
        "moved_result_count": moved_result_count,
    }


def retire_template_family(
    db: FingerprintDB,
    template_family_id: str,
    *,
    reason: str = "manual_retire",
    retire_fingerprints: bool = True,
) -> Optional[Dict[str, Any]]:
    family = db.get_template_family(template_family_id)
    if family is None:
        return None
    if family.status == TemplateStatus.retired:
        return {
            "template_family_id": template_family_id,
            "retired_fingerprint_count": 0,
            "family": family,
        }

    active_fingerprints = db.get_fingerprints_for_family(template_family_id)
    active_hashes = [record.hash for record in active_fingerprints]
    now = datetime.now(timezone.utc).isoformat()

    _record_family_version(db, template_family_id, f"{reason}:before_retire")
    retired_fingerprint_count = 0
    if retire_fingerprints:
        retired_fingerprint_count = db.retire_fingerprints(active_hashes)

    retired_family = family.model_copy(
        update={
            "status": TemplateStatus.retired,
            "updated_at": now,
        }
    )
    db.store_template_family(retired_family)
    _record_family_version(db, template_family_id, reason)

    return {
        "template_family_id": template_family_id,
        "retired_fingerprint_count": retired_fingerprint_count,
        "family": db.get_template_family(template_family_id),
    }


def check_degradation(db: FingerprintDB, hash_val: str, config: TemplateLifecycleConfig):
    """
    Compute rolling confidence; transition to degraded if below threshold.
    """
    records = db.get_all_active_fingerprints()
    record = next((r for r in records if r.hash == hash_val), None)

    if not record or record.status != TemplateStatus.established:
        return

    scores = [
        1.0 if passed else 0.0
        for passed in db.get_recent_validation_passes(hash_val, config.degradation_window)
    ]

    if len(scores) == config.degradation_window:
        rolling_confidence = sum(scores) / len(scores)
        if rolling_confidence < config.degradation_threshold:
            db.update_fingerprint_confidence(
                hash_val=hash_val,
                confidence=record.confidence,
                apply_count=record.apply_count,
                reject_count=record.reject_count,
                gt_apply_count=None,
                gt_reject_count=None,
                gt_confidence=None,
                status=TemplateStatus.degraded,
            )


def archive_template(db: FingerprintDB, hash_val: str, replaced_by: str, reason: str):
    """
    Move current template to fingerprint_history before replacement.
    """
    now = datetime.now(timezone.utc).isoformat()
    db.archive_fingerprint(
        hash_val=hash_val,
        valid_from="",  # Ideally fetch created_at from fingerprints
        valid_to=now,
        replaced_by=replaced_by,
        retirement_reason=reason,
    )
