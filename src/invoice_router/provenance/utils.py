"""Hashing and timestamp helpers used across the pipeline."""

import hashlib
import uuid
from datetime import datetime, timezone


def compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_string_hash(data: str) -> str:
    return compute_hash(data.encode("utf-8"))


def generate_request_id() -> str:
    return str(uuid.uuid4())


def get_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
