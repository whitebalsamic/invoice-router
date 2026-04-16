import logging
from datetime import datetime, timezone

from celery import Celery

from ..config import load_config
from ..infrastructure.filesystem.source import list_invoices
from ..infrastructure.persistence.redis import get_redis_client
from ..infrastructure.persistence.storage import FingerprintDB
from ..pipeline import process_single_invoice

logger = logging.getLogger(__name__)

settings, config = load_config(required_settings=("database_url", "redis_url"))

celery_app = Celery(
    "invoice_router_workers",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    worker_concurrency=config.processing.worker_concurrency,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)

# Atomic update lua script for Redis
_UPDATE_PROGRESS_LUA = """
local key = KEYS[1]
local is_success = tonumber(ARGV[1])
local score = tonumber(ARGV[2])

if is_success == 1 then
    local done = redis.call('HINCRBY', key, 'done', 1)
    local acc_str = redis.call('HGET', key, 'accuracy')
    local acc = 0.0
    if acc_str then
        acc = tonumber(acc_str)
    end
    acc = (acc * (done - 1) + score) / done
    redis.call('HSET', key, 'accuracy', tostring(acc))
else
    redis.call('HINCRBY', key, 'failed', 1)
end
return 1
"""


def _update_progress(job_id: str, success: bool, score: float):
    redis_client = get_redis_client(settings)
    key = f"job_progress:{job_id}"
    redis_client.eval(_UPDATE_PROGRESS_LUA, 1, key, 1 if success else 0, score)


@celery_app.task(bind=True)
def process_invoice(self, invoice_path: str, job_id: str, folder_path: str):
    db = FingerprintDB(settings.primary_database_target)
    try:
        res = process_single_invoice(invoice_path, settings, config, db)
        passed = bool(res.validation_passed) if res.validation_passed is not None else True
        score = 1.0 if passed else 0.0
        _update_progress(job_id, passed, score)

        db.upsert_job_progress(
            job_id=job_id,
            current_item=invoice_path,
            status="RUNNING",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        return res.id
    except Exception as e:
        logger.error(f"Error processing {invoice_path}: {e}")
        _update_progress(job_id, False, 0.0)
        return None


@celery_app.task(bind=True)
def process_batch(self, invoice_paths: list, job_id: str, folder_path: str):
    db = FingerprintDB(settings.primary_database_target)
    for inv in invoice_paths:
        try:
            res = process_single_invoice(inv, settings, config, db)
            passed = bool(res.validation_passed) if res.validation_passed is not None else True
            _update_progress(job_id, passed, 1.0 if passed else 0.0)

            db.upsert_job_progress(
                job_id=job_id,
                current_item=inv,
                status="RUNNING",
                started_at=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as e:
            logger.error(f"Batch error processing {inv}: {e}")
            _update_progress(job_id, False, 0.0)


@celery_app.task(bind=True)
def process_folder(self, folder_path: str, job_id: str):
    invoices = list_invoices(folder_path)
    batch_size = config.processing.batch_size

    redis_client = get_redis_client(settings)
    key = f"job_progress:{job_id}"
    redis_client.hset(key, "total", len(invoices))

    for i in range(0, len(invoices), batch_size):
        batch = invoices[i : i + batch_size]
        process_batch.delay(batch, job_id, folder_path)
