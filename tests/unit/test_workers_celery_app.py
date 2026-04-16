import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock


def _load_worker_module(monkeypatch, tmp_path):
    monkeypatch.setenv("INVOICE_INPUT_DIR", str(tmp_path / "invoices"))
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql://invoice_router:invoice_router@localhost:5432/invoice_router"
    )
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))

    module = importlib.import_module("invoice_router.workers.celery_app")
    return importlib.reload(module)


def test_update_progress_uses_atomic_redis_eval(monkeypatch, tmp_path):
    module = _load_worker_module(monkeypatch, tmp_path)
    redis_instance = MagicMock()
    monkeypatch.setattr(module, "get_redis_client", lambda _settings: redis_instance)

    module._update_progress("job-123", True, 0.75)

    redis_instance.eval.assert_called_once_with(
        module._UPDATE_PROGRESS_LUA,
        1,
        "job_progress:job-123",
        1,
        0.75,
    )


def test_process_invoice_updates_progress_and_job_status_on_success(monkeypatch, tmp_path):
    module = _load_worker_module(monkeypatch, tmp_path)
    db_instance = MagicMock()
    update_calls = []

    monkeypatch.setattr(
        module,
        "settings",
        SimpleNamespace(
            primary_database_target="postgresql://db", redis_url="redis://localhost:6379/0"
        ),
    )
    monkeypatch.setattr(module, "config", SimpleNamespace())
    monkeypatch.setattr(module, "FingerprintDB", lambda _dsn: db_instance)
    monkeypatch.setattr(
        module,
        "process_single_invoice",
        lambda *_args, **_kwargs: SimpleNamespace(validation_passed=False, id=42),
    )
    monkeypatch.setattr(
        module,
        "_update_progress",
        lambda job_id, success, score: update_calls.append((job_id, success, score)),
    )

    result = module.process_invoice.run("/tmp/invoice.png", "job-1", "/tmp")

    assert result == 42
    assert update_calls == [("job-1", False, 0.0)]
    db_instance.upsert_job_progress.assert_called_once()


def test_process_batch_continues_after_item_failure(monkeypatch, tmp_path):
    module = _load_worker_module(monkeypatch, tmp_path)
    db_instance = MagicMock()
    update_calls = []

    monkeypatch.setattr(
        module,
        "settings",
        SimpleNamespace(
            primary_database_target="postgresql://db", redis_url="redis://localhost:6379/0"
        ),
    )
    monkeypatch.setattr(module, "config", SimpleNamespace())
    monkeypatch.setattr(module, "FingerprintDB", lambda _dsn: db_instance)

    def fake_process(invoice_path, *_args, **_kwargs):
        if invoice_path.endswith("b.png"):
            raise RuntimeError("boom")
        return SimpleNamespace(validation_passed=True, id=1)

    monkeypatch.setattr(module, "process_single_invoice", fake_process)
    monkeypatch.setattr(
        module,
        "_update_progress",
        lambda job_id, success, score: update_calls.append((job_id, success, score)),
    )

    module.process_batch.run(["/tmp/a.png", "/tmp/b.png"], "job-2", "/tmp")

    assert update_calls == [("job-2", True, 1.0), ("job-2", False, 0.0)]
    assert db_instance.upsert_job_progress.call_count == 1


def test_process_folder_sets_total_and_enqueues_batches(monkeypatch, tmp_path):
    module = _load_worker_module(monkeypatch, tmp_path)
    redis_instance = MagicMock()
    queued_batches = []

    monkeypatch.setattr(module, "config", SimpleNamespace(processing=SimpleNamespace(batch_size=2)))
    monkeypatch.setattr(module, "list_invoices", lambda _path: ["a.png", "b.png", "c.png"])
    monkeypatch.setattr(module, "get_redis_client", lambda _settings: redis_instance)
    monkeypatch.setattr(
        module.process_batch,
        "delay",
        lambda batch, job_id, folder_path: queued_batches.append((batch, job_id, folder_path)),
    )

    module.process_folder.run("/tmp/folder", "job-3")

    redis_instance.hset.assert_called_once_with("job_progress:job-3", "total", 3)
    assert queued_batches == [
        (["a.png", "b.png"], "job-3", "/tmp/folder"),
        (["c.png"], "job-3", "/tmp/folder"),
    ]
