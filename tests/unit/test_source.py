from invoice_router.infrastructure.filesystem.source import (
    list_invoices,
    resolve_ground_truth_source_dir,
    summarize_ground_truth_sync,
    sync_ground_truth_from_source,
)


def test_resolve_ground_truth_source_dir_prefers_invoices_all_for_subset(tmp_path):
    dataset_dir = tmp_path / "invoices-small"
    source_dir = tmp_path / "invoices-all"
    dataset_dir.mkdir()
    source_dir.mkdir()

    assert resolve_ground_truth_source_dir(str(dataset_dir)) == source_dir


def test_resolve_ground_truth_source_dir_uses_dataset_root_for_named_dataset(monkeypatch, tmp_path):
    dataset_root = tmp_path / "datasets"
    dataset_dir = dataset_root / "invoices-test"
    source_dir = dataset_root / "invoices-all"
    dataset_dir.mkdir(parents=True)
    source_dir.mkdir()
    monkeypatch.setenv("DATASET_ROOT", str(dataset_root))

    assert resolve_ground_truth_source_dir("invoices-test") == source_dir


def test_list_invoices_resolves_named_dataset_from_dataset_root(monkeypatch, tmp_path):
    dataset_root = tmp_path / "datasets"
    dataset_dir = dataset_root / "invoices-small"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "one.png").write_bytes(b"img")
    (dataset_dir / "two.pdf").write_bytes(b"pdf")
    (dataset_dir / "skip.txt").write_text("x")
    monkeypatch.setenv("DATASET_ROOT", str(dataset_root))

    invoices = list_invoices("invoices-small")

    assert invoices == [str(dataset_dir / "one.png"), str(dataset_dir / "two.pdf")]


def test_summarize_ground_truth_sync_detects_mismatch(tmp_path):
    dataset_dir = tmp_path / "invoices-test"
    source_dir = tmp_path / "invoices-all"
    dataset_dir.mkdir()
    source_dir.mkdir()

    (dataset_dir / "a.png").write_bytes(b"img")
    (dataset_dir / "a.json").write_text('{"invoiceNumber":"1"}')
    (source_dir / "a.json").write_text('{"invoiceNumber":"2"}')

    summary = summarize_ground_truth_sync(str(dataset_dir), source_dir=source_dir)

    assert summary["status"] == "out_of_sync"
    assert summary["mismatched_count"] == 1
    assert summary["matching_count"] == 0


def test_sync_ground_truth_from_source_updates_local_json(tmp_path):
    dataset_dir = tmp_path / "invoices-small"
    source_dir = tmp_path / "invoices-all"
    dataset_dir.mkdir()
    source_dir.mkdir()

    (dataset_dir / "a.png").write_bytes(b"img")
    (dataset_dir / "a.json").write_text('{"invoiceNumber":"old"}')
    (source_dir / "a.json").write_text('{"invoiceNumber":"new"}')

    summary = sync_ground_truth_from_source(str(dataset_dir), source_dir=source_dir)

    assert summary["status"] == "in_sync"
    assert summary["updated_count"] == 1
    assert (dataset_dir / "a.json").read_text() == '{"invoiceNumber":"new"}'
