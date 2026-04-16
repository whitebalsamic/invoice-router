import json

from invoice_router.reporting.analysis_surface import (
    build_analysis_surface_report,
    write_analysis_surface_report,
)


def test_build_analysis_surface_report_excludes_large_data_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "blob.json").write_text('{"huge": true}\n')
    (tmp_path / "docs" / "assets" / "node_modules").mkdir(parents=True)
    (tmp_path / "docs" / "assets" / "node_modules" / "vendor.js").write_text("x" * 1000)

    report = build_analysis_surface_report(tmp_path)

    assert report["included_files"] == 1
    assert "src" in report["included_by_top_level"]
    assert "data" not in report["included_by_top_level"]
    assert "docs/assets/node_modules" in report["excluded_path_matches"]
    assert "docs/assets/package-lock.json" not in [
        item["path"] for item in report["largest_included_paths"]
    ]


def test_build_analysis_surface_report_excludes_known_non_code_files(tmp_path):
    (tmp_path / "docs" / "assets").mkdir(parents=True)
    (tmp_path / "docs" / "assets" / "package-lock.json").write_text("{}")
    (tmp_path / "dump.rdb").write_text("binary-looking-data")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('ok')\n")

    report = build_analysis_surface_report(tmp_path)

    included_paths = [item["path"] for item in report["largest_included_paths"]]
    assert "docs/assets/package-lock.json" not in included_paths
    assert "dump.rdb" not in included_paths
    assert "docs/assets/package-lock.json" in report["excluded_path_matches"]
    assert "dump.rdb" in report["excluded_path_matches"]


def test_write_analysis_surface_report_includes_delta(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n")
    baseline = {"included_files": 10, "included_lines": 50, "estimated_tokens": 200}

    report = build_analysis_surface_report(tmp_path, baseline_report=baseline)
    output_path = write_analysis_surface_report(report, tmp_path / "report.json")

    payload = json.loads(output_path.read_text())
    assert payload["delta_from_baseline"]["included_files"] == -9
    assert payload["delta_from_baseline"]["included_lines"] < 0
    assert payload["delta_from_baseline"]["estimated_tokens"] < 0
