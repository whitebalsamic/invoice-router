from click.testing import CliRunner

from invoice_router.cli.main import cli


def test_demo_command_generates_public_quickstart_artifacts(
    tmp_path, postgres_database_url, monkeypatch
):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("DATABASE_URL", postgres_database_url)
    monkeypatch.setenv("ANALYSIS_DATABASE_URL", postgres_database_url)

    workspace = tmp_path / "demo"
    runner = CliRunner()
    result = runner.invoke(cli, ["demo", "--workspace", str(workspace)])

    assert result.exit_code == 0
    assert "Output JSON:" in result.output
    assert (workspace / "datasets" / "demo-native-pdf" / "demo_invoice.pdf").exists()
    assert (workspace / "output" / "demo-native-pdf" / "demo_invoice.json").exists()
