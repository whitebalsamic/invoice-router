import invoice_router.reporting.measurements as measurements


def test_read_peak_rss_mb_normalizes_linux_kilobytes(monkeypatch):
    fake_resource = type(
        "FakeResource",
        (),
        {
            "RUSAGE_SELF": object(),
            "getrusage": staticmethod(lambda _scope: type("Usage", (), {"ru_maxrss": 2048})()),
        },
    )()

    monkeypatch.setattr(measurements, "resource", fake_resource)
    monkeypatch.setattr(measurements.sys, "platform", "linux")

    assert measurements.read_peak_rss_mb() == 2.0


def test_read_peak_rss_mb_normalizes_macos_bytes(monkeypatch):
    fake_resource = type(
        "FakeResource",
        (),
        {
            "RUSAGE_SELF": object(),
            "getrusage": staticmethod(
                lambda _scope: type("Usage", (), {"ru_maxrss": 2 * 1024 * 1024})()
            ),
        },
    )()

    monkeypatch.setattr(measurements, "resource", fake_resource)
    monkeypatch.setattr(measurements.sys, "platform", "darwin")

    assert measurements.read_peak_rss_mb() == 2.0


def test_read_peak_rss_mb_returns_none_when_resource_is_unavailable(monkeypatch):
    monkeypatch.setattr(measurements, "resource", None)

    assert measurements.read_peak_rss_mb() is None
