from types import SimpleNamespace

import invoice_router.infrastructure.persistence.redis as redis_client


def test_get_redis_pool_initializes_once_and_reuses_cached_pool(monkeypatch):
    monkeypatch.setattr(redis_client, "_pool", None, raising=False)
    calls = []

    def fake_from_url(url, max_connections, decode_responses):
        calls.append((url, max_connections, decode_responses))
        return object()

    monkeypatch.setattr(redis_client.redis.ConnectionPool, "from_url", fake_from_url)
    settings = SimpleNamespace(redis_url="redis://localhost:6379/0", redis_max_connections=17)

    pool1 = redis_client.get_redis_pool(settings)
    pool2 = redis_client.get_redis_pool(settings)

    assert pool1 is pool2
    assert calls == [("redis://localhost:6379/0", 17, True)]


def test_get_redis_client_wraps_cached_pool(monkeypatch):
    cached_pool = object()
    monkeypatch.setattr(redis_client, "_pool", cached_pool, raising=False)
    captured = {}

    def fake_redis(*, connection_pool):
        captured["pool"] = connection_pool
        return "redis-client"

    monkeypatch.setattr(redis_client.redis, "Redis", fake_redis)
    settings = SimpleNamespace(redis_url="redis://localhost:6379/0", redis_max_connections=5)

    client = redis_client.get_redis_client(settings)

    assert client == "redis-client"
    assert captured["pool"] is cached_pool
