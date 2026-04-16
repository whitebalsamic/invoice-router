from typing import Optional

import redis

from ...config import Settings

_pool: Optional[redis.ConnectionPool] = None


def get_redis_pool(settings: Settings) -> redis.ConnectionPool:
    """Initialize and return a Redis connection pool based on configuration."""
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_max_connections,
            decode_responses=True,
        )
    return _pool


def get_redis_client(settings: Settings) -> redis.Redis:
    """Get a Redis client instance from the pool."""
    pool = get_redis_pool(settings)
    return redis.Redis(connection_pool=pool)
