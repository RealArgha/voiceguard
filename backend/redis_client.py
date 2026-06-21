"""
Redis pub/sub + event cache helpers.

Each risk event is:
  - Published to  voiceguard:events:{session_id}   (real-time subscribers)
  - Prepended to  voiceguard:history:{session_id}  (last 50, 24-h TTL)

Set REDIS_URL in .env or environment.
Set REDIS_ENABLED=0 to disable (app still works without Redis).
"""

import json
import os

REDIS_URL     = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "1") == "1"

_redis = None


async def _get() :
    global _redis
    if _redis is None:
        import redis.asyncio as aioredis
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
    return _redis


async def init_redis() -> bool:
    if not REDIS_ENABLED:
        print("[redis] disabled (REDIS_ENABLED=0)")
        return False
    try:
        r = await _get()
        await r.ping()
        print(f"[redis] connected → {REDIS_URL}")
        return True
    except Exception as e:
        print(f"[redis] unavailable ({e}) — running without cache")
        global _redis
        _redis = None
        return False


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


def redis_available() -> bool:
    return REDIS_ENABLED and _redis is not None


async def publish_event(session_id: str, event: dict) -> None:
    if not redis_available():
        return
    try:
        r = await _get()
        payload = json.dumps(event)
        await r.publish(f"voiceguard:events:{session_id}", payload)
        hist_key = f"voiceguard:history:{session_id}"
        await r.lpush(hist_key, payload)
        await r.ltrim(hist_key, 0, 49)       # keep last 50
        await r.expire(hist_key, 86400)       # 24-hour TTL
    except Exception:
        pass                                  # never crash the audio loop


async def get_cached_events(session_id: str, n: int = 50) -> list[dict]:
    if not redis_available():
        return []
    try:
        r = await _get()
        raw = await r.lrange(f"voiceguard:history:{session_id}", 0, n - 1)
        return [json.loads(e) for e in raw]
    except Exception:
        return []
