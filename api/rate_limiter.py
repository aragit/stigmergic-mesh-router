"""Distributed, Redis-backed multi-dimension rate limiting.

Two simultaneous quotas are enforced per tenant:

* **Requests Per Minute (RPM)** — each request costs ``1``.
* **Tokens Per Minute (TPM)** — each request costs its estimated prompt-token
  count plus the requested ``max_tokens`` (input + capped output).

Both dimensions use the *same* atomic Lua sliding-window implementation:
a single ``EVALSHA`` (or ``EVAL``) call prunes expired buckets, computes the
weighted usage for the current window, and — only if the request fits —
records the cost.  The script is idempotent on rejection, so a rejected
request never mutates state.

Redis schema
------------
* **Key** ``ratelimit:<tenant_id>:<dimension>``
* **Type** Redis Hash
* **Fields** integer time-bucket index (``floor(now / window_seconds)``)
* **Value** accumulated cost within that bucket

The sliding-window approximation blends the previous bucket (decaying linearly
with elapsed time) with the current bucket, yielding a true sliding average
rather than a hard fixed-window cliff.
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple

from api.metrics import stigmergic_rate_limit_exceeded_total

logger = logging.getLogger(__name__)

# ── Atomic sliding-window Lua script ─────────────────────────────────────
#
# KEYS[1] = ratelimit key (hash)
# ARGV[1] = now (unix timestamp, seconds)
# ARGV[2] = window_seconds
# ARGV[3] = max_limit
# ARGV[4] = cost_increment
#
# Returns: [allowed (1/0), current_usage (float), remaining (float),
#           reset_time_seconds (int)]
SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local max_limit = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])

local bucket_dur = window
local current_bucket = math.floor(now / bucket_dur)
local prev_bucket = current_bucket - 1
local time_into = now - (current_bucket * bucket_dur)
local fraction = time_into / bucket_dur

local current_val = redis.call('HGET', key, tostring(current_bucket))
local prev_val = redis.call('HGET', key, tostring(prev_bucket))

if current_val then current_val = tonumber(current_val) end
if prev_val then prev_val = tonumber(prev_val) end

current_val = current_val or 0
prev_val = prev_val or 0

-- Weighted (sliding) usage across the window.
local weighted = prev_val * (1 - fraction) + current_val

if weighted + cost > max_limit then
  local reset = (current_bucket + 1) * bucket_dur - now
  return {0, weighted, max_limit - weighted, reset}
end

current_val = current_val + cost
redis.call('HSET', key, tostring(current_bucket), current_val)
redis.call('EXPIRE', key, window * 2)

weighted = prev_val * (1 - fraction) + current_val
local reset = (current_bucket + 1) * bucket_dur - now
return {1, weighted, max_limit - weighted, reset}
"""


class RedisRateLimiter:
    """Enforce per-tenant RPM and TPM quotas via atomic Redis Lua scripts.

    Both dimensions share a single :class:`RedisRateLimiter` instance; the
    tenant context carries the per-tier limits.
    """

    #: Length of the sliding window in seconds (1 minute).
    WINDOW_SECONDS: int = 60

    def __init__(self, redis_client: Optional[Any] = None) -> None:
        self._redis: Optional[Any] = redis_client
        self._script_sha: Optional[str] = None

    async def _ensure_script(self) -> None:
        """Lazy-load (cache) the sliding-window Lua script by SHA."""
        if self._redis is None:
            raise RuntimeError("Redis client is not configured for rate limiting")
        if self._script_sha is None:
            self._script_sha = await self._redis.script_load(SLIDING_WINDOW_LUA)

    async def _eval_window(
        self,
        key: str,
        now: int,
        window: int,
        max_limit: int,
        cost: int,
    ) -> Tuple[bool, float, float, int]:
        """Execute the sliding-window script for a single dimension.

        Returns
        -------
        tuple
            ``(allowed, current_usage, remaining, reset_seconds)``.
        """
        await self._ensure_script()
        try:
            result = await self._redis.evalsha(
                self._script_sha,
                1,
                key,
                now,
                window,
                max_limit,
                cost,
            )
        except Exception:
            # EVALSHA can fail with NOSCRIPT on a cold shard; fall back to EVAL.
            result = await self._redis.eval(
                SLIDING_WINDOW_LUA,
                1,
                key,
                now,
                window,
                max_limit,
                cost,
            )

        allowed = int(result[0]) == 1
        usage = float(result[1])
        remaining = float(result[2])
        reset = int(result[3])
        return allowed, usage, remaining, reset

    async def check_rate_limits(
        self,
        tenant: Any,
        prompt_token_cost: int = 1,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate RPM and TPM quotas for *tenant*.

        Parameters
        ----------
        tenant
            A :class:`~api.auth.TenantContext` (duck-typed; only
            ``tenant_id``, ``rpm_limit``, ``tpm_limit`` are accessed).
        prompt_token_cost
            Estimated input tokens for the request.  The total TPM cost is
            ``prompt_token_cost + max_output_tokens``; callers should pass
            ``max_tokens`` as the output ceiling when known.  Defaults to
            ``1`` for RPM-only checks.

        Returns
        -------
        tuple
            ``(allowed, info)`` where *info* is a dict carrying the
            ``limit``, ``remaining`` and ``reset`` of the *binding* dimension
            (the one that is exhausted first), plus per-dimension breakdowns
            suitable for the ``X-RateLimit-*`` response headers.
        """
        now = int(time.time())
        rpm_key = f"ratelimit:{tenant.tenant_id}:rpm"
        tpm_key = f"ratelimit:{tenant.tenant_id}:tpm"

        rpm_allowed, rpm_usage, rpm_remaining, rpm_reset = await self._eval_window(
            rpm_key, now, self.WINDOW_SECONDS, tenant.rpm_limit, 1
        )
        tpm_allowed, tpm_usage, tpm_remaining, tpm_reset = await self._eval_window(
            tpm_key, now, self.WINDOW_SECONDS, tenant.tpm_limit, prompt_token_cost
        )

        if not rpm_allowed:
            limit, remaining, reset = tenant.rpm_limit, rpm_remaining, rpm_reset
        elif not tpm_allowed:
            limit, remaining, reset = tenant.tpm_limit, tpm_remaining, tpm_reset
        elif rpm_remaining <= tpm_remaining:
            limit, remaining, reset = tenant.rpm_limit, rpm_remaining, rpm_reset
        else:
            limit, remaining, reset = tenant.tpm_limit, tpm_remaining, tpm_reset

        if not (rpm_allowed and tpm_allowed):
            if not rpm_allowed:
                stigmergic_rate_limit_exceeded_total.labels(
                    tenant_id=tenant.tenant_id, dimension="rpm"
                ).inc()
            if not tpm_allowed:
                stigmergic_rate_limit_exceeded_total.labels(
                    tenant_id=tenant.tenant_id, dimension="tpm"
                ).inc()
            logger.info(
                "Rate limit exceeded for tenant=%s rpm=%s tpm=%s",
                tenant.tenant_id, rpm_allowed, tpm_allowed,
            )

        info: Dict[str, Any] = {
            "allowed": rpm_allowed and tpm_allowed,
            "limit": limit,
            "remaining": remaining,
            "reset": reset,
            "rpm": {
                "limit": tenant.rpm_limit,
                "remaining": rpm_remaining,
                "reset": rpm_reset,
                "allowed": rpm_allowed,
            },
            "tpm": {
                "limit": tenant.tpm_limit,
                "remaining": tpm_remaining,
                "reset": tpm_reset,
                "allowed": tpm_allowed,
            },
        }
        return rpm_allowed and tpm_allowed, info
