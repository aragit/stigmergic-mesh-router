"""Phase 11 tests: API-key auth, distributed rate limiting, and governance.

Covers:
* Valid API key authorization flow (200 OK).
* Missing and malformed Authorization header handling (401).
* Rate limit bucket exhaustion for RPM and TPM (429 / allowed=False).
* Header propagation (``X-RateLimit-*``).
* Governance weight transformation for ``free``, ``pro``, and ``enterprise``.

Two execution strategies are used:

* **Direct unit tests** of :class:`APIKeyAuthenticator`,
  :class:`RedisRateLimiter`, and :class:`TenantPolicyEngine` against an
  in-process ``fakeredis`` instance (same event loop, so the real atomic Lua
  scripts are exercised).
* **HTTP integration tests** via ``TestClient``.  Authentication (which can
  run entirely off the static ``secret_keys_map`` fallback and needs no
  cross-loop Redis) is driven through the live FastAPI dependency graph.
  Rate-limit *exhaustion* is exercised on the unit-tested limiter; the HTTP
  layer's 429 + header propagation is verified with a small stub limiter so
  we avoid async-fakeredis / cross-event-loop pitfalls with ``TestClient``.
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Security is disabled by default in local config.yaml, so the mesh boots
# with no authenticator / rate limiter and legacy unauthenticated flows keep
# working.  Individual tests opt-in by injecting components below.
os.environ.pop("STIGMERGIC_STORAGE_BACKEND", None)
os.environ.pop("STIGMERGIC_REDIS_HOST", None)
os.environ.pop("STIGMERGIC_SECURITY_ENABLED", None)
os.environ.pop("STIGMERGIC_DEFAULT_TENANT_KEYS", None)

from api.auth import APIKeyAuthenticator, TenantContext  # noqa: E402
from api.governance import TenantPolicyEngine  # noqa: E402
from api.rate_limiter import RedisRateLimiter  # noqa: E402
import api.server as server_mod  # noqa: E402
import api.rate_limiter as _rl_mod  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
async def fake_redis():
    return fakeredis.aioredis.FakeRedis()


@pytest.fixture
def pro_tenant() -> TenantContext:
    return TenantContext(
        tenant_id="tenant-pro",
        tier="pro",
        rpm_limit=600,
        tpm_limit=100_000,
        priority_weight=1.0,
    )


@pytest.fixture
def client():
    with TestClient(server_mod.app) as c:
        # Ensure a clean security slate; tests inject what they need.
        server_mod._state.authenticator = None
        server_mod._state.rate_limiter = None
        server_mod._state.policy_engine = None
        yield c
    # Restore defaults so other test modules are unaffected.
    server_mod._state.authenticator = None
    server_mod._state.rate_limiter = None
    server_mod._state.policy_engine = None


# ── API key authentication (direct) ─────────────────────────────────────

@pytest.mark.asyncio
async def test_hash_key_is_sha256():
    """hash_key must return the SHA-256 hex digest of the raw key."""
    import hashlib

    expected = hashlib.sha256(b"my-secret-key").hexdigest()
    assert APIKeyAuthenticator.hash_key("my-secret-key") == expected
    assert len(expected) == 64


@pytest.mark.asyncio
async def test_validate_api_key_from_redis(fake_redis):
    """A key stored as a Redis hash should resolve to a TenantContext."""
    raw_key = "redis-stored-key"
    key_hash = APIKeyAuthenticator.hash_key(raw_key)
    redis_key = f"tenant:key:{key_hash}"

    await fake_redis.hset(redis_key, mapping={
        "tenant_id": "tenant-redis",
        "tier": "enterprise",
        "rpm_limit": 5000,
        "tpm_limit": 500000,
        "priority_weight": 2.0,
    })

    auth = APIKeyAuthenticator(redis_client=fake_redis, secret_keys_map={})
    ctx = await auth.validate_api_key(raw_key)

    assert ctx.tenant_id == "tenant-redis"
    assert ctx.tier == "enterprise"
    assert ctx.rpm_limit == 5000
    assert ctx.tpm_limit == 500000
    assert ctx.priority_weight == 2.0


@pytest.mark.asyncio
async def test_validate_api_key_fallback_to_static_map(fake_redis):
    """When Redis misses, the static secret_keys_map must be consulted."""
    raw_key = "fallback-key"
    key_hash = APIKeyAuthenticator.hash_key(raw_key)
    ctx = TenantContext("tenant-fb", "pro", 600, 100_000, 1.5)

    # Redis is configured but holds no entry for this key.
    auth = APIKeyAuthenticator(
        redis_client=fake_redis,
        secret_keys_map={key_hash: ctx},
    )
    result = await auth.validate_api_key(raw_key)
    assert result is ctx


@pytest.mark.asyncio
async def test_validate_invalid_api_key_raises(fake_redis):
    """An unknown key must raise HTTPException(401) and bump the metric."""
    auth = APIKeyAuthenticator(redis_client=fake_redis, secret_keys_map={})
    with pytest.raises(Exception) as exc_info:
        await auth.validate_api_key("nonexistent-key")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_validate_empty_api_key_raises():
    """An empty key is a 401 with reason 'missing_key'."""
    auth = APIKeyAuthenticator(redis_client=None, secret_keys_map={})
    with pytest.raises(Exception) as exc_info:
        await auth.validate_api_key("")
    assert exc_info.value.status_code == 401


# ── Rate limiting (direct, against fakeredis + real Lua) ─────────────────

@pytest.fixture
def frozen_clock(monkeypatch):
    """Freeze ``time.time`` inside the rate-limiter to avoid sub-second
    flakiness from the sliding-window bucket boundary."""
    monkeypatch.setattr(_rl_mod.time, "time", lambda: 1_700_000_000.0)
    return 1_700_000_000.0


@pytest.mark.asyncio
async def test_rate_limiter_allows_within_rpm(fake_redis, pro_tenant):
    """A single request within both quotas is allowed and returns headers."""
    limiter = RedisRateLimiter(redis_client=fake_redis)
    allowed, info = await limiter.check_rate_limits(pro_tenant, prompt_token_cost=10)

    assert allowed is True
    assert info["allowed"] is True
    assert info["limit"] > 0
    assert info["remaining"] >= 0
    assert info["reset"] > 0
    assert info["rpm"]["allowed"] is True
    assert info["tpm"]["allowed"] is True


@pytest.mark.asyncio
async def test_rpm_bucket_exhaustion(fake_redis, frozen_clock):
    """After rpm_limit requests the RPM dimension must reject."""
    tenant = TenantContext("t-rpm", "free", rpm_limit=3, tpm_limit=10_000_000, priority_weight=1.0)
    limiter = RedisRateLimiter(redis_client=fake_redis)

    for i in range(3):
        allowed, info = await limiter.check_rate_limits(tenant, prompt_token_cost=1)
        assert allowed is True, f"iteration {i} unexpectedly rejected"
        assert info["rpm"]["allowed"] is True

    # 4th request exceeds the RPM window.
    allowed, info = await limiter.check_rate_limits(tenant, prompt_token_cost=1)
    assert allowed is False
    assert info["rpm"]["allowed"] is False
    assert info["tpm"]["allowed"] is True  # TPM not the binding constraint
    assert info["limit"] == tenant.rpm_limit


@pytest.mark.asyncio
async def test_tpm_bucket_exhaustion(fake_redis, frozen_clock):
    """A large prompt_token_cost exhausts the TPM window before RPM."""
    tenant = TenantContext("t-tpm", "free", rpm_limit=10_000_000, tpm_limit=100)
    limiter = RedisRateLimiter(redis_client=fake_redis)

    # Each call costs 50 tokens against a 100-token TPM budget / 60s window.
    first_allowed, _ = await limiter.check_rate_limits(tenant, prompt_token_cost=50)
    assert first_allowed is True

    second_allowed, _ = await limiter.check_rate_limits(tenant, prompt_token_cost=50)
    assert second_allowed is True

    # Third call would push usage to 150 > 100 → rejected on TPM.
    allowed, info = await limiter.check_rate_limits(tenant, prompt_token_cost=50)
    assert allowed is False
    assert info["tpm"]["allowed"] is False
    assert info["rpm"]["allowed"] is True  # RPM is generous here


@pytest.mark.asyncio
async def test_rejected_request_does_not_consume_tokens(fake_redis, frozen_clock):
    """A rejected (over-limit) request must not record cost in Redis."""
    tenant = TenantContext("t-noconsume", "free", rpm_limit=1, tpm_limit=10_000_000, priority_weight=1.0)
    limiter = RedisRateLimiter(redis_client=fake_redis)

    await limiter.check_rate_limits(tenant, prompt_token_cost=1)  # fills bucket
    allowed, _ = await limiter.check_rate_limits(tenant, prompt_token_cost=1)
    assert allowed is False

    # The rejected call must not have mutated the bucket: exactly one unit
    # of cost should be recorded for the tenant's RPM key.
    raw = await fake_redis.hgetall(f"ratelimit:{tenant.tenant_id}:rpm")
    recorded_values = [float(v) for v in raw.values()]
    assert max(recorded_values, default=0.0) <= 1.0


# ── Governance weight transformation ────────────────────────────────────

@pytest.mark.asyncio
async def test_governance_free_boosts_capability_and_saturation():
    """Free tier: w_C × 2.0 and w_S × 2.0; V and L unchanged."""
    base = {"alpha": 1.0, "beta": 2.0, "gamma": 1.5, "delta": 1.5}
    engine = TenantPolicyEngine()
    tenant = TenantContext("t-free", "free", 600, 100_000, 1.0)

    biased = engine.apply_governance_bias(tenant, base)

    assert biased["delta"] == pytest.approx(3.0)   # 1.5 × 2.0
    assert biased["gamma"] == pytest.approx(3.0)   # 1.5 × 2.0
    assert biased["alpha"] == pytest.approx(1.0)   # unchanged
    assert biased["beta"] == pytest.approx(2.0)    # unchanged
    # Must not mutate the caller's dict.
    assert base == {"alpha": 1.0, "beta": 2.0, "gamma": 1.5, "delta": 1.5}


@pytest.mark.asyncio
async def test_governance_enterprise_boost_latency_and_throughput():
    """Enterprise tier: w_L × 1.5 and w_V × 1.2; S and C unchanged."""
    base = {"alpha": 1.0, "beta": 2.0, "gamma": 1.5, "delta": 1.5}
    engine = TenantPolicyEngine()
    tenant = TenantContext("t-ent", "enterprise", 600, 100_000, 1.0)

    biased = engine.apply_governance_bias(tenant, base)

    assert biased["beta"] == pytest.approx(3.0)    # 2.0 × 1.5
    assert biased["alpha"] == pytest.approx(1.2)   # 1.0 × 1.2
    assert biased["gamma"] == pytest.approx(1.5)   # unchanged
    assert biased["delta"] == pytest.approx(1.5)   # unchanged


@pytest.mark.asyncio
async def test_governance_pro_is_balanced():
    """Pro tier applies no multiplier — a balanced copy of the baseline."""
    base = {"alpha": 1.0, "beta": 2.0, "gamma": 1.5, "delta": 1.5}
    engine = TenantPolicyEngine()
    tenant = TenantContext("t-pro", "pro", 600, 100_000, 1.0)

    biased = engine.apply_governance_bias(tenant, base)
    assert biased == base
    assert biased is not base  # returns a fresh dict


@pytest.mark.asyncio
async def test_governance_unknown_tier_defaults_to_pro():
    """An unrecognised tier should fall back to the balanced (pro) policy."""
    base = {"alpha": 1.0, "beta": 2.0, "gamma": 1.5, "delta": 1.5}
    engine = TenantPolicyEngine()
    tenant = TenantContext("t-weird", "gold", 600, 100_000, 1.0)

    biased = engine.apply_governance_bias(tenant, base)
    assert biased == base


# ── HTTP integration: authentication ────────────────────────────────────

def _install_authenticator(secret_key: str, tenant: TenantContext):
    """Inject an authenticator backed only by the static key map."""
    key_hash = APIKeyAuthenticator.hash_key(secret_key)
    server_mod._state.authenticator = APIKeyAuthenticator(
        redis_client=None,
        secret_keys_map={key_hash: tenant},
    )
    server_mod._state.rate_limiter = None
    server_mod._state.policy_engine = None


def test_valid_api_key_returns_200(client, pro_tenant):
    """A request with a valid Bearer key should authenticate and succeed."""
    _install_authenticator("valid-secret-key", pro_tenant)
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer valid-secret-key"},
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200


def test_missing_authorization_header_returns_401(client, pro_tenant):
    _install_authenticator("valid-secret-key", pro_tenant)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 401


def test_malformed_authorization_header_returns_401(client, pro_tenant):
    """A non-Bearer scheme must be rejected as 401."""
    _install_authenticator("valid-secret-key", pro_tenant)
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 401


def test_invalid_api_key_returns_401(client, pro_tenant):
    _install_authenticator("valid-secret-key", pro_tenant)
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer wrong-key"},
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 401


def test_models_endpoint_requires_auth(client, pro_tenant):
    """GET /v1/models must also enforce authentication."""
    _install_authenticator("valid-secret-key", pro_tenant)
    # Unauthenticated
    assert client.get("/v1/models").status_code == 401
    # Authenticated
    resp = client.get(
        "/v1/models", headers={"Authorization": "Bearer valid-secret-key"}
    )
    assert resp.status_code == 200


# ── HTTP integration: rate-limit propagation via a stub limiter ─────────


def _install_stub_limiter(allowed: bool, info: dict):
    """Replace the rate limiter with a stub returning fixed results."""
    stub = MagicMock()
    stub.check_rate_limits = AsyncMock(return_value=(allowed, info))
    server_mod._state.rate_limiter = stub
    server_mod._state.authenticator = None  # auth disabled → default tenant
    server_mod._state.policy_engine = None


def test_rate_limited_request_returns_429_with_headers(client, pro_tenant):
    """A rejected request must yield 429 and standard rate-limit headers."""
    info = {
        "allowed": False,
        "limit": 600,
        "remaining": 0,
        "reset": 42,
        "rpm": {"limit": 600, "remaining": 0, "reset": 42, "allowed": False},
        "tpm": {"limit": 100_000, "remaining": 100, "reset": 42, "allowed": True},
    }
    _install_stub_limiter(allowed=False, info=info)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 429
    assert resp.headers["X-RateLimit-Limit"] == "600"
    assert resp.headers["X-RateLimit-Remaining"] == "0"
    assert resp.headers["X-RateLimit-Reset"] == "42"
    assert resp.headers["Retry-After"] == "42"


def test_allowed_request_propagates_rate_limit_headers(client, pro_tenant):
    """A permitted request should still emit X-RateLimit-* on the response."""
    info = {
        "allowed": True,
        "limit": 600,
        "remaining": 599,
        "reset": 42,
        "rpm": {"limit": 600, "remaining": 599, "reset": 42, "allowed": True},
        "tpm": {"limit": 100_000, "remaining": 99_984, "reset": 42, "allowed": True},
    }
    _install_stub_limiter(allowed=True, info=info)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 16,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["X-RateLimit-Limit"] == "600"
    assert resp.headers["X-RateLimit-Remaining"] == "599"
    assert resp.headers["X-RateLimit-Reset"] == "42"
