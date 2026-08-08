"""Tenant authentication and API-key verification for the ingress layer.

Provides:

* :class:`TenantContext` — a lightweight dataclass carrying the identity and
  quota/tier metadata for the tenant on whose behalf a request is being served.
* :class:`APIKeyAuthenticator` — resolves a raw API key (extracted from the
  ``Authorization: Bearer <key>`` header) into a :class:`TenantContext` by
  consulting the Redis key store first and a static fallback map second.
* :func:`get_current_tenant` — a FastAPI dependency that performs Bearer-token
  extraction, delegates to the authenticator, and emits 401 on any failure.

Design notes
------------
When ``security.enabled`` is ``False`` in ``config.yaml`` (the local-dev /
test default) the authenticator instance is left unconfigured and
:func:`get_current_tenant` short-circuits to a permissive default tenant so
that existing unauthenticated flows are not broken.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from fastapi import Header, HTTPException
from fastapi.security import SecurityScopes
from starlette.status import HTTP_401_UNAUTHORIZED

from api.metrics import stigmergic_auth_failures_total

logger = logging.getLogger(__name__)

# Fallback tenant used when security is disabled (local-dev / unit-test mode).
_DEFAULT_TENANT = "local-dev"
_DEFAULT_TIER = "pro"
_DEFAULT_RPM = 999_999_999
_DEFAULT_TPM = 9_999_999_999
_DEFAULT_PRIORITY = 1.0


@dataclass
class TenantContext:
    """Identity + quota metadata for the authenticated tenant."""

    tenant_id: str
    tier: str  # "free" | "pro" | "enterprise"
    rpm_limit: int
    tpm_limit: int
    priority_weight: float = 1.0


def default_tenant() -> TenantContext:
    """Return a permissive :class:`TenantContext` for unauthenticated mode."""
    return TenantContext(
        tenant_id=_DEFAULT_TENANT,
        tier=_DEFAULT_TIER,
        rpm_limit=_DEFAULT_RPM,
        tpm_limit=_DEFAULT_TPM,
        priority_weight=_DEFAULT_PRIORITY,
    )


class APIKeyAuthenticator:
    """Resolve raw API keys into :class:`TenantContext` objects.

    Lookup order for a given raw key:

    1. **Redis** — ``HGETALL tenant:key:<sha256_hash>``.  When Redis is
       reachable this is the authoritative store and is updated
       dynamically by operators without a restart.
    2. **Static fallback** — the ``secret_keys_map`` dict (typically seeded
       from a Kubernetes ``Secret`` at startup) is consulted when the Redis
       lookup misses or Redis is unavailable, guaranteeing the router can
       still boot in degraded environments.
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        secret_keys_map: Optional[Dict[str, TenantContext]] = None,
    ) -> None:
        self._redis: Optional[Any] = redis_client
        self._secret_keys_map: Dict[str, TenantContext] = (
            secret_keys_map or {}
        )

    @staticmethod
    def hash_key(raw_key: str) -> str:
        """Return the SHA-256 hex digest of *raw_key*."""
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_to_context(raw: Dict[bytes, bytes]) -> Optional[TenantContext]:
        """Convert a Redis hash response into a :class:`TenantContext`."""
        if not raw:
            return None

        def _field(name: str, default: str) -> str:
            val = raw.get(name.encode())
            if val is None:
                return default
            return val.decode("utf-8") if isinstance(val, bytes) else str(val)

        try:
            return TenantContext(
                tenant_id=_field("tenant_id", "unknown"),
                tier=_field("tier", "pro"),
                rpm_limit=int(_field("rpm_limit", "600")),
                tpm_limit=int(_field("tpm_limit", "100000")),
                priority_weight=float(_field("priority_weight", "1.0")),
            )
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_bearer(authorization: Optional[str]) -> Optional[Tuple[str, str]]:
        """Split ``Authorization`` header into ``(scheme, token)``.

        Returns ``None`` when the header is absent or malformed.
        """
        if not authorization:
            return None
        parts = authorization.split(None, 1)
        if len(parts) != 2:
            return None
        scheme, token = parts[0], parts[1]
        if scheme.lower() != "bearer" or not token:
            return None
        return scheme, token

    async def validate_api_key(self, api_key: str) -> TenantContext:
        """Resolve *api_key* into a :class:`TenantContext`.

        Raises
        ------
        HTTPException(401)
            If the key cannot be resolved through either Redis or the
            static fallback map.  The corresponding Prometheus failure
            counter is incremented before raising.
        """
        if not api_key:
            stigmergic_auth_failures_total.labels(reason="missing_key").inc()
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )

        key_hash = self.hash_key(api_key)
        redis_key = f"tenant:key:{key_hash}"

        # 1. Authoritative lookup in Redis.
        if self._redis is not None:
            try:
                raw = await self._redis.hgetall(redis_key)
                ctx = self._hash_to_context(raw)
                if ctx is not None:
                    return ctx
            except Exception as exc:
                # Redis transient failure — fall through to the static map.
                logger.warning("Redis key lookup failed (%s); using fallback", exc)

        # 2. Static fallback map.
        ctx = self._secret_keys_map.get(key_hash)
        if ctx is None:
            stigmergic_auth_failures_total.labels(reason="invalid_key").inc()
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )
        return ctx


# ── FastAPI dependency ────────────────────────────────────────────────

async def get_current_tenant(
    security_scopes: SecurityScopes,
    authorization: Optional[str] = Header(None),
) -> TenantContext:
    """FastAPI dependency that extracts and validates the caller's tenant.

    When no authenticator is configured (security disabled) a permissive
    default tenant is returned so that local-dev and legacy test flows
    continue to function.
    """
    from api.server import _state

    authenticator: Optional[APIKeyAuthenticator] = _state.authenticator

    if authenticator is None:
        return default_tenant()

    parsed = APIKeyAuthenticator._parse_bearer(authorization)
    if parsed is None:
        reason = "missing_key" if not authorization else "malformed_key"
        stigmergic_auth_failures_total.labels(reason=reason).inc()
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header",
        )

    _, token = parsed
    try:
        return await authenticator.validate_api_key(token)
    except HTTPException:
        raise
    except Exception:
        stigmergic_auth_failures_total.labels(reason="invalid_key").inc()
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
