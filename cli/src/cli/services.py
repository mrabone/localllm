import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx

from cli.config import settings

logger = logging.getLogger(__name__)

# Registry key used to store validation timestamps — never a session name.
_CACHE_KEY = "_cache"


def _registry_path() -> Path:
    return Path(os.path.expanduser(settings.sessions_registry))


def _load_registry() -> dict:
    path = _registry_path()
    if not path.exists():
        return {}
    try:
        import json

        return json.loads(path.read_text())
    except Exception:
        logger.warning("Sessions registry is corrupt; starting fresh.")
        return {}


def _save_registry(registry: dict) -> None:
    import json

    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2))


def _get_session_uuid(registry: dict, name: str) -> Optional[uuid.UUID]:
    """Return the UUID for *name* from the registry, or None."""
    raw = registry.get(name)
    if raw is None:
        return None
    try:
        return uuid.UUID(raw)
    except (ValueError, AttributeError):
        return None


def _set_session_uuid(registry: dict, name: str, session_id: uuid.UUID) -> None:
    registry[name] = str(session_id)


def _is_cache_valid(registry: dict, name: str) -> bool:
    """Return True if the cached validation timestamp for *name* is still fresh."""
    ttl = settings.session_cache_ttl
    if ttl <= 0:
        return False
    cache = registry.get(_CACHE_KEY, {})
    entry = cache.get(name, {})
    validated_at = entry.get("validated_at", 0.0)
    return (time.time() - validated_at) < ttl


def _update_cache(registry: dict, name: str) -> None:
    if _CACHE_KEY not in registry:
        registry[_CACHE_KEY] = {}
    registry[_CACHE_KEY][name] = {"validated_at": time.time()}


def _invalidate_cache(registry: dict, name: str) -> None:
    registry.get(_CACHE_KEY, {}).pop(name, None)


# ---------------------------------------------------------------------------
# Server communication
# ---------------------------------------------------------------------------


def _session_alive(client: httpx.Client, session_id: uuid.UUID) -> bool:
    """Return True if the server confirms the session exists.

    Uses HEAD to avoid the cost of loading message history.
    """
    try:
        resp = client.head(f"{settings.server_url}/sessions/{session_id}")
        return resp.status_code == 200
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Cannot reach server at {settings.server_url}: {exc}"
        ) from exc


def _create_remote_session(client: httpx.Client) -> uuid.UUID:
    """Ask the server to create a new session and return its UUID."""
    try:
        resp = client.post(f"{settings.server_url}/sessions")
        resp.raise_for_status()
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Cannot reach server at {settings.server_url}: {exc}"
        ) from exc
    return uuid.UUID(resp.json()["session_id"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_sessions() -> list[str]:
    """Return all named sessions stored in the registry (sorted)."""
    registry = _load_registry()
    return sorted(k for k in registry if k != _CACHE_KEY)


def get_or_create_session(
    client: httpx.Client,
    name: Optional[str] = None,
) -> tuple[uuid.UUID, str]:
    """Return ``(session_id, name)`` for the requested session.

    Behaviour
    ---------
    - If *name* is given and exists in the registry, validate it with the
      server (using the TTL cache to avoid redundant requests) and return it.
    - If *name* is given but does not exist yet, create a new session on the
      server, store it under *name*, and return it.
    - If *name* is ``None``, always create a fresh session (no name is stored)
      and print a hint about ``--session`` so the user knows how to resume it.
    """
    registry = _load_registry()

    if name is not None:
        session_id = _get_session_uuid(registry, name)

        if session_id is not None:
            # Fast path: trust the cache without a server call.
            if _is_cache_valid(registry, name):
                logger.debug("Session '%s' served from cache (TTL not expired).", name)
                return session_id, name

            # Validate with server using lightweight HEAD request.
            if _session_alive(client, session_id):
                _update_cache(registry, name)
                _save_registry(registry)
                return session_id, name

            # Session gone (e.g. DB wiped) — create a replacement.
            logger.info(
                "Session '%s' (%s) not found on server; creating a new one.",
                name,
                session_id,
            )
            _invalidate_cache(registry, name)

        # Either session was missing or just invalidated — create a new one.
        session_id = _create_remote_session(client)
        _set_session_uuid(registry, name, session_id)
        _update_cache(registry, name)
        _save_registry(registry)
        logger.info("Created session '%s' (%s).", name, session_id)
        return session_id, name

    # No name provided — always start fresh and tell the user.
    session_id = _create_remote_session(client)
    existing = list_sessions()
    hint = (
        f"To resume a previous session, use: make run-cli SESSION_ARGS='--session <name>'\n"
        f"Available sessions: {', '.join(existing)}"
        if existing
        else "To allow you to revisit this session later, use: make run-cli SESSION_ARGS='--session <name>'"
    )
    print(f"Starting a new session. {hint}\n")
    logger.info("Created anonymous session %s.", session_id)
    return session_id, ""
