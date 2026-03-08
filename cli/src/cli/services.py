import json
import logging
import time
import uuid
from pathlib import Path

import httpx

from cli.config import settings

logger = logging.getLogger(__name__)

# Registry key used to store validation timestamps — never a session name.
_CACHE_KEY = "_cache"


class SessionRegistry:
    """Manages the on-disk JSON registry that maps session names to UUIDs.

    Encapsulates all file I/O, UUID parsing, and TTL-based cache validation so
    that the public ``get_or_create_session`` function can focus purely on the
    higher-level session lifecycle logic.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict = self._load()

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except Exception:
            logger.warning("Sessions registry is corrupt; starting fresh.")
            return {}

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

    def get_uuid(self, name: str) -> uuid.UUID | None:
        """Return the UUID for *name*, or None if absent or unparseable."""
        raw = self._data.get(name)
        if raw is None:
            return None
        try:
            return uuid.UUID(raw)
        except (ValueError, AttributeError):
            return None

    def set_uuid(self, name: str, session_id: uuid.UUID) -> None:
        self._data[name] = str(session_id)

    def is_cache_valid(self, name: str) -> bool:
        """Return True if the cached validation timestamp for *name* is still fresh."""
        ttl = settings.session_cache_ttl
        if ttl <= 0:
            return False
        entry = self._data.get(_CACHE_KEY, {}).get(name, {})
        validated_at = entry.get("validated_at", 0.0)
        return (time.time() - validated_at) < ttl

    def update_cache(self, name: str) -> None:
        if _CACHE_KEY not in self._data:
            self._data[_CACHE_KEY] = {}
        self._data[_CACHE_KEY][name] = {"validated_at": time.time()}

    def invalidate_cache(self, name: str) -> None:
        self._data.get(_CACHE_KEY, {}).pop(name, None)

    def names(self) -> list[str]:
        """Return all named session keys (excludes the internal cache key)."""
        return sorted(k for k in self._data if k != _CACHE_KEY)


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


def list_sessions() -> list[str]:
    """Return all named sessions stored in the registry (sorted)."""
    registry = SessionRegistry(Path(settings.sessions_registry).expanduser())
    return registry.names()


def get_or_create_session(
    client: httpx.Client,
    name: str | None = None,
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
    registry = SessionRegistry(Path(settings.sessions_registry).expanduser())

    if name is not None:
        session_id = registry.get_uuid(name)

        if session_id is not None:
            # Fast path: trust the cache without a server call.
            if registry.is_cache_valid(name):
                logger.debug("Session '%s' served from cache (TTL not expired).", name)
                return session_id, name

            # Validate with server using lightweight HEAD request.
            if _session_alive(client, session_id):
                registry.update_cache(name)
                registry.save()
                return session_id, name

            # Session gone (e.g. DB wiped) — create a replacement.
            logger.info(
                "Session '%s' (%s) not found on server; creating a new one.",
                name,
                session_id,
            )
            registry.invalidate_cache(name)

        # Either session was missing or just invalidated — create a new one.
        session_id = _create_remote_session(client)
        registry.set_uuid(name, session_id)
        registry.update_cache(name)
        registry.save()
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
