import logging

import psycopg2
import psycopg2.pool

logger = logging.getLogger(__name__)

_POOL_MIN_CONN = 2
_POOL_MAX_CONN = 10

# Module-level connection pool.  Initialised lazily on the first call to
# _get_conn() so that imports don't require a live database.  Replaced by
# _init_pool() during server lifespan startup for explicit control over pool sizing.
_pool: psycopg2.pool.ThreadedConnectionPool | None = None
_pool_dsn: str | None = None


def _init_pool(dsn: str) -> None:
    """Create (or recreate) the module-level connection pool for *dsn*.

    Safe to call multiple times; a new pool is only created when the DSN
    changes.  Intended to be called once during server lifespan startup so
    that a warm pool is ready before the first request arrives.
    """
    global _pool, _pool_dsn
    if _pool is not None and _pool_dsn == dsn:
        return
    if _pool is not None:
        try:
            _pool.closeall()
        except Exception:
            pass
    logger.info(
        "Initialising psycopg2 connection pool (min=%d, max=%d).",
        _POOL_MIN_CONN,
        _POOL_MAX_CONN,
    )
    _pool = psycopg2.pool.ThreadedConnectionPool(
        minconn=_POOL_MIN_CONN,
        maxconn=_POOL_MAX_CONN,
        dsn=dsn,
    )
    _pool_dsn = dsn
    logger.info("psycopg2 connection pool ready.")


def _close_pool() -> None:
    """Close all pooled connections.  Call during server lifespan shutdown."""
    global _pool, _pool_dsn
    if _pool is not None:
        try:
            _pool.closeall()
            logger.info("psycopg2 connection pool closed.")
        except Exception as exc:
            logger.warning("Error closing connection pool: %s", exc)
        finally:
            _pool = None
            _pool_dsn = None


class _PooledConn:
    """Context manager that checks out a connection and returns it to the pool.

    Falls back to a direct ``psycopg2.connect()`` when no pool is available
    (e.g. during tests that patch ``_get_conn`` or run without a live DB).
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn = None
        self._from_pool = False

    def __enter__(self):
        if _pool is not None and _pool_dsn == self._dsn:
            self._conn = _pool.getconn()
            self._from_pool = True
        else:
            self._conn = psycopg2.connect(self._dsn)
            self._from_pool = False
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn is not None:
            if self._from_pool and _pool is not None:
                try:
                    if exc_type is None:
                        self._conn.commit()
                    else:
                        self._conn.rollback()
                except Exception:
                    pass
                _pool.putconn(self._conn)
            else:
                try:
                    if exc_type is None:
                        self._conn.commit()
                    else:
                        self._conn.rollback()
                finally:
                    self._conn.close()
        return False


def _get_conn(dsn: str) -> "_PooledConn":
    """Return a ``_PooledConn`` context manager for *dsn*.

    Usage::

        with _get_conn(dsn) as conn, conn.cursor() as cur:
            cur.execute(...)

    Kept as a named function so that unit tests can patch ``_get_conn``
    and inject mock connections without needing to be aware of pool internals.
    """
    return _PooledConn(dsn)
