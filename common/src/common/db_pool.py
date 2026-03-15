import logging
from dataclasses import dataclass

import psycopg2
import psycopg2.pool

logger = logging.getLogger(__name__)

_POOL_MIN_CONN = 2
_POOL_MAX_CONN = 10


class _ConnectionPool:
    _instance: "_ConnectionPool | None" = None

    def __init__(self, pool: psycopg2.pool.ThreadedConnectionPool, dsn: str) -> None:
        self.pool = pool
        self.dsn = dsn

    @classmethod
    def initialise(cls, dsn: str) -> "_ConnectionPool":
        if cls._instance is not None and cls._instance.dsn == dsn:
            return cls._instance
        if cls._instance is not None:
            try:
                cls._instance.pool.closeall()
            except Exception as exc:
                logger.warning("Error closing existing connection pool: %s", exc)
        logger.info(
            "Initialising psycopg2 connection pool (min=%d, max=%d).",
            _POOL_MIN_CONN,
            _POOL_MAX_CONN,
        )
        pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=_POOL_MIN_CONN,
            maxconn=_POOL_MAX_CONN,
            dsn=dsn,
        )
        cls._instance = cls(pool=pool, dsn=dsn)
        logger.info("psycopg2 connection pool ready.")
        return cls._instance

    @classmethod
    def get(cls) -> "_ConnectionPool":
        if cls._instance is None:
            raise RuntimeError("Connection pool not initialised")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        if cls._instance is not None:
            try:
                cls._instance.pool.closeall()
                logger.info("psycopg2 connection pool closed.")
            except Exception as exc:
                logger.warning("Error closing connection pool: %s", exc)
            finally:
                cls._instance = None


def init_pool(dsn: str) -> None:
    """Create (or recreate) the connection pool for *dsn*.

    Safe to call multiple times; a new pool is only created when the DSN
    changes.  Intended to be called once during server lifespan startup so
    that a warm pool is ready before the first request arrives.
    """
    _ConnectionPool.initialise(dsn)


def close_pool() -> None:
    """Close all pooled connections.  Call during server lifespan shutdown."""
    _ConnectionPool.reset()


class _PooledConn:
    """Context manager that checks out a connection and returns it to the pool.

    Falls back to a direct ``psycopg2.connect()`` when no pool is available
    (e.g. during tests that patch ``get_conn`` or run without a live DB).
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn = None
        self._from_pool = False

    def __enter__(self):
        pool_state = _ConnectionPool._instance
        if pool_state is not None and pool_state.dsn == self._dsn:
            self._conn = pool_state.pool.getconn()
            self._from_pool = True
        else:
            self._conn = psycopg2.connect(self._dsn)
            self._from_pool = False
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn is not None:
            pool_state = _ConnectionPool._instance
            if self._from_pool and pool_state is not None:
                try:
                    if exc_type is None:
                        self._conn.commit()
                    else:
                        self._conn.rollback()
                except Exception:
                    pass
                pool_state.pool.putconn(self._conn)
            else:
                try:
                    if exc_type is None:
                        self._conn.commit()
                    else:
                        self._conn.rollback()
                finally:
                    self._conn.close()
        return False


def get_conn(dsn: str) -> "_PooledConn":
    """Return a ``_PooledConn`` context manager for *dsn*.

    Usage::

        with get_conn(dsn) as conn, conn.cursor() as cur:
            cur.execute(...)

    Kept as a named function so that unit tests can patch ``get_conn``
    and inject mock connections without being aware of pool internals.
    """
    return _PooledConn(dsn)
