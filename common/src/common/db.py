from common.config import SharedSettings


def build_pg_dsn(settings: SharedSettings) -> str:
    """Build a libpq connection string from shared settings.

    Returns a DSN in keyword=value format suitable for passing directly to
    ``psycopg2.connect()`` or the pool initialiser.
    """
    return (
        f"host={settings.pg_host} port={settings.pg_port} "
        f"dbname={settings.pg_database} user={settings.pg_user} "
        f"password={settings.pg_password}"
    )
