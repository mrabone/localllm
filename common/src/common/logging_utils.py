import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Logging formatter that emits each log record as a single JSON line.

    Fields emitted: timestamp (UTC ISO 8601), level, logger, message.
    An exc_info key is added when an exception is attached to the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_record: dict = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


UVICORN_LOG_CONFIG: dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": f"{JsonFormatter.__module__}.{JsonFormatter.__qualname__}",
        },
    },
    "handlers": {
        "json_stdout": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "json",
        },
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["json_stdout"],
            "level": "INFO",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["json_stdout"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["json_stdout"],
        "level": "INFO",
    },
}


def setup_logging() -> None:
    """Configure the root logger to emit structured JSON to stdout.

    Replaces any existing handlers so that this function is safe to call
    multiple times (e.g. in tests) without producing duplicate log lines.

    For uvicorn-based services, pass ``UVICORN_LOG_CONFIG`` to
    ``uvicorn.run(log_config=...)`` instead.  It configures both the root
    logger and uvicorn's own loggers to use the same JSON format.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)
    root_logger.addHandler(handler)
