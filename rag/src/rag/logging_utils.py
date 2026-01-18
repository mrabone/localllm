import sys
import json
import logging
from datetime import datetime


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_record)


class LoggingSetup:
    @staticmethod
    def setup_logging():
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        # Remove any existing handlers to avoid duplicate logs
        for h in root_logger.handlers:
            root_logger.removeHandler(h)
        root_logger.addHandler(handler)
