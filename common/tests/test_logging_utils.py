import json
import logging
import logging.config

from common.logging_utils import UVICORN_LOG_CONFIG, JsonFormatter, setup_logging


class TestJsonFormatter:
    def test_formats_as_valid_json(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello world",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello world"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test"
        assert "timestamp" in parsed

    def test_includes_exc_info_when_present(self):
        formatter = JsonFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg="failed",
                args=(),
                exc_info=sys.exc_info(),
            )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exc_info" in parsed
        assert "ValueError" in parsed["exc_info"]


class TestSetupLogging:
    def test_configures_root_logger_with_json_formatter(self):
        setup_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)

    def test_calling_twice_does_not_duplicate_handlers(self):
        setup_logging()
        setup_logging()
        root = logging.getLogger()
        assert len(root.handlers) == 1


class TestUvicornLogConfig:
    def test_dict_config_applies_without_error(self):
        logging.config.dictConfig(UVICORN_LOG_CONFIG)
        root = logging.getLogger()
        assert isinstance(root.handlers[0].formatter, JsonFormatter)

    def test_uvicorn_logger_uses_json_formatter(self):
        logging.config.dictConfig(UVICORN_LOG_CONFIG)
        uvicorn_logger = logging.getLogger("uvicorn")
        assert len(uvicorn_logger.handlers) == 1
        assert isinstance(uvicorn_logger.handlers[0].formatter, JsonFormatter)

    def test_uvicorn_access_logger_uses_json_formatter(self):
        logging.config.dictConfig(UVICORN_LOG_CONFIG)
        access_logger = logging.getLogger("uvicorn.access")
        assert len(access_logger.handlers) == 1
        assert isinstance(access_logger.handlers[0].formatter, JsonFormatter)
