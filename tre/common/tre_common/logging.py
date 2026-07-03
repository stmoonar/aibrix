from __future__ import annotations

import json
import logging as std_logging
from typing import Any


class JsonFormatter(std_logging.Formatter):
    def format(self, record: std_logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, self.datefmt),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_json_logging(level: str | int = "INFO") -> None:
    handler = std_logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = std_logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
