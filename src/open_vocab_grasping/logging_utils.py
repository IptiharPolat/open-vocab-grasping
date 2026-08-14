from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {
        "time": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    logger.info(json.dumps(payload, ensure_ascii=False, default=str))

