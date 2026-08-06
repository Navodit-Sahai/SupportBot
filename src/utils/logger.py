"""
Minimal structured logger. Prints to stdout and appends to a jsonl log file.
Every node uses `log_event(node, event, **kw)` for traceability.
"""
from __future__ import annotations
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from config import LOG_DIR


LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = LOG_DIR / "agent.jsonl"


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("support_agent")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    return logger


LOG = _get_logger()


def log_event(node: str, event: str, **fields: Any) -> None:
    """Emit a structured log line. Used by every node."""
    payload = {"ts": time.time(), "node": node, "event": event, **fields}
    LOG.info(f"{node} :: {event} :: {_short_kv(fields)}")
    try:
        with _LOG_FILE.open("a") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        # Never let logging break the request path.
        pass


def _short_kv(fields: dict) -> str:
    parts = []
    for k, v in fields.items():
        s = str(v)
        if len(s) > 80:
            s = s[:77] + "..."
        parts.append(f"{k}={s}")
    return " ".join(parts)


class Timer:
    """Context manager that records elapsed ms into a dict."""
    def __init__(self, bucket: dict, key: str):
        self.bucket, self.key = bucket, key
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, *exc):
        self.bucket[self.key] = round((time.perf_counter() - self.t0) * 1000, 1)
