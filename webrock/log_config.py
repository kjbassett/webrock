"""Logging configuration for webrock.

Call setup_logging() at process startup for standalone use. When embedded in an
application (e.g. stonks), the host configures the root logger instead — webrock
loggers under the ``webrock.*`` hierarchy will inherit that configuration.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import queue

_listener: logging.handlers.QueueListener | None = None


def setup_logging(log_dir: str = "logs") -> None:
    """Configure webrock logging to file + console. No-op if already called."""
    global _listener
    if _listener is not None:
        return

    os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    app_file = logging.handlers.RotatingFileHandler(
        filename=os.path.join(log_dir, "webrock.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    app_file.setLevel(logging.DEBUG)
    app_file.setFormatter(fmt)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    log_queue: queue.Queue = queue.Queue(maxsize=-1)
    _listener = logging.handlers.QueueListener(
        log_queue, app_file, console, respect_handler_level=True
    )
    _listener.start()

    queue_handler = logging.handlers.QueueHandler(log_queue)
    webrock_logger = logging.getLogger("webrock")
    webrock_logger.setLevel(logging.DEBUG)
    webrock_logger.addHandler(queue_handler)


def shutdown_logging() -> None:
    """Stop the queue listener gracefully."""
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None
