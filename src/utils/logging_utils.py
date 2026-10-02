"""
Logging utilities for cmu-10799-diffusion.

Provides unified logging setup for training, sampling, and evaluation.
"""

import os
import sys
import logging
from typing import Optional


class _FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit.

    Python buffers stdout in block mode when not writing to a terminal
    (e.g. nohup, pipe, subprocess). Without an explicit flush the output
    only appears when the 8 KB buffer fills — making long-running jobs look
    silent. This subclass flushes unconditionally so log lines appear in
    real time regardless of how the process was launched.
    """

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_logger(
    log_dir: str,
    name: str = 'main',
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Set up logger that writes to both console and file.

    Configures the *root* logger with both handlers so that child loggers
    created anywhere in the package (e.g. ``logging.getLogger(__name__)``
    inside schic_em.py) propagate their messages to the same sinks without
    any extra wiring in each module.

    Args:
        log_dir: Directory to save log files
        name: Logger name (returned logger; only used for the log file name)
        log_file: Log filename (default: {name}.log)
        level: Logging level

    Returns:
        Logger instance (same as logging.getLogger(name))
    """
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    simple_formatter = logging.Formatter('%(message)s')

    # File handler
    if log_file is None:
        log_file = f'{name}.log'
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(file_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(detailed_formatter)

    # Console handler — flushes on every record so output is real-time
    console_handler = _FlushingStreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)

    # Configure the ROOT logger so all child loggers (src.methods.schic_em,
    # src.data.schic, …) propagate here automatically.  Clear any handlers
    # that a previous call or basicConfig may have installed.
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Return the named logger for callers that want to keep a reference.
    # propagate=True (default) so it still reaches the root handlers above.
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def log_section(logger: logging.Logger, title: str, width: int = 80):
    """Log a section header."""
    logger.info("")
    logger.info("=" * width)
    logger.info(title)
    logger.info("=" * width)
    logger.info("")
