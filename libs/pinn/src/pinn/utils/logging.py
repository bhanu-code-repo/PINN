"""Logging configuration — loguru console and file sinks.

Copyright 2026 Bhanu Thakur. All rights reserved.
"""

import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm


def setup_logging(
    log_dir: str | Path | None = None,
    level: str = "INFO",
    file_level: str = "DEBUG",
) -> Path | None:
    """Configure loguru for console (tqdm-safe) and optional file logging.

    The console sink writes through :func:`tqdm.write` so log lines do not
    mangle active progress bars. If ``log_dir`` is given, a timestamped log
    file is also written there with full ``DEBUG`` detail and rotation.

    Call this once at application startup (CLI entry point, notebook top cell).
    Library code should simply use ``from loguru import logger`` and log —
    it inherits whatever sinks the application configured.

    Args:
        log_dir: Directory for the log file. Created if missing. ``None``
            disables file logging.
        level: Minimum level for the console sink.
        file_level: Minimum level for the file sink.

    Returns:
        The path of the created log file, or ``None`` if file logging is off.
    """
    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end="", file=sys.stderr),
        level=level,
        colorize=True,
    )

    if log_dir is None:
        return None

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run_{time:YYYY-MM-DD_HH-mm-ss}.log"
    logger.add(log_file, level=file_level, rotation="50 MB", enqueue=True)
    # Resolve the actual file loguru created (it substitutes {time} itself).
    created = sorted(log_dir.glob("run_*.log"))
    return created[-1] if created else None
