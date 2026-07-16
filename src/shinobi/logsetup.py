"""Run-log file setup for `AppConfig.log` (`shinobi.config.LogConfig`).

Standard library-vs-application split: shinobi modules only ever *emit*
records through the ``shinobi.*`` logger hierarchy (which carries a
`NullHandler`, attached in the package `__init__`, so an unconfigured
run stays silent); nothing in the package touches the root logger. The
file handler lives here and is attached by the CLI, once per invocation,
when ``log.file`` is set. Programmatic users who want shinobi's records
attach their own handlers to the ``shinobi`` logger instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

from shinobi.config import LogConfig

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"

# The one file handler this module manages. Kept so a repeat call (e.g.
# several CliRunner invocations in one test process) replaces the handler
# instead of stacking duplicates onto the shared `shinobi` logger.
_file_handler: logging.FileHandler | None = None


def setup_file_logging(log: LogConfig) -> Path | None:
    """Attach a file handler for the run log described by `log`, replacing
    any handler a previous call attached.

    Args:
        log: The resolved log settings. `log.file` of None means file
            logging is off -- any previously-attached handler is removed
            and nothing new is attached. Otherwise the log file is
            `log.dir/log.file` (dir created if needed; an absolute
            `log.file` wins over `log.dir`, per pathlib join semantics),
            filtered at `log.level`.

    Returns:
        The log file's path, or None when file logging is off.
    """
    global _file_handler
    logger = logging.getLogger("shinobi")
    if _file_handler is not None:
        logger.removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None
    if not log.file:
        return None

    log_dir = Path(log.dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / log.file
    handler = logging.FileHandler(path)
    handler.setLevel(log.level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    # Without an explicit level the `shinobi` logger inherits the root
    # logger's WARNING and would drop INFO/DEBUG records before the
    # handler ever sees them.
    logger.setLevel(log.level)
    _file_handler = handler
    return path
