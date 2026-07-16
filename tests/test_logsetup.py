import logging

import pytest

from shinobi.config import LogConfig
from shinobi.logsetup import setup_file_logging

logger = logging.getLogger("shinobi.run")


@pytest.fixture(autouse=True)
def _detach_file_logging():
    """Leave the shared `shinobi` logger handler-free after each test."""
    yield
    setup_file_logging(LogConfig())


def test_no_file_is_a_noop(tmp_path):
    assert setup_file_logging(LogConfig(dir=str(tmp_path))) is None
    assert list(tmp_path.iterdir()) == []


def test_records_land_in_the_configured_file(tmp_path):
    path = setup_file_logging(LogConfig(dir=str(tmp_path), file="run.log"))
    assert path == tmp_path / "run.log"
    logger.info("step greet: starting")
    text = path.read_text()
    assert "INFO step greet: starting" in text


def test_dir_is_created_if_missing(tmp_path):
    path = setup_file_logging(LogConfig(dir=str(tmp_path / "logs" / "deep"), file="run.log"))
    logger.info("hello")
    assert "hello" in path.read_text()


def test_level_filters_records(tmp_path):
    path = setup_file_logging(LogConfig(dir=str(tmp_path), file="run.log", level="WARNING"))
    logger.info("quiet please")
    logger.error("step x: failed (returncode 1)")
    text = path.read_text()
    assert "quiet please" not in text
    assert "ERROR step x: failed (returncode 1)" in text


def test_resetup_replaces_handler_instead_of_stacking(tmp_path):
    first = setup_file_logging(LogConfig(dir=str(tmp_path), file="first.log"))
    second = setup_file_logging(LogConfig(dir=str(tmp_path), file="second.log"))
    logger.info("only once, only here")
    assert "only once, only here" not in first.read_text()
    assert second.read_text().count("only once, only here") == 1


def test_setup_with_no_file_detaches_previous_handler(tmp_path):
    path = setup_file_logging(LogConfig(dir=str(tmp_path), file="run.log"))
    assert setup_file_logging(LogConfig()) is None
    logger.info("into the void")
    assert "into the void" not in path.read_text()


def test_unconfigured_logging_emits_nothing_to_stderr(capsys):
    # The package NullHandler must keep logging's last-resort stderr
    # handler out of unconfigured runs, even for WARNING+ records.
    logger.error("step x: failed (returncode 1)")
    assert capsys.readouterr().err == ""


def test_bad_level_rejected_at_config_load():
    with pytest.raises(ValueError, match="invalid log level"):
        LogConfig(level="CHATTY")


def test_level_is_normalized_to_uppercase():
    assert LogConfig(level="debug").level == "DEBUG"
