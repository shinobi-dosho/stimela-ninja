from shinobi.config import AppConfig


def test_defaults_when_no_file_or_env(tmp_path, monkeypatch):
    monkeypatch.delenv("SHINOBI_BACKEND__DEFAULT", raising=False)
    cfg = AppConfig.load(config_file=tmp_path / "missing.yml")
    assert cfg.backend.default == "native"
    # Opt-out, not opt-in: root-by-default is Docker's own footgun.
    assert cfg.backend.run_as_host_user is True


def test_run_as_host_user_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("SHINOBI_BACKEND__RUN_AS_HOST_USER", raising=False)
    config_file = tmp_path / "config.yml"
    config_file.write_text("backend:\n  run_as_host_user: false\n")
    cfg = AppConfig.load(config_file=config_file)
    assert cfg.backend.run_as_host_user is False


def test_config_file_overrides_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("SHINOBI_BACKEND__DEFAULT", raising=False)
    config_file = tmp_path / "config.yml"
    config_file.write_text("backend:\n  default: docker\n")
    cfg = AppConfig.load(config_file=config_file)
    assert cfg.backend.default == "docker"


def test_env_overrides_config_file(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yml"
    config_file.write_text("backend:\n  default: docker\n")
    monkeypatch.setenv("SHINOBI_BACKEND__DEFAULT", "podman")
    cfg = AppConfig.load(config_file=config_file)
    assert cfg.backend.default == "podman"


def test_log_file_defaults_to_none(tmp_path):
    cfg = AppConfig.load(config_file=tmp_path / "missing.yml")
    assert cfg.log.file is None


def test_log_override_merges_with_config_file(tmp_path, monkeypatch):
    monkeypatch.delenv("SHINOBI_LOG__DIR", raising=False)
    config_file = tmp_path / "config.yml"
    config_file.write_text("log:\n  dir: logs\n  level: WARNING\n")
    cfg = AppConfig.load(config_file=config_file, log={"file": "run.log", "level": "DEBUG"})
    # Partial overrides deep-merge with the file: untouched keys survive.
    assert cfg.log.file == "run.log"
    assert cfg.log.level == "DEBUG"
    assert cfg.log.dir == "logs"


def test_explicit_override_wins_over_everything(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yml"
    config_file.write_text("backend:\n  default: docker\n")
    monkeypatch.setenv("SHINOBI_BACKEND__DEFAULT", "podman")
    cfg = AppConfig.load(config_file=config_file, backend={"default": "apptainer"})
    assert cfg.backend.default == "apptainer"
