from shinobi.config import AppConfig


def test_defaults_when_no_file_or_env(tmp_path, monkeypatch):
    monkeypatch.delenv("SHINOBI_BACKEND__DEFAULT", raising=False)
    cfg = AppConfig.load(config_file=tmp_path / "missing.yml")
    assert cfg.backend.default == "native"


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


def test_explicit_override_wins_over_everything(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yml"
    config_file.write_text("backend:\n  default: docker\n")
    monkeypatch.setenv("SHINOBI_BACKEND__DEFAULT", "podman")
    cfg = AppConfig.load(config_file=config_file, backend={"default": "apptainer"})
    assert cfg.backend.default == "apptainer"
