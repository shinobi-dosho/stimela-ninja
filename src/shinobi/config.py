"""Application configuration: layered defaults < config file < env vars <
CLI overrides, all validated by the same pydantic models used everywhere
else in shinobi. No OmegaConf/scabha/munch/benedict stack -- just
pydantic-settings, reusing the validation library the cab schemas already
depend on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

DEFAULT_CONFIG_FILE = Path.home() / ".shinobi" / "config.yml"


class BackendConfig(BaseModel):
    default: str = "native"
    # Run docker/podman containers as the host UID/GID (`--user uid:gid`,
    # HOME=workdir) instead of root, so bind-mounted outputs come out
    # host-owned. Defaults to True -- unlike this file's other options,
    # this one is opt-OUT: root-by-default is Docker's own footgun, not
    # behavior worth preserving silently. Set to False for images that
    # specifically require running as root. No-op for apptainer, which
    # already runs as the host user.
    run_as_host_user: bool = True


class ExecutionConfig(BaseModel):
    # How many recipe steps may run concurrently. Default 1 (sequential) --
    # parallelism is opt-in: at 1 the scheduler reproduces exact
    # declaration-order execution, and no MUTABLE input can be shared across
    # concurrently-running steps (see AGENTS.md's recipe-execution note).
    max_workers: int = 1


class CacheConfig(BaseModel):
    # Step-level skip-if-unchanged caching (shinobi.cache). Disabled by
    # default -- same "opt-in, zero cost for existing users" shape as
    # `backend.default`/`execution.max_workers`.
    enabled: bool = False
    dir: str = ".shinobi/cache"


class LogConfig(BaseModel):
    dir: str = "."
    level: str = "INFO"
    # Live-echo a running cab's stdout/stderr to the terminal as it runs
    # (native/container backends only -- see shinobi.backends._stream).
    # Default on: `ninja run --quiet` opts out for one invocation.
    stream: bool = True


class _YamlFileSource(PydanticBaseSettingsSource):
    """Reads a YAML file (if it exists) as a settings source."""

    def __init__(self, settings_cls: type[BaseSettings], yaml_file: Path):
        super().__init__(settings_cls)
        self._data: dict[str, Any] = {}
        if yaml_file.exists():
            self._data = yaml.safe_load(yaml_file.read_text()) or {}

    def get_field_value(self, field, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data


class AppConfig(BaseSettings):
    """Precedence, highest to lowest: CLI overrides > env vars (SHINOBI_*)
    > config file > built-in defaults.
    """

    model_config = SettingsConfigDict(env_prefix="SHINOBI_", env_nested_delimiter="__")

    _config_file: ClassVar[Path] = DEFAULT_CONFIG_FILE

    backend: BackendConfig = Field(default_factory=BackendConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        yaml_source = _YamlFileSource(settings_cls, cls._config_file)
        return (init_settings, env_settings, yaml_source)

    @classmethod
    def load(cls, config_file: str | Path | None = None, **cli_overrides: Any) -> "AppConfig":
        cls._config_file = Path(config_file) if config_file else DEFAULT_CONFIG_FILE
        return cls(**cli_overrides)
