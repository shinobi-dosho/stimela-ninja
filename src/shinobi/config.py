"""Application configuration: layered defaults < config file < env vars <
CLI overrides, all validated by the same pydantic models used everywhere
else in shinobi. No OmegaConf/scabha/munch/benedict stack -- just
pydantic-settings, reusing the validation library the cab schemas already
depend on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from shinobi.resources import ResourceBudget

DEFAULT_CONFIG_FILE = Path.home() / ".shinobi" / "config.yml"


class VenvConfig(BaseModel):
    """Settings for the `venv` backend (`shinobi.backends.venv`).

    `default` is the venv used when a step declares none of its own (a path,
    or a key into `envs`); `None` means "no default", so a `venv`-backend step
    with nothing declared falls back to native. `envs` maps short names to
    venv paths so recipes/config can refer to a venv by name rather than a
    machine-specific absolute path. A venv path is a deployment concern, so
    these live here (or on a Scope in Python), never in a shared cab repo.
    """

    default: str | None = None
    envs: dict[str, str] = Field(default_factory=dict)


class BackendConfig(BaseModel):
    """Settings controlling which execution backend cabs run under."""

    default: str = "native"
    # Run docker/podman containers as the host UID/GID (`--user uid:gid`,
    # HOME=workdir) instead of root, so bind-mounted outputs come out
    # host-owned. Defaults to True -- unlike this file's other options,
    # this one is opt-OUT: root-by-default is Docker's own footgun, not
    # behavior worth preserving silently. Set to False for images that
    # specifically require running as root. No-op for apptainer, which
    # already runs as the host user, and `--user`-free under a rootless
    # podman, which already runs the container as the invoking user (asking
    # for a uid inside its user namespace would break every bind-mount
    # write instead of fixing ownership -- see `_rootless`).
    run_as_host_user: bool = True
    # `venv` backend settings; reachable as SHINOBI_BACKEND__VENV__DEFAULT /
    # __ENVS via the env_nested_delimiter below.
    venv: VenvConfig = Field(default_factory=VenvConfig)


class ExecutionConfig(BaseModel):
    """Settings controlling recipe step scheduling."""

    # How many recipe steps may run concurrently. Default 1 (sequential) --
    # parallelism is opt-in: at 1 the scheduler reproduces exact
    # declaration-order execution, and no MUTABLE input can be shared across
    # concurrently-running steps (see AGENTS.md's recipe-execution note).
    max_workers: int = 1
    # The machine budget steps are admitted against, on top of `max_workers`
    # (see `shinobi.resources`). Only consulted when some step actually
    # declares a footprint, so the default costs nothing; `"auto"` then
    # detects the real cgroup-aware limit rather than trusting /proc/meminfo.
    resources: ResourceBudget = Field(default_factory=ResourceBudget)
    # Whether a *backend* turns a declaration into a real limit, on top of
    # admission control (which is unaffected by this and always applies).
    # `"auto"` emits only the limits the host can actually enforce -- a
    # rootless runtime on a session with only `memory` delegated gets
    # `--memory` and not `--cpus`, instead of failing outright and losing
    # both. `"always"` emits everything declared and lets the runtime fail
    # loudly if it cannot apply it; `"never"` emits nothing, for a site that
    # wants scheduler-side admission only. See `shinobi.resources`.
    enforce_resources: Literal["auto", "always", "never"] = "auto"
    # Whether a step about to run deletes the previous run's product from
    # each declared output path it writes directly to
    # (`sandbox.clear_stale_outputs`). On by default: re-running a step is
    # supposed to replace its products, which is what a sandboxed relative
    # output already gets from harvest, and without this an absolute one
    # instead lands on top of the last run -- a hard failure with CASA-family
    # tools and a silently corrupt product with anything that appends.
    # Set false for a workspace where nothing may be deleted without the
    # tool itself doing it.
    clear_stale_outputs: bool = True


class SnapshotConfig(BaseModel):
    """Settings controlling mutation-chain snapshots (`shinobi.snapshots`)."""

    # How snapshots are taken, when they are taken at all (they ride on the
    # cache being enabled -- see `CacheConfig.enabled`).
    #
    # - "auto": use the clone ladder (`shinobi.clonefs`), so a reflink-
    #   capable filesystem pays almost nothing per snapshot.
    # - "copy": force full copies. For measuring the real cost, or for a
    #   filesystem whose clone support is suspect.
    # - "off": no snapshots, no journal, no restores -- the escape hatch,
    #   leaving exactly the shipped skip-cache behaviour.
    mode: Literal["auto", "copy", "off"] = "auto"


class CacheConfig(BaseModel):
    """Settings controlling step-level skip-if-unchanged caching."""

    # Step-level skip-if-unchanged caching (shinobi.cache). Disabled by
    # default -- same "opt-in, zero cost for existing users" shape as
    # `backend.default`/`execution.max_workers`.
    enabled: bool = False
    dir: str = ".shinobi/cache"
    snapshots: SnapshotConfig = Field(default_factory=SnapshotConfig)
    # Add a bounded content sample (first+last 4 KiB per file) to boundary
    # path fingerprints -- see `shinobi.cache._hash_path`. Off by default
    # because turning it on changes every boundary-input key, so the first
    # run after enabling recomputes that whole layer.
    content_sample: bool = False


class LogConfig(BaseModel):
    """Settings controlling logging and live output streaming."""

    dir: str = "."
    # Filename for the run log, created under `dir`. None (the default)
    # disables file logging -- same opt-in shape as cache/provenance.
    file: str | None = None
    level: str = "INFO"
    # Live-echo a running cab's stdout/stderr to the terminal as it runs
    # (native/container backends only -- see shinobi.backends._stream).
    # Default on: `ninja run --quiet` opts out for one invocation.
    stream: bool = True
    # How many lines of a cab's stdout/stderr to hold in memory, from the
    # start and from the end of each stream; the middle is elided with a
    # marker (see shinobi.backends._stream.LineBuffer). Raise these to keep
    # more of a chatty tool's output in the StepResult; 0 for either end
    # drops that end entirely. Lines matching the cab's own wranglers are
    # retained regardless, so capping does not cost output values.
    capture_head_lines: int = 5_000
    capture_tail_lines: int = 5_000

    @field_validator("capture_head_lines", "capture_tail_lines")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        """A negative cap is meaningless and would silently behave as 0
        (`max(0, ...)` in the buffer), so it is rejected at config load
        where the offending YAML/env key can still be named.
        """
        if value < 0:
            raise ValueError(f"line capture limits cannot be negative, got {value}")
        return value

    @field_validator("level")
    @classmethod
    def _normalize_level(cls, value: str) -> str:
        """Uppercase and validate `level` at config load, so a bad name in
        the YAML/env fails with a clear message instead of a ValueError
        deep inside `logging.Handler.setLevel`.
        """
        level = value.upper()
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"invalid log level {value!r} (expected DEBUG, INFO, WARNING, ERROR, or CRITICAL)")
        return level


class SandboxConfig(BaseModel):
    """Settings controlling per-step sandbox execution (`shinobi.sandbox`)."""

    # Opt-in. When enabled, each subprocess-backed step (native cabs,
    # containerized cabs/pysteps) runs with its cwd inside a private scratch
    # directory; on success only declared outputs (path-typed output fields
    # plus `Scope.harvest` globs) are moved back to the workspace and the
    # rest is deleted. Off by default -- same opt-in shape as cache/provenance.
    enabled: bool = False
    # Scratch root the per-step directories are created under. Relative to
    # the invocation cwd so it lives on the same filesystem as the workspace
    # -- harvest moves outputs by rename, and multi-GB products must never
    # pay a cross-filesystem copy.
    dir: str = ".shinobi/work"


class ProvenanceConfig(BaseModel):
    """Settings controlling reproducible-run provenance (`shinobi.provenance`)."""

    # Opt-in. When enabled, two things happen together: container images are
    # digest-pinned before running (pin-then-run -- so what executes is what
    # gets recorded, but the run now needs a registry round-trip and executes
    # `repo@sha256:...` instead of `repo:tag`), and a static run manifest is
    # written per top-level run. Off by default so the pinning behaviour is
    # never a surprise; turn on with `ninja run --provenance` or config.
    enabled: bool = False
    dir: str = ".shinobi/runs"


class _YamlFileSource(PydanticBaseSettingsSource):
    """Reads a YAML file (if it exists) as a settings source."""

    def __init__(self, settings_cls: type[BaseSettings], yaml_file: Path):
        """Load `yaml_file` (if it exists) into the source's data.

        Args:
            settings_cls: The `BaseSettings` subclass this source feeds.
            yaml_file: Path to the YAML config file. Missing files are
                treated as empty config, not an error.
        """
        super().__init__(settings_cls)
        self._data: dict[str, Any] = {}
        if yaml_file.exists():
            self._data = yaml.safe_load(yaml_file.read_text()) or {}

    def get_field_value(self, field, field_name: str) -> tuple[Any, str, bool]:
        """Look up a single field's value, per `PydanticBaseSettingsSource`.

        Args:
            field: The pydantic field metadata (unused; required by the
                base class interface).
            field_name: Name of the top-level settings field to look up.

        Returns:
            A `(value, field_name, is_complex)` tuple, `is_complex` always
            False.
        """
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        """Return the full parsed YAML data as this source's settings dict."""
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
    provenance: ProvenanceConfig = Field(default_factory=ProvenanceConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Set the settings source precedence: init > env vars > YAML file.

        Args:
            settings_cls: The `BaseSettings` subclass being configured.
            init_settings: Source for values passed directly to `__init__`.
            env_settings: Source for `SHINOBI_*` environment variables.
            dotenv_settings: Unused; `.env` files are not supported.
            file_secret_settings: Unused; Docker/Kubernetes secret files
                are not supported.

        Returns:
            The ordered tuple of settings sources pydantic-settings should
            consult, highest precedence first.
        """
        yaml_source = _YamlFileSource(settings_cls, cls._config_file)
        return (init_settings, env_settings, yaml_source)

    @classmethod
    def load(cls, config_file: str | Path | None = None, **cli_overrides: Any) -> "AppConfig":
        """Build an `AppConfig`, layering defaults, config file, env, and overrides.

        Args:
            config_file: Path to a YAML config file. Defaults to
                `DEFAULT_CONFIG_FILE` (`~/.shinobi/config.yml`) if not given.
            **cli_overrides: Explicit values that take precedence over the
                config file and environment variables.

        Returns:
            A fully-resolved `AppConfig` instance.
        """
        cls._config_file = Path(config_file) if config_file else DEFAULT_CONFIG_FILE
        return cls(**cli_overrides)
