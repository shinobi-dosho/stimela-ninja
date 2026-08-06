"""Naming and validating the launcher venv a `--remote` run activates.

`shinobi.offload.ssh` launches `ninja run` on a remote host, and `--venv
sync` lets it *provision* the environment it launches into rather than
hoping someone left one at `venv/` (see `docs/design_remote_venv.md`).
This module is the half of that with no side effects: it names an
environment from its inputs, reads back the one piece of evidence that such
an environment was fully built, and builds -- as strings -- the commands
that do the work. **Nothing here talks to a remote host.** The round-trips
belong to `ssh.resolve_remote_venv`, which is where the ordering, the
failure handling and the ssh live.

That split is worth keeping. Every remote command in this feature is a
shell string assembled from paths, and the interesting bugs in it are
quoting, ordering and flags -- all of which can be asserted on a string,
by a test that needs no host at all.

Two ideas carry the whole design.

**`env_id` names inputs, never bytes.** It is a hash of the lock, the
pyproject, the extras/groups selection, the python constraint, the
provisioning mode and the remote platform (§4.2 of the design). Identical
inputs on identical platforms can still install different compiled
artifacts, so nothing may read an `env_id` as a claim about what is
installed -- that is what `venv_digest` is for, and even that is
version-parity only (`backends/venv.py:14-20`). What `env_id` buys is that
two different environments cannot land in one directory, and that an older
revision's environment is still sitting there under its own name.

**The sentinel is the only evidence of completion.** A directory named
`<env_id>` proves nothing: a `rename()` onto an *empty* directory succeeds
silently, so a crashed provision or a stray `mkdir` leaves a plausible-
looking path with nothing in it. Every existence question is therefore
asked of `.shinobi-env.json` *inside* the directory, written last, and
`read_sentinel` is the single place that asks it.

Three read outcomes, because two of them are not the same question:

- `ABSENT` -- no file, unparseable, or missing required keys. There is
  nothing here; provision.
- `FOREIGN` -- parses, but carries a `schema` this client does not know.
  A *newer* ninja owns this directory. Activation declines it (an old
  client must not guess at fields it has never seen), and provisioning
  refuses rather than overwriting it.
- `PRESENT` -- a sentinel this client understands. Still not usable on its
  own: `platform_matches` has to agree before activation, which is what
  keeps two hosts sharing one networked `remote.path` from activating each
  other's builds under a name asserting they match.

The platform triple deliberately describes the **host**, not the venv's
interpreter. It is `<machine>/<libc>/<host python X.Y>`, probed with a
plain `python3` and no `uv`, because `use` mode has to validate a sentinel
on a host that may have no `uv` at all -- and because `env_id` contains the
triple, so the triple cannot be computed by asking `uv` what it would build.
The venv's *own* interpreter version is a separate, informational
`venv_python` recorded at provisioning time. Keeping them apart matters:
uv can download an interpreter that is not the host's, so a triple claiming
to describe the venv's python would be wrong exactly when it was
interesting.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

# The sentinel schema this client writes and understands. Bump when a field
# changes meaning; a reader that does not know a schema treats the sentinel
# as FOREIGN rather than guessing (see the module docstring).
SCHEMA = 1

SENTINEL_NAME = ".shinobi-env.json"

# Everything this feature owns lives under one directory, so an operator
# clearing provisioned environments by hand has exactly one path to remove.
VENVS_SUBDIR = ".shinobi/venvs"

# The two ways a lock can be turned into an environment. Part of `env_id`
# because the same `requirements.txt` bytes mean different things to
# `uv sync` and `uv pip sync`.
MODE_UV_SYNC = "uv-sync"
MODE_UV_PIP_SYNC = "uv-pip-sync"
MODES = (MODE_UV_SYNC, MODE_UV_PIP_SYNC)

# What `--venv` asks for. `sync` -- provision the environment if its sentinel
# is absent -- provisions it.
VENV_OFF = "off"
VENV_USE = "use"
VENV_SYNC = "sync"
VENV_MODES = (VENV_OFF, VENV_USE, VENV_SYNC)


# ---------------------------------------------------------------------------
# Platform identity
# ---------------------------------------------------------------------------


# One statement per element, joined with `; ` at import time rather than
# written as one long line: this is Python source that will be read back by a
# human debugging a host whose probe misbehaved, and `python3 -c` gives no
# line numbers to help them.
_PROBE_PY = "; ".join(
    (
        "import platform as p, sys",
        "lv = p.libc_ver()",
        'libc = ("%s-%s" % lv) if lv[0] and lv[1] else "unknown"',
        'ver = "%d.%d" % sys.version_info[:2]',
        'print("%s/%s/%s" % (p.machine() or "unknown", libc, ver))',
    )
)

# Every line the probe means to be read carries one of these. A remote login
# shell prints banners, module-system notices and MOTDs, and those arrive
# interleaved with -- not merely before -- a multi-command probe's own
# output, so position is not something a parser can rely on.
_MARK_PLATFORM = "shinobi-platform:"
_MARK_UV = "shinobi-uv:"

# The remote command `parse_probe` reads. `python3` rather than
# `uv python find`: this has to work in `use` mode on a host with no uv, and
# `env_id` embeds the result, so it cannot be obtained by asking uv what it
# would build. The uv version rides along in the same round-trip -- `sync`
# needs to know uv is there *before* anything is copied to the host, and it
# is a sentinel field either way.
PLATFORM_PROBE = f"python3 -c {shlex.quote(_PROBE_PY)}"
PROBE_COMMAND = f"printf '{_MARK_PLATFORM}%s\\n' \"$({PLATFORM_PROBE})\"; printf '{_MARK_UV}%s\\n' \"$(uv --version 2>/dev/null || true)\""

# One field of the triple. Anchored and conservative: the triple is
# interpolated into a directory name via `env_id`'s hash, but it is also
# compared and printed, and a newline or a slash in a component would make
# two different triples render identically. `\Z` rather than `$`, which
# matches *before* a trailing newline and so would admit the one character
# most likely to arrive on the end of a probe's output.
_TRIPLE_FIELD = re.compile(r"\A[A-Za-z0-9._+-]+\Z")


@dataclass(frozen=True)
class PlatformTriple:
    """The host identity that participates in `env_id`.

    Attributes:
        machine: `platform.machine()` -- `x86_64`, `aarch64`, ...
        libc: `glibc-2.39`, or `unknown` where `platform.libc_ver()` says
            nothing (musl reports empty strings, and a musl host that read
            as `glibc-` would be claiming a match it does not have).
        python: The host's default `python3`, major.minor only. Patch
            releases do not change ABI compatibility, and including them
            would re-provision every environment on a distro point update.
    """

    machine: str
    libc: str
    python: str

    def __post_init__(self) -> None:
        for name, value in (("machine", self.machine), ("libc", self.libc), ("python", self.python)):
            if not _TRIPLE_FIELD.match(value):
                raise ValueError(f"platform triple {name} {value!r} is not a plain [A-Za-z0-9._+-] token")

    def __str__(self) -> str:
        return f"{self.machine}/{self.libc}/{self.python}"

    @classmethod
    def parse(cls, text: str) -> PlatformTriple:
        """Parse the `machine/libc/python` rendering back into a triple.

        Raises:
            ValueError: If `text` is not exactly three well-formed fields.
        """
        parts = text.strip().split("/")
        if len(parts) != 3:
            raise ValueError(f"platform triple must be 'machine/libc/python', got {text!r}")
        return cls(machine=parts[0], libc=parts[1], python=parts[2])


@dataclass(frozen=True)
class RemoteProbe:
    """What one round-trip establishes about the host before anything is
    copied to it.

    Attributes:
        platform: The triple that goes into `env_id`.
        uv_version: `uv --version`'s output, or None where uv is not on the
            remote PATH. None is not an error here -- `use` mode does not
            need uv at all, and only `sync` turns it into a refusal.
    """

    platform: PlatformTriple
    uv_version: str | None


def parse_probe(stdout: str) -> RemoteProbe:
    """Read `PROBE_COMMAND`'s output.

    Scans for marked lines rather than reading by position. A remote login
    shell's banners and module notices interleave with the probe's own
    output, so "the last line" stops being a reliable address as soon as the
    probe emits more than one thing.

    Raises:
        ValueError: If no line carries the platform marker, or it does not
            parse as a triple. A missing uv marker is not an error.
    """
    platform: PlatformTriple | None = None
    uv_version: str | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(_MARK_PLATFORM):
            platform = PlatformTriple.parse(line[len(_MARK_PLATFORM) :])
        elif line.startswith(_MARK_UV):
            uv_version = line[len(_MARK_UV) :].strip() or None
    if platform is None:
        raise ValueError(f"probe produced no {_MARK_PLATFORM!r} line; got: {stdout.strip()!r}")
    return RemoteProbe(platform=platform, uv_version=uv_version)


# ---------------------------------------------------------------------------
# env_id
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvInputs:
    """Everything `env_id` is a function of. See §4.2 of the design.

    `lock` and `pyproject` are raw **bytes**, hashed as they are. A
    whitespace-only edit therefore re-provisions -- accepted deliberately:
    a false re-provision costs time and disk and never costs correctness,
    the inverse mistake does, and both files are machine-written.

    Attributes:
        lock: `uv.lock` (or `requirements.txt`) bytes.
        pyproject: `pyproject.toml` bytes, or `b""` under
            `MODE_UV_PIP_SYNC`, which does not read one. `uv sync` requires
            it and it selects what gets installed, so it cannot be left out
            of the identity.
        extras: `--extra` selections.
        groups: `--group` selections.
        python_request: The `--python` constraint, or `""`.
        mode: One of `MODES`.
        platform: The remote host's triple.
    """

    lock: bytes
    pyproject: bytes
    extras: tuple[str, ...]
    groups: tuple[str, ...]
    python_request: str
    mode: str
    platform: PlatformTriple

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")


def _selection(inputs: EnvInputs) -> bytes:
    """The extras/groups selection, canonically ordered and tagged.

    Sorted so that `--extra a --extra b` and `--extra b --extra a` are one
    environment. Tagged (`extra=`/`group=`) and NUL-separated so that an
    extra and a group of the same name stay distinguishable -- `uv` treats
    them as different things, and an untagged join would let
    `--extra dev` collide with `--group dev`.
    """
    parts = [f"extra={e}" for e in sorted(inputs.extras)] + [f"group={g}" for g in sorted(inputs.groups)]
    return "\0".join(parts).encode()


def env_id(inputs: EnvInputs) -> str:
    """The content-addressed name of the environment `inputs` describe.

    16 hex characters of a sha256 over every field, NUL-separated. NUL
    rather than a printable separator because it cannot occur in any of the
    text fields, so no combination of values can be made to hash as another.
    """
    h = hashlib.sha256()
    for field in (
        inputs.lock,
        inputs.pyproject,
        _selection(inputs),
        inputs.python_request.encode(),
        inputs.mode.encode(),
        str(inputs.platform).encode(),
    ):
        h.update(field)
        h.update(b"\0")
    return h.hexdigest()[:16]


def venv_dir(remote_path: str, env_id_: str) -> str:
    """The directory an environment named `env_id_` lives in."""
    return f"{remote_path}/{VENVS_SUBDIR}/{env_id_}"


def sentinel_path(remote_path: str, env_id_: str) -> str:
    """The sentinel inside `venv_dir` -- the only thing worth testing for."""
    return f"{venv_dir(remote_path, env_id_)}/{SENTINEL_NAME}"


def staging_dir(remote_path: str, token: str) -> str:
    """Where an in-progress provision lives, before it earns its name.

    The `.partial-` prefix is the convention the snapshot writer already uses
    (`design_cache_tiers.md` §9), and the leading dot keeps it out of a glob
    over finished environments.
    """
    return f"{remote_path}/{VENVS_SUBDIR}/.partial-{token}"


# ---------------------------------------------------------------------------
# Provisioning: the scripts, built here and run by `offload.ssh`
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LockSource:
    """The local files a `sync` copies over and provisions from.

    Attributes:
        project_dir: What gets rsynced -- `uv sync` reads the lock *and* the
            pyproject beside it, so the pair travels together.
        lock: `uv.lock` or `requirements.txt`.
        pyproject: `pyproject.toml`, or None under `MODE_UV_PIP_SYNC`.
        mode: One of `MODES`, decided by which lock was found.
    """

    project_dir: Path
    lock: Path
    pyproject: Path | None
    mode: str

    def rel_paths(self) -> list[Path]:
        """The pair, relative to `project_dir`, for `sync_to_remote`."""
        paths = [self.lock] if self.pyproject is None else [self.lock, self.pyproject]
        return [p.relative_to(self.project_dir) for p in paths]

    def read(self) -> tuple[bytes, bytes]:
        """`(lock bytes, pyproject bytes)`, the two hashed inputs.

        An absent pyproject hashes as `b""` rather than being skipped, so
        `uv-pip-sync` and a hypothetical `uv-sync` over an empty pyproject
        stay distinguishable through the `mode` field either way.
        """
        return self.lock.read_bytes(), (self.pyproject.read_bytes() if self.pyproject else b"")


def discover_lock(start: Path) -> LockSource | None:
    """Find the lock governing `start`, walking up to the filesystem root.

    Walks up rather than looking only beside the target because a recipe
    lives in a repository, not in a directory of its own -- caracal2 keeps
    its pipelines under `src/` and its `uv.lock` at the root.

    A `uv.lock` **plus** the `pyproject.toml` beside it wins over a
    `requirements.txt` at the same level: it is the stronger statement, and
    `uv sync --frozen` over it cannot re-resolve. A `uv.lock` with no
    `pyproject.toml` beside it is not a project `uv sync` can build, so it is
    passed over rather than reported as something that will fail later.

    Returns None if nothing is found, which is not an error -- it is the
    normal state for `use` mode.
    """
    for directory in (start if start.is_dir() else start.parent, *(start if start.is_dir() else start.parent).parents):
        lock, pyproject = directory / "uv.lock", directory / "pyproject.toml"
        if lock.is_file() and pyproject.is_file():
            return LockSource(project_dir=directory, lock=lock, pyproject=pyproject, mode=MODE_UV_SYNC)
        requirements = directory / "requirements.txt"
        if requirements.is_file():
            return LockSource(project_dir=directory, lock=requirements, pyproject=None, mode=MODE_UV_PIP_SYNC)
    return None


def lock_source_for(path: Path) -> LockSource:
    """The `LockSource` a user named explicitly with `--venv-lock`.

    Raises:
        ValueError: If the file is neither a `uv.lock` with a `pyproject.toml`
            beside it nor a `requirements.txt`. Named explicitly, so an
            unusable one is refused rather than silently walked past the way
            `discover_lock` would.
    """
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"--venv-lock {path} does not exist")
    if path.name == "uv.lock":
        pyproject = path.parent / "pyproject.toml"
        if not pyproject.is_file():
            raise ValueError(f"--venv-lock {path} has no pyproject.toml beside it; `uv sync` needs both")
        return LockSource(project_dir=path.parent, lock=path, pyproject=pyproject, mode=MODE_UV_SYNC)
    if path.name == "requirements.txt":
        return LockSource(project_dir=path.parent, lock=path, pyproject=None, mode=MODE_UV_PIP_SYNC)
    raise ValueError(f"--venv-lock must name a uv.lock or a requirements.txt, got {path.name!r}")


def read_sentinel_command(path: str) -> str:
    """Read a sentinel, treating "not there" as empty rather than as failure.

    `cat` of a missing file is a non-zero exit and a message on stderr, and
    the overwhelmingly common case -- no environment provisioned yet -- is
    not an error worth either.
    """
    return f"cat {shlex.quote(path)} 2>/dev/null || true"


# Marks the one line of `provision_command`'s output that is data.
_MARK_VENV_PYTHON = "shinobi-venv-python:"


def provision_command(staging: str, mode: str) -> str:
    """Build the venv inside `staging`, and report its interpreter version.

    The two steps and their order are the load-bearing part, and the obvious
    one-step alternative is broken in a way that only shows up after
    publication. Verified on uv 0.11.21: letting `uv sync --frozen` create
    its own venv produces no `relocatable` marker in `pyvenv.cfg` and a
    console-script shim that hardcodes the absolute interpreter path -- even
    under `UV_VENV_RELOCATABLE=1`, and `uv sync` has no `--relocatable` flag
    at all. After the rename in `publish_command` such a script dies with
    `exec: /.../.venv/bin/python: not found`. `ninja` *is* a console script,
    so that is the launcher failing to launch. Creating the venv first with
    `uv venv --relocatable` and populating it with `uv sync --active` keeps
    the marker, emits a `dirname $0`-relative shim, and survives the rename.

    `--frozen` is not optional either: it forbids re-resolution, so the
    remote cannot drift to a version set the lock does not name.

    **`--no-install-project` is not an optimisation, it is a correctness
    fix**, and the bug it fixes is invisible until after publication. `uv
    sync` installs the *project* as an **editable** install -- a `.pth` file
    pointing at the directory it was run in. Here that directory is
    `.partial-<token>`, which is removed the moment provisioning finishes, so
    the published environment imports the project and gets
    `ModuleNotFoundError`. Verified end to end: the console script survives
    the rename (that is what `--relocatable` bought) and then dies importing
    the package it exists to run. Two further reasons the project has no
    business being installed here: its source is not among `env_id`'s inputs,
    so installing it would make the environment depend on bytes its own name
    does not cover; and nothing in this path needs it -- the launcher is
    `ninja`, which arrives as a *dependency* of the recipe repository, and
    the recipe file itself is rsynced separately and run by path.

    The consequence, which belongs in the user docs: a repository whose own
    console script is the launcher (caracal driving `caracal run`) does not
    get that script from a `sync`. That wants a flag, not a silent
    reintroduction of a path-dependent install.

    Nothing here passes `--no-dev` or selects groups. What a `sync` builds is
    deliberately what a plain `uv sync` builds in the recipe's own repository
    -- the environment its maintainers run -- rather than a subset ninja has
    decided is enough.
    """
    venv = f"{staging}/.venv"
    q_staging, q_venv = shlex.quote(staging), shlex.quote(venv)
    if mode == MODE_UV_PIP_SYNC:
        populate = f"uv pip sync --python {q_venv}/bin/python requirements.txt"
    else:
        populate = f"VIRTUAL_ENV={q_venv} uv sync --frozen --active --no-install-project"
    return "; ".join(
        (
            "set -e",
            f"cd {q_staging}",
            f"uv venv --relocatable {q_venv}",
            populate,
            f"printf '{_MARK_VENV_PYTHON}%s\\n' \"$({q_venv}/bin/python -c 'import platform; print(platform.python_version())')\"",
        )
    )


def parse_provision_output(stdout: str) -> str | None:
    """The venv interpreter's `X.Y.Z`, or None if the marker never arrived.

    None rather than an exception: this is one informational sentinel field,
    and a host whose login shell ate the line has still built a working
    environment.
    """
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith(_MARK_VENV_PYTHON):
            return line[len(_MARK_VENV_PYTHON) :].strip() or None
    return None


# What `publish_command` prints, and what each outcome authorises.
PUBLISHED = "published"
COLLIDED = "collided"

_HEREDOC = "SHINOBI_SENTINEL_EOF"

_PUBLISH_PY = "\n".join(
    (
        "import os, sys",
        "try:",
        "    os.rename(sys.argv[1], sys.argv[2])",
        f"    print({PUBLISHED!r})",
        "except OSError:",
        f"    print({COLLIDED!r})",
    )
)


def publish_command(staging: str, final: str, sentinel_json: str) -> str:
    """Write the sentinel last, then rename the venv onto its real name.

    Two things here are not interchangeable with the obvious shell.

    **`os.rename`, not `mv`.** A plain `mv src dst` where `dst` is an
    existing directory moves `src` *inside* it -- verified -- so a collision
    would silently produce `<env_id>/.venv` and every later launch would find
    no sentinel at `<env_id>`. `mv -T` has rename semantics but is GNU
    coreutils only. `os.rename` is exactly rename(2), and python3 is already
    required by the probe.

    **A non-zero exit is not how a collision is reported.** rename(2) onto a
    *non-empty* directory fails with ENOTEMPTY, which is the good case -- a
    concurrent launch got there first with, by construction, an equivalent
    environment. It is reported as `COLLIDED` on stdout so the caller can
    re-read the sentinel and adopt it, rather than as a failure the launch
    has to distinguish from a real one.

    The sentinel goes in via a quoted heredoc, so nothing in the JSON is
    expanded by the remote shell. Newline-separated rather than `; `-joined
    for that reason alone: a heredoc's terminator has to end a line, and a
    `; ` after it would be a stray leading semicolon on the next.
    """
    if _HEREDOC in sentinel_json:
        raise ValueError(f"sentinel JSON contains the heredoc terminator {_HEREDOC!r}")
    if not sentinel_json.endswith("\n"):
        sentinel_json += "\n"
    venv = f"{staging}/.venv"
    return "\n".join(
        (
            "set -e",
            f"mkdir -p {shlex.quote(str(PurePosixPath(final).parent))}",
            f"cat > {shlex.quote(f'{venv}/{SENTINEL_NAME}')} <<'{_HEREDOC}'",
            sentinel_json + _HEREDOC,
            f"python3 -c {shlex.quote(_PUBLISH_PY)} {shlex.quote(venv)} {shlex.quote(final)}",
        )
    )


def cleanup_command(staging: str) -> str:
    """Remove a staging directory, on success or on any failure.

    Scoped to a `.partial-<token>` path this process generated, and never to
    anything under a final name -- Invariant 8 is that a pre-existing
    directory at a final path is never deleted by this feature.
    """
    if "/.partial-" not in staging:
        raise ValueError(f"refusing to remove {staging!r}: not a .partial- staging directory")
    return f"rm -rf {shlex.quote(staging)}"


# ---------------------------------------------------------------------------
# The sentinel
# ---------------------------------------------------------------------------


class SentinelStatus(Enum):
    """Why a sentinel read did or did not yield a usable environment.

    `ABSENT` and `FOREIGN` are deliberately not merged: "nothing is here,
    build it" and "a newer ninja built this, keep your hands off it" lead to
    opposite actions under `sync`.
    """

    ABSENT = "absent"
    FOREIGN = "foreign"
    PRESENT = "present"


@dataclass(frozen=True)
class Sentinel:
    """A parsed, known-schema `.shinobi-env.json`.

    The hash fields duplicate what `env_id` already covers. That is the
    point: `env_id` is 16 characters that cannot be read backwards, and when
    a run has to explain *why* it is not reusing an environment, the
    sentinel is what it reads the answer out of.

    Attributes:
        env_id: The name of the directory this sits in.
        lock_sha256: Full digest of the lock bytes.
        pyproject_sha256: Full digest of the pyproject bytes.
        extras: The `--extra` selection, as provisioned.
        groups: The `--group` selection, as provisioned.
        python_request: The `--python` constraint, or `""`.
        mode: One of `MODES`.
        platform_triple: The host this was built on, rendered.
        venv_digest: `backends/venv.py`'s version-parity digest of what
            landed, or None where it could not be taken. Never fabricated:
            an honest null, matching `venv_digest`'s own contract.
        venv_python: The venv interpreter's `X.Y.Z`, which is *not*
            `platform_triple`'s python (uv may install its own).
        created: ISO-8601, informational only. Nothing branches on it --
            an age policy would need a clock two hosts agree on.
        uv_version: The uv that built this, for diagnosing a build that
            behaved differently than the same inputs did elsewhere.
    """

    env_id: str
    lock_sha256: str
    pyproject_sha256: str
    extras: tuple[str, ...]
    groups: tuple[str, ...]
    python_request: str
    mode: str
    platform_triple: str
    venv_digest: str | None = None
    venv_python: str | None = None
    created: str = ""
    uv_version: str | None = None

    def to_json(self) -> str:
        """Render for writing. `schema` first, sorted keys, trailing newline.

        Sorted so two provisions of the same environment produce identical
        bytes, which makes a hand-diff of two hosts' sentinels readable.
        """
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "env_id": self.env_id,
            "lock_sha256": self.lock_sha256,
            "pyproject_sha256": self.pyproject_sha256,
            "extras": list(self.extras),
            "groups": list(self.groups),
            "python_request": self.python_request,
            "mode": self.mode,
            "platform_triple": self.platform_triple,
            "venv_digest": self.venv_digest,
            "venv_python": self.venv_python,
            "created": self.created,
            "uv_version": self.uv_version,
        }
        return json.dumps(payload, sort_keys=True, indent=1) + "\n"


@dataclass(frozen=True)
class SentinelRead:
    """The outcome of `read_sentinel`.

    Attributes:
        status: See `SentinelStatus`.
        sentinel: The parsed sentinel, iff `status is PRESENT`.
        detail: A human-readable reason, always populated for the two
            non-PRESENT statuses. This ends up in the message a `sync`
            refusal prints, and "there is a directory but no sentinel" and
            "the JSON is truncated" send an operator to different places.
    """

    status: SentinelStatus
    sentinel: Sentinel | None = None
    detail: str = ""


_REQUIRED = ("env_id", "lock_sha256", "pyproject_sha256", "mode", "platform_triple")


def read_sentinel(text: str | None) -> SentinelRead:
    """Parse sentinel `text`, or None where the file does not exist.

    Never raises. Every malformed shape is a read outcome, because this runs
    against whatever is on a remote filesystem -- a truncated write, a
    half-rsynced file, an operator's stray edit -- and a traceback there
    would abort a launch over a file whose whole purpose is to be optional.
    """
    if text is None or not text.strip():
        return SentinelRead(SentinelStatus.ABSENT, detail="no sentinel file")

    try:
        payload = json.loads(text)
    except ValueError as exc:
        return SentinelRead(SentinelStatus.ABSENT, detail=f"sentinel is not valid JSON ({exc})")
    if not isinstance(payload, dict):
        return SentinelRead(SentinelStatus.ABSENT, detail="sentinel is not a JSON object")

    # Schema is checked before anything else is read. A future sentinel's
    # other fields are not this client's to interpret, even the ones whose
    # names it recognises.
    schema = payload.get("schema")
    if schema != SCHEMA:
        return SentinelRead(
            SentinelStatus.FOREIGN,
            detail=f"sentinel schema {schema!r}, this ninja understands {SCHEMA}",
        )

    missing = [key for key in _REQUIRED if not isinstance(payload.get(key), str) or not payload[key]]
    if missing:
        return SentinelRead(
            SentinelStatus.ABSENT,
            detail=f"sentinel is missing required field(s): {', '.join(missing)}",
        )
    if payload["mode"] not in MODES:
        return SentinelRead(SentinelStatus.ABSENT, detail=f"sentinel mode {payload['mode']!r} is not one of {MODES}")

    def _strs(key: str) -> tuple[str, ...]:
        value = payload.get(key)
        if not isinstance(value, list):
            return ()
        return tuple(v for v in value if isinstance(v, str))

    def _opt(key: str) -> str | None:
        value = payload.get(key)
        return value if isinstance(value, str) else None

    return SentinelRead(
        SentinelStatus.PRESENT,
        sentinel=Sentinel(
            env_id=payload["env_id"],
            lock_sha256=payload["lock_sha256"],
            pyproject_sha256=payload["pyproject_sha256"],
            extras=_strs("extras"),
            groups=_strs("groups"),
            python_request=payload.get("python_request") if isinstance(payload.get("python_request"), str) else "",
            mode=payload["mode"],
            platform_triple=payload["platform_triple"],
            venv_digest=_opt("venv_digest"),
            venv_python=_opt("venv_python"),
            created=payload.get("created") if isinstance(payload.get("created"), str) else "",
            uv_version=_opt("uv_version"),
        ),
    )


def platform_matches(sentinel: Sentinel, platform: PlatformTriple) -> bool:
    """Whether `sentinel` describes an environment built for `platform`.

    Invariant 3 of the design: activation refuses a sentinel whose triple
    does not match the current host. Compared as rendered text -- a triple
    that will not even parse is a mismatch, not an exception, since the only
    action either way is to decline the environment.
    """
    return sentinel.platform_triple == str(platform)


def sha256_hex(data: bytes) -> str:
    """`sha256:`-prefixed digest, the form the sentinel records."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


# ---------------------------------------------------------------------------
# `--venv` and the boolean pair it replaces
# ---------------------------------------------------------------------------


def resolve_venv_mode(venv: str | None, add_venv: bool | None) -> tuple[str, str | None]:
    """Reconcile `--venv` with the deprecated `--add-venv/--no-add-venv`.

    Both arguments are tri-state, and `None` means *not given* -- which is
    the only way to tell "the caller asked for `use`" from "nobody asked for
    anything", and therefore the only way to notice a caller asking for two
    different things at once.

    Returns the resolved mode plus a deprecation notice for the caller to
    emit, or None. Returning the notice rather than warning here keeps this
    pure: the CLI wants it on stderr in click's voice, a library caller
    wants a `DeprecationWarning`, and a test wants neither.

    Raises:
        ValueError: If both flags are given and disagree, or if `venv` is
            not one of `VENV_MODES`. Disagreement is refused rather than
            resolved by precedence -- whichever way it went, half of the
            people who wrote it would get the other environment, silently,
            on a host they cannot see.
    """
    if venv is not None and venv not in VENV_MODES:
        raise ValueError(f"--venv must be one of {VENV_MODES}, got {venv!r}")

    if add_venv is None:
        return (venv or VENV_USE), None

    alias = VENV_USE if add_venv else VENV_OFF
    legacy_flag = "--add-venv" if add_venv else "--no-add-venv"
    if venv is not None and venv != alias:
        raise ValueError(f"--venv {venv} and {legacy_flag} ask for different things; {legacy_flag} is the deprecated spelling of --venv {alias}")
    return alias, f"{legacy_flag} is deprecated -- use --venv {alias}"
