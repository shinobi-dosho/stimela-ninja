"""Naming and validating the launcher venv a `--remote` run activates.

`shinobi.offload.ssh` launches `ninja run` on a remote host and, today,
activates whatever `venv/` or `.venv/` it finds under `remote.path`
(`ssh._venv_activation`). `docs/design_remote_venv.md` proposes letting it
*provision* that environment instead. This module is the naming half of
that: it computes the identity of a launcher environment from its inputs,
and it reads back the one piece of evidence that such an environment was
fully built. **Nothing here provisions anything, and nothing here talks to
a remote host** -- every function is pure, and the ssh round-trips that
feed them belong to the caller.

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
# is absent -- is design step 4 and is deliberately not a value yet: a flag
# that accepts a word it cannot honour is worse than one that refuses it.
VENV_OFF = "off"
VENV_USE = "use"
VENV_MODES = (VENV_OFF, VENV_USE)


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

# The remote command whose single line of stdout `parse_platform_probe`
# reads. `python3` rather than `uv python find`: this has to work in `use`
# mode on a host with no uv, and `env_id` embeds the result, so it cannot be
# obtained by asking uv what it would build.
PLATFORM_PROBE = f"python3 -c {shlex.quote(_PROBE_PY)}"

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


def parse_platform_probe(stdout: str) -> PlatformTriple:
    """Read `PLATFORM_PROBE`'s output.

    Takes the last non-empty line rather than the whole of stdout: the probe
    runs under `bash -lc`, and a remote login shell that prints a banner or a
    module-system notice would otherwise turn a working host into an
    unparseable one.

    Raises:
        ValueError: If no line parses as a triple.
    """
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("platform probe produced no output")
    return PlatformTriple.parse(lines[-1])


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
