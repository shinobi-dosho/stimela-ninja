import dataclasses
import json
import subprocess

import pytest

from shinobi.offload.remote_venv import (
    MODE_UV_PIP_SYNC,
    MODE_UV_SYNC,
    PLATFORM_PROBE,
    SCHEMA,
    VENV_OFF,
    VENV_USE,
    EnvInputs,
    PlatformTriple,
    Sentinel,
    SentinelStatus,
    env_id,
    parse_platform_probe,
    platform_matches,
    read_sentinel,
    resolve_venv_mode,
    sentinel_path,
    sha256_hex,
    venv_dir,
)

TRIPLE = PlatformTriple(machine="x86_64", libc="glibc-2.39", python="3.11")

BASE = EnvInputs(
    lock=b"# uv.lock\n",
    pyproject=b"[project]\nname = 'r'\n",
    extras=(),
    groups=(),
    python_request="",
    mode=MODE_UV_SYNC,
    platform=TRIPLE,
)


# -- PlatformTriple --


def test_triple_renders_and_parses_round_trip():
    assert str(TRIPLE) == "x86_64/glibc-2.39/3.11"
    assert PlatformTriple.parse("x86_64/glibc-2.39/3.11") == TRIPLE


@pytest.mark.parametrize("text", ["x86_64/glibc-2.39", "x86_64/glibc-2.39/3.11/extra", "", "x86_64"])
def test_triple_parse_rejects_wrong_field_count(text):
    with pytest.raises(ValueError, match="machine/libc/python"):
        PlatformTriple.parse(text)


@pytest.mark.parametrize("bad", ["x86 64", "x86_64\n", "a/b", ""])
def test_triple_rejects_fields_that_would_not_round_trip(bad):
    """A space, a newline or an embedded slash would let two different
    triples render to one string -- and the rendering is what `env_id`
    hashes and what `platform_matches` compares."""
    with pytest.raises(ValueError, match="plain"):
        PlatformTriple(machine=bad, libc="glibc-2.39", python="3.11")


# -- the platform probe --


def test_platform_probe_runs_and_parses_on_this_host():
    """The probe is a string handed to a remote shell, so nothing else in the
    suite would notice it going stale (a renamed `platform` attribute, a
    quoting slip). Run it locally for real."""
    proc = subprocess.run(["bash", "-lc", PLATFORM_PROBE], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    triple = parse_platform_probe(proc.stdout)
    assert triple.machine and triple.python
    assert triple.python.count(".") == 1  # major.minor only, no patch


def test_platform_probe_uses_no_uv():
    """`use` mode validates a sentinel on hosts that may have no uv, and
    `env_id` embeds the triple -- so it cannot be obtained from uv."""
    assert "uv " not in PLATFORM_PROBE


def test_parse_platform_probe_ignores_login_shell_noise():
    """`bash -lc` runs the login profile, and plenty of clusters print a
    banner or a module notice before the command's own output."""
    noisy = "Welcome to cluster login02\nLoading modules...\nx86_64/glibc-2.39/3.11\n"
    assert parse_platform_probe(noisy) == TRIPLE


def test_parse_platform_probe_rejects_empty_output():
    with pytest.raises(ValueError, match="no output"):
        parse_platform_probe("   \n\n")


# -- env_id --


def test_env_id_is_stable_and_short():
    assert env_id(BASE) == env_id(dataclasses.replace(BASE))
    assert len(env_id(BASE)) == 16
    assert env_id(BASE).isalnum()


@pytest.mark.parametrize(
    "field,value",
    [
        ("lock", b"# uv.lock -- different\n"),
        ("pyproject", b"[project]\nname = 'other'\n"),
        ("extras", ("cuda",)),
        ("groups", ("dev",)),
        ("python_request", "3.12"),
        ("mode", MODE_UV_PIP_SYNC),
        ("platform", PlatformTriple(machine="aarch64", libc="glibc-2.39", python="3.11")),
    ],
)
def test_every_input_moves_the_env_id(field, value):
    """§4.2 lists six inputs; an input that does not move the id is an input
    two different environments can collide on."""
    assert env_id(dataclasses.replace(BASE, **{field: value})) != env_id(BASE)


def test_env_id_ignores_extras_ordering():
    a = dataclasses.replace(BASE, extras=("cuda", "viz"))
    b = dataclasses.replace(BASE, extras=("viz", "cuda"))
    assert env_id(a) == env_id(b)


def test_env_id_distinguishes_an_extra_from_a_group_of_the_same_name():
    """uv treats them as different selections, so an untagged join would let
    `--extra dev` and `--group dev` share one directory."""
    assert env_id(dataclasses.replace(BASE, extras=("dev",))) != env_id(dataclasses.replace(BASE, groups=("dev",)))


def test_env_id_separator_cannot_be_forged_across_fields():
    """The fields are NUL-separated, so no run of text in one field can be
    made to hash as a different split across two."""
    a = dataclasses.replace(BASE, python_request="3.12", mode=MODE_UV_SYNC)
    b = dataclasses.replace(BASE, python_request="3.12" + MODE_UV_SYNC, mode=MODE_UV_SYNC)
    assert env_id(a) != env_id(b)


def test_platform_is_part_of_the_id_so_two_hosts_cannot_collide():
    """Invariant 2: two hosts sharing one networked `remote.path` compute
    different ids and cannot activate each other's builds."""
    other_host = dataclasses.replace(BASE, platform=PlatformTriple("aarch64", "glibc-2.31", "3.11"))
    assert env_id(other_host) != env_id(BASE)


def test_env_inputs_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        dataclasses.replace(BASE, mode="pip-install")


# -- paths --


def test_paths_put_the_sentinel_inside_the_directory():
    """The whole point: existence is asked of a file inside the venv, never
    of the venv directory, which can exist empty."""
    eid = env_id(BASE)
    assert venv_dir("/scratch/run1", eid) == f"/scratch/run1/.shinobi/venvs/{eid}"
    assert sentinel_path("/scratch/run1", eid).startswith(venv_dir("/scratch/run1", eid) + "/")
    assert sentinel_path("/scratch/run1", eid).endswith("/.shinobi-env.json")


# -- the sentinel --


def _sentinel(**overrides) -> Sentinel:
    base = Sentinel(
        env_id=env_id(BASE),
        lock_sha256=sha256_hex(BASE.lock),
        pyproject_sha256=sha256_hex(BASE.pyproject),
        extras=(),
        groups=(),
        python_request="",
        mode=MODE_UV_SYNC,
        platform_triple=str(TRIPLE),
        venv_digest="sha256:abc",
        venv_python="3.11.9",
        created="2026-08-06T00:00:00Z",
        uv_version="0.11.21",
    )
    return dataclasses.replace(base, **overrides)


def test_sentinel_round_trips_through_json():
    original = _sentinel()
    read = read_sentinel(original.to_json())
    assert read.status is SentinelStatus.PRESENT
    assert read.sentinel == original


def test_sentinel_writes_its_schema():
    assert json.loads(_sentinel().to_json())["schema"] == SCHEMA


def test_sentinel_json_is_byte_stable():
    """Two provisions of one environment should produce identical bytes, so
    diffing two hosts' sentinels shows real differences only."""
    assert _sentinel().to_json() == _sentinel().to_json()


@pytest.mark.parametrize(
    "text,expected_detail",
    [
        (None, "no sentinel file"),
        ("", "no sentinel file"),
        ("   \n", "no sentinel file"),
        ("{not json", "not valid JSON"),
        ('["a", "b"]', "not a JSON object"),
    ],
)
def test_missing_or_unreadable_reads_as_absent(text, expected_detail):
    read = read_sentinel(text)
    assert read.status is SentinelStatus.ABSENT
    assert read.sentinel is None
    assert expected_detail in read.detail


def test_an_empty_directory_reads_as_absent():
    """The hole this closes: `rename()` onto an *empty* directory succeeds
    silently, so a crashed provision leaves a plausible path. Reading the
    directory is what `read_sentinel(None)` models."""
    assert read_sentinel(None).status is SentinelStatus.ABSENT


@pytest.mark.parametrize("schema", [2, 0, "1", None])
def test_an_unknown_schema_is_foreign_not_absent(schema):
    """A newer ninja owns that directory. `use` declines it; `sync` refuses
    rather than provisioning over it. Merging this into ABSENT would make
    `sync` overwrite a newer client's environment."""
    payload = json.loads(_sentinel().to_json())
    payload["schema"] = schema
    read = read_sentinel(json.dumps(payload))
    assert read.status is SentinelStatus.FOREIGN
    assert read.sentinel is None
    assert str(SCHEMA) in read.detail


@pytest.mark.parametrize("key", ["env_id", "lock_sha256", "pyproject_sha256", "mode", "platform_triple"])
def test_a_sentinel_missing_a_required_field_reads_as_absent(key):
    payload = json.loads(_sentinel().to_json())
    del payload[key]
    read = read_sentinel(json.dumps(payload))
    assert read.status is SentinelStatus.ABSENT
    assert key in read.detail


def test_a_sentinel_with_an_unknown_mode_reads_as_absent():
    payload = json.loads(_sentinel().to_json())
    payload["mode"] = "pip-install"
    assert read_sentinel(json.dumps(payload)).status is SentinelStatus.ABSENT


def test_optional_fields_degrade_to_none_rather_than_failing():
    """`venv_digest` is allowed to be null -- `backends/venv.py` returns an
    honest null rather than a fabricated digest, and the sentinel carries
    that through instead of turning it into an unreadable file."""
    payload = json.loads(_sentinel().to_json())
    for key in ("venv_digest", "venv_python", "uv_version"):
        payload[key] = None
    payload["extras"] = "not-a-list"
    read = read_sentinel(json.dumps(payload))
    assert read.status is SentinelStatus.PRESENT
    assert read.sentinel.venv_digest is None
    assert read.sentinel.extras == ()


def test_read_sentinel_never_raises_on_arbitrary_bytes():
    """It reads whatever is on a remote filesystem. A traceback here would
    abort a launch over a file whose whole purpose is to be optional."""
    for text in ("\x00\x01", "null", "12", '{"schema": 1}', '{"schema": 1, "env_id": 5}'):
        assert read_sentinel(text).status in (SentinelStatus.ABSENT, SentinelStatus.FOREIGN)


# -- platform_matches --


def test_platform_matches_only_the_host_it_was_built_on():
    assert platform_matches(_sentinel(), TRIPLE)
    assert not platform_matches(_sentinel(), PlatformTriple("aarch64", "glibc-2.39", "3.11"))
    assert not platform_matches(_sentinel(), PlatformTriple("x86_64", "glibc-2.31", "3.11"))
    assert not platform_matches(_sentinel(), PlatformTriple("x86_64", "glibc-2.39", "3.12"))


def test_platform_matches_treats_an_unparseable_triple_as_a_mismatch():
    assert not platform_matches(_sentinel(platform_triple="whatever"), TRIPLE)


# -- resolve_venv_mode --


def test_nothing_asked_for_means_use():
    assert resolve_venv_mode(None, None) == (VENV_USE, None)


@pytest.mark.parametrize("mode", [VENV_OFF, VENV_USE])
def test_venv_alone_is_taken_at_its_word(mode):
    assert resolve_venv_mode(mode, None) == (mode, None)


@pytest.mark.parametrize("add_venv,expected,flag", [(True, VENV_USE, "--add-venv"), (False, VENV_OFF, "--no-add-venv")])
def test_the_deprecated_pair_maps_and_reports_itself(add_venv, expected, flag):
    mode, notice = resolve_venv_mode(None, add_venv)
    assert mode == expected
    assert flag in notice
    assert f"--venv {expected}" in notice


@pytest.mark.parametrize("add_venv,agreeing", [(True, VENV_USE), (False, VENV_OFF)])
def test_both_spellings_agreeing_is_not_an_error(add_venv, agreeing):
    mode, notice = resolve_venv_mode(agreeing, add_venv)
    assert mode == agreeing
    assert notice  # still deprecated, still says so


@pytest.mark.parametrize("venv,add_venv", [(VENV_OFF, True), (VENV_USE, False)])
def test_the_two_spellings_disagreeing_is_refused(venv, add_venv):
    """Not resolved by precedence. Either precedence rule silently gives
    half the people who write it the environment they did not ask for."""
    with pytest.raises(ValueError, match="different things"):
        resolve_venv_mode(venv, add_venv)


@pytest.mark.parametrize("bad", ["sync", "on", "", "USE"])
def test_an_unknown_venv_mode_is_refused(bad):
    """`sync` included deliberately: it is design step 4, and until it
    provisions, accepting the word would mean doing something else."""
    with pytest.raises(ValueError, match="--venv must be one of"):
        resolve_venv_mode(bad, None)
