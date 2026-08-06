import dataclasses
import json
import subprocess

import pytest

from shinobi.offload.remote_venv import (
    MODE_UV_PIP_SYNC,
    MODE_UV_SYNC,
    PLATFORM_PROBE,
    PROBE_COMMAND,
    SCHEMA,
    VENV_OFF,
    VENV_SYNC,
    VENV_USE,
    EnvInputs,
    PlatformTriple,
    Sentinel,
    SentinelStatus,
    cleanup_command,
    discover_lock,
    env_id,
    lock_source_for,
    parse_probe,
    parse_provision_output,
    provision_command,
    publish_command,
    platform_matches,
    read_sentinel,
    resolve_venv_mode,
    sentinel_path,
    sha256_hex,
    staging_dir,
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


def test_probe_runs_and_parses_on_this_host():
    """The probe is a string handed to a remote shell, so nothing else in the
    suite would notice it going stale (a renamed `platform` attribute, a
    quoting slip). Run it locally for real."""
    proc = subprocess.run(["bash", "-lc", PROBE_COMMAND], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    probe = parse_probe(proc.stdout)
    assert probe.platform.machine and probe.platform.python
    assert probe.platform.python.count(".") == 1  # major.minor only, no patch


def test_the_platform_half_of_the_probe_uses_no_uv():
    """`use` mode validates a sentinel on hosts that may have no uv, and
    `env_id` embeds the triple -- so it cannot be obtained from uv."""
    assert "uv " not in PLATFORM_PROBE


def test_probe_reports_a_missing_uv_as_none_not_a_failure():
    """`use` needs no uv at all. Only `sync` turns its absence into a
    refusal, and it does that itself."""
    probe = parse_probe("shinobi-platform:x86_64/glibc-2.39/3.11\nshinobi-uv:\n")
    assert probe.platform == TRIPLE
    assert probe.uv_version is None


def test_probe_survives_a_host_with_no_uv_for_real():
    """Not a hypothetical: `uv --version` on a host without uv writes to
    stderr and exits non-zero, and the probe has to keep going -- `sync`
    turns that into its own refusal, and `use` ignores it entirely.

    `bash -c`, not `-lc` as the real call uses: the login profile is exactly
    what puts `~/.local/bin/uv` back on PATH, so a login shell cannot be made
    to stand in for a host that has no uv."""
    proc = subprocess.run(["bash", "-c", PROBE_COMMAND], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stderr
    assert parse_probe(proc.stdout).uv_version is None


def test_parse_probe_ignores_login_shell_noise():
    """`bash -lc` runs the login profile, and plenty of clusters print a
    banner or a module notice. Marked lines rather than positional ones,
    because that noise interleaves with a multi-command probe's output
    rather than politely preceding all of it."""
    noisy = "Welcome to cluster login02\nshinobi-platform:x86_64/glibc-2.39/3.11\nLoading modules...\nshinobi-uv:uv 0.11.21\n"
    probe = parse_probe(noisy)
    assert probe.platform == TRIPLE
    assert probe.uv_version == "uv 0.11.21"


def test_parse_probe_rejects_output_with_no_platform_line():
    with pytest.raises(ValueError, match="shinobi-platform"):
        parse_probe("   \nshinobi-uv:uv 0.11.21\n")


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


@pytest.mark.parametrize("bad", ["on", "", "USE", "Sync"])
def test_an_unknown_venv_mode_is_refused(bad):
    with pytest.raises(ValueError, match="--venv must be one of"):
        resolve_venv_mode(bad, None)


def test_sync_is_a_mode_now():
    assert resolve_venv_mode(VENV_SYNC, None) == (VENV_SYNC, None)


@pytest.mark.parametrize("add_venv", [True, False])
def test_sync_disagrees_with_either_deprecated_spelling(add_venv):
    """`--add-venv` was only ever able to say "activate something". Neither
    of its forms is a way of asking to *build* one."""
    with pytest.raises(ValueError, match="different things"):
        resolve_venv_mode(VENV_SYNC, add_venv)


# -- lock discovery --


def _project(root, *, lock=True, pyproject=True, requirements=False):
    if lock:
        (root / "uv.lock").write_text("# lock\n")
    if pyproject:
        (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    if requirements:
        (root / "requirements.txt").write_text("click\n")
    return root


def test_discover_lock_walks_up_from_the_target(tmp_path):
    """A recipe lives in a repository, not in a directory of its own --
    caracal2 keeps its pipelines under `src/` and its uv.lock at the root."""
    _project(tmp_path)
    deep = tmp_path / "src" / "pipelines"
    deep.mkdir(parents=True)
    target = deep / "recipe.py"
    target.write_text("")
    source = discover_lock(target)
    assert source.project_dir == tmp_path
    assert source.lock == tmp_path / "uv.lock"
    assert source.mode == MODE_UV_SYNC


def test_discover_lock_prefers_a_uv_lock_to_a_requirements_txt(tmp_path):
    """`uv sync --frozen` over a uv.lock cannot re-resolve; `uv pip sync`
    over a requirements.txt is the weaker statement."""
    _project(tmp_path, requirements=True)
    assert discover_lock(tmp_path).mode == MODE_UV_SYNC


def test_discover_lock_passes_over_a_uv_lock_with_no_pyproject(tmp_path):
    """Not a project `uv sync` can build, so it is walked past rather than
    reported as something that will fail several minutes later."""
    _project(tmp_path, pyproject=False, requirements=True)
    source = discover_lock(tmp_path)
    assert source.mode == MODE_UV_PIP_SYNC
    assert source.pyproject is None


def test_discover_lock_finds_nothing_without_erroring(tmp_path):
    """The normal state for `use` mode, and not a failure."""
    assert discover_lock(tmp_path / "nothing-here.py") is None


def test_lock_source_reads_both_halves(tmp_path):
    _project(tmp_path)
    lock, pyproject = discover_lock(tmp_path).read()
    assert lock == b"# lock\n"
    assert pyproject.startswith(b"[project]")


def test_a_requirements_source_hashes_an_empty_pyproject(tmp_path):
    """Absent rather than skipped, so `mode` stays the thing that
    distinguishes the two kinds of environment."""
    _project(tmp_path, lock=False, pyproject=False, requirements=True)
    assert discover_lock(tmp_path).read()[1] == b""


def test_lock_source_for_accepts_a_named_uv_lock(tmp_path):
    _project(tmp_path)
    assert lock_source_for(tmp_path / "uv.lock").mode == MODE_UV_SYNC


def test_lock_source_for_refuses_a_uv_lock_with_no_pyproject(tmp_path):
    """Named explicitly, so it is refused rather than silently walked past
    the way `discover_lock` would."""
    _project(tmp_path, pyproject=False)
    with pytest.raises(ValueError, match="no pyproject.toml beside it"):
        lock_source_for(tmp_path / "uv.lock")


def test_lock_source_for_refuses_a_file_that_is_neither(tmp_path):
    other = tmp_path / "environment.yml"
    other.write_text("")
    with pytest.raises(ValueError, match="uv.lock or a requirements.txt"):
        lock_source_for(other)


# -- the provisioning scripts --


def test_provision_creates_the_venv_before_populating_it():
    """The order is the whole mechanism: `uv sync` creating its own venv
    produces a console-script shim with an absolute interpreter path, which
    dies after the rename. `uv sync` has no --relocatable of its own."""
    script = provision_command("/p/.shinobi/venvs/.partial-abc", MODE_UV_SYNC)
    assert script.index("uv venv --relocatable") < script.index("uv sync")


def test_provision_forbids_re_resolution():
    """Without --frozen the remote can drift to a version set the lock does
    not name, under an env_id asserting it did not."""
    assert "--frozen" in provision_command("/p/.partial-abc", MODE_UV_SYNC)


def test_provision_does_not_install_the_project():
    """`uv sync` installs the project *editable*, pointing at the staging
    directory this feature deletes -- so the published environment would
    import it and get ModuleNotFoundError. Verified end to end."""
    assert "--no-install-project" in provision_command("/p/.partial-abc", MODE_UV_SYNC)


def test_provision_uses_pip_sync_for_a_requirements_txt():
    script = provision_command("/p/.partial-abc", MODE_UV_PIP_SYNC)
    assert "uv pip sync" in script and "requirements.txt" in script
    assert "--frozen" not in script  # uv pip sync has no such flag


def test_provision_reports_the_venv_interpreter():
    assert parse_provision_output("noise\nshinobi-venv-python:3.11.9\n") == "3.11.9"


def test_a_missing_interpreter_line_is_none_not_a_failure():
    """One informational sentinel field. A host whose login shell ate the
    line has still built a working environment."""
    assert parse_provision_output("built it\n") is None


def test_publish_renames_rather_than_moving():
    """A plain `mv src dst` onto an existing directory moves src *inside*
    it -- verified -- so a collision would silently produce
    `<env_id>/.venv` and every later launch would find no sentinel."""
    script = publish_command("/p/.partial-abc", "/p/.shinobi/venvs/deadbeef", "{}\n")
    assert "os.rename" in script
    assert "mv " not in script


def test_publish_writes_the_sentinel_before_the_rename():
    """Written last inside the staging venv, so it becomes visible at the
    final path in the same instant the directory does."""
    script = publish_command("/p/.partial-abc", "/p/final", '{"schema": 1}\n')
    assert script.index(".shinobi-env.json") < script.index("os.rename")


def test_publish_refuses_a_sentinel_that_would_break_out_of_the_heredoc():
    with pytest.raises(ValueError, match="heredoc terminator"):
        publish_command("/p/.partial-abc", "/p/final", '{"x": "SHINOBI_SENTINEL_EOF"}')


def test_cleanup_refuses_anything_that_is_not_a_staging_directory():
    """Invariant 8: a pre-existing directory at a final path is never
    deleted by this feature. `rm -rf` built from a variable deserves a
    guard that does not depend on the caller getting it right."""
    with pytest.raises(ValueError, match="not a .partial- staging directory"):
        cleanup_command("/scratch/run1/.shinobi/venvs/deadbeef")


def test_cleanup_accepts_a_real_staging_directory():
    assert "rm -rf" in cleanup_command(staging_dir("/scratch/run1", "abc123"))


def test_staging_is_hidden_and_marked_partial():
    """`.partial-` is the convention the snapshot writer already uses, and
    the leading dot keeps it out of a glob over finished environments."""
    assert "/.partial-abc123" in staging_dir("/scratch/run1", "abc123")
