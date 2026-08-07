"""Tests for interrupt teardown (`shinobi.backends._stream`).

The hermetic half drives the signal ladder against plain processes. The
live half is at the bottom, one test per container runtime, and is the only
half that can prove the thing that matters: a container is not always a
descendant of the process that launched it, so "we killed our child" and
"the work stopped" are different claims. Under docker and podman the first
is true while the second is false -- a mock cannot show that, because a mock
is written from the same belief that produced the bug.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time

import pytest

from shinobi.backends import _stream
from shinobi.backends._stream import run_streaming, terminate_all

# A payload that ignores the polite signals, standing in for a tool sitting in
# a long C-level call (casacore does exactly this for the duration of an
# mstransform, and Python never gets round to running its handler).
STUBBORN = "import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nprint('up', flush=True)\ntime.sleep(600)\n"


@pytest.fixture(autouse=True)
def _fast_ladder(monkeypatch):
    """Shorten the grace periods; the ladder's *order* is what is under test,
    not how patient it is. Also resets the process-wide interrupt state, which
    is module-level and would otherwise leak between tests."""
    monkeypatch.setattr(_stream, "TERM_GRACE_SECONDS", 0.5)
    monkeypatch.setattr(_stream, "KILL_GRACE_SECONDS", 2.0)
    _stream._IMPATIENT.clear()
    _stream._TERMINATE_CALLS = 0
    yield
    _stream._IMPATIENT.clear()
    _stream._TERMINATE_CALLS = 0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---- the child gets its own session -----------------------------------------


def test_child_runs_in_its_own_session():
    """Without this, the terminal's Ctrl-C goes to the tool and to ninja at
    the same time, and ninja starts unwinding (and deleting) while the tool
    is still running.

    The assertion has to come back *through* `run_streaming` -- an earlier
    version of this test compared `os.getpgrp()` with itself and then spawned
    its own `Popen(start_new_session=True)`, so it passed against a
    `run_streaming` that did no such thing.
    """
    run = run_streaming([sys.executable, "-c", "import os; print(os.getpgrp())"], label="t", stream=False)
    assert run.returncode == 0
    child_pgid = int(run.stdout.strip())
    assert child_pgid != os.getpgrp(), "the child shares ninja's process group"
    assert child_pgid != os.getsid(0), "the child shares ninja's session"


# ---- the signal ladder ------------------------------------------------------


def test_teardown_kills_a_child_that_ignores_sigterm():
    proc = subprocess.Popen([sys.executable, "-c", STUBBORN], stdout=subprocess.PIPE, text=True, start_new_session=True)
    assert proc.stdout.readline().strip() == "up"
    child = _stream._Child(proc=proc, label="stubborn", runtime_stop=None)

    _stream._teardown(child)

    assert proc.poll() is not None, "teardown returned with the child still alive"
    assert not _alive(proc.pid)


def test_teardown_reaches_a_grandchild_the_direct_child_left_behind():
    """The apptainer shape: what must die is not the process we launched but
    what it started. Signalling the process group is what covers that."""
    spawner = "import subprocess, sys, time\np = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\nprint(p.pid, flush=True)\ntime.sleep(600)\n"
    proc = subprocess.Popen([sys.executable, "-c", spawner], stdout=subprocess.PIPE, text=True, start_new_session=True)
    grandchild = int(proc.stdout.readline().strip())
    assert _alive(grandchild)

    _stream._teardown(_stream._Child(proc=proc, label="spawner", runtime_stop=None))

    for _ in range(50):
        if not _alive(grandchild):
            break
        time.sleep(0.1)
    assert not _alive(grandchild), "the grandchild outlived teardown"


def test_teardown_calls_the_runtime_stop_first():
    """docker/podman order: `rm -f` is not a fallback for when signals fail,
    it is the only thing that reaches the container at all."""
    calls: list[str] = []
    proc = subprocess.Popen([sys.executable, "-c", STUBBORN], stdout=subprocess.PIPE, text=True, start_new_session=True)
    assert proc.stdout.readline().strip() == "up"
    rs = _stream.RuntimeStop(stop=lambda: calls.append("stop"), running=lambda: False)

    assert _stream._teardown(_stream._Child(proc=proc, label="c", runtime_stop=rs)) is True
    assert calls[0] == "stop"
    assert proc.poll() is not None


def test_teardown_retries_the_runtime_stop_while_the_container_lives():
    """`docker rm -f` on a container the daemon has not created yet exits 0.
    A teardown that asked once and trusted that exit code reported success
    over a container that then started -- so the ask is repeated until
    `running()` says otherwise."""
    proc = subprocess.Popen([sys.executable, "-c", "print('up', flush=True)"], stdout=subprocess.PIPE, text=True, start_new_session=True)
    proc.stdout.readline()
    proc.wait()
    asks = []
    # "running" until the second ask -- the create-then-stop race, compressed.
    rs = _stream.RuntimeStop(stop=lambda: asks.append(1), running=lambda: len(asks) < 2)

    assert _stream._teardown(_stream._Child(proc=proc, label="c", runtime_stop=rs)) is True
    assert len(asks) >= 2, "teardown asked once and took the answer on faith"


def test_teardown_reports_failure_when_the_container_will_not_die():
    """The signal ladder can succeed against the client while the container
    survives. Teardown must return False so nothing downstream tidies up
    around it."""
    proc = subprocess.Popen([sys.executable, "-c", "print('up', flush=True)"], stdout=subprocess.PIPE, text=True, start_new_session=True)
    proc.stdout.readline()
    proc.wait()
    rs = _stream.RuntimeStop(stop=lambda: None, running=lambda: True)  # never stops

    assert _stream._teardown(_stream._Child(proc=proc, label="c", runtime_stop=rs)) is False


def test_a_failing_runtime_stop_is_reported_not_swallowed():
    """For docker/podman `stop()` is the entire mechanism; if it fails and the
    container is still there, teardown has not succeeded."""
    proc = subprocess.Popen([sys.executable, "-c", "print('up', flush=True)"], stdout=subprocess.PIPE, text=True, start_new_session=True)
    proc.stdout.readline()
    proc.wait()

    def _stop() -> None:
        raise RuntimeError("daemon is wedged")

    rs = _stream.RuntimeStop(stop=_stop, running=lambda: True)
    assert _stream._teardown(_stream._Child(proc=proc, label="c", runtime_stop=rs)) is False  # and does not raise


def test_teardown_never_signals_our_own_process_group():
    """The fallback path (no new session) must not killpg *us* -- that would
    take ninja and the user's shell job down with the step."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])  # same group as us
    try:
        _stream._teardown(_stream._Child(proc=proc, label="same-group", runtime_stop=None))
        assert proc.poll() is not None  # the child alone died
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


# ---- run_streaming's own guarantee ------------------------------------------


def test_run_streaming_tears_down_on_interrupt(monkeypatch):
    """A KeyboardInterrupt out of the wait must not propagate past a live
    child: `steps.pyfunc` deletes the runner's I/O directory as this unwinds,
    and that directory is open inside the container."""
    launched: list[int] = []
    real_popen = subprocess.Popen

    def _spy(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        launched.append(proc.pid)
        return proc

    def _interrupt(self, *a, **k):
        raise KeyboardInterrupt

    # The wait being interrupted is the join on the reader threads -- for
    # `stream=False` as much as `stream=True`, since capping what is held in
    # memory took the non-streaming path off `communicate()` too.
    monkeypatch.setattr(threading.Thread, "join", _interrupt)
    monkeypatch.setattr(subprocess, "Popen", _spy)

    with pytest.raises(KeyboardInterrupt):
        run_streaming([sys.executable, "-c", STUBBORN], label="t", stream=False)

    assert launched, "nothing was launched"
    assert not _alive(launched[0]), "run_streaming propagated the interrupt with its child still alive"


def test_run_streaming_raises_teardown_incomplete_when_it_cannot_stop_the_child(monkeypatch):
    """The signal that must reach `steps.pyfunc`: do not delete the workspace,
    something is still using it."""

    def _interrupt(self, *a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(threading.Thread, "join", _interrupt)
    monkeypatch.setattr(_stream, "_teardown", lambda child: False)

    with pytest.raises(_stream.TeardownIncomplete) as caught:
        run_streaming([sys.executable, "-c", "import time; time.sleep(5)"], label="t", stream=False)
    assert isinstance(caught.value.__cause__, KeyboardInterrupt), "the original interrupt was lost"


def test_second_interrupt_skips_the_grace_periods():
    """A user leaning on Ctrl-C is asking for escalation. The flag has to be
    set when teardown is *asked for*, not when it finishes -- set at the end
    (as this first was) it could only ever be read by a teardown that no
    longer needed it."""
    assert not _stream._IMPATIENT.is_set()
    terminate_all(reason="first")
    assert not _stream._IMPATIENT.is_set(), "one interrupt should still get the full ladder"
    terminate_all(reason="second")
    assert _stream._IMPATIENT.is_set()


def test_a_fresh_burst_of_work_restores_the_grace_period():
    """Otherwise, in a notebook or any long-lived process, one interrupted run
    permanently strips the SIGTERM grace from every run after it."""
    terminate_all(reason="first")
    terminate_all(reason="second")
    assert _stream._IMPATIENT.is_set()

    run_streaming([sys.executable, "-c", "print('hi')"], label="t", stream=False)
    assert not _stream._IMPATIENT.is_set()


def test_registry_is_emptied_on_the_normal_path():
    run_streaming([sys.executable, "-c", "print('hi')"], label="t", stream=False)
    assert not _stream._LIVE


def test_terminate_all_reaches_a_child_of_another_thread():
    """The recipe case: the interrupt lands in the main thread while workers
    sit blocked on children of their own, which is exactly the set that
    `terminate_all` exists to reach."""
    import threading

    started = threading.Event()
    result: list[object] = []

    def _worker():
        started.set()
        try:
            result.append(run_streaming([sys.executable, "-c", STUBBORN], label="worker", stream=False))
        except BaseException as exc:  # noqa: BLE001 -- recorded for the assertion
            result.append(exc)

    thread = threading.Thread(target=_worker)
    thread.start()
    started.wait(5)
    for _ in range(50):  # wait for the child to be registered
        if _stream._LIVE:
            break
        time.sleep(0.1)
    assert _stream._LIVE, "the worker's child never registered"
    pid = next(iter(_stream._LIVE))

    assert terminate_all(reason="test") == 1

    thread.join(timeout=15)
    assert not thread.is_alive(), "the worker never unwound"
    assert not _alive(pid)


def test_terminate_all_with_nothing_running_is_a_no_op():
    assert terminate_all(reason="test") == 0


# ---- live: does the container actually stop? --------------------------------

BUSYBOX = "docker.io/library/busybox:latest"
# `sleep` in its own container, found from the host by its distinctive
# duration -- **one marker per runtime**, so a case that does leak fails its
# own assertion instead of the next case's. (It cost a debugging round to
# learn that a single shared marker makes every later parametrisation report
# the first one's leak.) Long enough that anything still alive after teardown
# is a leak and not a race.
SLEEP_MARKERS = {"docker": "999501", "podman": "999502", "apptainer": "999503", "singularity": "999504"}


def _has(runtime: str) -> bool:
    if not shutil.which(runtime):
        return False
    probe = ["image", "inspect", BUSYBOX] if runtime in ("docker", "podman") else ["--version"]
    return subprocess.run([runtime, *probe], capture_output=True).returncode == 0


def _container_argv(runtime: str, name: str, marker: str) -> list[str]:
    if runtime in ("docker", "podman"):
        return [runtime, "run", "--rm", "--name", name, BUSYBOX, "sleep", marker]
    return [runtime, "exec", f"docker://{BUSYBOX}", "sleep", marker]


def _payload_running(marker: str) -> bool:
    """Is the *payload* up -- not merely the client that will start it?

    Exact match on `sleep <marker>`, because a substring test also matches the
    launcher's own `docker run --rm --name ... sleep <marker>` argv. That
    mistake made this test's readiness gate trip the instant the client was
    spawned, sometimes before the daemon had created the container at all, so
    teardown ran against a container that did not exist yet and the test flaked
    (observed: 1 failure in 6 consecutive runs). It also flattered the fix --
    a `stop=None` control would have "leaked" for the same reason, whatever
    the code did.
    """
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    for line in ps.splitlines():
        parts = line.split()
        # `sleep <marker>` and nothing else -- but matched on the *basename*,
        # because the runtimes disagree on the form: docker's payload shows as
        # `sleep 999501`, apptainer's as `/bin/sleep 999503`. Requiring the
        # bare form made the apptainer gate time out and leak its container.
        if len(parts) == 2 and os.path.basename(parts[0]) == "sleep" and parts[1] == marker:
            return True
    return False


def _container_up(runtime: str, name: str) -> bool:
    """Does the runtime itself say the container is running? The host-side
    `ps` check cannot distinguish "not started yet" from "already gone", and
    for docker/podman the daemon is the only authority on either."""
    if runtime not in ("docker", "podman"):
        return True
    out = subprocess.run([runtime, "ps", "--quiet", "--filter", f"name=^{name}$"], capture_output=True, text=True)
    return bool(out.stdout.strip())


@pytest.mark.parametrize("runtime", ["docker", "podman", "apptainer", "singularity"])
def test_live_interrupt_leaves_no_container_running(runtime, monkeypatch):
    """The regression, end to end, per runtime.

    Skipped where the runtime isn't installed. Worth the wall-clock when it
    does run: docker and podman pass here *only* because of the `--name` +
    `rm -f` path, and would pass a signals-only implementation's unit tests
    while leaking a real container.
    """
    if not _has(runtime):
        pytest.skip(f"{runtime} (or {BUSYBOX}) not available")
    monkeypatch.setattr(_stream, "TERM_GRACE_SECONDS", 2.0)
    monkeypatch.setattr(_stream, "KILL_GRACE_SECONDS", 5.0)

    from shinobi.backends.container import container_stopper, new_container_name

    marker = SLEEP_MARKERS[runtime]
    assert not _payload_running(marker), f"a {marker} payload leaked from an earlier run -- kill it before trusting this test"
    name = new_container_name()
    argv = _container_argv(runtime, name, marker)
    stop = container_stopper(runtime, name)

    # Run it the way a recipe worker does -- on another thread, blocked on the
    # container -- and interrupt from here, which is where a Ctrl-C lands.
    # (Patching `Popen.communicate` to raise would be the shorter route and is
    # a trap: this test shells out to `ps`, so the patch recurses into itself.)
    import threading

    done = threading.Event()

    def _worker():
        try:
            run_streaming(argv, label=runtime, stream=False, stop=stop)
        except BaseException:  # noqa: BLE001,S110 -- teardown is what's asserted
            pass
        finally:
            done.set()

    thread = threading.Thread(target=_worker)
    thread.start()
    try:
        # Both conditions: the payload visible on the host *and* the runtime
        # agreeing the container is up. Either alone leaves the window this
        # test kept falling into.
        for _ in range(300):
            if _payload_running(marker) and _container_up(runtime, name):
                break
            time.sleep(0.1)
        else:
            pytest.fail(f"{runtime}: the container never started")

        assert terminate_all(reason="test") == 1

        for _ in range(100):
            if not _payload_running(marker):
                break
            time.sleep(0.1)
        assert not _payload_running(marker), f"{runtime}: the container survived teardown"
        assert done.wait(30), f"{runtime}: run_streaming never unwound"
    finally:
        # Unconditional, and before the join: a test that fails its readiness
        # gate still has a live container and a worker blocked on it, and
        # leaving those behind poisons every later run of this file (the
        # marker guard above would then fail for the wrong reason).
        terminate_all(reason="test cleanup")
        if runtime in ("docker", "podman"):
            subprocess.run([runtime, "rm", "-f", name], capture_output=True)
        thread.join(timeout=30)
        for _ in range(50):
            if not _payload_running(marker):
                break
            time.sleep(0.1)


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_live_container_is_named_so_it_can_be_stopped(runtime):
    """docker/podman get a `--name` for exactly one reason: without a handle
    there is no way to stop a container that is not our descendant."""
    if not _has(runtime):
        pytest.skip(f"{runtime} (or {BUSYBOX}) not available")

    from shinobi.backends.container import DockerBackend, PodmanBackend, new_container_name
    from shinobi.loaders import build_model
    from shinobi.steps.schema import Cab

    backend = (DockerBackend if runtime == "docker" else PodmanBackend)(workdir=os.getcwd(), run_as_host_user=False)
    cab = Cab(name="t", command="true", image=BUSYBOX, inputs_model=build_model("In", {}), outputs_model=build_model("Out", {}))
    argv, _ = backend._wrap(cab, ["true"], {}, container_name=new_container_name())
    assert "--name" in argv
    assert argv[argv.index("--name") + 1].startswith("shinobi-")


def test_apptainer_likes_get_no_stopper():
    """They need no handle: the runtime parent and payload stay in our
    process group, where signalling the group does tear the container down."""
    from shinobi.backends.container import container_stopper

    assert container_stopper("apptainer", "shinobi-x") is None
    assert container_stopper("singularity", "shinobi-x") is None
    assert container_stopper("docker", "shinobi-x") is not None
    assert container_stopper("podman", "shinobi-x") is not None
    assert container_stopper("docker", None) is None
