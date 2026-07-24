import threading

import pytest

from shinobi.resources import Budget, ResourceBudget, Resources, detect_budget, format_size, parse_size

# -- size parsing --


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1024", 1024),
        ("1kB", 1000),
        ("1KiB", 1024),
        ("200GB", 200 * 1000**3),
        ("250GiB", 250 * 1024**3),
        ("2T", 2 * 1000**4),
        ("1.5 GiB", int(1.5 * 1024**3)),
        (4096, 4096),
    ],
)
def test_parse_size_accepts_decimal_and_binary_units(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize("bad", ["", "GB", "12 parsecs", "-5", "1.2.3GB", True])
def test_parse_size_rejects_junk(bad):
    with pytest.raises(ValueError):
        parse_size(bad)


def test_format_size_uses_binary_units():
    assert format_size(270335016960) == "251.8GiB"
    assert format_size(512) == "512B"


def test_memory_field_accepts_a_size_string():
    assert Resources(memory="200GB").memory == 200 * 1000**3
    assert Resources().is_empty()
    assert not Resources(cpus=1).is_empty()


# -- budget detection --


def _write_cgroup_tree(root, *, leaf, limits, controllers="cpu memory"):
    """Build a fake cgroup v2 tree.

    `leaf` is the cgroup path in /proc/self/cgroup; `limits` maps a
    cgroup-relative path to the {filename: contents} to write there. Every
    level not named in `limits` gets an explicit "no constraint here", which
    is what makes the ancestor walk the thing under test rather than an
    accident of which files happen to exist.
    """
    (root / "proc/self").mkdir(parents=True)
    (root / "proc/self/cgroup").write_text(f"0::{leaf}\n")
    (root / "proc/meminfo").write_text("MemTotal:       988000000 kB\n")
    cg = root / "sys/fs/cgroup"
    cg.mkdir(parents=True)
    (cg / "cgroup.controllers").write_text(f"{controllers}\n")
    components = [p for p in leaf.split("/") if p]
    for depth in range(len(components) + 1):
        level = cg.joinpath(*components[:depth])
        level.mkdir(parents=True, exist_ok=True)
        (level / "memory.max").write_text("max\n")
        (level / "cpu.max").write_text("max 100000\n")
    for rel, files in limits.items():
        for name, contents in files.items():
            (cg / rel if rel else cg).joinpath(name).write_text(contents)


def test_detect_budget_walks_the_ancestor_chain(tmp_path):
    """The cap that matters is on an ancestor, not the leaf.

    This is the regression that decides whether the whole feature works
    where it was needed: a fair-share quota lives on `user-<uid>.slice`,
    several levels above the cgroup a process actually runs in. Reading only
    the leaf finds "max", falls through to /proc/meminfo, and reports the
    host's full memory -- which is exactly the bug (a tool sizing itself
    from /proc/meminfo and getting OOM-killed) that this exists to prevent.
    """
    _write_cgroup_tree(
        tmp_path,
        leaf="/user.slice/user-20001.slice/session.scope/app.scope",
        limits={"user.slice/user-20001.slice": {"memory.max": "270335016960\n", "cpu.max": "1600000 100000\n"}},
    )
    budget, source = detect_budget(tmp_path)
    assert source == "cgroup v2"
    assert budget.memory == 270335016960
    assert budget.cpus == 16.0


def test_detect_budget_takes_the_tightest_limit_in_the_chain(tmp_path):
    _write_cgroup_tree(
        tmp_path,
        leaf="/a/b/c",
        limits={
            "a": {"memory.max": "8000000000\n"},
            "a/b": {"memory.max": "2000000000\n"},  # tightest
            "a/b/c": {"memory.max": "4000000000\n"},
        },
    )
    budget, _ = detect_budget(tmp_path)
    assert budget.memory == 2000000000


def test_detect_budget_falls_back_to_meminfo_when_nothing_constrains(tmp_path):
    _write_cgroup_tree(tmp_path, leaf="/a/b", limits={})
    budget, source = detect_budget(tmp_path)
    assert source == "host"
    assert budget.memory == 988000000 * 1024
    assert budget.cpus  # from sched_getaffinity


def test_detect_budget_reads_cgroup_v1(tmp_path):
    (tmp_path / "proc/self").mkdir(parents=True)
    (tmp_path / "proc/self/cgroup").write_text("4:memory:/user.slice/user-1000.slice\n3:cpu,cpuacct:/user.slice\n")
    (tmp_path / "proc/meminfo").write_text("MemTotal:       988000000 kB\n")
    mem = tmp_path / "sys/fs/cgroup/memory/user.slice/user-1000.slice"
    mem.mkdir(parents=True)
    # The unlimited sentinel on the ancestor must not win over a real limit.
    (mem.parent / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    (mem / "memory.limit_in_bytes").write_text("1000000000\n")
    budget, source = detect_budget(tmp_path)
    assert source == "cgroup v1"
    assert budget.memory == 1000000000


def test_resource_budget_config_modes(tmp_path):
    _write_cgroup_tree(tmp_path, leaf="/a", limits={"a": {"memory.max": "1000000000\n"}})

    explicit, source = ResourceBudget(cpus=4, memory="2GiB").resolve(tmp_path)
    assert (explicit.cpus, explicit.memory) == (4.0, 2 * 1024**3)
    assert "config" in source

    unbounded, _ = ResourceBudget(cpus="unbounded", memory="unbounded").resolve(tmp_path)
    assert unbounded.is_empty()

    auto, source = ResourceBudget().resolve(tmp_path)
    assert auto.memory == 1000000000
    assert "cgroup v2" in source


# -- the Budget state machine --


def test_budget_grants_until_full_then_refuses():
    budget = Budget(Resources(cpus=8, memory="100GiB"))
    assert budget.try_acquire(Resources(memory="60GiB"), "a")[0]
    assert not budget.try_acquire(Resources(memory="60GiB"), "b")[0]
    budget.release(Resources(memory="60GiB"))
    budget.abandon()  # the refused caller gives up its ticket
    assert budget.try_acquire(Resources(memory="60GiB"), "b")[0]


def test_undeclared_demand_is_always_free():
    budget = Budget(Resources(memory="1GiB"))
    assert budget.try_acquire(Resources(memory="1GiB"), "big")[0]
    for _ in range(10):
        assert budget.try_acquire(Resources(), "free")[0]
    assert budget.reserved.memory == 1024**3


def test_demand_larger_than_the_whole_budget_is_granted_not_parked():
    """It could never fit, so waiting could only hang. Run it and warn."""
    budget = Budget(Resources(memory="10GiB"))
    granted, _ = budget.try_acquire(Resources(memory="500GiB"), "ddfacet")
    assert granted
    # ...and it now blocks everything behind it, rather than sharing.
    assert not budget.try_acquire(Resources(memory="1GiB"), "sibling")[0]


def test_unconstrained_dimension_never_refuses():
    budget = Budget(Resources(cpus=4, memory=None))
    for _ in range(5):
        assert budget.try_acquire(Resources(memory="1TiB"), "greedy")[0]


def test_fifo_fairness_serves_the_waiting_step_first():
    """A large step must not be starved by a stream of smaller ones.

    Sibling recipes share one budget, so "block in declaration order" inside
    one recipe is not enough: without a queue, small steps from other
    branches could keep the reserved total above zero forever and the large
    step would never be admitted.
    """
    budget = Budget(Resources(memory="100GiB"))
    budget.try_acquire(Resources(memory="50GiB"), "holder")

    big_refused = threading.Event()

    def big_waiter():
        # Takes a ticket by being refused.
        assert not budget.try_acquire(Resources(memory="80GiB"), "big")[0]
        big_refused.set()

    thread = threading.Thread(target=big_waiter)
    thread.start()
    thread.join()
    assert big_refused.is_set()

    # A small step that *would* fit is held back, because someone is queued
    # ahead of it.
    assert not budget.try_acquire(Resources(memory="10GiB"), "small")[0]


def test_wait_for_change_does_not_miss_a_release():
    """The refuse-then-park window must not lose a wakeup.

    `try_acquire` and `wait_for_change` take the lock separately, so a
    release landing between them would park forever if the generation
    counter were not carried across.
    """
    budget = Budget(Resources(memory="10GiB"))
    budget.try_acquire(Resources(memory="8GiB"), "holder")
    granted, generation = budget.try_acquire(Resources(memory="8GiB"), "waiter")
    assert not granted

    # The release happens *before* the wait -- the lost-wakeup window.
    budget.release(Resources(memory="8GiB"))

    finished = threading.Event()
    threading.Thread(target=lambda: (budget.wait_for_change(generation), finished.set())).start()
    assert finished.wait(timeout=5), "wait_for_change missed a release and parked"


def test_wait_for_change_wakes_on_a_later_release():
    budget = Budget(Resources(memory="10GiB"))
    budget.try_acquire(Resources(memory="8GiB"), "holder")
    _, generation = budget.try_acquire(Resources(memory="8GiB"), "waiter")

    woken = threading.Event()
    threading.Thread(target=lambda: (budget.wait_for_change(generation), woken.set())).start()
    assert not woken.wait(timeout=0.2)  # still parked
    budget.release(Resources(memory="8GiB"))
    assert woken.wait(timeout=5)


def test_abandon_frees_the_queue_for_later_arrivals():
    budget = Budget(Resources(memory="10GiB"))
    budget.try_acquire(Resources(memory="6GiB"), "holder")

    def queue_then_die():
        assert not budget.try_acquire(Resources(memory="9GiB"), "doomed")[0]
        budget.abandon()  # what _run_recipe's exit path does

    thread = threading.Thread(target=queue_then_die)
    thread.start()
    thread.join()

    # Without abandon() the dead scheduler's ticket would block this forever.
    assert budget.try_acquire(Resources(memory="4GiB"), "later")[0]
