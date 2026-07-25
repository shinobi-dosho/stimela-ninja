import threading

import pytest

from shinobi.resources import Budget, ResourceBudget, Resources, delegated_controllers, detect_budget, format_size, parse_size

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


# -- delegated controllers (what a rootless runtime can actually enforce) --


def _delegate(root, *, leaf, per_level):
    """A fake cgroup v2 tree whose levels advertise `per_level` controllers.

    `per_level` maps a cgroup-relative path (`""` for the root) to the
    `cgroup.controllers` contents at that level -- the file systemd's
    `Delegate=` ultimately drives.
    """
    _write_cgroup_tree(root, leaf=leaf, limits={})
    cg = root / "sys/fs/cgroup"
    for rel, controllers in per_level.items():
        level = cg / rel if rel else cg
        level.mkdir(parents=True, exist_ok=True)  # a named level need not be on the leaf path
        (level / "cgroup.controllers").write_text(f"{controllers}\n")
    delegated_controllers.cache_clear()
    return str(root)


def test_delegated_controllers_reports_a_partial_delegation(tmp_path):
    """Issue #39's host: root has everything, the user manager has `memory
    pids` and no `cpu`.
    """
    root = _delegate(
        tmp_path,
        leaf="/user.slice/user-20001.slice/user@20001.service/app.slice/vte.scope",
        per_level={
            "": "cpuset cpu io memory hugetlb pids rdma misc",
            "user.slice/user-20001.slice/user@20001.service": "memory pids",
            "user.slice/user-20001.slice/user@20001.service/app.slice": "memory pids",
            "user.slice/user-20001.slice/user@20001.service/app.slice/vte.scope": "memory pids",
        },
    )
    assert delegated_controllers(root) == frozenset({"memory", "pids"})


def test_delegated_controllers_ignores_the_callers_own_level(tmp_path):
    """The measured layout of the dev machine: the user manager has `cpu`
    (and apptainer really does apply `--cpus` there), while the calling shell
    sits under an `app.slice` that never enabled it. Reading the caller's own
    cgroup would drop a limit that works -- a silent loss of enforcement,
    which is the one error direction that matters.
    """
    root = _delegate(
        tmp_path,
        leaf="/user.slice/user-1000.slice/user@1000.service/app.slice/app-terminal.slice/vte-spawn.scope",
        per_level={
            "": "cpuset cpu io memory pids",
            "user.slice/user-1000.slice/user@1000.service": "cpu memory pids",
            "user.slice/user-1000.slice/user@1000.service/app.slice": "memory pids",
            "user.slice/user-1000.slice/user@1000.service/app.slice/app-terminal.slice": "memory pids",
            "user.slice/user-1000.slice/user@1000.service/app.slice/app-terminal.slice/vte-spawn.scope": "memory pids",
        },
    )
    assert delegated_controllers(root) == frozenset({"cpu", "memory", "pids"})


def test_delegated_controllers_reports_nothing_delegated(tmp_path):
    """An empty set at the boundary is an answer ("nothing is delegated"),
    and a different one from "couldn't tell".
    """
    root = _delegate(
        tmp_path,
        leaf="/user.slice/user-1000.slice/user@1000.service/leaf.scope",
        per_level={"": "cpu memory pids", "user.slice/user-1000.slice/user@1000.service": ""},
    )
    assert delegated_controllers(root) == frozenset()


def test_delegated_controllers_reads_the_manager_beside_an_ssh_session(tmp_path):
    """The measured layout of the batch host: an `ssh` login sits in
    `session-<n>.scope`, a *sibling* of `user@<uid>.service` rather than a
    descendant. The manager is right there and says `memory pids`, so
    answering "couldn't tell" would emit `--cpus` on a host known to abort
    on it -- and `ssh` is how batch runs are started.
    """
    root = _delegate(
        tmp_path,
        leaf="/user.slice/user-20001.slice/session-c4116.scope",
        per_level={
            "": "cpuset cpu io memory hugetlb pids rdma misc",
            "user.slice/user-20001.slice": "memory pids",
            "user.slice/user-20001.slice/user@20001.service": "memory pids",
            "user.slice/user-20001.slice/session-c4116.scope": "memory pids",
        },
    )
    assert delegated_controllers(root) == frozenset({"memory", "pids"})


def test_delegated_controllers_prefers_a_manager_the_caller_is_inside(tmp_path):
    """When the caller really is under a `user@<uid>.service`, that one is
    the answer -- the sibling lookup is a fallback, not an override.
    """
    root = _delegate(
        tmp_path,
        leaf="/user.slice/user-1000.slice/user@1000.service/app.slice/vte.scope",
        per_level={
            "": "cpu memory pids",
            "user.slice/user-1000.slice/user@1000.service": "cpu memory pids",
            "user.slice/user-1000.slice/user@1000.service/app.slice": "memory pids",
            "user.slice/user-1000.slice/user@1000.service/app.slice/vte.scope": "memory pids",
        },
    )
    assert delegated_controllers(root) == frozenset({"cpu", "memory", "pids"})


def test_delegated_controllers_is_unknown_when_no_manager_is_running(tmp_path):
    """A user slice with no `user@<uid>.service` beside it -- systemd never
    started one. Nothing to read, so nothing is claimed.
    """
    root = _delegate(
        tmp_path,
        leaf="/user.slice/user-20001.slice/session-c1.scope",
        per_level={"": "cpu memory pids", "user.slice/user-20001.slice": "memory pids"},
    )
    assert delegated_controllers(root) is None


def test_delegated_controllers_is_unknown_without_a_user_manager(tmp_path):
    """No `user@<uid>.service` in the path and no user slice either -- a
    container, or a system service. Nothing to read, so nothing is claimed.
    """
    root = _delegate(tmp_path, leaf="/system.slice/some.service", per_level={"": "cpu memory pids"})
    assert delegated_controllers(root) is None


def test_delegated_controllers_is_unknown_without_cgroup_v2(tmp_path):
    """No unified hierarchy in view means "can't tell" -- which callers must
    treat as "emit everything and fail loudly", never as "nothing works".
    """
    (tmp_path / "proc/self").mkdir(parents=True)
    (tmp_path / "proc/self/cgroup").write_text("4:memory:/user.slice\n3:cpu,cpuacct:/user.slice\n")
    delegated_controllers.cache_clear()
    assert delegated_controllers(str(tmp_path)) is None


def test_delegated_controllers_is_unknown_without_a_unified_entry(tmp_path):
    root = _delegate(tmp_path, leaf="/user.slice/user@1000.service/a", per_level={"": "cpu memory"})
    (tmp_path / "proc/self/cgroup").write_text("4:memory:/user.slice\n")  # v1-only entry
    delegated_controllers.cache_clear()
    assert delegated_controllers(root) is None


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


def test_all_cpus_takes_the_whole_pool_and_excludes_a_sibling():
    """The point of the sentinel: a step that will use every core must not
    read as *free* just because a recipe cannot know the number.
    """
    budget = Budget(Resources(cpus=8))
    assert budget.try_acquire(Resources(cpus="all"), "wsclean")[0]
    assert budget.reserved.cpus == 8
    assert not budget.try_acquire(Resources(cpus=1), "sibling")[0]


def test_all_cpus_is_granted_rather_than_deadlocking_against_itself():
    """Resolving to exactly the total (not more) is what keeps this a *wait*
    rather than the `_exceeds_total` escape hatch, which would wave both
    steps through unconstrained -- turning the strongest declaration into
    the weakest.
    """
    budget = Budget(Resources(cpus=8))
    assert budget.try_acquire(Resources(cpus="all"), "first")[0]
    assert not budget.try_acquire(Resources(cpus="all"), "second")[0]
    budget.release(Resources(cpus="all"))  # released as declared, resolved the same way
    budget.abandon()
    assert budget.reserved.cpus == 0
    assert budget.try_acquire(Resources(cpus="all"), "second")[0]


def test_all_cpus_against_an_unconstrained_pool_is_free():
    """Nothing can exhaust a pool with no CPU total, so there is nothing for
    the step to be held back by."""
    budget = Budget(Resources(cpus=None, memory="10GiB"))
    for _ in range(5):
        assert budget.try_acquire(Resources(cpus="all"), "greedy")[0]


def test_all_cpus_is_not_a_backend_limit():
    """A cap of "the whole machine" is not a cap, and no backend may invent
    a number for it (see `Resources.enforceable_cpus`)."""
    assert Resources(cpus="all").enforceable_cpus is None
    assert Resources(cpus=4).enforceable_cpus == 4
    assert not Resources(cpus="all").is_empty()
    assert Resources(cpus="all").describe() == "cpus=all"


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


# -- backfill --


def test_backfill_admits_a_unit_that_leaves_room_for_the_head():
    """Numbers chosen so the two clauses do not coincide: with 50 GiB
    running, a 40 GiB candidate fits in what is free (50) but would leave the
    70 GiB head unable to start; a 25 GiB one leaves room for it.
    """
    budget = Budget(Resources(memory="100GiB"))
    budget.try_acquire(Resources(memory="50GiB"), "running")
    head = Resources(memory="70GiB")
    assert not budget.try_acquire(head, "head")[0]  # parked, and now holds a ticket

    assert not budget.try_backfill(Resources(memory="40GiB"), head, "too-big")
    assert budget.try_backfill(Resources(memory="25GiB"), head, "fits")


def test_backfill_is_cumulative_across_grants():
    """The bug an earlier draft shipped: bounding each grant against the
    total on its own admits both 30 GiB units, and the 70 GiB head then
    cannot start when the original 40 GiB holder finishes.
    """
    budget = Budget(Resources(memory="100GiB"))
    running = Resources(memory="40GiB")
    budget.try_acquire(running, "running")
    head = Resources(memory="70GiB")
    assert not budget.try_acquire(head, "head")[0]

    assert budget.try_backfill(Resources(memory="30GiB"), head, "b1")
    assert not budget.try_backfill(Resources(memory="30GiB"), head, "b2")

    # The whole point: once the work the head was already waiting for
    # finishes, the head starts -- exactly as it would have with no backfill.
    budget.release(running)
    assert budget.try_acquire(head, "head")[0]


def test_releasing_a_backfill_frees_the_cumulative_room_again():
    budget = Budget(Resources(memory="100GiB"))
    budget.try_acquire(Resources(memory="40GiB"), "running")
    head = Resources(memory="70GiB")
    assert not budget.try_acquire(head, "head")[0]

    first = Resources(memory="30GiB")
    assert budget.try_backfill(first, head, "b1")
    assert not budget.try_backfill(Resources(memory="30GiB"), head, "b2")

    budget.release(first, backfilled=True)
    assert budget.try_backfill(Resources(memory="30GiB"), head, "b3")


def test_undeclared_units_backfill_without_limit():
    """An undeclared step reserves nothing, so it can never delay the head
    and must not consume cumulative room. This is the common case -- a
    plain step stuck behind a sized one.
    """
    budget = Budget(Resources(memory="100GiB"))
    budget.try_acquire(Resources(memory="90GiB"), "running")
    head = Resources(memory="70GiB")
    assert not budget.try_acquire(head, "head")[0]

    for _ in range(5):
        assert budget.try_backfill(Resources(), head, "free")


def test_backfill_refuses_when_a_sibling_is_queued_ahead():
    """Backfill must not jump the cross-recipe FIFO queue -- that is the
    starvation the tickets exist to prevent.
    """
    budget = Budget(Resources(memory="100GiB"))
    budget.try_acquire(Resources(memory="40GiB"), "running")

    def older_sibling():
        assert not budget.try_acquire(Resources(memory="80GiB"), "sibling")[0]

    thread = threading.Thread(target=older_sibling)
    thread.start()
    thread.join()
    # The sibling exits still holding its ticket. Harmless here because the
    # budget dies with the test, but on a real scheduler that is exactly the
    # leak `abandon()` exists to prevent -- a ticket from a dead waiter would
    # block every later arrival forever (see `_return_reservations`).

    # This thread has no ticket at all, and a sibling holds an older one.
    assert not budget.try_backfill(Resources(memory="10GiB"), Resources(memory="70GiB"), "b")


def test_backfill_never_touches_the_waiter_queue():
    """The head's parked ticket is what holds its place; a backfill grant on
    the same thread must not pop it the way `try_acquire` does.
    """
    budget = Budget(Resources(memory="100GiB"))
    budget.try_acquire(Resources(memory="40GiB"), "running")
    head = Resources(memory="70GiB")
    assert not budget.try_acquire(head, "head")[0]
    ticket = dict(budget._waiters)
    assert ticket, "the refusal should have queued this thread"

    assert budget.try_backfill(Resources(memory="30GiB"), head, "granted")
    assert budget._waiters == ticket
    assert not budget.try_backfill(Resources(memory="30GiB"), head, "refused")
    assert budget._waiters == ticket


def test_backfill_resolves_the_cpus_all_sentinel_on_both_sides():
    """`cpus="all"` is symbolic until measured against the pool, so both the
    parked head's demand and the candidate's have to be resolved before the
    room-to-spare arithmetic -- otherwise a string reaches it.

    Semantically: a head claiming every core leaves room for nothing that
    declares CPU, but still cannot be delayed by work that declares none.
    """
    budget = Budget(Resources(cpus=8, memory="100GiB"))
    budget.try_acquire(Resources(cpus=4), "running")
    head = Resources(cpus="all")
    assert not budget.try_acquire(head, "head")[0]

    assert not budget.try_backfill(Resources(cpus=2), head, "sized")
    assert budget.try_backfill(Resources(), head, "undeclared")

    # ... and a candidate that itself wants every core cannot slip past a
    # head that needs only some of them.
    other = Budget(Resources(cpus=8, memory="100GiB"))
    other.try_acquire(Resources(cpus=6), "running")
    small_head = Resources(cpus=4)
    assert not other.try_acquire(small_head, "head")[0]
    assert not other.try_backfill(Resources(cpus="all"), small_head, "greedy")


def test_backfill_treats_the_dimensions_independently():
    """`_leaves_room_for` skips a dimension the candidate leaves unset, which
    is what makes "work that consumes no CPU cannot delay anyone on CPU"
    true. Pin the cross-dimension contract before someone simplifies that
    filter away.
    """
    budget = Budget(Resources(cpus=8, memory="100GiB"))
    budget.try_acquire(Resources(memory="60GiB"), "running")
    memory_head = Resources(memory="70GiB")
    assert not budget.try_acquire(memory_head, "head")[0]

    # A CPU-only candidate competes with the memory-bound head for nothing.
    assert budget.try_backfill(Resources(cpus=4), memory_head, "cpu-only")

    # ... but it is still held to the CPU dimension against a CPU-bound head.
    other = Budget(Resources(cpus=8, memory="100GiB"))
    other.try_acquire(Resources(cpus=6), "running")
    cpu_head = Resources(cpus=6)
    assert not other.try_acquire(cpu_head, "head")[0]
    assert not other.try_backfill(Resources(cpus=4), cpu_head, "cpu-only")  # 6 + 4 > 8
