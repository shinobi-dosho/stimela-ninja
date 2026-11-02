"""Placed graphs preserve leaf resource admission and refuse unsupported loops."""

import threading
import time

import pytest
from pydantic import BaseModel

from shinobi import Recipe, SlurmPlacement, pystep
from shinobi.config import AppConfig
from shinobi.offload._codec import BundleError
from shinobi.resources import Budget, Resources


class Empty(BaseModel):
    pass


def config(**resources):
    return AppConfig(backend={"default": "native"}, cache={"enabled": False}, provenance={"enabled": False}, execution={"resources": resources})


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHINOBI_OWNERSHIP_REGISTRY", str(tmp_path / "registry.json"))


def measured_step(demand, events, *, fail=False):
    lock = threading.Lock()

    @pystep()
    def work(number: int) -> Empty:
        with lock:
            events.append(("start", number))
        time.sleep(0.06)
        with lock:
            events.append(("end", number))
        if fail:
            raise RuntimeError("deliberate admission failure")
        return Empty()

    work.step.resources = demand
    return work


def root():
    return Recipe(name="admission", inputs_model=Empty, outputs_model=Empty, max_workers=2)


@pytest.mark.parametrize("dimension", ["cpus", "memory"])
@pytest.mark.parametrize("scatter", [False, True])
def test_direct_units_serialize_against_configured_budget(dimension, scatter):
    events = []
    work = measured_step(Resources(**{dimension: 1}), events)
    recipe = root()
    if scatter:
        recipe.add_step("work", work, number=[0, 1], scatter=["number"], placement=SlurmPlacement())
    else:
        recipe.add_step("first", work, number=0, placement=SlurmPlacement())
        recipe.add_step("second", work, number=1)
    assert recipe(_config=config(**{dimension: 1})).success
    assert events == [("start", 0), ("end", 0), ("start", 1), ("end", 1)]


def test_nested_units_share_budget_without_parent_reservation():
    events = []
    work = measured_step(Resources(cpus=1), events)
    branch = root()
    branch.add_step("first", work, number=0)
    branch.add_step("second", work, number=1)
    recipe = root()
    recipe.add_step("branch", branch, placement=SlurmPlacement())
    recipe.add_step("direct", work, number=2)
    assert recipe(_config=config(cpus=1)).success
    assert sorted(number for action, number in events if action == "end") == [0, 1, 2]
    assert [action for action, _ in events] == ["start", "end"] * 3


def test_oversized_demand_retains_existing_warning_policy(caplog):
    events = []
    work = measured_step(Resources(cpus=2), events)
    recipe = root()
    recipe.max_workers = 1
    recipe.add_step("work", work, number=0, placement=SlurmPlacement())
    budget = Budget(Resources(cpus=1))
    assert recipe(_config=config(cpus=1), _budget=budget).success
    assert "more than the whole budget" in caplog.text
    granted, _ = budget.try_acquire(Resources(cpus=1))
    assert granted
    budget.release(Resources(cpus=1))


@pytest.mark.parametrize("failure", ["leaf", "submit"])
def test_failure_returns_shared_reservations(failure, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    events = []
    recipe = root()
    recipe.add_step("first", measured_step(Resources(cpus=1), events, fail=failure == "leaf"), number=0, placement=SlurmPlacement())
    recipe.add_step("second", measured_step(Resources(cpus=1), events), number=1)
    budget = Budget(Resources(cpus=2 if failure == "submit" else 1))
    if failure == "submit":
        original = ThreadPoolExecutor.submit
        calls = 0

        def submit(pool, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("deliberate submit failure")
            return original(pool, *args, **kwargs)

        monkeypatch.setattr(ThreadPoolExecutor, "submit", submit)
    with pytest.raises(Exception, match="failure|failed"):
        recipe(_config=config(cpus=2), _budget=budget)
    assert not any(number == 1 for _, number in events)
    granted, _ = budget.try_acquire(budget.total)
    assert granted
    budget.release(budget.total)
    assert not budget._waiters


@pytest.mark.parametrize("nested", [False, True])
def test_loops_refused_before_independent_earlier_work(tmp_path, nested):
    from tests.test_steps_loops import ConvergeAfter, make_recipe
    from shinobi.steps import register_step_backend

    calls = []

    @pystep()
    def earlier() -> Empty:
        calls.append("launched")
        return Empty()

    recipe = root()
    recipe.add_step("earlier", earlier, placement=SlurmPlacement())
    backend = ConvergeAfter(after=1)
    register_step_backend("mixed-loop-review", backend)
    loop = make_recipe("mixed-loop-review", tmp_path, max_iter=3)
    if nested:
        recipe.add_step("loop", loop, ms=tmp_path / "source.ms")
    else:
        recipe.inputs_model = loop.inputs_model
        recipe.steps.extend(loop.steps)
    with pytest.raises(BundleError, match="loop.*unsupported"):
        recipe(_config=config(cpus=1), **({"ms": tmp_path / "source.ms"} if not nested else {}))
    assert calls == []
    assert backend.ran == []
    assert not list(tmp_path.iterdir())


def test_cancelled_submission_returns_reservation(monkeypatch):
    from concurrent.futures import Future, ThreadPoolExecutor, CancelledError

    calls = []
    recipe = root()
    recipe.add_step("work", measured_step(Resources(cpus=1), calls), number=0, placement=SlurmPlacement())
    budget = Budget(Resources(cpus=1))

    def cancelled(*args, **kwargs):
        future = Future()
        future.cancel()
        future.set_running_or_notify_cancel()
        return future

    monkeypatch.setattr(ThreadPoolExecutor, "submit", cancelled)
    with pytest.raises(CancelledError):
        recipe(_config=config(cpus=1), _budget=budget)
    assert calls == []
    assert budget.try_acquire(Resources(cpus=1))[0]
    budget.release(Resources(cpus=1))
    assert not budget._waiters


def test_submit_failure_drains_leaf_before_joining_waiting_nested_recipe(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    events = []
    work = measured_step(Resources(cpus=1), events)
    branch = root()
    branch.add_step("nested", work, number=1)
    recipe = root()
    recipe.max_workers = 3
    recipe.add_step("first", work, number=0, placement=SlurmPlacement())
    recipe.add_step("branch", branch)
    recipe.add_step("third", measured_step(Resources(), events), number=2)
    budget = Budget(Resources(cpus=1))
    original = ThreadPoolExecutor.submit

    def submit(pool, *args, **kwargs):
        if len(args) > 2 and getattr(args[2], "name", None) == "admission.third":
            raise RuntimeError("third submit failure")
        return original(pool, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "submit", submit)
    with pytest.raises(RuntimeError, match="third submit failure"):
        recipe(_config=config(cpus=1), _budget=budget)
    assert sorted(number for action, number in events if action == "end") == [0, 1]
    assert budget.try_acquire(Resources(cpus=1))[0]
    budget.release(Resources(cpus=1))
    assert not budget._waiters


@pytest.mark.parametrize("dimension", ["cpus", "memory"])
def test_backward_dependencies_release_parent_ticket_when_composites_fill_slots(dimension):
    """A's completion makes lower-index B/D ready ahead of the refused C."""
    ticketed = threading.Event()
    completed = []
    demand = Resources(**{dimension: 1})

    class BoundedBudget(Budget):
        def try_acquire(self, request, label=""):
            granted, generation = super().try_acquire(request, label)
            if label == "admission.C" and not granted:
                ticketed.set()
            return granted, generation

        def wait_for_change(self, since, timeout=None):
            # Turn a stranded child into a bounded test failure, rather than
            # leaving the executor's teardown permanently blocked.
            super().wait_for_change(since, timeout=timeout if timeout is not None else 1)
            if timeout is None and self._generation == since:
                raise TimeoutError("parent FIFO ticket stranded a nested leaf")

    @pystep()
    def first() -> Empty:
        assert ticketed.wait(2), "C was not refused while A held the budget"
        completed.append("A")
        return Empty()

    @pystep()
    def leaf(label: str) -> Empty:
        completed.append(label)
        return Empty()

    first.step.resources = demand
    leaf.step.resources = demand
    recipe = root()
    for name in ("B", "D"):
        branch = root()
        branch.add_step("leaf", leaf, label=name)
        recipe.add_step(name, branch, placement=SlurmPlacement() if name == "B" else None)
        recipe.steps[-1].after = ["A"]
    recipe.add_step("A", first)
    recipe.add_step("C", leaf, label="C")
    budget = BoundedBudget(demand)
    assert recipe(_config=config(**{dimension: 1}), _budget=budget).success
    assert completed[0] == "A"
    assert sorted(completed) == ["A", "B", "C", "D"]
    assert not budget._waiters
    assert budget.try_acquire(demand)[0]
    budget.release(demand)
