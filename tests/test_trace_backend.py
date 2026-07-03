import pytest

from shinobi.backends import registered_backend_classes
from shinobi.backends.native import NativeBackend
from shinobi.backends.trace import TraceBackend, patch_all_backends
from shinobi.schema import CabDef, ParamSchema


def make_cab(**kwargs) -> CabDef:
    kwargs.setdefault("name", "tool")
    kwargs.setdefault("command", "tool")
    return CabDef(**kwargs)


def test_trace_backend_records_step_and_returns_success():
    tracer = TraceBackend()
    cab = make_cab(outputs={"image": ParamSchema(dtype="File")})

    result = tracer.run(cab, ["tool"], {})

    assert result.success
    assert result.returncode == 0
    assert len(tracer.steps) == 1
    assert tracer.steps[0].name == "tool"
    assert tracer.steps[0].depends_on == set()


def test_trace_backend_placeholder_output_is_usable_as_next_input():
    tracer = TraceBackend()
    cab = make_cab(outputs={"image": ParamSchema(dtype="File")})

    result = tracer.run(cab, ["tool"], {})
    # exactly what a recipe would do: pass result.image into the next call
    tracer.run(cab, ["tool"], {"path": result.outputs["image"]})

    assert len(tracer.steps) == 2
    assert tracer.steps[1].depends_on == {0}


def test_trace_backend_no_dependency_chains_to_previous_step():
    tracer = TraceBackend()
    cab = make_cab()

    tracer.run(cab, ["tool"], {})
    tracer.run(cab, ["tool"], {"unrelated": "value"})

    assert tracer.steps[1].depends_on == {0}


def test_patch_all_backends_intercepts_a_real_backend_class():
    tracer = TraceBackend()
    cab = make_cab()

    with patch_all_backends(tracer):
        native = NativeBackend()
        native.run(cab, ["/bin/false"], {})  # would fail for real, but is traced instead

    assert len(tracer.steps) == 1
    assert tracer.steps[0].name == "tool"


def test_patch_all_backends_restores_originals_after_context():
    original = NativeBackend.run
    with patch_all_backends(TraceBackend()):
        assert NativeBackend.run is not original
    assert NativeBackend.run is original


def test_patch_all_backends_restores_originals_even_if_body_raises():
    original = NativeBackend.run
    with pytest.raises(ValueError), patch_all_backends(TraceBackend()):
        raise ValueError("boom")
    assert NativeBackend.run is original


def test_patch_all_backends_covers_every_registered_class():
    tracer = TraceBackend()
    originals = {cls: cls.run for cls in registered_backend_classes()}
    with patch_all_backends(tracer):
        for cls, original in originals.items():
            assert cls.run is not original
    for cls, original in originals.items():
        assert cls.run is original
