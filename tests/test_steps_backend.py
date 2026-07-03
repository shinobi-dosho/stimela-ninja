from pydantic import BaseModel

from shinobi.steps.backend import NativeStepBackend, RecordingStepBackend
from shinobi.steps.schema import CabDef


class TextInputs(BaseModel):
    text: str = "hi"


class CommandOutputs(BaseModel):
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def test_native_step_backend_runs_a_real_command():
    cab = CabDef(name="echo", command="/bin/echo", inputs_model=TextInputs, outputs_model=CommandOutputs)
    backend = NativeStepBackend()

    raw = backend.run(cab, TextInputs(text="hello there"))

    assert raw["returncode"] == 0
    assert "hello there" in raw["stdout"]


def test_native_step_backend_skips_none_and_omits_false_bool_flags():
    class Opts(BaseModel):
        text: str | None = None
        verbose: bool = False

    cab = CabDef(name="echo", command="/bin/echo", inputs_model=Opts, outputs_model=CommandOutputs)
    backend = NativeStepBackend()

    raw = backend.run(cab, Opts(text=None, verbose=False))
    assert raw["returncode"] == 0
    assert raw["stdout"].strip() == ""


def test_native_step_backend_emits_true_bool_as_a_bare_flag():
    class Opts(BaseModel):
        verbose: bool = False

    cab = CabDef(name="echo", command="/bin/echo", inputs_model=Opts, outputs_model=CommandOutputs)
    backend = NativeStepBackend()

    raw = backend.run(cab, Opts(verbose=True))
    assert raw["stdout"].strip() == "--verbose"


def test_recording_step_backend_records_without_executing():
    cab = CabDef(name="echo", command="/bin/echo", inputs_model=TextInputs, outputs_model=CommandOutputs)
    backend = RecordingStepBackend()

    result = backend.run(cab, TextInputs(text="hello"))

    assert result == {}
    assert len(backend.calls) == 1
    recorded_defn, recorded_inputs = backend.calls[0]
    assert recorded_defn is cab
    assert recorded_inputs.text == "hello"
