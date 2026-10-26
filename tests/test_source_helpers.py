"""Production runner proof for bounded imported Python source helpers."""

import importlib.util
import inspect
import json
import pickle
import subprocess
import sys
import types
from pathlib import Path

import pytest
from pydantic import BaseModel

from shinobi.cache import _identity
from shinobi.source_helpers import SourceHelperError, resolve_source_helpers
from shinobi.steps.pyfunc import _RUNNER_TEMPLATE, _ctx_shim
from shinobi.steps.schema import Scope


TARGET = """from pydantic import BaseModel
from .bridge import helper
class Output(BaseModel):
    result: int
def target(x: int) -> Output:
    return Output(result=helper(x))
"""
BRIDGE = """from .nested.leaf import leaf
def helper(x):
    return leaf(x) + 1
"""
LEAF = """def leaf(x):
    return x * 2
"""


def package(tmp_path, monkeypatch, target=TARGET, bridge=BRIDGE, leaf=LEAF, extra=None, nested_const=False):
    root = tmp_path / "helperpkg"
    (root / "nested").mkdir(parents=True)
    for name in ("__init__.py", "nested/__init__.py", "compiled_guard.py"):
        (root / name).write_text("raise RuntimeError('host package initializer or compiled dependency executed')\n")
    (root / "target.py").write_text(target)
    (root / "bridge.py").write_text(bridge)
    (root / "nested/leaf.py").write_text(leaf)
    for name, source in (extra or {}).items():
        (root / name).write_text(source)
    for key in list(sys.modules):
        if key == "helperpkg" or key.startswith("helperpkg."):
            monkeypatch.delitem(sys.modules, key)
    for name, directory in (("helperpkg", root), ("helperpkg.nested", root / "nested")):
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        if name == "helperpkg.nested" and nested_const:
            module.CONST = None
        monkeypatch.setitem(sys.modules, name, module)
    # Host setup loads only inert fixture function modules too. The child
    # must independently reconstruct package parents without executing init.
    guard = types.ModuleType("helperpkg.compiled_guard")
    monkeypatch.setitem(sys.modules, guard.__name__, guard)
    spec = importlib.util.spec_from_file_location("helperpkg.target", root / "target.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module.target, root


def runner(tmp_path, func, inputs, *, ctx=False, manifest=None, bypass=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    inputs_path = tmp_path / "inputs.pkl"
    inputs_path.write_bytes(pickle.dumps(inputs, protocol=4))
    closure = None if bypass else resolve_source_helpers(func)
    path = tmp_path / "runner.py"
    path.write_text(
        _RUNNER_TEMPLATE.format(
            stub_prefixes={"shinobi", "pydantic", "dosho", func.__module__.split(".")[0]},
            helper_manifest=(closure.manifest if closure is not None else {}) if manifest is None else manifest,
            target_module=func.__module__ if closure is not None else None,
            source_root=None,
            source_file=inspect.getfile(func),
            qualname_parts=func.__qualname__.split("."),
            func_name=func.__name__,
            inputs_path=str(inputs_path),
            outputs_path=str(tmp_path / "outputs.json"),
            ctx_shim=_ctx_shim() if ctx else "",
            ctx_arg="ctx, " if ctx else "",
        )
    )
    return path


def execute(path):
    return subprocess.run([sys.executable, str(path)], capture_output=True, text=True)


def test_runner_relative_imports_transitive_helpers_and_no_initializers(tmp_path, monkeypatch):
    func, root = package(tmp_path, monkeypatch)
    closure = resolve_source_helpers(func)
    assert set(closure.manifest) == {"helperpkg.target", "helperpkg.bridge", "helperpkg.nested.leaf"}
    path = runner(tmp_path / "io", func, {"x": 2})
    result = execute(path)
    assert result.returncode == 0, result.stderr
    assert json.loads((path.parent / "outputs.json").read_text()) == {"result": 5}
    assert str(root / "__init__.py") not in closure.manifest


def test_runner_refuses_source_changed_after_discovery(tmp_path, monkeypatch):
    func, root = package(tmp_path, monkeypatch)
    path = runner(tmp_path / "io", func, {"x": 2})
    (root / "nested/leaf.py").write_text(LEAF.replace("* 2", "* 3"))
    result = execute(path)
    assert result.returncode != 0
    assert "source changed after helper discovery" in result.stderr


def test_helper_source_changes_identity_and_relocation_does_not(tmp_path, monkeypatch):
    func, root = package(tmp_path / "first", monkeypatch)
    scope = Scope(name="test", inputs_model=BaseModel, outputs_model=BaseModel)
    first = _identity(scope, func, source_helpers=resolve_source_helpers(func))
    (root / "nested/leaf.py").write_text(LEAF.replace("* 2", "* 3"))
    assert _identity(scope, func, source_helpers=resolve_source_helpers(func)) != first
    relocated, _ = package(tmp_path / "relocated", monkeypatch)
    assert _identity(scope, relocated, source_helpers=resolve_source_helpers(relocated)) == first


def test_module_imports_are_captured_and_same_module_identity_stays_historical(tmp_path, monkeypatch):
    target = TARGET.replace("return Output(result=helper(x))", "return Output(result=x)")
    func, _ = package(tmp_path / "same", monkeypatch, target=target)
    scope = Scope(name="test", inputs_model=BaseModel, outputs_model=BaseModel)
    # Importing a sibling executes its module even if a callable does not use
    # the alias or shadows it with an input. Native identity still stays old.
    assert resolve_source_helpers(func) is not None
    assert _identity(scope, func) == ["func", inspect.getsource(func)]
    target = TARGET.replace("def target(x: int)", "def target(helper: int)").replace("helper(x)", "helper")
    func, _ = package(tmp_path / "shadow", monkeypatch, target=target)
    assert resolve_source_helpers(func) is not None
    target = TARGET.replace("from .bridge import helper", "def helper(x):\n    return x * 2")
    func, _ = package(tmp_path / "local", monkeypatch, target=target)
    assert _identity(scope, func) == ["func", inspect.getsource(func)]


@pytest.mark.parametrize("damage", ["missing", "escape", "nonfunction", "module", "dynamic"])
def test_unsupported_source_dependencies_refuse(tmp_path, monkeypatch, damage):
    func, root = package(tmp_path, monkeypatch)
    if damage == "missing":
        (root / "bridge.py").unlink()
    elif damage == "escape":
        outside = tmp_path / "outside.py"
        outside.write_text(BRIDGE)
        (root / "bridge.py").unlink()
        (root / "bridge.py").symlink_to(outside)
    elif damage == "nonfunction":
        (root / "bridge.py").write_text("helper = 42\n")
    elif damage == "module":
        (root / "target.py").write_text(TARGET.replace("from .bridge import helper", "import helperpkg.bridge as helper"))
    else:
        (root / "target.py").write_text(TARGET.replace("return Output(result=helper(x))", "return __import__('helperpkg.bridge')"))
    with pytest.raises(SourceHelperError):
        resolve_source_helpers(func)


CASA = """import json
class Array:
    def __init__(self, values, flag):
        self.values = values
        self.shape = (1, 1, len(values)) if flag else (len(values),)
    def tobytes(self):
        return bytes(self.values)
class table:
    def open(self, path, nomodify=True):
        assert nomodify is True
        from pathlib import Path
        self.data = json.loads((Path(path) / 'flags.json').read_text())
    def close(self): pass
    def colnames(self): return ['FLAG', 'FLAG_ROW']
    def getkeywords(self): return {}
    def getcolkeywords(self, column): return {}
    def nrows(self): return len(self.data['FLAG_ROW'])
    def getcol(self, column, startrow=0, nrow=None):
        return Array(self.data[column][startrow:startrow+nrow], column == 'FLAG')
"""


def test_actual_saved_history_verifier_runs_real_helper_in_production_runner(tmp_path, monkeypatch):
    import dosho

    companion = pytest.importorskip("dosho.cabs.casa_selfcal", reason="requires companion Dosho history integration")
    _path_flag_digest = companion._path_flag_digest

    io = tmp_path / "io"
    io.mkdir()
    (io / "casatools.py").write_text(CASA)
    spec = importlib.util.spec_from_file_location("casatools", io / "casatools.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)

    class Ctx:
        def import_func(self, name, package):
            assert (name, package) == ("table", "casatools")
            return native.table

    vis = tmp_path / "obs.ms"
    vis.mkdir()
    history = Path(str(vis) + ".flagversions")
    version = history / "flags.ninja_legacy"
    version.mkdir(parents=True)
    flags = version / "flags.json"
    flags.write_text(json.dumps({"FLAG": [0, 1], "FLAG_ROW": [0, 0]}))
    expected = _path_flag_digest(Ctx(), version)
    func = dosho.get("verify-saved-flag-history")
    inputs = {"vis": vis, "history": history, "versionname": "ninja_legacy", "expected_digest": expected}
    path = runner(io, func, inputs, ctx=True)
    result = execute(path)
    assert result.returncode == 0, result.stderr
    assert json.loads((io / "outputs.json").read_text()) == {"valid": True, "version_digest": expected}
    path = runner(io, func, {**inputs, "expected_digest": "0" * 64}, ctx=True)
    assert "differs from cached SAVE" in execute(path).stderr
    flags.write_text(json.dumps({"FLAG": [1, 1], "FLAG_ROW": [0, 0]}))
    path = runner(io, func, inputs, ctx=True)
    assert "differs from cached SAVE" in execute(path).stderr


@pytest.mark.parametrize(
    "body",
    [
        "return Output(result=helper(x) + sum(helper for helper in [1, 2]))",
        "def nested():\n        helper = 3\n        return helper\n    return Output(result=helper(x) + nested())",
    ],
)
def test_lexical_comprehension_and_nested_locals_do_not_hide_global_helper(tmp_path, monkeypatch, body):
    func, _ = package(tmp_path, monkeypatch, target=TARGET.replace("return Output(result=helper(x))", body))
    assert resolve_source_helpers(func) is not None
    path = runner(tmp_path / "io", func, {"x": 2})
    result = execute(path)
    assert result.returncode == 0, result.stderr
    assert json.loads((path.parent / "outputs.json").read_text()) == {"result": 8}


def test_loaded_helper_module_from_conflicting_tree_refuses(tmp_path, monkeypatch):
    func, _ = package(tmp_path, monkeypatch)
    monkeypatch.setattr(sys.modules["helperpkg.bridge"], "__file__", str(tmp_path / "foreign.py"))
    with pytest.raises(SourceHelperError, match="conflicting"):
        resolve_source_helpers(func)


def test_dynamic_external_ctx_import_preserved_and_runtime_stub_refuses(tmp_path, monkeypatch):
    target = TARGET.replace("from .bridge import helper\n", "").replace(
        "def target(x: int) -> Output:\n    return Output(result=helper(x))",
        "def target(ctx, module: str) -> Output:\n    return Output(result=len(ctx.import_module(module).join('a', 'bc')))",
    )
    func, _ = package(tmp_path, monkeypatch, target=target)
    assert resolve_source_helpers(func) is None
    path = runner(tmp_path / "io", func, {"module": "os.path"}, ctx=True)
    result = execute(path)
    assert result.returncode == 0, result.stderr
    path = runner(tmp_path / "io", func, {"module": "helperpkg.compiled_guard"}, ctx=True)
    result = execute(path)
    assert result.returncode != 0 and "runtime module resolved to a package stub" in result.stderr


def test_literal_ctx_same_package_helper_reference_refuses(tmp_path, monkeypatch):
    target = TARGET.replace("return Output(result=helper(x))", "return ctx.import_func('helper', 'helperpkg.bridge')(x)")
    func, _ = package(tmp_path, monkeypatch, target=target)
    with pytest.raises(SourceHelperError, match="ctx same-package"):
        resolve_source_helpers(func)


def test_actual_save_keeps_exact_historical_identity():
    import dosho

    func = dosho.get("flagmanager-save")
    scope = Scope(name="save", inputs_model=BaseModel, outputs_model=BaseModel)
    assert resolve_source_helpers(func) is None
    assert _identity(scope, func) == ["func", inspect.getsource(func)]


def test_container_dispatch_uses_exact_readonly_source_files(tmp_path, monkeypatch):
    from unittest.mock import patch
    from shinobi import pystep

    func, root = package(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    captured = []

    def launch(argv, **kwargs):
        captured.append(argv)
        return execute(Path(next(argument for argument in argv if argument.endswith("runner.py"))))

    ref = pystep(image="native:test", backend="docker")(func)
    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=launch):
        result = ref(x=2)
    assert result.outputs.result == 5
    argv = captured[0]
    for path in (root / "target.py", root / "bridge.py", root / "nested/leaf.py"):
        assert f"{path}:{path}:ro" in argv
    assert f"{root}:{root}" not in argv
    assert all("compiled_guard" not in argument for argument in argv)


def test_imported_helper_top_level_initializer_has_transitive_exact_sources(tmp_path, monkeypatch):
    bridge = "from .nested.leaf import leaf\nFACTOR = leaf(3)\ndef helper(x):\n    return x * FACTOR\n"
    func, _ = package(tmp_path, monkeypatch, bridge=bridge)
    closure = resolve_source_helpers(func)
    assert "helperpkg.nested.leaf" in closure.manifest
    path = runner(tmp_path / "io", func, {"x": 2})
    result = execute(path)
    assert result.returncode == 0, result.stderr
    assert json.loads((path.parent / "outputs.json").read_text()) == {"result": 12}


@pytest.mark.parametrize("shape", ["if", "try", "class", "module"])
def test_admitted_helper_unsupported_import_shapes_refuse_before_runner(tmp_path, monkeypatch, shape):
    from unittest.mock import patch
    from shinobi import pystep

    if shape == "if":
        bridge = "if True:\n    from .nested.leaf import leaf\ndef helper(x):\n    return 0 if leaf(x) == 4 else 1\n"
    elif shape == "try":
        bridge = "try:\n    from .nested.leaf import leaf\nexcept ImportError:\n    pass\ndef helper(x):\n    return 0 if leaf(x) == 4 else 1\n"
    elif shape == "class":
        bridge = "class Holder:\n    from .nested.leaf import leaf\ndef helper(x):\n    return 0 if Holder.leaf(x) == 4 else 1\n"
    else:
        bridge = "import helperpkg.nested.leaf\ndef helper(x):\n    return 0\n"
    func, _ = package(tmp_path, monkeypatch, bridge=bridge)
    assert func(2).result == 0
    ref = pystep(image="native:test", backend="docker")(func)
    with patch("shinobi.steps.pyfunc.run_streaming") as launched:
        with pytest.raises(SourceHelperError, match="same-package.*import"):
            ref(x=2)
    launched.assert_not_called()


ROOT_STATE = "import math as module\nVALUE = 3\nclass Thing:\n    value = 4\n"


@pytest.mark.parametrize("mode", ["native", "venv"])
def test_cached_native_and_venv_keep_same_package_state_imports(tmp_path, monkeypatch, make_venv, mode):
    from shinobi import pystep
    from shinobi.steps.dispatch import _dispatch

    target = TARGET.replace("from .bridge import helper", "from helperpkg import VALUE, Thing, module").replace(
        "return Output(result=helper(x))", "return Output(result=VALUE + Thing.value + int(module.sqrt(x)))"
    )
    # Set up the host's source package without executing its root initializer.
    root = types.ModuleType("helperpkg")
    exec(ROOT_STATE, root.__dict__)
    target_root = tmp_path / "source/helperpkg"
    target_root.mkdir(parents=True)
    root.__path__ = [str(target_root)]
    monkeypatch.setitem(sys.modules, "helperpkg", root)
    path = target_root / "target.py"
    path.write_text(target)
    spec = importlib.util.spec_from_file_location("helperpkg.target", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    venv = make_venv(package=("helperpkg", "1.0", ROOT_STATE)) if mode == "venv" else None
    # An image is deliberately present: the actual backend override must win.
    ref = pystep(image="native:test", venv=str(venv) if venv else None, backend="docker")(module.target)
    kwargs = {"backend": mode, "cache": True, "cache_dir": str(tmp_path / "cache"), "x": 4}
    first = _dispatch(ref.step, ref.func, **kwargs)
    second = _dispatch(ref.step, ref.func, **kwargs)
    assert first.success and first.outputs.result == 9
    assert second.cached and second.cache_key == first.cache_key
    assert _identity(ref.step, ref.func) == ["func", inspect.getsource(module.target)]


def test_container_helper_key_and_cache_off_evidence_are_attempt_local(tmp_path, monkeypatch):
    from unittest.mock import patch
    from shinobi import pystep
    from shinobi.steps.dispatch import _dispatch

    func, root = package(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    ref = pystep(image="native:test", backend="docker")(func)

    def launch(argv, **kwargs):
        return execute(Path(next(argument for argument in argv if argument.endswith("runner.py"))))

    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=launch):
        first = _dispatch(ref.step, ref.func, cache=True, cache_dir=str(tmp_path / "cache"), x=2)
        (root / "nested/leaf.py").write_text(LEAF.replace("* 2", "* 3"))
        uncached = _dispatch(ref.step, ref.func, cache=False, x=2)
        second = _dispatch(ref.step, ref.func, cache=True, cache_dir=str(tmp_path / "cache"), x=2)
    assert first.outputs.result == 5 and uncached.outputs.result == second.outputs.result == 7
    assert first.cache_key != second.cache_key
    assert not hasattr(func, "__shinobi_source_helper_identity__")


def test_target_conditional_import_refuses_without_direct_helper_binding(tmp_path, monkeypatch):
    from unittest.mock import patch
    from shinobi import pystep

    target = TARGET.replace("from .bridge import helper", "if True:\n    from .nested.leaf import leaf").replace(
        "return Output(result=helper(x))", "return Output(result=0 if leaf(x) == 4 else 1)"
    )
    func, _ = package(tmp_path, monkeypatch, target=target)
    assert func(2).result == 0
    ref = pystep(image="native:test", backend="docker")(func)
    with patch("shinobi.steps.pyfunc.run_streaming") as launched:
        with pytest.raises(SourceHelperError, match="conditional/class.*target"):
            ref(x=2)
    launched.assert_not_called()


def test_generic_host_orchestration_with_image_keeps_historical_cache(tmp_path, monkeypatch):
    from shinobi.results import StepResult
    from shinobi.steps.dispatch import _dispatch
    import shinobi.source_helpers

    def unexpected(*args, **kwargs):
        raise AssertionError("host orchestration must not use container source resolver")

    monkeypatch.setattr(shinobi.source_helpers, "resolve_source_helpers", unexpected)

    class Empty(BaseModel):
        pass

    scope = Scope(name="host-orchestration", inputs_model=Empty, outputs_model=Empty, image="native:test", backend="docker")

    def orchestration(ctx):
        return StepResult(name=scope.name, returncode=0, inputs=Empty(), outputs=Empty())

    first = _dispatch(scope, orchestration, cache=True, cache_dir=str(tmp_path / "cache"))
    second = _dispatch(scope, orchestration, cache=True, cache_dir=str(tmp_path / "cache"))
    assert first.success and second.cached and first.cache_key == second.cache_key


@pytest.mark.parametrize("shape", ["if", "try", "class", "with", "loop", "default", "decorator"])
def test_recursive_initializer_dynamic_imports_refuse_before_runner(tmp_path, monkeypatch, shape):
    from unittest.mock import patch
    from shinobi import pystep

    dynamic = "importlib.import_module('helperpkg.nested.leaf')"
    prefix = "import importlib\n"
    suffix = "def helper(x):\n    return 0 if leaf(x) == 4 else 1\n"
    if shape == "if":
        bridge = prefix + f"if True:\n    leaf = {dynamic}.leaf\n" + suffix
    elif shape == "try":
        bridge = prefix + f"try:\n    leaf = {dynamic}.leaf\nexcept ImportError:\n    pass\n" + suffix
    elif shape == "class":
        bridge = prefix + f"class Holder:\n    leaf = {dynamic}.leaf\nleaf = Holder.leaf\n" + suffix
    elif shape == "with":
        bridge = prefix + f"with open(__file__) as source:\n    leaf = {dynamic}.leaf\n" + suffix
    elif shape == "loop":
        bridge = prefix + f"for ignored in [0]:\n    leaf = {dynamic}.leaf\n" + suffix
    elif shape == "default":
        bridge = prefix + f"def helper(x, leaf={dynamic}.leaf):\n    return 0 if leaf(x) == 4 else 1\n"
    else:
        bridge = prefix + f"@{dynamic}.keep\ndef helper(x):\n    return 0 if x == 2 else 1\n"
    func, _ = package(tmp_path, monkeypatch, bridge=bridge, leaf=LEAF + "\ndef keep(function):\n    return function\n")
    assert func(2).result == 0
    ref = pystep(image="native:test", backend="docker")(func)
    with patch("shinobi.steps.pyfunc.run_streaming") as launched:
        with pytest.raises(SourceHelperError, match="dynamic initializer"):
            ref(x=2)
    launched.assert_not_called()


def test_target_global_initializer_gets_real_helper_source_and_cache_identity(tmp_path, monkeypatch):
    from unittest.mock import patch
    from shinobi import pystep
    from shinobi.steps.dispatch import _dispatch

    target = TARGET.replace("from .bridge import helper", "from .bridge import helper as build\nSTATE = build(2)").replace(
        "return Output(result=helper(x))", "return Output(result=STATE)"
    )
    func, root = package(tmp_path, monkeypatch, target=target)
    assert func(0).result == 5
    assert resolve_source_helpers(func) is not None
    monkeypatch.chdir(tmp_path)
    ref = pystep(image="native:test", backend="docker")(func)

    def launch(argv, **kwargs):
        return execute(Path(next(argument for argument in argv if argument.endswith("runner.py"))))

    with patch("shinobi.steps.pyfunc.run_streaming", side_effect=launch):
        first = _dispatch(ref.step, ref.func, cache=True, cache_dir=str(tmp_path / "cache"), x=0)
        (root / "nested/leaf.py").write_text(LEAF.replace("* 2", "* 3"))
        second = _dispatch(ref.step, ref.func, cache=True, cache_dir=str(tmp_path / "cache"), x=0)
    assert first.outputs.result == 5 and second.outputs.result == 7
    assert first.cache_key != second.cache_key


@pytest.mark.parametrize("computation", ["missing.VALUE(2)", "1 if missing.VALUE else 0", "1 if missing.VALUE == 0 else 0", "missing.VALUE + 1", "str(missing.VALUE)"])
def test_runner_package_placeholder_computation_fails_loudly(tmp_path, monkeypatch, computation):
    target = TARGET.replace("from .bridge import helper", "import helperpkg.compiled_guard as missing").replace(
        "return Output(result=helper(x))", "return Output(result=" + computation + ")"
    )
    func, _ = package(tmp_path, monkeypatch, target=target)
    # Deliberately bypass discovery to exercise the runner's last-line defense.
    path = runner(tmp_path / "io", func, {"x": 2}, bypass=True)
    result = execute(path)
    assert result.returncode != 0 and "package placeholder used for runtime computation" in result.stderr


def test_exact_utf8_crlf_source_bytes_survive_runner_digest_checks(tmp_path, monkeypatch):
    import hashlib

    func, root = package(tmp_path, monkeypatch)
    for source in (root / "target.py", root / "bridge.py", root / "nested/leaf.py"):
        source.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    closure = resolve_source_helpers(func)
    for source in (closure.target, *closure.helpers):
        assert source.digest == hashlib.sha256(source.path.read_bytes()).hexdigest()
    path = runner(tmp_path / "io", func, {"x": 2})
    result = execute(path)
    assert result.returncode == 0, result.stderr
    assert json.loads((path.parent / "outputs.json").read_text()) == {"result": 5}


@pytest.mark.parametrize("dependency", ["unmapped-module", "synthetic-parent"])
def test_exact_import_map_refuses_aliased_dynamic_fallback(tmp_path, monkeypatch, dependency):
    selected = "helperpkg.missing" if dependency == "unmapped-module" else "helperpkg.nested"
    bridge = (
        "import importlib\nfrom .nested.leaf import leaf\nload = importlib.import_module\n"
        f"if True:\n    VALUE = load('{selected}').CONST is None\n"
        "def helper(x):\n    return 0 if VALUE else 1\n"
    )
    func, _ = package(tmp_path, monkeypatch, bridge=bridge, extra={"missing.py": "CONST = None\n"}, nested_const=True)
    assert func(2).result == 0
    closure = resolve_source_helpers(func)
    assert selected not in closure.manifest
    path = runner(tmp_path / "io", func, {"x": 2})
    result = execute(path)
    assert result.returncode != 0
    if dependency == "unmapped-module":
        assert "unmapped same-package source import: helperpkg.missing" in result.stderr
    else:
        assert "has no attribute 'CONST'" in result.stderr


def test_caught_unmapped_source_import_cannot_publish_fallback_result(tmp_path, monkeypatch):
    bridge = (
        "import importlib\nfrom .nested.leaf import leaf\nload = importlib.import_module\n"
        "try:\n    VALUE = load('helperpkg.missing').CONST is None\nexcept ImportError:\n    VALUE = False\n"
        "def helper(x):\n    return 0 if VALUE else 1\n"
    )
    func, _ = package(tmp_path, monkeypatch, bridge=bridge, extra={"missing.py": "CONST = None\n"})
    assert func(2).result == 0
    path = runner(tmp_path / "io", func, {"x": 2})
    result = execute(path)
    assert result.returncode != 0
    assert "unsupported same-package source dependencies were caught" in result.stderr
    assert not (path.parent / "outputs.json").exists()


def test_caught_source_digest_failure_cannot_publish_fallback_result(tmp_path, monkeypatch):
    target = TARGET.replace("from .bridge import helper\n", "").replace(
        "return Output(result=helper(x))",
        "try:\n        from .nested.leaf import leaf\n    except ImportError:\n        return Output(result=0)\n    return Output(result=leaf(x))",
    )
    func, root = package(tmp_path, monkeypatch, target=target)
    path = runner(tmp_path / "io", func, {"x": 2})
    (root / "nested/leaf.py").write_text(LEAF.replace("* 2", "* 3"))
    result = execute(path)
    assert result.returncode != 0
    assert "unsupported same-package source dependencies were caught" in result.stderr
    assert not (path.parent / "outputs.json").exists()


def test_caught_unmapped_import_during_serialization_cannot_publish_result(tmp_path, monkeypatch):
    target = TARGET.replace(
        "    result: int\n",
        "    result: int\n"
        "    def model_dump(self, **kwargs):\n"
        "        import importlib\n"
        "        load = importlib.import_module\n"
        "        try:\n"
        "            load('helperpkg.missing')\n"
        "        except ImportError:\n"
        "            pass\n"
        "        return {'result': self.result}\n",
    )
    func, _ = package(tmp_path, monkeypatch, target=target, extra={"missing.py": "CONST = None\n"})
    path = runner(tmp_path / "io", func, {"x": 2})
    result = execute(path)
    assert result.returncode != 0
    assert "unsupported same-package source dependencies were caught" in result.stderr
    assert not (path.parent / "outputs.json").exists()
