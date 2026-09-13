"""Snapshot trusted pystep source without importing or executing its modules.

Only Python files reachable through literal imports under filesystem roots
are included. Other imports are an explicit requirement of the execution
environment. Dynamic imports of local helpers require an explicit inclusion.
The snapshot is code, with the same trust boundary as the original recipe;
it is never executed merely by reading a bundle.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import keyword
from pathlib import Path, PurePosixPath
from typing import Callable

from pydantic import model_validator

from shinobi.offload._codec import BundleError, WireModel


def source_tree_digest(root: Path) -> str:
    """Content identity for a staged source tree, excluding bytecode caches."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _entry_module(entry: str) -> str:
    """The sole import address of a source file within its explicit root."""
    path = PurePosixPath(entry).with_suffix("")
    parts = path.parts[:-1] if path.name == "__init__" else path.parts
    if not parts or any(not p.isidentifier() or keyword.iskeyword(p) for p in parts):
        raise BundleError(f"source entry {entry!r} does not have an importable module address")
    return ".".join(parts)


class CodeFile(WireModel):
    path: str
    source: str

    @model_validator(mode="after")
    def _relative_path(self) -> CodeFile:
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or "\\" in self.path or str(path) != self.path or path.suffix != ".py":
            raise BundleError(f"unsafe source path {self.path!r}")
        return self


class CodeBundle(WireModel):
    """Captured source and callable address, independent of the original files."""

    module: str
    qualname: str
    entry: str
    files: tuple[CodeFile, ...]
    environment_imports: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_files(self) -> CodeBundle:
        paths = [f.path for f in self.files]
        if len(set(paths)) != len(paths) or self.entry not in paths:
            raise BundleError("source snapshot has duplicate files or no entry module")
        expected = _entry_module(self.entry)
        if self.module != expected:
            raise BundleError(f"callable module {self.module!r} does not match source entry {self.entry!r} ({expected!r}); alias-loaded and __main__ callables are unsupported")
        if not all(p.isidentifier() for p in self.qualname.split(".")):
            raise BundleError(f"callable {self.qualname!r} is not addressable in a module")
        return self

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    def write(self, root: Path) -> Path:
        """Materialize into a new directory; never replace an existing snapshot."""
        root.mkdir(parents=True, exist_ok=False)
        for file in self.files:
            path = root / file.path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(file.source)
        return root / self.entry


def capture_code(func: Callable, *, roots: tuple[Path, ...] = (), include: tuple[str, ...] = ()) -> CodeBundle:
    """Capture a module-level callable and its static local imports.

    ``roots`` are explicit Python import roots, not package directories.
    They are mandatory: guessing from a module name could accidentally
    capture an entire site-packages tree. ``include``
    names additional local modules used through dynamic imports. No named
    package is imported to find a filesystem root.
    """
    if not inspect.isfunction(func) or "<locals>" in func.__qualname__ or func.__closure__:
        raise BundleError("a bundled pystep must be a module-addressable function without captured closure state")
    try:
        source = Path(inspect.getfile(func)).resolve(strict=True)
    except (TypeError, OSError) as exc:
        raise BundleError(f"pystep {func.__qualname__!r} has no readable source file") from exc
    if not roots:
        raise BundleError("pystep packaging requires explicit code_roots (Python import roots)")
    roots = tuple(root.resolve(strict=True) for root in roots)
    containing = next((root for root in roots if source.is_relative_to(root)), None)
    if containing is None:
        raise BundleError(f"pystep source {source} is outside supplied code roots")
    # A source snapshot cannot transport a live module's configured state.
    # Imported names and definitions are reproducible from source; a global
    # data value must be an unchanged immutable literal assignment.
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    declarations = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            declarations[node.name] = None
        elif isinstance(node, ast.Import):
            declarations.update({a.asname or a.name.split(".")[0]: None for a in node.names})
        elif isinstance(node, ast.ImportFrom):
            declarations.update({a.asname or a.name: None for a in node.names})
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    declarations[target.id] = node.value
    for name, value in inspect.getclosurevars(func).globals.items():
        if name not in declarations:
            raise BundleError(f"pystep global {name!r} is not declared by its captured source")
        assignment = declarations[name]
        if assignment is not None:
            try:
                literal = ast.literal_eval(assignment)
            except (ValueError, TypeError, SyntaxError) as exc:
                raise BundleError(f"pystep global {name!r} has runtime state; pass it as a declared input") from exc
            if type(value) not in (str, int, float, bool, type(None)) or type(value) is not type(literal) or value != literal:
                raise BundleError(f"pystep global {name!r} is mutable or changed; pass it as a declared input")
    entry = source.relative_to(containing).as_posix()
    pending = [(source, containing)]
    captured: dict[str, CodeFile] = {}
    external: set[str] = set()

    def find(module: str, *, required: bool = False) -> bool:
        if not module or not all(part.isidentifier() for part in module.split(".")):
            raise BundleError(f"invalid source module {module!r}")
        relative = Path(*module.split("."))
        for root in roots:
            for candidate in (root / relative.with_suffix(".py"), root / relative / "__init__.py"):
                if candidate.is_file():
                    if not candidate.resolve().is_relative_to(root):
                        raise BundleError(f"source import {module!r} escapes its code root through a symlink")
                    pending.append((candidate, root))
                    return True
        if required:
            raise BundleError(f"local source import {module!r} is missing from the supplied roots")
        external.add(module)
        return False

    for module in include:
        find(module, required=True)
    while pending:
        path, root = pending.pop()
        relative = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=relative)
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise BundleError(f"cannot capture Python source {path}: {exc}") from exc
        if relative in captured:
            if captured[relative].source != text:
                raise BundleError(f"code roots provide conflicting modules at {relative!r}")
            continue
        captured[relative] = CodeFile(path=relative, source=text)
        # Package initializers are part of the code being frozen too.
        parent = path.parent
        while parent != root:
            init = parent / "__init__.py"
            if init.is_file():
                if not init.resolve().is_relative_to(root):
                    raise BundleError(f"package initializer {init} escapes its source root")
                pending.append((init, root))
            parent = parent.parent
        module = relative.removesuffix(".py").replace("/", ".")
        package = module.rsplit(".", 1)[0] if "." in module else ""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    find(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    if not package:
                        raise BundleError(f"relative import in {relative!r} needs its package import root")
                    try:
                        target = importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                    except ImportError as exc:
                        raise BundleError(f"relative import escapes code root in {relative!r}") from exc
                    find(target, required=True)
                else:
                    target = node.module or ""
                    find(target)
                # `from package import helper` can name a module or an
                # ordinary attribute. Only actual local files are bundled.
                for alias in node.names:
                    if alias.name != "*":
                        candidate = target + "." + alias.name
                        before = set(external)
                        find(candidate)
                        external.intersection_update(before)
    return CodeBundle(module=func.__module__, qualname=func.__qualname__, entry=entry,
                      files=tuple(captured[p] for p in sorted(captured)), environment_imports=tuple(sorted(external)))
