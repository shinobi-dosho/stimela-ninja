"""Exact source closure for imported same-package Python function helpers.

Container runners cannot import the host package and its compiled wheels. This
bounded resolver admits literal function imports only, without importing any
module or executing package initializers. Offloaded source bundles and venv
imports retain their separate execution contracts.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
import symtable
import sys
from pathlib import Path

from shinobi.offload.code import _entry_module


class SourceHelperError(ValueError):
    """A same-package runtime dependency cannot be transported exactly."""


@dataclass(frozen=True)
class SourceFile:
    module: str
    path: Path
    source: str

    @property
    def digest(self):
        return hashlib.sha256(self.source.encode()).hexdigest()


@dataclass(frozen=True)
class SourceHelpers:
    target: SourceFile
    helpers: tuple[SourceFile, ...]

    @property
    def identity(self):
        return [(file.module, file.source) for file in (self.target, *self.helpers)]

    @property
    def manifest(self):
        return {file.module: (str(file.path), file.digest) for file in (self.target, *self.helpers)}


def resolve_source_helpers(func) -> SourceHelpers | None:
    """Follow literal function dependencies; return None for historical identity.

    Same-module definitions are already loaded with the target file, so they
    are traversed but never added as imported modules. A helper may re-export
    another helper through a literal import. Module objects, stars and imported
    nonfunction state are unsupported as same-package runtime dependencies.
    """
    func = inspect.unwrap(func)
    module = getattr(func, "__module__", "")
    if not inspect.isfunction(func) or module == "__main__" or "." not in module:
        return None
    source = Path(inspect.getfile(func)).absolute()
    parts = module.split(".")
    root = source.parent
    for _ in parts[:-1]:
        root = root.parent
    package = parts[0]
    files = {}
    trees = {}
    symbols = {}
    seen = set()
    imported = set()
    expanded = set()

    def load(name):
        if name in files:
            return trees[name]
        path = root.joinpath(*name.split(".")).with_suffix(".py")
        if not path.is_file() or path.resolve() != path:
            raise SourceHelperError(f"same-package source module {name!r} is missing, non-Python or escapes its exact source root")
        if _entry_module(path.relative_to(root).as_posix()) != name:
            raise SourceHelperError(f"conflicting source module address {name!r}")
        loaded = sys.modules.get(name)
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file is not None and Path(loaded_file).resolve() != path:
            raise SourceHelperError(f"conflicting loaded source module {name!r}")
        try:
            text = path.read_bytes().decode("utf-8")
            tree = ast.parse(text, filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise SourceHelperError(f"unreadable same-package source module {name!r}: {exc}") from exc
        files[name] = SourceFile(name, path, text)
        trees[name] = tree
        symbols[name] = symtable.symtable(text, str(path), "exec")
        return tree

    def bindings(name):
        definitions = {}
        imports = {}
        for node in load(name).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions[node.name] = node
            elif isinstance(node, ast.ImportFrom):
                try:
                    origin = importlib.util.resolve_name("." * node.level + (node.module or ""), name.rsplit(".", 1)[0]) if node.level else node.module or ""
                except ImportError as exc:
                    raise SourceHelperError(f"relative source import escapes package {name!r}") from exc
                for alias in node.names:
                    imports[alias.asname or alias.name] = (origin, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name.split(".")[0]] = (alias.name, None)
        return definitions, imports

    def visit(name, symbol):
        address = (name, symbol)
        if address in seen:
            return
        seen.add(address)
        definitions, imports = bindings(name)
        node = definitions.get(symbol)
        if node is None:
            if symbol in imports:
                follow(name, *imports[symbol])
                return
            raise SourceHelperError(f"same-package helper {name}.{symbol} has no source function definition")
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise SourceHelperError(f"same-package helper {name}.{symbol} is not a Python function")
        # The ordinary target loader already executes same-module definitions.
        # Traverse their runtime bodies to find imported helper calls too.
        table = next((child for child in symbols[name].get_children() if child.get_name() == node.name and child.get_lineno() == node.lineno), None)
        if table is None:
            raise SourceHelperError(f"cannot establish lexical bindings for {name}.{symbol}")
        global_names = set()
        pending_scopes = [table]
        while pending_scopes:
            lexical = pending_scopes.pop()
            global_names.update(item.get_name() for item in lexical.get_symbols() if item.is_global() and item.is_referenced())
            pending_scopes.extend(lexical.get_children())
        for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if isinstance(child, ast.Call):
                called = child.func
                importer = (
                    isinstance(called, ast.Name)
                    and (called.id == "__import__" and called.id in global_names or imports.get(called.id) == ("importlib", "import_module"))
                    or isinstance(called, ast.Attribute)
                    and called.attr == "import_module"
                    and isinstance(called.value, ast.Name)
                    and (called.value.id == "ctx" or imports.get(called.value.id) == ("importlib", None))
                )
                value = child.args[0].value if child.args and isinstance(child.args[0], ast.Constant) else None
                if importer and isinstance(value, str) and (value == package or value.startswith(package + ".") or value.startswith(".")):
                    raise SourceHelperError("dynamic same-package source imports are unsupported; import a Python function explicitly")
                if isinstance(called, ast.Attribute) and isinstance(called.value, ast.Name) and called.value.id == "ctx" and called.attr in {"import_func", "import_callable"}:
                    argument = child.args[1] if len(child.args) > 1 else next((item.value for item in child.keywords if item.arg == "module"), None)
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str) and (argument.value == package or argument.value.startswith(package + ".")):
                        raise SourceHelperError("ctx same-package source imports are unsupported; import a Python function explicitly")
            if isinstance(child, ast.ImportFrom):
                if child.level:
                    origin = importlib.util.resolve_name("." * child.level + (child.module or ""), name.rsplit(".", 1)[0])
                else:
                    origin = child.module or ""
                for alias in child.names:
                    follow(name, origin, alias.name)
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    if alias.name.startswith(package + "."):
                        raise SourceHelperError(f"same-package module import {alias.name!r} is unsupported; import a Python function explicitly")
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id not in global_names:
                    continue
                if child.id in imports:
                    follow(name, *imports[child.id])
                elif child.id in definitions and isinstance(definitions[child.id], (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit(name, child.id)

    def expand_imports(name):
        """Analyze executable module initialization, with one shared traversal.

        Control flow, class bodies, defaults and decorators execute on import.
        Ordinary function/lambda bodies do not: their dependency analysis stays
        in visit(). Unsupported source imports must never become package stubs.
        """
        if name in expanded:
            return
        expanded.add(name)
        definitions, imports = bindings(name)

        def expression(node, metadata=False):
            if node is None:
                return
            if isinstance(node, ast.Lambda):
                for value in (*node.args.defaults, *node.args.kw_defaults):
                    expression(value, metadata)
                if not metadata:
                    raise SourceHelperError(f"unsupported lambda initializer in {name!r}")
                return
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                origin, symbol = imports.get(node.id, ("", ""))
                if origin.startswith(package + "."):
                    follow(name, origin, symbol)
                elif origin == package and not metadata:
                    raise SourceHelperError(f"unsupported package-stub initializer dependency in {name!r}")
                if node.id in definitions and isinstance(definitions[node.id], (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit(name, node.id)
            if isinstance(node, ast.Call):
                called = node.func
                dynamic = (
                    isinstance(called, ast.Name)
                    and (called.id == "__import__" or imports.get(called.id) == ("importlib", "import_module"))
                    or isinstance(called, ast.Attribute)
                    and called.attr == "import_module"
                    and isinstance(called.value, ast.Name)
                    and imports.get(called.value.id) == ("importlib", None)
                )
                if dynamic:
                    argument = node.args[0] if node.args else next((item.value for item in node.keywords if item.arg == "name"), None)
                    if (
                        not isinstance(argument, ast.Constant)
                        or not isinstance(argument.value, str)
                        or argument.value == package
                        or argument.value.startswith(package + ".")
                        or argument.value.startswith(".")
                    ):
                        raise SourceHelperError(f"unsupported dynamic initializer dependency in {name!r}")
            for child in ast.iter_child_nodes(node):
                expression(child, metadata)

        def statements(nodes, nested=False, metadata=False):
            for node in nodes:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith(package + "."):
                            raise SourceHelperError(f"same-package module import in source {name!r} is unsupported")
                        imports[alias.asname or alias.name.split(".")[0]] = (alias.name, None)
                    continue
                if isinstance(node, ast.ImportFrom):
                    origin = importlib.util.resolve_name("." * node.level + (node.module or ""), name.rsplit(".", 1)[0]) if node.level else node.module or ""
                    if nested and (origin == package or origin.startswith(package + ".")):
                        raise SourceHelperError(f"conditional/class/nested same-package import in source {name!r} is unsupported")
                    for alias in node.names:
                        imports[alias.asname or alias.name] = (origin, alias.name)
                        if origin.startswith(package + "."):
                            follow(name, origin, alias.name)
                    continue
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for value in node.decorator_list:
                        expression(value, metadata=True)
                    for value in (*node.args.defaults, *node.args.kw_defaults):
                        expression(value, metadata)
                    for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                        expression(argument.annotation, metadata=True)
                    expression(node.returns, metadata=True)
                    continue
                if isinstance(node, ast.ClassDef):
                    for value in (*node.bases, *node.decorator_list, *(item.value for item in node.keywords)):
                        expression(value, metadata=True)
                    model = any(isinstance(base, ast.Name) and base.id == "BaseModel" for base in node.bases)
                    statements(node.body, nested=True, metadata=model)
                    continue
                # Recurse every executable statement field, including if/try,
                # handlers, with, loops and match cases; never ast.walk a def.
                for _field, value in ast.iter_fields(node):
                    values = value if isinstance(value, list) else [value]
                    for child in values:
                        if isinstance(child, ast.stmt):
                            statements([child], nested=True, metadata=metadata)
                        elif isinstance(child, (ast.ExceptHandler, ast.match_case)):
                            expression(getattr(child, "type", None), metadata)
                            expression(getattr(child, "guard", None), metadata)
                            statements(child.body, nested=True, metadata=metadata)
                        elif isinstance(child, ast.AST):
                            expression(child, metadata)

        statements(load(name).body)

    def follow(owner, origin, symbol):
        if origin == package:
            if symbol is not None and symbol not in {"pystep", "step", "Recipe", "Cab", "define_cab"}:
                raise SourceHelperError(f"same-package root/module import {origin}.{symbol} is unsupported; package initializers must not execute")
            return
        if not origin.startswith(package + "."):
            return
        if symbol is None or symbol == "*":
            raise SourceHelperError(f"same-package module/star import {origin!r} is unsupported; import a Python function explicitly")
        imported.add(origin)
        expand_imports(origin)
        visit(origin, symbol)

    # Preserve old in-process/local identities when no imported helper exists.
    # Source address validation is material only once dependencies are found.
    expected = root.joinpath(*parts).with_suffix(".py")
    if expected != source:
        return None
    expand_imports(module)
    visit(module, func.__qualname__.split(".")[0])
    imported.discard(module)
    if not imported:
        return None
    expand_imports(module)
    if source.resolve() != source:
        raise SourceHelperError("target source escapes its exact import root")
    return SourceHelpers(files[module], tuple(files[name] for name in sorted(imported)))
