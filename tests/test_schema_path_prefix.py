"""`ParamMeta.path_prefix`: declaring that a string input is a path stem.

The convention it formalises predates it -- `declared_output_dirs` documents
why a tool's output stem is typed `str` rather than `File` (a path dtype would
be absolutized into the sandbox and the tool would write outside it). What did
not exist was any way to *say* a field is one, so nothing could check that the
matching write target had been declared. These tests pin both halves: that the
marker changes no behaviour, and that the check catches the omission.
"""

from __future__ import annotations

import pytest

from shinobi.loaders import build_model
from shinobi.steps.schema import Cab, ParamMeta, declared_output_dirs, path_fields


def _models():
    return (
        build_model("I", {"prefix": ("str", False, None)}),
        build_model("O", {"dirty": ("File", False, None)}),
    )


def _cab(**kw) -> Cab:
    inputs_model, outputs_model = _models()
    kw.setdefault("field_meta", {})
    return Cab(name="c", command="tool", inputs_model=inputs_model, outputs_model=outputs_model, **kw)


def test_marker_does_not_make_the_field_a_path():
    """The whole point: it is a declaration, not a type change. If this ever
    starts failing, the sandbox will begin absolutizing output stems and tools
    will write outside it -- the exact bug the str convention avoids.
    """
    cab = _cab(field_meta={"prefix": ParamMeta(path_prefix=True), "dirty": ParamMeta(implicit="{prefix}-dirty.fits")})
    assert "prefix" not in path_fields(cab.inputs_model)


def test_marker_does_not_change_declared_output_dirs():
    """Mounting still comes from the output side, marked or not."""
    meta = {"dirty": ParamMeta(implicit="{prefix}-dirty.fits")}
    plain = _cab(field_meta=meta)
    marked = _cab(field_meta={**meta, "prefix": ParamMeta(path_prefix=True)})
    args = {"prefix": "/data/out/img"}
    assert declared_output_dirs(plain, args) == declared_output_dirs(marked, args)


def test_an_output_implicit_template_satisfies_the_check():
    _cab(field_meta={"prefix": ParamMeta(path_prefix=True), "dirty": ParamMeta(implicit="{prefix}-dirty.fits")})


def test_a_harvest_pattern_satisfies_the_check():
    _cab(harvest=["{prefix}-*.fits"], field_meta={"prefix": ParamMeta(path_prefix=True)})


def test_a_scratch_pattern_satisfies_the_check():
    _cab(scratch=["{prefix}.log"], field_meta={"prefix": ParamMeta(path_prefix=True)})


def test_an_unreferenced_stem_is_rejected():
    """The failure the marker exists to catch: the tool runs, writes, and the
    products are silently left inside the container or outside the harvest.
    """
    with pytest.raises(ValueError, match="marked path_prefix but named by no write declaration"):
        _cab(field_meta={"prefix": ParamMeta(path_prefix=True)})


def test_an_input_side_implicit_does_not_satisfy_the_check():
    """Only the *output* side declares where the tool writes. An `implicit` on
    the input itself is a supplied value, not a write target.
    """
    with pytest.raises(ValueError, match="marked path_prefix"):
        _cab(field_meta={"prefix": ParamMeta(path_prefix=True, implicit="{prefix}-x")})


def test_unmarked_stems_are_left_alone():
    """Opt-in. Every cab written before this field existed still builds."""
    _cab(field_meta={"prefix": ParamMeta(), "dirty": ParamMeta(implicit="{prefix}-dirty.fits")})
    _cab()
