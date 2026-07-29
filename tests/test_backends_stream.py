"""Tests for `shinobi.backends._stream.run_streaming` -- the shared
subprocess helper `native`/`container` backends (and `steps.pyfunc`'s own
inline container-subprocess call) use for live stdout/stderr echo without
changing the blocking `Backend.run() -> BackendRun` contract.
"""

import sys

import pytest

from shinobi.backends._stream import display_label, run_streaming

_ARGV = [
    sys.executable,
    "-c",
    "import sys; print('out-line'); print('err-line', file=sys.stderr)",
]


def test_stream_false_matches_subprocess_run_capture_output():
    result = run_streaming(_ARGV, label="t", stream=False)
    assert result.returncode == 0
    assert result.stdout == "out-line\n"
    assert result.stderr == "err-line\n"


def test_stream_true_still_returns_full_captured_output():
    """The BackendRun contract (full text, same as non-streaming) must be
    unchanged -- _run_cab/wranglers/_fill_outputs/_run_recipe all depend
    on getting the complete text back, not just partial echoed lines.
    """
    result = run_streaming(_ARGV, label="t", stream=True)
    assert result.returncode == 0
    assert result.stdout == "out-line\n"
    assert result.stderr == "err-line\n"


def test_stream_true_echoes_lines_with_label_prefix(capsys):
    run_streaming(_ARGV, label="my-step", stream=True)
    captured = capsys.readouterr()
    assert "[my-step] out-line" in captured.out
    assert "[my-step] err-line" in captured.err


def test_stream_false_echoes_nothing(capsys):
    run_streaming(_ARGV, label="my-step", stream=False)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_nonzero_returncode_preserved_when_streaming():
    result = run_streaming([sys.executable, "-c", "import sys; sys.exit(3)"], label="t", stream=True)
    assert result.returncode == 3
    assert not result.success


# ---- display_label ----------------------------------------------------


@pytest.mark.parametrize(
    "cache_path,expected",
    [
        ("myrecipe.stepA", "stepA"),
        ("myrecipe.stepA.stepB", "stepA.stepB"),
        ("caracal_pipeline_241f3ea5a2b77240.selfcal.cycle1_solve", "selfcal.cycle1_solve"),
        # a scattered slice keeps its index -- that is per-line information
        ("myrecipe.transform[2].split", "transform[2].split"),
        # no root to drop: a step run outside any recipe
        ("standalone_cab", "standalone_cab"),
        ("", ""),
    ],
)
def test_display_label_drops_only_the_root_scope(cache_path, expected):
    assert display_label(cache_path) == expected
