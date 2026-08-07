"""Tests for `shinobi.backends._stream.run_streaming` -- the shared
subprocess helper `native`/`container` backends (and `steps.pyfunc`'s own
inline container-subprocess call) use for live stdout/stderr echo without
changing the blocking `Backend.run() -> BackendRun` contract.
"""

import sys

import pytest

from shinobi.backends import _stream
from shinobi.backends._stream import LineBuffer, display_label, elision_marker, run_streaming, set_capture_limits

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


# --- capture capping (LineBuffer) -------------------------------------


def test_uncapped_output_is_returned_byte_for_byte():
    """The overwhelmingly common case: a run short enough to fit under the
    cap must be indistinguishable from the uncapped behaviour, terminators
    and all.
    """
    buf = LineBuffer(head_max=10, tail_max=10)
    for line in ("a\n", "b\n", "c"):
        buf.append(line)
    assert buf.text() == "a\nb\nc"
    assert buf.dropped == 0
    assert buf.total == 3


def test_middle_is_elided_and_the_marker_counts_what_went():
    buf = LineBuffer(head_max=2, tail_max=2)
    for i in range(10):
        buf.append(f"line{i}\n")
    assert buf.text() == "line0\nline1\n" + elision_marker(6) + "\nline8\nline9\n"
    assert buf.dropped == 6
    assert buf.total == 10


def test_head_and_tail_are_the_ends_not_a_window():
    """Guards against the obvious wrong implementation -- a single deque,
    which keeps the tail and silently loses the banner and the resolved
    parameters at the top, the part most worth having.
    """
    buf = LineBuffer(head_max=1, tail_max=1)
    for i in range(5):
        buf.append(f"line{i}\n")
    text = buf.text()
    assert text.startswith("line0\n")
    assert text.endswith("line4\n")


def test_wrangler_matching_lines_survive_the_middle():
    """The whole point of `keep_matching`: a dropped line a wrangler would
    have matched is a missing *output value*, not just a less readable log.
    """
    buf = LineBuffer(head_max=1, tail_max=1, keep_matching=(r"^RESULT: ",))
    buf.append("banner\n")
    for i in range(20):
        buf.append(f"progress {i}\n")
    buf.append("RESULT: flux=1.25\n")
    for i in range(20):
        buf.append(f"more progress {i}\n")
    buf.append("done\n")
    text = buf.text()
    assert "RESULT: flux=1.25\n" in text
    assert text.startswith("banner\n")
    assert text.endswith("done\n")
    assert "progress 5" not in text
    assert not buf.matches_dropped


def test_retained_matches_keep_their_position_relative_to_the_ends():
    buf = LineBuffer(head_max=1, tail_max=1, keep_matching=("KEEP",))
    buf.append("first\n")
    buf.append("drop-a\n")
    buf.append("KEEP me\n")
    buf.append("drop-b\n")
    buf.append("last\n")
    assert buf.text() == "first\n" + elision_marker(1) + "\nKEEP me\n" + elision_marker(1) + "\nlast\n"


def test_keep_max_is_itself_bounded_and_reported():
    """A wrangler pattern matching nearly every line must not reintroduce
    the unbounded list `keep_matching` exists inside of.
    """
    buf = LineBuffer(head_max=1, tail_max=1, keep_matching=("line",), keep_max=3)
    for i in range(100):
        buf.append(f"line{i}\n")
    assert buf.matches_dropped
    assert buf.dropped > 0
    assert len(buf.text().splitlines()) < 100


def test_uncompilable_keep_pattern_is_skipped_not_raised():
    """A bad wrangler regex is the cab author's bug, and `apply_wranglers`
    reports it far better than a pump thread dying mid-read would.
    """
    buf = LineBuffer(head_max=1, tail_max=1, keep_matching=("[unclosed",))
    buf.append("a\n")
    buf.append("b\n")
    buf.append("c\n")
    assert buf.text() == "a\n" + elision_marker(1) + "\nc\n"


def test_zero_limits_elide_everything_but_still_say_so():
    buf = LineBuffer(head_max=0, tail_max=0)
    for i in range(4):
        buf.append(f"line{i}\n")
    assert buf.text() == elision_marker(4) + "\n"
    assert buf.dropped == 4


def test_empty_stream_is_empty_not_a_marker():
    assert LineBuffer(head_max=0, tail_max=0).text() == ""


def test_run_streaming_caps_a_chatty_process_and_reports_the_drop():
    argv = [sys.executable, "-c", "import sys\nfor i in range(5000): print(f'line{i}')"]
    result = run_streaming(argv, label="t", stream=False, head_lines=5, tail_lines=5)
    assert result.returncode == 0
    assert result.stdout.startswith("line0\n")
    assert result.stdout.endswith("line4999\n")
    assert result.stdout_dropped == 4990
    assert not result.wrangler_lines_dropped
    # The cap is the point: what comes back is bounded by the limits, not
    # by how much the process felt like emitting.
    assert len(result.stdout.splitlines()) == 11  # 5 + marker + 5


def test_run_streaming_keeps_wrangler_lines_from_a_chatty_process():
    argv = [
        sys.executable,
        "-c",
        "import sys\nfor i in range(2000): print(f'noise{i}')\nprint('RESULT: 42')\nfor i in range(2000): print(f'noise{i}')",
    ]
    result = run_streaming(argv, label="t", stream=False, head_lines=2, tail_lines=2, keep_matching=(r"^RESULT: ",))
    assert "RESULT: 42\n" in result.stdout
    assert result.stdout_dropped > 0


def test_capture_limits_are_process_wide_and_settable():
    argv = [sys.executable, "-c", "import sys\nfor i in range(50): print(f'line{i}')"]
    original = (_stream._head_limit, _stream._tail_limit)
    try:
        set_capture_limits(2, 2)
        result = run_streaming(argv, label="t", stream=False)
        assert result.stdout_dropped == 46
    finally:
        set_capture_limits(*original)


def test_negative_capture_limits_clamp_to_zero():
    original = (_stream._head_limit, _stream._tail_limit)
    try:
        set_capture_limits(-5, -5)
        assert _stream._head_limit == 0
        assert _stream._tail_limit == 0
    finally:
        set_capture_limits(*original)
