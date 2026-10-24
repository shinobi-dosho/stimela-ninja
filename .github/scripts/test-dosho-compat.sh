#!/usr/bin/env bash
# Exercise the released publication gate or an explicitly selected source pair.
set -euo pipefail

lane=${1:?pass released or main}
dosho_ref=${2:-main}
case "$lane" in
    released|main) ;;
    *) echo "unknown dosho compatibility lane: $lane" >&2; exit 2 ;;
esac

repo=$(git rev-parse --show-toplevel)
cd "$repo"
mkdir -p .work-cache/compat/tmp
export TMPDIR="$repo/.work-cache/compat/tmp"
export UV_CACHE_DIR="$repo/.work-cache/compat/cache"
uv sync --locked --python 3.12 --group dev --group examples

if [[ "$lane" == main ]]; then
    checkout=$(mktemp -d "$repo/.work-cache/compat/dosho.XXXXXXXX")
    trap 'rm -rf "$checkout"' EXIT
    git clone --filter=blob:none --no-checkout https://github.com/shinobi-dosho/dosho "$checkout"
    git -C "$checkout" fetch --depth=1 origin -- "$dosho_ref"
    git -C "$checkout" checkout --detach FETCH_HEAD
    bash .github/scripts/install-candidate-pair.sh "$checkout"
else
uv run --no-sync python - <<'PY'
import importlib.metadata as metadata
from pathlib import Path

import dosho
import shinobi

repo = Path.cwd().resolve()
assert Path(shinobi.__file__).resolve().is_relative_to(repo / "src"), shinobi.__file__
assert metadata.version("dosho") == dosho.__version__

print(f"shinobi: {shinobi.__file__}")
print(f"dosho {dosho.__version__}: {dosho.__file__}")
PY

fi

if [[ "$lane" == main ]]; then
    uv run --no-sync pytest -q -o pythonpath= tests/test_dosho_compat.py tests/test_neutral_adaptation.py

    # Focused catalogue contracts run against the installed candidate wheels.
    # Dosho's full suite has additional runtime dependencies and is owned there.
    uv run --no-sync pytest -q -o pythonpath= \
        "$checkout/tests/test_registry.py" \
        "$checkout/tests/test_builder.py" \
        "$checkout/tests/test_strict_ms_example.py"
else
    # The simulation dispatch test requires simms and real MS I/O. These
    # cover recipe construction, graph rendering, and CubiCal/WSClean argv.
    uv run --no-sync pytest -q \
        tests/test_dosho_compat.py \
        tests/test_example_simulation.py \
        -k 'not test_recipe_dispatches_with_correct_argv_shape'
fi
