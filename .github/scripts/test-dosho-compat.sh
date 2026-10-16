#!/usr/bin/env bash
# Exercise this checkout against the locked dosho release or a fresh dosho main.
set -euo pipefail

lane=${1:?pass released or main}
case "$lane" in
    released|main) ;;
    *) echo "unknown dosho compatibility lane: $lane" >&2; exit 2 ;;
esac

repo=$(git rev-parse --show-toplevel)
cd "$repo"
uv sync --locked --python 3.12 --group dev --group examples

if [[ "$lane" == main ]]; then
    checkout=${RUNNER_TEMP:-/tmp}/dosho-compat-main
    git clone --filter=blob:none https://github.com/shinobi-dosho/dosho "$checkout"
    echo "dosho main: $(git -C "$checkout" rev-parse HEAD)"
    # --no-deps prevents dosho's optional shinobi requirement (and its dev
    # source redirect) from replacing the editable checkout under test.
    uv pip install --no-deps "$checkout"
    export DOSHO_COMPAT_CHECKOUT="$checkout"
fi

uv run --no-sync python - <<'PY'
import importlib.metadata as metadata
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

import dosho
import shinobi

repo = Path.cwd().resolve()
assert Path(shinobi.__file__).resolve().is_relative_to(repo / "src"), shinobi.__file__
assert metadata.version("dosho") == dosho.__version__

checkout = os.environ.get("DOSHO_COMPAT_CHECKOUT")
if checkout:
    direct = json.loads(metadata.distribution("dosho").read_text("direct_url.json") or "{}")
    source = Path(unquote(urlparse(direct.get("url", "")).path)).resolve()
    assert source == Path(checkout).resolve(), direct

print(f"shinobi: {shinobi.__file__}")
print(f"dosho {dosho.__version__}: {dosho.__file__}")
PY

if [[ "$lane" == main ]]; then
    uv run --no-sync pytest -q tests/test_dosho_compat.py

    # Dosho's own focused contracts, run with this shinobi checkout instead of
    # its lockfile's Git-sourced shinobi. The local simulation example still
    # uses CubiCal's old `ms` output; its dosho-main migration is separate.
    # Dosho's full suite has additional runtime dependencies and is owned there.
    uv run --no-sync pytest -q \
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
