#!/usr/bin/env bash
# Overlay only the two explicit candidate wheels onto an already locked env.
set -euo pipefail
candidate=${1:?pass the compatible Dosho candidate checkout}
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
candidate=$(cd "$candidate" && pwd)
cd "$repo"
mkdir -p .work-cache/candidate-pair/tmp
export TMPDIR="$repo/.work-cache/candidate-pair/tmp"
export UV_CACHE_DIR="$repo/.work-cache/candidate-pair/cache"
pair_dir=$(mktemp -d "$repo/.work-cache/candidate-pair/build.XXXXXXXX")
trap 'rm -rf "$pair_dir"' EXIT
for source in "$repo" "$candidate"; do
    echo "Candidate source $source HEAD: $(git -C "$source" rev-parse HEAD)"
    if [[ -n "$(git -C "$source" status --porcelain)" ]]; then
        echo "Working tree has local changes; HEAD does not identify those edits."
    fi
done
uv build --wheel --directory "$repo" --out-dir "$pair_dir/shinobi"
uv build --wheel --directory "$candidate" --out-dir "$pair_dir/dosho"
shopt -s nullglob
ninja_wheels=("$pair_dir/shinobi/"*.whl)
dosho_wheels=("$pair_dir/dosho/"*.whl)
[[ ${#ninja_wheels[@]} == 1 && ${#dosho_wheels[@]} == 1 ]] || { echo "Expected one wheel per candidate" >&2; exit 1; }
pair_python=$(uv run --no-sync python -c 'import sys; print(sys.executable)')
uv pip install --python "$pair_python" --no-deps --reinstall "${ninja_wheels[0]}" "${dosho_wheels[0]}"
uv pip check --python "$pair_python"
uv run --no-sync python - "$repo" "$candidate" "${ninja_wheels[0]}" "${dosho_wheels[0]}" <<'VERIFY'
import hashlib
import importlib
import importlib.metadata as metadata
import json
import sys
import sysconfig
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlparse

sites = {Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")}
for distribution, module, source, wheel in (
    ("stimela-ninja", "shinobi", sys.argv[1], sys.argv[3]),
    ("dosho", "dosho", sys.argv[2], sys.argv[4]),
):
    source, wheel = Path(source).resolve(), Path(wheel).resolve()
    expected = tomllib.loads((source / "pyproject.toml").read_text())["project"]["version"]
    dist = metadata.distribution(distribution)
    assert dist.version == expected, (distribution, dist.version, expected)
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    url = urlparse(direct.get("url", ""))
    assert url.scheme == "file" and url.netloc in ("", "localhost"), direct
    assert Path(unquote(url.path)).resolve() == wheel, (distribution, direct, str(wheel))
    archive = direct.get("archive_info", {})
    digest = archive.get("hashes", {}).get("sha256")
    if digest is None and archive.get("hash", "").startswith("sha256="):
        digest = archive["hash"].split("=", 1)[1]
    if digest is not None:
        assert digest == hashlib.sha256(wheel.read_bytes()).hexdigest(), distribution
    imported = Path(importlib.import_module(module).__file__).resolve()
    assert any(imported.is_relative_to(site) for site in sites), (module, imported, sites)
    assert any(Path(dist.locate_file("")).resolve().is_relative_to(site) for site in sites), dist
    print(f"Verified {distribution} {dist.version}: {imported}; wheel={wheel}", flush=True)
VERIFY
