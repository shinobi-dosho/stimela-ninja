"""Download cult-cargo cab definitions from GitHub.

Provides functionality to download cult-cargo YAML cab definitions from the
caracal-pipeline/cult-cargo repository, with support for version selection
(latest stable tag, specific tag, branch, or commit SHA).
"""

from __future__ import annotations

import json
import re
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from packaging.version import Version


CULTCARGO_REPO = "caracal-pipeline/cult-cargo"
GITHUB_API_URL = f"https://api.github.com/repos/{CULTCARGO_REPO}/tags"
GITHUB_ARCHIVE_URL = f"https://github.com/{CULTCARGO_REPO}/archive"


def resolve_latest_version() -> str:
    """Query GitHub for cult-cargo tags and return the latest v* semver tag.

    Returns:
        The tag name (e.g., "v0.2.1")

    Raises:
        RuntimeError: If no v* tags found or API rate-limited
    """
    try:
        req = urllib.request.Request(GITHUB_API_URL, headers={"User-Agent": "shinobi"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            tags_data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            raise RuntimeError(
                f"GitHub API rate limit exceeded. "
                f"Try specifying --version <tag-or-branch> to skip the API call."
            ) from e
        raise RuntimeError(f"Failed to query GitHub tags: {e}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error querying GitHub: {e}") from e

    # Filter for v* semver tags
    v_tags = []
    for tag_info in tags_data:
        name = tag_info["name"]
        if name.startswith("v") and re.match(r"^v\d+(\.\d+)*$", name):
            try:
                v_tags.append((name, Version(name[1:])))
            except Exception:
                continue

    if not v_tags:
        raise RuntimeError(
            f"No v* semver tags found in {CULTCARGO_REPO}. "
            f"Try specifying --version <branch-or-tag>."
        )

    # Sort by version, return the tag name
    v_tags.sort(key=lambda x: x[1], reverse=True)
    return v_tags[0][0]


def download_cultcargo(
    dest_dir: Path,
    version: str = "latest",
    exclude_images: bool = True,
) -> dict:
    """Download cult-cargo cab definitions from GitHub.

    Args:
        dest_dir: Destination directory (will be created if needed)
        version: Version to download ("latest", tag name, branch name, or commit SHA)
        exclude_images: If True, exclude the images/ subdirectory

    Returns:
        Dict with keys: version, file_count, dest_dir

    Raises:
        RuntimeError: On download/extract errors
    """
    # Resolve "latest" to actual tag
    if version == "latest":
        version = resolve_latest_version()

    # Download tarball
    archive_url = f"{GITHUB_ARCHIVE_URL}/{version}.tar.gz"
    try:
        req = urllib.request.Request(archive_url, headers={"User-Agent": "shinobi"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            tarball_data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RuntimeError(
                f"Version '{version}' not found in {CULTCARGO_REPO}. "
                f"Check the tag/branch/commit SHA."
            ) from e
        raise RuntimeError(f"Failed to download {archive_url}: {e}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error downloading tarball: {e}") from e

    # Extract to temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        tarball_path = Path(tmpdir) / "cultcargo.tar.gz"
        tarball_path.write_bytes(tarball_data)

        try:
            with tarfile.open(tarball_path, "r:gz") as tar:
                tar.extractall(tmpdir)
        except (tarfile.TarError, OSError) as e:
            raise RuntimeError(f"Failed to extract tarball: {e}") from e

        # Find extracted directory (name varies: cult-cargo-v0.2.1/, cult-cargo-master/, etc.)
        extracted_dirs = [
            d for d in Path(tmpdir).iterdir()
            if d.is_dir() and d.name.startswith("cult-cargo-")
        ]
        if not extracted_dirs:
            raise RuntimeError("No cult-cargo directory found in tarball")
        if len(extracted_dirs) > 1:
            raise RuntimeError(f"Multiple cult-cargo directories found: {extracted_dirs}")

        extracted_root = extracted_dirs[0]
        cultcargo_src = extracted_root / "cultcargo"

        if not cultcargo_src.exists():
            raise RuntimeError(f"cultcargo/ directory not found in {extracted_root}")

        # Copy cultcargo/ subtree to dest_dir, excluding images/
        dest_dir.mkdir(parents=True, exist_ok=True)

        file_count = 0
        for item in cultcargo_src.rglob("*"):
            if item.is_file():
                # Check if we should exclude this file
                if exclude_images and "images" in item.parts:
                    continue

                # Compute relative path from cultcargo_src
                rel_path = item.relative_to(cultcargo_src)
                dest_path = dest_dir / rel_path

                # Create parent dirs
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy file
                shutil.copy2(item, dest_path)
                file_count += 1

    return {
        "version": version,
        "file_count": file_count,
        "dest_dir": str(dest_dir),
    }
