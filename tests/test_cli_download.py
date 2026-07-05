"""Tests for the download CLI command and download module."""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from shinobi.cli import main
from shinobi.download import download_cultcargo, resolve_latest_version


def _create_mock_tarball(
    version: str,
    files: dict[str, str],
    include_images: bool = False,
) -> bytes:
    """Create a mock tarball with the given files.

    Args:
        version: Version string (used in directory name)
        files: Dict of {relative_path: content}
        include_images: If True, add some files in images/ directory

    Returns:
        Tarball bytes
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        base_dir = f"cult-cargo-{version}"

        for rel_path, content in files.items():
            full_path = f"{base_dir}/cultcargo/{rel_path}"
            data = content.encode()
            info = tarfile.TarInfo(name=full_path)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        if include_images:
            # Add some files in images/ directory
            for name in ["wsclean/Dockerfile", "casa/Dockerfile"]:
                full_path = f"{base_dir}/cultcargo/images/{name}"
                data = b"FROM ubuntu:20.04\n"
                info = tarfile.TarInfo(name=full_path)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

    return buf.getvalue()


class TestResolveLatestVersion:
    """Tests for resolve_latest_version()."""

    def test_resolve_latest_version_picks_highest_semver(self):
        """Should pick the highest v* semver tag."""
        mock_response = [
            {"name": "v0.1.0"},
            {"name": "v0.2.1"},
            {"name": "v0.2.0"},
            {"name": "v0.1.5"},
        ]

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_response).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = resolve_latest_version()

        assert result == "v0.2.1"

    def test_resolve_latest_version_ignores_non_semver_tags(self):
        """Should ignore tags that don't match v* semver pattern."""
        mock_response = [
            {"name": "v0.1.0"},
            {"name": "release-1.0"},
            {"name": "v0.2.0"},
            {"name": "beta"},
            {"name": "v0.1.5-rc1"},  # Has suffix, should be ignored
        ]

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_response).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = resolve_latest_version()

        assert result == "v0.2.0"

    def test_resolve_latest_version_no_v_tags_raises(self):
        """Should raise RuntimeError if no v* tags found."""
        mock_response = [
            {"name": "release-1.0"},
            {"name": "beta"},
        ]

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(mock_response).encode()
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            with pytest.raises(RuntimeError, match="No v\\* semver tags found"):
                resolve_latest_version()

    def test_resolve_latest_version_rate_limit_error(self):
        """Should raise RuntimeError with helpful message on rate limit."""
        import urllib.error

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                url="https://api.github.com",
                code=403,
                msg="rate limit exceeded",
                hdrs={},
                fp=None,
            )

            with pytest.raises(RuntimeError, match="rate limit exceeded"):
                resolve_latest_version()


class TestDownloadCultcargo:
    """Tests for download_cultcargo()."""

    def test_download_with_specific_version(self):
        """Should download and extract files for a specific version."""
        mock_tarball = _create_mock_tarball(
            version="v0.2.1",
            files={
                "wsclean.yml": "name: wsclean\n",
                "genesis/cult-cargo-base.yml": "base: true\n",
                "casa/flagdata.yml": "name: flagdata\n",
            },
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_tarball
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            with tempfile.TemporaryDirectory() as tmpdir:
                dest = Path(tmpdir) / "cabs"
                result = download_cultcargo(dest_dir=dest, version="v0.2.1")

                assert result["version"] == "v0.2.1"
                assert result["file_count"] == 3
                assert result["dest_dir"] == str(dest)
                assert (dest / "wsclean.yml").exists()
                assert (dest / "genesis" / "cult-cargo-base.yml").exists()
                assert (dest / "casa" / "flagdata.yml").exists()

    def test_download_excludes_images_directory(self):
        """Should exclude files in the images/ directory."""
        mock_tarball = _create_mock_tarball(
            version="v0.2.1",
            files={
                "wsclean.yml": "name: wsclean\n",
                "genesis/cult-cargo-base.yml": "base: true\n",
            },
            include_images=True,
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_tarball
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            with tempfile.TemporaryDirectory() as tmpdir:
                dest = Path(tmpdir) / "cabs"
                result = download_cultcargo(dest_dir=dest, version="v0.2.1")

                # Should have 2 files (wsclean.yml and genesis/cult-cargo-base.yml)
                assert result["file_count"] == 2
                assert (dest / "wsclean.yml").exists()
                assert (dest / "genesis" / "cult-cargo-base.yml").exists()
                # images/ directory should not exist
                assert not (dest / "images").exists()

    def test_download_creates_dest_dir(self):
        """Should create destination directory if it doesn't exist."""
        mock_tarball = _create_mock_tarball(
            version="v0.2.1",
            files={"wsclean.yml": "name: wsclean\n"},
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_tarball
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            with tempfile.TemporaryDirectory() as tmpdir:
                dest = Path(tmpdir) / "nested" / "path" / "cabs"
                assert not dest.exists()

                result = download_cultcargo(dest_dir=dest, version="v0.2.1")

                assert dest.exists()
                assert (dest / "wsclean.yml").exists()

    def test_download_overwrites_existing_files(self):
        """Should overwrite existing files without error."""
        mock_tarball = _create_mock_tarball(
            version="v0.2.1",
            files={"wsclean.yml": "name: wsclean\nversion: 2\n"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "cabs"
            dest.mkdir()

            # Create existing file with old content
            (dest / "wsclean.yml").write_text("name: wsclean\nversion: 1\n")

            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_resp = MagicMock()
                mock_resp.read.return_value = mock_tarball
                mock_resp.__enter__ = lambda s: s
                mock_resp.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_resp

                result = download_cultcargo(dest_dir=dest, version="v0.2.1")

            # File should be overwritten with new content
            assert (dest / "wsclean.yml").read_text() == "name: wsclean\nversion: 2\n"

    def test_download_404_error(self):
        """Should raise RuntimeError on 404."""
        import urllib.error

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.HTTPError(
                url="https://github.com",
                code=404,
                msg="Not Found",
                hdrs={},
                fp=None,
            )

            with tempfile.TemporaryDirectory() as tmpdir:
                dest = Path(tmpdir) / "cabs"
                with pytest.raises(RuntimeError, match="Version 'v99.99.99' not found"):
                    download_cultcargo(dest_dir=dest, version="v99.99.99")

    def test_download_with_latest_version(self):
        """Should resolve 'latest' to actual tag and download."""
        mock_tags_response = [
            {"name": "v0.1.0"},
            {"name": "v0.2.1"},
        ]

        mock_tarball = _create_mock_tarball(
            version="v0.2.1",
            files={"wsclean.yml": "name: wsclean\n"},
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            # First call: tags API
            tags_resp = MagicMock()
            tags_resp.read.return_value = json.dumps(mock_tags_response).encode()
            tags_resp.__enter__ = lambda s: s
            tags_resp.__exit__ = MagicMock(return_value=False)

            # Second call: tarball download
            tarball_resp = MagicMock()
            tarball_resp.read.return_value = mock_tarball
            tarball_resp.__enter__ = lambda s: s
            tarball_resp.__exit__ = MagicMock(return_value=False)

            mock_urlopen.side_effect = [tags_resp, tarball_resp]

            with tempfile.TemporaryDirectory() as tmpdir:
                dest = Path(tmpdir) / "cabs"
                result = download_cultcargo(dest_dir=dest, version="latest")

        assert result["version"] == "v0.2.1"
        assert result["file_count"] == 1

    def test_download_preserves_nested_images_directory(self):
        """Should only exclude the top-level images/ dir, not nested ones."""
        mock_tarball = _create_mock_tarball(
            version="v0.2.1",
            files={
                "wsclean.yml": "name: wsclean\n",
                "somecab/images/icon.yml": "icon: true\n",
            },
            include_images=True,
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_tarball
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            with tempfile.TemporaryDirectory() as tmpdir:
                dest = Path(tmpdir) / "cabs"
                result = download_cultcargo(dest_dir=dest, version="v0.2.1")

                # Top-level images/ (from include_images=True) is excluded...
                assert not (dest / "images").exists()
                # ...but a nested dir that happens to be named images/ is not.
                assert (dest / "somecab" / "images" / "icon.yml").exists()
                assert result["file_count"] == 2

    def test_download_rejects_symlink_escaping_destination(self):
        """Should fail cleanly if the tarball contains a symlink pointing outside dest."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            base = "cult-cargo-v0.2.1/cultcargo"

            info = tarfile.TarInfo(name=f"{base}/wsclean.yml")
            data = b"name: wsclean\n"
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

            link_info = tarfile.TarInfo(name=f"{base}/escape.yml")
            link_info.type = tarfile.SYMTYPE
            link_info.linkname = "/etc/passwd"
            tar.addfile(link_info)

        mock_tarball = buf.getvalue()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_tarball
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            with tempfile.TemporaryDirectory() as tmpdir:
                dest = Path(tmpdir) / "cabs"
                with pytest.raises(RuntimeError, match="Failed to extract tarball"):
                    download_cultcargo(dest_dir=dest, version="v0.2.1")

    def test_download_rejects_path_traversal_entry(self):
        """Should fail cleanly if a tarball entry tries to write outside the extraction dir."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            data = b"evil\n"
            info = tarfile.TarInfo(name="../../evil.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        mock_tarball = buf.getvalue()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_tarball
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            with tempfile.TemporaryDirectory() as tmpdir:
                dest = Path(tmpdir) / "cabs"
                with pytest.raises(RuntimeError, match="Failed to extract tarball"):
                    download_cultcargo(dest_dir=dest, version="v0.2.1")

    def test_download_skips_in_tree_symlink(self):
        """Should skip symlinks even when their target is inside the extracted tree."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            base = "cult-cargo-v0.2.1/cultcargo"

            data = b"name: wsclean\n"
            info = tarfile.TarInfo(name=f"{base}/wsclean.yml")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

            link_info = tarfile.TarInfo(name=f"{base}/alias.yml")
            link_info.type = tarfile.SYMTYPE
            link_info.linkname = "wsclean.yml"
            tar.addfile(link_info)

        mock_tarball = buf.getvalue()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = mock_tarball
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            with tempfile.TemporaryDirectory() as tmpdir:
                dest = Path(tmpdir) / "cabs"
                result = download_cultcargo(dest_dir=dest, version="v0.2.1")

                assert (dest / "wsclean.yml").exists()
                assert not (dest / "alias.yml").exists()
                assert result["file_count"] == 1


class TestDownloadCLI:
    """Tests for the CLI download command."""

    def test_cli_download_cult_cargo(self):
        """Should download cult-cargo cabs via CLI."""
        mock_tarball = _create_mock_tarball(
            version="v0.2.1",
            files={
                "wsclean.yml": "name: wsclean\n",
                "genesis/cult-cargo-base.yml": "base: true\n",
            },
        )

        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_resp = MagicMock()
                mock_resp.read.return_value = mock_tarball
                mock_resp.__enter__ = lambda s: s
                mock_resp.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_resp

                result = runner.invoke(main, ["download", "--cult-cargo", "--version", "v0.2.1"])

            assert result.exit_code == 0
            assert "Downloaded cult-cargo v0.2.1" in result.output
            assert "Files: 2" in result.output
            assert ".shinobi/cabs/cultcargo" in result.output

            # Verify files were created
            assert Path(".shinobi/cabs/cultcargo/wsclean.yml").exists()
            assert Path(".shinobi/cabs/cultcargo/genesis/cult-cargo-base.yml").exists()

    def test_cli_download_custom_dest_dir(self):
        """Should download to custom destination directory."""
        mock_tarball = _create_mock_tarball(
            version="v0.2.1",
            files={"wsclean.yml": "name: wsclean\n"},
        )

        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_resp = MagicMock()
                mock_resp.read.return_value = mock_tarball
                mock_resp.__enter__ = lambda s: s
                mock_resp.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_resp

                result = runner.invoke(
                    main,
                    ["download", "--cult-cargo", "--version", "v0.2.1", "--dest-dir", "./my-cabs"],
                )

            assert result.exit_code == 0
            assert "my-cabs" in result.output
            assert Path("./my-cabs/wsclean.yml").exists()

    def test_cli_download_no_source_specified(self):
        """Should error if no source flag is provided."""
        runner = CliRunner()
        result = runner.invoke(main, ["download"])

        assert result.exit_code != 0
        assert "No source specified" in result.output

    def test_cli_download_error_handling(self):
        """Should handle download errors gracefully."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            with patch("urllib.request.urlopen") as mock_urlopen:
                import urllib.error

                mock_urlopen.side_effect = urllib.error.HTTPError(
                    url="https://github.com",
                    code=404,
                    msg="Not Found",
                    hdrs={},
                    fp=None,
                )

                result = runner.invoke(main, ["download", "--cult-cargo", "--version", "v99.99.99"])

            assert result.exit_code != 0
            assert "not found" in result.output.lower()
