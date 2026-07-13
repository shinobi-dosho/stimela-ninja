from click.testing import CliRunner

from shinobi.cli import main


def _seed(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    cache = tmp_path / "cache"
    runs.mkdir()
    cache.mkdir()
    (runs / "r.run.json").write_text("{}")
    (cache / "manifest.json").write_text("{}")
    monkeypatch.setenv("SHINOBI_PROVENANCE__DIR", str(runs))
    monkeypatch.setenv("SHINOBI_CACHE__DIR", str(cache))
    return runs, cache


def test_clean_removes_runs_and_cache(tmp_path, monkeypatch):
    runs, cache = _seed(tmp_path, monkeypatch)
    result = CliRunner().invoke(main, ["clean"])
    assert result.exit_code == 0, result.output
    assert not runs.exists() and not cache.exists()


def test_clean_dry_run_deletes_nothing(tmp_path, monkeypatch):
    runs, cache = _seed(tmp_path, monkeypatch)
    result = CliRunner().invoke(main, ["clean", "--dry-run"])
    assert result.exit_code == 0
    assert "would remove" in result.output
    assert runs.exists() and cache.exists()


def test_clean_selective(tmp_path, monkeypatch):
    runs, cache = _seed(tmp_path, monkeypatch)
    result = CliRunner().invoke(main, ["clean", "--no-cache"])
    assert result.exit_code == 0
    assert not runs.exists()
    assert cache.exists()  # --no-cache left it alone


def test_clean_missing_dirs_is_graceful(tmp_path, monkeypatch):
    monkeypatch.setenv("SHINOBI_PROVENANCE__DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("SHINOBI_CACHE__DIR", str(tmp_path / "gone"))
    result = CliRunner().invoke(main, ["clean"])
    assert result.exit_code == 0
    assert "nothing at" in result.output
