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


def _seed_sandboxes(tmp_path, monkeypatch):
    work = tmp_path / "work"
    (work / "step-abc123").mkdir(parents=True)
    (work / "step-abc123" / "junk.log").write_text("junk")
    monkeypatch.setenv("SHINOBI_SANDBOX__DIR", str(work))
    return work


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


def test_clean_removes_leftover_sandboxes_by_default(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    work = _seed_sandboxes(tmp_path, monkeypatch)
    result = CliRunner().invoke(main, ["clean"])
    assert result.exit_code == 0, result.output
    assert not work.exists()


def test_clean_no_sandboxes_leaves_them(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    work = _seed_sandboxes(tmp_path, monkeypatch)
    result = CliRunner().invoke(main, ["clean", "--no-sandboxes"])
    assert result.exit_code == 0, result.output
    assert work.exists()


def test_clean_missing_dirs_is_graceful(tmp_path, monkeypatch):
    monkeypatch.setenv("SHINOBI_PROVENANCE__DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("SHINOBI_CACHE__DIR", str(tmp_path / "gone"))
    result = CliRunner().invoke(main, ["clean"])
    assert result.exit_code == 0
    assert "nothing at" in result.output


def _seed_launch(tmp_path, recipe):
    launch_dir = tmp_path / ".shinobi" / recipe
    launch_dir.mkdir(parents=True)
    (launch_dir / "handle.json").write_text("{}")
    (launch_dir / f"{recipe}.out").write_text("")
    return launch_dir


def test_clean_launches_removes_handle_dir(tmp_path, monkeypatch):
    launch_dir = _seed_launch(tmp_path, "myrecipe")
    monkeypatch.setenv("SHINOBI_PROVENANCE__DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("SHINOBI_CACHE__DIR", str(tmp_path / "gone"))
    result = CliRunner().invoke(
        main, ["clean", "--no-runs", "--no-cache", "--launches", "--workdir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert not launch_dir.exists()


def test_clean_launches_off_by_default(tmp_path, monkeypatch):
    launch_dir = _seed_launch(tmp_path, "myrecipe")
    monkeypatch.setenv("SHINOBI_PROVENANCE__DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("SHINOBI_CACHE__DIR", str(tmp_path / "gone"))
    result = CliRunner().invoke(main, ["clean", "--workdir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert launch_dir.exists()


def test_clean_launches_dry_run(tmp_path, monkeypatch):
    launch_dir = _seed_launch(tmp_path, "myrecipe")
    monkeypatch.setenv("SHINOBI_PROVENANCE__DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("SHINOBI_CACHE__DIR", str(tmp_path / "gone"))
    result = CliRunner().invoke(
        main,
        ["clean", "--no-runs", "--no-cache", "--launches", "--dry-run", "--workdir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "would remove" in result.output
    assert launch_dir.exists()


def test_clean_launches_multiple_recipes(tmp_path, monkeypatch):
    a = _seed_launch(tmp_path, "recipe-a")
    b = _seed_launch(tmp_path, "recipe-b")
    monkeypatch.setenv("SHINOBI_PROVENANCE__DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("SHINOBI_CACHE__DIR", str(tmp_path / "gone"))
    result = CliRunner().invoke(
        main, ["clean", "--no-runs", "--no-cache", "--launches", "--workdir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert not a.exists() and not b.exists()


def test_clean_launches_no_matches_is_graceful(tmp_path, monkeypatch):
    monkeypatch.setenv("SHINOBI_PROVENANCE__DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("SHINOBI_CACHE__DIR", str(tmp_path / "gone"))
    result = CliRunner().invoke(
        main, ["clean", "--no-runs", "--no-cache", "--launches", "--workdir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "nothing at" in result.output
