import threading

from click.testing import CliRunner

from shinobi.cache import CacheManifest
from shinobi.cli import _clear_step_cache, main
from shinobi.snapshots import get_journal
from shinobi.storage import SharedFileLock


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
    assert not runs.exists()
    assert (cache / "manifest.json.lock").exists()
    assert (cache / "snapshots" / "chains.json.lock").exists()
    assert not (cache / "manifest.json").exists()


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


def test_clean_waits_for_metadata_transaction_and_preserves_lock_domain(tmp_path, monkeypatch):
    _runs, cache = _seed(tmp_path, monkeypatch)
    manifest = CacheManifest(cache / "manifest.json")
    manifest.update(lambda data: data.update(seed={"cache_key": "key"}))
    inode = manifest.lock_path.stat().st_ino
    entered = threading.Event()
    release = threading.Event()

    def hold(data):
        entered.set()
        release.wait(timeout=10)
        data["held"] = {"cache_key": "held"}

    holder = threading.Thread(target=lambda: manifest.update(hold))
    holder.start()
    assert entered.wait(timeout=10)
    outcome = {}

    def clean_cache():
        _clear_step_cache(cache)
        outcome["complete"] = True

    cleaner = threading.Thread(target=clean_cache)
    cleaner.start()
    assert cleaner.is_alive()
    release.set()
    holder.join(timeout=10)
    cleaner.join(timeout=10)
    assert outcome["complete"] is True
    assert manifest.lock_path.stat().st_ino == inode
    manifest.update(lambda data: data.update(after={"cache_key": "after"}))
    assert set(manifest.read()) == {"after"}


def test_clean_preserves_the_recovery_lock_domain(tmp_path, monkeypatch):
    _runs, cache = _seed(tmp_path, monkeypatch)
    journal = get_journal(str(cache))
    lock = SharedFileLock(journal.root / "recovery")
    lock.acquire()
    inode = lock.path.stat().st_ino
    outcome = {}

    def clean_cache():
        _clear_step_cache(cache)
        outcome["complete"] = True

    cleaner = threading.Thread(target=clean_cache)
    cleaner.start()
    cleaner.join(timeout=0.05)
    assert cleaner.is_alive()
    lock.release()
    cleaner.join(timeout=10)

    assert outcome["complete"] is True
    assert journal.recovery_lock_path.stat().st_ino == inode


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
    result = CliRunner().invoke(main, ["clean", "--no-runs", "--no-cache", "--launches", "--workdir", str(tmp_path)])
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
    result = CliRunner().invoke(main, ["clean", "--no-runs", "--no-cache", "--launches", "--workdir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert not a.exists() and not b.exists()


def test_clean_launches_no_matches_is_graceful(tmp_path, monkeypatch):
    monkeypatch.setenv("SHINOBI_PROVENANCE__DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("SHINOBI_CACHE__DIR", str(tmp_path / "gone"))
    result = CliRunner().invoke(main, ["clean", "--no-runs", "--no-cache", "--launches", "--workdir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "nothing at" in result.output
