"""The clone ladder: every rung must produce byte-identical content, and
must keep producing it after the source is rewritten. Only the space bill
is allowed to differ (assumption A1).
"""

import os
from pathlib import Path

import pytest

from shinobi import clonefs
from shinobi.clonefs import CloneTier, can_afford, clone_tree, decisions, free_space, probe, reset_probe_cache, tree_size


@pytest.fixture(autouse=True)
def _fresh_probes():
    reset_probe_cache()
    yield
    reset_probe_cache()


def _tree(root: Path) -> Path:
    """A miniature MS: a directory of small files, a subtable, a symlink."""
    root.mkdir()
    (root / "table.dat").write_bytes(b"visibilities" * 100)
    (root / "table.f0").write_bytes(b"\x00" * 4096)
    sub = root / "ANTENNA"
    sub.mkdir()
    (sub / "table.dat").write_bytes(b"antennas")
    (root / "link.dat").symlink_to("table.dat")
    return root


def _snapshot_of(root: Path) -> dict[str, bytes | str]:
    """Content of every entry under `root`, symlinks by their target."""
    out: dict[str, bytes | str] = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            full = Path(dirpath) / name
            rel = str(full.relative_to(root))
            out[rel] = os.readlink(full) if full.is_symlink() else full.read_bytes()
    return out


@pytest.fixture(params=[True, False], ids=["with-cfr", "without-cfr"])
def _copy_file_range_availability(request, monkeypatch):
    """Run the content tests both with and without `os.copy_file_range`.

    Its absence is not hypothetical: CPython compiles the binding in only
    when the C library had it at configure time, so a portable
    redistributable build (what `uv python install` fetches, and what CI
    runs) lacks it on a kernel that supports it perfectly well. That is
    exactly the kind of environment difference A1 says must cost space and
    never content, so both answers are exercised rather than whichever the
    test machine happens to have.

    Only *absence* can be faked. Claiming the syscall exists on an
    interpreter that lacks it just walks the code into the AttributeError
    the gate was added to prevent -- testing the fake rather than the
    behaviour -- so that direction is skipped instead.
    """
    if request.param and not hasattr(os, "copy_file_range"):
        pytest.skip("this interpreter was built without os.copy_file_range, so its presence cannot be simulated")
    monkeypatch.setattr(clonefs, "_has_copy_file_range", lambda: request.param)
    return request.param


@pytest.mark.parametrize("tier", list(CloneTier))
def test_every_rung_reproduces_the_tree_exactly(tmp_path, tier, _copy_file_range_availability):
    """A1: a reflink, a copy_file_range clone and a full copy differ only in
    space. If any rung produced different bytes the whole design would rest
    on which filesystem happened to be underneath.
    """
    src = _tree(tmp_path / "data.ms")
    dst = tmp_path / "snap.ms"
    clone_tree(src, dst, tier=tier)
    assert _snapshot_of(dst) == _snapshot_of(src)


@pytest.mark.parametrize("tier", list(CloneTier))
def test_a_snapshot_does_not_move_when_the_source_is_rewritten_in_place(tmp_path, tier, _copy_file_range_availability):
    """The property that rules hardlinks out of the ladder entirely.

    An MS is rewritten *in place* -- casacore writes into existing table
    files rather than replacing them -- so a snapshot that shared the source
    inode would silently follow the mutation and restore the very corruption
    it was taken to undo. This is the regression test for that: mutate the
    source the way the real thing does (open for update, seek, write) and the
    snapshot must not notice.
    """
    src = _tree(tmp_path / "data.ms")
    dst = tmp_path / "snap.ms"
    clone_tree(src, dst, tier=tier)
    before = _snapshot_of(dst)

    with open(src / "table.f0", "r+b") as fh:
        fh.seek(0)
        fh.write(b"CALIBRATED")
    with open(src / "ANTENNA" / "table.dat", "r+b") as fh:
        fh.seek(0)
        fh.write(b"XX")

    assert _snapshot_of(dst) == before
    assert _snapshot_of(dst) != _snapshot_of(src)


@pytest.mark.parametrize("tier", list(CloneTier))
def test_mtimes_are_preserved_on_every_rung(tmp_path, tier, _copy_file_range_availability):
    """A boundary path's fingerprint is `[relpath, mtime_ns, size]`, and a
    generation-0 snapshot is *named* by the hash of that fingerprint. A rung
    that kept the bytes but re-dated the files would make a restored tree
    fingerprint as a different dataset.
    """
    src = _tree(tmp_path / "data.ms")
    os.utime(src / "table.dat", (1_000_000, 1_000_000))
    dst = tmp_path / "snap.ms"
    clone_tree(src, dst, tier=tier)
    assert (dst / "table.dat").stat().st_mtime_ns == (src / "table.dat").stat().st_mtime_ns


def test_a_single_file_clones_as_a_file(tmp_path):
    src = tmp_path / "image.fits"
    src.write_bytes(b"SIMPLE  =                    T")
    clone_tree(src, tmp_path / "snap.fits")
    assert (tmp_path / "snap.fits").read_bytes() == src.read_bytes()


def test_symlinks_are_recreated_not_followed(tmp_path):
    """Following them would inflate the snapshot and lose the distinction on
    restore -- the copy is meant to be a faithful image of the tree.
    """
    src = _tree(tmp_path / "data.ms")
    dst = tmp_path / "snap.ms"
    clone_tree(src, dst)
    assert (dst / "link.dat").is_symlink()
    assert os.readlink(dst / "link.dat") == "table.dat"


def test_clone_refuses_to_overwrite_an_existing_tree(tmp_path):
    """Snapshot names are content identities; silently overwriting one would
    let a wrong state inherit a right name.
    """
    src = _tree(tmp_path / "data.ms")
    (tmp_path / "snap.ms").mkdir()
    with pytest.raises(FileExistsError):
        clone_tree(src, tmp_path / "snap.ms")


# --- the probe ladder, with the filesystem faked out ---


def test_probe_reports_ficlone_when_the_ioctl_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(clonefs, "_try_ficlone", lambda directory: True)
    monkeypatch.setattr(clonefs, "_mount_info", lambda path: ("/", "xfs"))
    assert probe(tmp_path) is CloneTier.FICLONE
    assert "FICLONE" in decisions()[0].reason


def test_probe_reports_probable_clone_on_zfs_with_block_cloning_on(tmp_path, monkeypatch):
    """ZFS hooks cloning into copy_file_range only, so the reflink ioctl
    fails there and the naive ladder would drop straight to a full copy.
    """
    monkeypatch.setattr(clonefs, "_try_ficlone", lambda directory: False)
    monkeypatch.setattr(clonefs, "_mount_info", lambda path: ("/tank", "zfs"))
    monkeypatch.setattr(clonefs, "_zfs_bclone_enabled", lambda: True)
    monkeypatch.setattr(clonefs, "_has_copy_file_range", lambda: True)
    assert probe(tmp_path) is CloneTier.COPY_FILE_RANGE
    assert "not verifiable" in decisions()[0].reason


def test_probe_falls_to_copy_on_zfs_with_block_cloning_disabled(tmp_path, monkeypatch):
    """`zfs_bclone_enabled` was turned off by default after the 2.2.0-era
    corruption bug, so its state is read rather than assumed.
    """
    monkeypatch.setattr(clonefs, "_try_ficlone", lambda directory: False)
    monkeypatch.setattr(clonefs, "_mount_info", lambda path: ("/tank", "zfs"))
    monkeypatch.setattr(clonefs, "_zfs_bclone_enabled", lambda: False)
    assert probe(tmp_path) is CloneTier.COPY
    assert "disabled" in decisions()[0].reason


def test_probe_falls_to_copy_on_a_filesystem_with_no_clone_support(tmp_path, monkeypatch):
    monkeypatch.setattr(clonefs, "_try_ficlone", lambda directory: False)
    monkeypatch.setattr(clonefs, "_mount_info", lambda path: ("/scratch", "lustre"))
    assert probe(tmp_path) is CloneTier.COPY
    assert "lustre" in decisions()[0].reason


def test_probe_is_memoized_per_filesystem(tmp_path, monkeypatch):
    """A run probes each filesystem once -- the probe writes two scratch
    files, and doing that per snapshot would be absurd.
    """
    calls = {"n": 0}

    def counting_probe(directory):
        calls["n"] += 1
        return True

    monkeypatch.setattr(clonefs, "_try_ficlone", counting_probe)
    monkeypatch.setattr(clonefs, "_mount_info", lambda path: ("/", "xfs"))
    probe(tmp_path)
    probe(tmp_path / "somewhere" / "deeper")
    assert calls["n"] == 1


def test_probe_of_a_nonexistent_path_uses_its_nearest_existing_ancestor(tmp_path, monkeypatch):
    """Snapshots are probed by their *destination*, which by definition does
    not exist yet.
    """
    monkeypatch.setattr(clonefs, "_try_ficlone", lambda directory: True)
    monkeypatch.setattr(clonefs, "_mount_info", lambda path: ("/", "xfs"))
    assert probe(tmp_path / "not" / "there" / "yet") is CloneTier.FICLONE


def test_the_real_probe_answers_without_raising(tmp_path):
    """Whatever the test machine's filesystem is, the probe must return a
    rung rather than blow up -- and must not leave its scratch files behind.
    """
    assert probe(tmp_path) in set(CloneTier)
    assert not list(tmp_path.glob(".shinobi-clone-probe-*"))


# --- space preflight ---


def test_a_block_sharing_rung_is_always_affordable(tmp_path, monkeypatch):
    src = _tree(tmp_path / "data.ms")
    monkeypatch.setattr(clonefs, "free_space", lambda path: 0)
    affordable, needed, _available = can_afford(src, tmp_path, tier=CloneTier.FICLONE)
    assert affordable and needed == 0


def test_a_full_copy_is_refused_when_it_would_not_fit(tmp_path, monkeypatch):
    """The refusal that stops a 2 TB tree half-filling a filesystem and
    taking the workspace down with it.
    """
    src = _tree(tmp_path / "data.ms")
    monkeypatch.setattr(clonefs, "free_space", lambda path: 10)
    affordable, needed, available = can_afford(src, tmp_path, tier=CloneTier.COPY)
    assert not affordable
    assert needed > available == 10


def test_tree_size_counts_files_not_directories(tmp_path):
    src = _tree(tmp_path / "data.ms")
    expected = sum(p.lstat().st_size for p in src.rglob("*") if not p.is_dir())
    assert tree_size(src) == expected


def test_tree_size_of_a_missing_path_is_zero(tmp_path):
    assert tree_size(tmp_path / "gone") == 0


def test_free_space_reports_a_real_number(tmp_path):
    assert free_space(tmp_path) > 0


def test_unknown_free_space_does_not_block_work(tmp_path, monkeypatch):
    """An undeterminable budget must not be reported as an empty one, or a
    stat failure would refuse every snapshot on the machine.
    """
    monkeypatch.setattr(clonefs.os, "statvfs", lambda path: (_ for _ in ()).throw(OSError()))
    assert free_space(tmp_path) > (1 << 60)


def test_a_python_without_copy_file_range_does_not_get_the_zfs_rung(tmp_path, monkeypatch):
    """ZFS exposes block cloning through `copy_file_range` and nothing else,
    so an interpreter without it cannot reach that rung -- and must be told
    why rather than silently reporting a clone it never performed.
    """
    monkeypatch.setattr(clonefs, "_try_ficlone", lambda directory: False)
    monkeypatch.setattr(clonefs, "_mount_info", lambda path: ("/tank", "zfs"))
    monkeypatch.setattr(clonefs, "_zfs_bclone_enabled", lambda: True)
    monkeypatch.setattr(clonefs, "_has_copy_file_range", lambda: False)
    assert probe(tmp_path) is CloneTier.COPY
    assert "without copy_file_range" in decisions()[0].reason
