"""Copy a file tree as cheaply as the filesystem allows, without ever
letting "cheaply" change what the copy contains.

This exists for `shinobi.snapshots`, which keeps point-in-time copies of
multi-gigabyte measurement sets so a mutation chain can be rolled back.
Storage is scarce relative to that data, so a design whose steady state is
N full copies of a 2 TB MS is unusable -- hence a ladder of increasingly
cheap mechanisms, probed at runtime.

**The ladder affects space only, never content.** Every rung produces a
tree that is byte-identical to the source at the moment of the copy, and
stays that way when the source is subsequently rewritten. That invariant
is the whole contract, and it is why there is no hardlink rung: a
hardlinked "copy" shares the inode, so a step that rewrites the MS in
place -- which is precisely the thing being protected against, and which
casacore does by writing into existing table files rather than replacing
them -- would rewrite the snapshot along with it. The snapshot would then
silently hold post-mutation state, and restoring it would reinstate the
corruption it was taken to undo. A reflink and a `copy_file_range` clone
are copy-on-write at the *block* level and do not have this problem; a
hardlink is not a clone at all.

The rungs:

1. **`FICLONE`** -- a whole-file reflink, one ioctl. XFS with
   ``reflink=1`` and Btrfs. Probed by actually attempting the ioctl on two
   scratch files in the target directory, never by parsing anyone's
   stderr: the answer depends on the mount, not on the filesystem name.
2. **`copy_file_range`** -- ZFS >= 2.2 hooks block cloning into this and
   *only* into this (a reflink ioctl fails there, which is why
   ``cp --reflink=always`` does not work on ZFS while plain ``cp`` may
   clone for free). Whether a given call actually cloned or fell back to a
   read/write loop is not observable from userspace on ZFS -- there is no
   ``filefrag`` equivalent -- so this rung is recorded as *probable*
   sharing, and gated on ``zfs_bclone_enabled`` being on (it was disabled
   by default after the 2.2.0-era corruption bug, so its state has to be
   read at runtime rather than assumed).
3. **`copy`** -- an ordinary read/write copy. Always correct, always full
   price. Callers that care about capacity preflight against free space
   before asking for one.

``--reflink=auto`` has no analogue here on purpose: silently degrading to
a full copy is the single worst behaviour under capacity scarcity, so a
caller either gets a cheap copy or gets told the price.

Every probe decision (which filesystem, which rung, and why) is recorded
and reported by ``ninja cache check`` -- when the space arithmetic looks
wrong on some future Lustre or ZFS deployment, the answer should be in a
report rather than in a re-run with debug logging.
"""

from __future__ import annotations

import errno
import fcntl
import logging
import os
import shutil
import stat
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger("shinobi.clonefs")


# _IOW(0x94, 9, int) -- the whole-file reflink ioctl. Hardcoded because it is
# a stable part of the Linux ABI and the alternative is a C extension.
FICLONE = 0x40049409


class CloneTier(str, Enum):
    """How cheaply a filesystem can copy a file. Ordered best-first."""

    FICLONE = "ficlone"  # verified reflink: shares blocks, CoW on write
    COPY_FILE_RANGE = "copy_file_range"  # probable clone (ZFS); sharing unverifiable
    COPY = "copy"  # full read/write copy, full price

    @property
    def shares_blocks(self) -> bool:
        """Whether this rung is expected to share blocks with the source.

        "Expected", not "guaranteed": `COPY_FILE_RANGE` may or may not have
        cloned, and userspace cannot tell. Used for space *reporting*, never
        for a correctness decision.
        """
        return self is not CloneTier.COPY


@dataclass(frozen=True)
class CloneDecision:
    """Why a filesystem got the rung it got -- reported by `ninja cache check`."""

    device: int
    mountpoint: str
    fstype: str
    tier: CloneTier
    reason: str


_decisions: dict[int, CloneDecision] = {}
_decisions_lock = threading.Lock()


def decisions() -> list[CloneDecision]:
    """Every probe decision made so far, for `ninja cache check`."""
    with _decisions_lock:
        return sorted(_decisions.values(), key=lambda d: d.mountpoint)


def reset_probe_cache() -> None:
    """Forget every memoized probe. For tests; a real run probes once."""
    with _decisions_lock:
        _decisions.clear()


def _mount_info(path: Path) -> tuple[str, str]:
    """`(mountpoint, fstype)` for whatever filesystem `path` sits on.

    Read from `/proc/self/mountinfo` by longest-matching mountpoint rather
    than shelling out to `stat -f`, and degrading to `("", "unknown")`
    rather than raising -- the fstype only ever selects which rung to
    *attempt*, so not knowing it costs a probe, not correctness.
    """
    try:
        target = os.path.realpath(path)
        best = ("", "unknown")
        best_len = -1
        with open("/proc/self/mountinfo") as fh:
            for line in fh:
                fields = line.split()
                try:
                    sep = fields.index("-")
                except ValueError:
                    continue
                mountpoint, fstype = fields[4], fields[sep + 1]
                if (target == mountpoint or target.startswith(mountpoint.rstrip("/") + "/")) and len(mountpoint) > best_len:
                    best, best_len = (mountpoint, fstype), len(mountpoint)
        return best
    except OSError:
        return ("", "unknown")


def _zfs_bclone_enabled() -> bool:
    """Whether ZFS block cloning is switched on in this kernel module.

    Read at runtime rather than assumed: it shipped enabled in 2.2.0, was
    disabled by default after the data-corruption bug found shortly after,
    and distributions have not converged. Absent file (not ZFS, or an older
    module) reads as off.
    """
    try:
        return Path("/sys/module/zfs/parameters/zfs_bclone_enabled").read_text().strip() not in ("0", "")
    except OSError:
        return False


def _try_ficlone(directory: Path) -> bool:
    """Attempt a real `FICLONE` between two scratch files in `directory`.

    An actual attempt, because reflink support is a property of the mount
    and its options, not of the filesystem's name: XFS supports it only
    when made with `reflink=1`, and a bind mount or an overlay can differ
    from the underlying device.
    """
    # A uuid, not the pid: two processes on *different hosts* sharing one
    # directory (which is the normal case on the shared filesystems this
    # ladder exists for) can hold the same pid, and would then probe over
    # each other's scratch files.
    tag = uuid.uuid4().hex[:12]
    src = directory / f".shinobi-clone-probe-{tag}.src"
    dst = directory / f".shinobi-clone-probe-{tag}.dst"
    try:
        src.write_bytes(b"shinobi clone capability probe\n")
        src_fd = os.open(src, os.O_RDONLY)
        try:
            dst_fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                fcntl.ioctl(dst_fd, FICLONE, src_fd)
                return True
            finally:
                os.close(dst_fd)
        finally:
            os.close(src_fd)
    except OSError:
        return False
    finally:
        for probe in (src, dst):
            try:
                probe.unlink()
            except OSError:
                pass


def probe(path: Path) -> CloneTier:
    """The best rung available for copies made *within* `path`'s filesystem.

    Memoized per device, so a run probes each filesystem once. `path` need
    not exist; the nearest existing ancestor is probed, since that is the
    filesystem the copy will land on anyway.
    """
    directory = Path(path)
    while not directory.exists() and directory != directory.parent:
        directory = directory.parent
    if not directory.is_dir():
        directory = directory.parent

    try:
        device = directory.stat().st_dev
    except OSError:
        return CloneTier.COPY

    with _decisions_lock:
        existing = _decisions.get(device)
    if existing is not None:
        return existing.tier

    mountpoint, fstype = _mount_info(directory)
    if _try_ficlone(directory):
        tier, reason = CloneTier.FICLONE, "FICLONE ioctl succeeded on a two-file probe"
    elif fstype == "zfs" and _zfs_bclone_enabled():
        tier, reason = CloneTier.COPY_FILE_RANGE, "ZFS with zfs_bclone_enabled=1: copy_file_range probably clones, sharing not verifiable from userspace"
    elif fstype == "zfs":
        tier, reason = CloneTier.COPY, "ZFS with block cloning disabled (zfs_bclone_enabled=0)"
    else:
        tier, reason = CloneTier.COPY, f"no reflink support on {fstype or 'unknown'}"

    decision = CloneDecision(device=device, mountpoint=mountpoint, fstype=fstype, tier=tier, reason=reason)
    with _decisions_lock:
        _decisions.setdefault(device, decision)
    logger.debug("clone capability for %s (%s): %s -- %s", mountpoint or directory, fstype, tier.value, reason)
    return tier


def _copy_file(src: Path, dst: Path, tier: CloneTier) -> None:
    """Copy one regular file at the best rung, degrading on the spot.

    Degradation is per *file* and silent by design: a rung failing mid-tree
    (a filesystem that reflinks most files but refuses one, an
    `EOPNOTSUPP` from a nested mount) must still produce a complete,
    correct tree. Only the space bill changes, which is exactly what A1
    licenses.
    """
    src_fd = os.open(src, os.O_RDONLY)
    try:
        size = os.fstat(src_fd).st_size
        dst_fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            _copy_fd(src_fd, dst_fd, size, tier)
        finally:
            os.close(dst_fd)
    finally:
        os.close(src_fd)
    # Outside the fast-path returns on purpose: a reflinked file that kept
    # the source's *blocks* but not its mtime would refingerprint as a
    # different dataset, and a generation-0 snapshot is named by exactly that
    # fingerprint.
    shutil.copystat(src, dst)


def _copy_fd(src_fd: int, dst_fd: int, size: int, tier: CloneTier) -> None:
    """Fill `dst_fd` from `src_fd`, trying `tier` first and degrading."""
    if tier is CloneTier.FICLONE:
        try:
            fcntl.ioctl(dst_fd, FICLONE, src_fd)
            return
        except OSError:
            os.ftruncate(dst_fd, 0)
    if tier in (CloneTier.FICLONE, CloneTier.COPY_FILE_RANGE):
        try:
            offset = 0
            while offset < size:
                written = os.copy_file_range(src_fd, dst_fd, size - offset, offset, offset)
                if written == 0:
                    break
                offset += written
            if offset >= size:
                return
        except OSError as exc:
            if exc.errno not in (errno.EXDEV, errno.EINVAL, errno.EOPNOTSUPP, errno.ENOSYS, errno.EPERM):
                raise
        os.ftruncate(dst_fd, 0)
    os.lseek(src_fd, 0, os.SEEK_SET)
    os.lseek(dst_fd, 0, os.SEEK_SET)
    while chunk := os.read(src_fd, 1 << 20):
        while chunk:
            chunk = chunk[os.write(dst_fd, chunk) :]


def clone_tree(src: Path, dst: Path, tier: CloneTier | None = None) -> None:
    """Copy `src` to `dst` -- a regular file, or a whole directory tree.

    `dst` must not already exist. Metadata (mode, mtime) is preserved,
    because a boundary path's fingerprint is built from mtimes and a
    restored tree that re-dated every file would look like a different
    dataset to anything that fingerprints it later.

    Symlinks are recreated as symlinks rather than followed -- the copy is
    meant to be a faithful image of the tree, and following them would
    both inflate it and lose the distinction on restore.
    """
    src, dst = Path(src), Path(dst)
    tier = probe(dst.parent) if tier is None else tier

    st = src.lstat()
    if stat.S_ISLNK(st.st_mode):
        os.symlink(os.readlink(src), dst)
        return
    if stat.S_ISREG(st.st_mode):
        dst.parent.mkdir(parents=True, exist_ok=True)
        _copy_file(src, dst, tier)
        return
    if not stat.S_ISDIR(st.st_mode):
        return  # fifo/socket/device: nothing meaningful to snapshot

    dst.mkdir(parents=True, exist_ok=False)
    for entry in os.scandir(src):
        clone_tree(Path(entry.path), dst / entry.name, tier)
    shutil.copystat(src, dst)


def tree_size(path: Path) -> int:
    """Apparent bytes under `path`, for space preflight and eviction
    accounting.

    Apparent size, not allocated blocks: on a clone-capable filesystem the
    two diverge wildly (a freshly cloned 2 TB tree allocates almost
    nothing), and the number is used to answer "would a *full copy* fit",
    which is the pessimistic question worth asking. It therefore
    overestimates reclaimable space on CoW filesystems -- a documented
    imprecision, not a bug.
    """
    try:
        st = os.lstat(path)
    except OSError:
        return 0
    if not stat.S_ISDIR(st.st_mode):
        return st.st_size
    total = 0
    stack = [os.fspath(path)]
    while stack:
        try:
            scan = os.scandir(stack.pop())
        except OSError:
            continue
        with scan:
            for entry in scan:
                try:
                    est = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISDIR(est.st_mode):
                    stack.append(entry.path)
                else:
                    total += est.st_size
    return total


def free_space(path: Path) -> int:
    """Bytes available to an unprivileged writer at `path`, or a very large
    number if that cannot be determined (an unknown budget must not be
    reported as an empty one and block real work).
    """
    directory = Path(path)
    while not directory.exists() and directory != directory.parent:
        directory = directory.parent
    try:
        st = os.statvfs(directory)
        return st.f_bavail * st.f_frsize
    except OSError:
        return 1 << 62


def can_afford(src: Path, dst_parent: Path, tier: CloneTier | None = None) -> tuple[bool, int, int]:
    """`(affordable, needed, available)` for copying `src` under `dst_parent`.

    A block-sharing rung needs essentially nothing, so it is always
    affordable; a full copy needs the whole apparent size. Callers preflight
    with this and refuse loudly rather than half-filling a filesystem with a
    2 TB tree they cannot complete -- the failure mode that would otherwise
    take the workspace down with it.
    """
    tier = probe(dst_parent) if tier is None else tier
    available = free_space(dst_parent)
    if tier.shares_blocks:
        return True, 0, available
    needed = tree_size(src)
    return needed <= available, needed, available
