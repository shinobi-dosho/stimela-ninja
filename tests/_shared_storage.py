from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from shinobi.dataset_backends import SharedStorageQualification


def qualified_storage(root: Path, *, storage_root: Path | None = None) -> Path:
    """Attest a test storage root, retaining an isolated qualification file.

    Broad product reservations can put the ownership record in a parent of
    the workflow workspace; callers explicitly include that parent.
    """

    path = root / "shared-storage-qualification.json"
    record = SharedStorageQualification(
        qualification_id=uuid4(),
        storage_root=(storage_root or root).resolve(),
        verified_at=time.time(),
        workers=("test-worker-a", "test-worker-b"),
        evidence="pytest fixture standing in for a completed physical M2 probe",
    )
    path.write_text(record.model_dump_json(indent=2) + "\n")
    return path
