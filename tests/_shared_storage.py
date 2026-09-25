from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

from shinobi.dataset_backends import SharedStorageQualification


def qualified_storage(root: Path) -> Path:
    """Create test-only persisted evidence for one temporary storage root."""

    path = root / "shared-storage-qualification.json"
    record = SharedStorageQualification(
        qualification_id=uuid4(),
        storage_root=root.resolve(),
        verified_at=time.time(),
        workers=("test-worker-a", "test-worker-b"),
        evidence="pytest fixture standing in for a completed physical M2 probe",
    )
    path.write_text(record.model_dump_json(indent=2) + "\n")
    return path
