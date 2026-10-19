"""Selected format operations for the shared detached lifecycle driver."""

from __future__ import annotations


class LifecycleOperations:
    def __init__(self, fits: bool):
        self.fits = fits
        if fits:
            from shinobi import fits_lifecycle as module

            self.lifecycle = module.FITSLifecycle
            self.store = module.FITSLifecycleStore
            self.snapshot = module.FITSSnapshot
            self.capability = module.fits_backend_capability
        else:
            from shinobi import dataset_lifecycle as module
            from shinobi.dataset_backends import dataset_backend_capability

            self.lifecycle = module.DatasetLifecycle
            self.store = module.DatasetLifecycleStore
            self.snapshot = module.DatasetLifecycleSnapshot
            self.capability = dataset_backend_capability
        self.module = module

    def resolve(self, scope, values, *, workspace, mutation, cache_dir, producer_step_path=None):
        return self.module.resolve_lifecycle_snapshot(
            scope, values, workspace=workspace, mutation=mutation, **({"cache_dir": cache_dir, "producer_step_path": producer_step_path} if self.fits else {})
        )

    def observe(self, accesses, roots, workspace):
        if self.fits:
            return self.module.observe_accesses(self.module.current_accesses(tuple(a for a in accesses if a.root in roots)))
        return self.module.observe_roots(roots, workspace)

    def recovery_matches(self, before, after):
        if before is None or after is None:
            return before is after
        if self.fits:
            return before.byte_profile == after.byte_profile and before.sha256 == after.sha256
        return self.module.structural_signature(before) == self.module.structural_signature(after) and self.module.member_fingerprint(before) == self.module.member_fingerprint(
            after
        )
