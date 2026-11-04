"""Declaration discovery is reused only within one construction/planning pass."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from shinobi import MSv2
from shinobi._annotations import annotation_cache
from shinobi import datasets
from shinobi.datasets import DatasetDeclarationError, dataset_declarations, executable_dataset_fields
from shinobi.steps.schema import path_fields


def test_discovery_reuses_walks_without_sharing_mutable_results(monkeypatch):
    class Inputs(BaseModel):
        ms: MSv2

    walks = 0
    original = datasets.walk_model_annotations

    def counted(model):
        nonlocal walks
        walks += 1
        yield from original(model)

    monkeypatch.setattr(datasets, "walk_model_annotations", counted)
    with annotation_cache():
        for discover in (dataset_declarations, executable_dataset_fields, path_fields):
            result = discover(Inputs)
            assert set(result) == {"ms"}
            result.clear()
            with annotation_cache():
                assert set(discover(Inputs)) == {"ms"}
        # Dataset discovery keeps the executable boundary's error type distinct.
        assert walks == 2
    assert set(dataset_declarations(Inputs)) == {"ms"}
    assert walks == 3


@pytest.mark.parametrize("interrupted", [False, True])
def test_new_pass_discovers_rebuilt_schema_after_normal_or_exceptional_exit(interrupted):
    class Inputs(BaseModel):
        value: int

    class Interrupted(Exception):
        pass

    try:
        with annotation_cache():
            assert path_fields(Inputs) == set()
            assert dataset_declarations(Inputs) == {}
            assert executable_dataset_fields(Inputs) == {}
            if interrupted:
                raise Interrupted
    except Interrupted:
        pass

    Inputs.model_fields["value"].annotation = MSv2
    Inputs.model_rebuild(force=True)
    # Outside a pass discovery must also be fresh.
    assert path_fields(Inputs) == {"value"}
    with annotation_cache():
        assert path_fields(Inputs) == {"value"}
        assert set(dataset_declarations(Inputs)) == {"value"}
        assert set(executable_dataset_fields(Inputs)) == {"value"}


def test_discovery_cache_preserves_validation_options_and_error_types():
    class Nested(BaseModel):
        ms: MSv2

    class Inputs(BaseModel):
        nested: Nested
        ordinary: Path

    with annotation_cache():
        assert set(executable_dataset_fields(Inputs, reject_unsupported=False)) == {"nested.ms"}
        with pytest.raises(DatasetDeclarationError, match="nested"):
            executable_dataset_fields(Inputs)
        with pytest.raises(ValueError, match="nested"):
            executable_dataset_fields(Inputs, error=ValueError)
        assert set(executable_dataset_fields(Inputs, reject_unsupported=False)) == {"nested.ms"}
