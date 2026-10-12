"""Fresh/cache product validation retains FITS profiles and explicit selectors."""

from pathlib import Path

import pytest

from pydantic import BaseModel

from shinobi import FITSImage, FITSTable, FITSAccess, FITSHDU, pystep
from shinobi.products import ProductFamily, ProductMember, framework_type
from shinobi.fits_products import validate_fits_outputs
from shinobi.datasets import dataset_declarations
from shinobi.cache import compute_cache_key, product_contract
from shinobi.steps.schema import Scope
from shinobi.steps.dispatch import _dispatch
from shinobi.exceptions import DatasetLifecycleViolationError
from shinobi.loaders._modelgen import dtype_to_type


class Images(BaseModel):
    images: ProductFamily[FITSImage]


class Tables(BaseModel):
    tables: ProductFamily[FITSTable]


class Empty(BaseModel):
    pass


fits = pytest.importorskip("astropy.io.fits")
np = pytest.importorskip("numpy")

# Gate the helper module before importing it: it also skips without Astropy.
from tests.test_fits import image  # noqa: E402


def test_exact_canonical_family_types():
    assert framework_type(ProductFamily[FITSImage]) == ("family", FITSImage)
    assert framework_type(ProductMember[FITSTable]) == ("member", FITSTable)
    assert dtype_to_type("ProductFamily[FITSImage]") is ProductFamily[FITSImage]
    assert dtype_to_type("ProductFamily[FITSTable]") is ProductFamily[FITSTable]
    assert any(one.profile == "fits-image/v1" for one in dataset_declarations(Images).values())


def test_table_family_requires_selector_and_changes_contract(tmp_path):
    path = tmp_path / "t.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns([], name="CAT")]).writeto(path)
    outputs = Tables(tables=ProductFamily[FITSTable](resolved=True, members=(ProductMember[FITSTable](coordinates={}, value=path),)))
    default = Scope(name="tables", inputs_model=Empty, outputs_model=Tables)
    with pytest.raises(DatasetLifecycleViolationError):
        validate_fits_outputs(default, outputs)
    selected = Scope(name="tables", inputs_model=Empty, outputs_model=Tables, fits_accesses=[FITSAccess(field="tables", mode="read", hdu=FITSHDU(index=1))])
    validate_fits_outputs(selected, outputs)
    assert product_contract(default, {}) != product_contract(selected, {})
    assert compute_cache_key(default, test_exact_canonical_family_types, {}) != compute_cache_key(selected, test_exact_canonical_family_types, {})


def test_fresh_and_cache_corrupt_image(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "x.fits"

    @pystep()
    def emit(file: str) -> Images:
        file = Path(file)
        image(file)
        return Images(images=ProductFamily[FITSImage](resolved=True, members=(ProductMember[FITSImage](coordinates={}, value=file),)))

    kwargs = {"file": str(path), "cache": True, "cache_dir": str(tmp_path / "cache")}
    assert _dispatch(emit.step, emit.func, **kwargs).success
    assert _dispatch(emit.step, emit.func, **kwargs).cached
    path.write_text("invalid")
    assert not _dispatch(emit.step, emit.func, **kwargs).cached


def test_invalid_fresh_family_refuses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @pystep()
    def emit(file: str) -> Images:
        file = Path(file)
        file.write_text("invalid")
        return Images(images=ProductFamily[FITSImage](resolved=True, members=(ProductMember[FITSImage](coordinates={}, value=file),)))

    with pytest.raises(DatasetLifecycleViolationError):
        _dispatch(emit.step, emit.func, file=str(tmp_path / "bad.fits"))


def test_typed_product_codec_roundtrip():
    from shinobi.offload._codec import pack, unpack, TypeSpec

    for annotation in (ProductFamily[FITSImage], ProductFamily[FITSTable], ProductMember[FITSImage]):
        assert TypeSpec.capture(annotation).restore() is annotation
    value = ProductFamily[FITSImage](resolved=True, members=(ProductMember[FITSImage](coordinates={}, value=Path("x.fits")),))
    assert type(unpack(pack(value))) is type(value)
    assert unpack(pack(value)) == value


def test_yaml_optional_list_and_table_selector():
    from shinobi.loaders.yaml_cab import loads

    cab = loads("""cabs:
  images:
    command: cat
    inputs:
      files: {dtype: "List[FITSImage]", required: true}
      extra: {dtype: FITSImage, default: null}
    fits_accesses:
      - {field: files, mode: read}
  tables:
    command: cat
    inputs:
      table: {dtype: FITSTable, required: true}
    fits_accesses:
      - field: table
        mode: read
        hdu: {extname: CATALOG, extver: 2}
""")
    assert cab["images"].inputs_model(files=["a.fits"]).extra is None
    assert cab["tables"].fits_accesses[0].hdu == FITSHDU(extname="CATALOG", extver=2)


def test_ms_writer_invalid_typed_fits_prevents_commit(tmp_path, monkeypatch):
    pytest.importorskip("casacore.tables")
    from shinobi import MSv2, DatasetAccess, DatasetColumns, FamilySpec, MemberRule
    from shinobi.steps.schema import ParamMeta
    from tests._dataset_fixtures import make_ms, scans
    from shinobi.snapshots import get_journal, chain_id

    monkeypatch.chdir(tmp_path)
    ms = make_ms(tmp_path / "input.ms")
    before = scans(ms)

    import sys
    from shinobi import Cab

    class Inputs(BaseModel):
        script: str
        ms: MSv2

    cab = Cab(
        name="emit_invalid",
        command=f"{sys.executable} -c",
        inputs_model=Inputs,
        outputs_model=Images,
        dataset_accesses=[DatasetAccess(field="ms", mode="write", columns=DatasetColumns(write=("SCAN_NUMBER",)))],
        field_meta={
            "script": ParamMeta(positional_head=True),
            "ms": ParamMeta(positional=True),
            "images": ParamMeta(family=FamilySpec(coordinates={}, rules=(MemberRule(path="bad.fits", required=True),))),
        },
    )
    script = """from pathlib import Path
import sys
import numpy as np
from casacore.tables import table
with table(sys.argv[1], readonly=False, ack=False) as ms:
    ms.putcol('SCAN_NUMBER', np.full(ms.nrows(), 55, dtype=np.int32))
Path('bad.fits').write_bytes(b'not FITS')
"""
    with pytest.raises(DatasetLifecycleViolationError):
        _dispatch(cab, None, script=script, ms=ms, cache_dir=str(tmp_path / "cache"))
    assert scans(ms) == before
    assert not get_journal(str(tmp_path / "cache")).get(chain_id(ms)).marker


def test_read_bytes_only_hash_unwired_list_elements(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    @pystep(fits_accesses=[FITSAccess(field="files", mode="read")])
    def reader(files: list[FITSImage]) -> None:
        pass

    paths = [image(tmp_path / f"{index}.fits") for index in range(2)]
    keys = {"files": ["producer", None]}

    def key():
        return compute_cache_key(reader.step, reader.func, {"files": paths}, input_keys=keys)

    before = key()
    with fits.open(paths[0], mode="update") as hdus:
        hdus[0].data[:] = 4
    assert key() == before
    with fits.open(paths[1], mode="update") as hdus:
        hdus[0].data[:] = 4
    assert key() != before


def test_ms_writer_direct_fits_create_refuses_before_launch(tmp_path, monkeypatch):
    pytest.importorskip("casacore.tables")
    from shinobi import MSv2, Cab, DatasetAccess, DatasetColumns
    from shinobi.steps.schema import ParamMeta
    from tests._dataset_fixtures import make_ms, scans
    import sys

    monkeypatch.chdir(tmp_path)

    class Inputs(BaseModel):
        script: str
        ms: MSv2

    class Outputs(BaseModel):
        out: FITSImage = Path("bad.fits")

    cab = Cab(
        name="mixed",
        command=f"{sys.executable} -c",
        inputs_model=Inputs,
        outputs_model=Outputs,
        field_meta={"script": ParamMeta(positional_head=True), "ms": ParamMeta(positional=True)},
        dataset_accesses=[DatasetAccess(field="ms", mode="write", columns=DatasetColumns(write=("SCAN_NUMBER",)))],
        fits_accesses=[FITSAccess(field="out", mode="create")],
    )
    ms = make_ms(tmp_path / "input.ms")
    before = scans(ms)
    from shinobi.exceptions import DatasetLifecycleUnavailableError

    with pytest.raises(DatasetLifecycleUnavailableError):
        _dispatch(cab, None, ms=ms, script="from pathlib import Path; Path('bad.fits').write_bytes(b'partial')", cache_dir=str(tmp_path / "cache"))
    assert not Path("bad.fits").exists()
    assert scans(ms) == before


@pytest.mark.parametrize("annotation", [ProductFamily[Path], ProductMember[Path]])
def test_selector_on_non_fits_product_refuses(annotation):
    from pydantic import create_model, ValidationError

    model = create_model("PlainProducts", items=(annotation, ...))
    with pytest.raises(ValidationError):
        Scope(name="plain", inputs_model=Empty, outputs_model=model, fits_accesses=[FITSAccess(field="items", mode="read", hdu=FITSHDU(index=1))])


@pytest.mark.parametrize("annotation", [ProductFamily[FITSImage] | None, list[ProductMember[FITSTable]]])
def test_selector_on_actual_optional_or_list_fits_product(annotation):
    from pydantic import create_model

    model = create_model("FitsProducts", items=(annotation, ...))
    Scope(name="fits", inputs_model=Empty, outputs_model=model, fits_accesses=[FITSAccess(field="items", mode="read", hdu=FITSHDU(index=1))])


def test_yaml_selector_on_plain_product_refuses():
    from shinobi.loaders.yaml_cab import loads

    with pytest.raises(ValueError):
        loads("""cabs:
  plain:
    command: cat
    outputs:
      items: {dtype: "ProductFamily[Path]"}
    fits_accesses:
      - {field: items, mode: read, hdu: {index: 1}}
""")


@pytest.mark.parametrize("annotation", [dict[str, ProductMember[FITSImage]], ProductMember[FITSImage] | int, list[ProductMember[FITSImage] | int]])
@pytest.mark.parametrize("selector", [False, True])
def test_unsupported_fits_product_shapes_refuse_declaration(annotation, selector):
    from pydantic import create_model, ValidationError

    model = create_model("UnsupportedProducts", out=(annotation, ...))
    with pytest.raises(ValidationError):
        Scope(name="unsupported", inputs_model=Empty, outputs_model=model, fits_accesses=[FITSAccess(field="out", mode="read", hdu=FITSHDU(index=1))] if selector else [])


@pytest.mark.parametrize("dtype", ["Dict[str, ProductMember[FITSImage]]", "Union[ProductMember[FITSImage], int]"])
def test_yaml_unsupported_fits_product_shapes_refuse(dtype):
    from shinobi.loaders.yaml_cab import loads

    with pytest.raises(ValueError):
        loads(f"""cabs:
  unsupported:
    command: cat
    outputs:
      out: {{dtype: "{dtype}"}}
    fits_accesses:
      - {{field: out, mode: read, hdu: {{index: 1}}}}
""")


@pytest.mark.parametrize("annotation,value", [(dict[str, ProductMember[FITSImage]], {}), (ProductMember[FITSImage] | int, 42)])
def test_output_validation_rejects_unsupported_product_branches(annotation, value):
    from pydantic import create_model
    from types import SimpleNamespace

    model = create_model("UnsupportedProducts", out=(annotation, ...))
    scope = SimpleNamespace(outputs_model=model, fits_accesses=[])
    with pytest.raises(DatasetLifecycleViolationError, match="product shape"):
        validate_fits_outputs(scope, model(out=value))


def test_nested_model_product_wrapper_refuses():
    from pydantic import ValidationError

    class Wrapper(BaseModel):
        member: ProductMember[FITSImage]

    class Outputs(BaseModel):
        out: Wrapper

    with pytest.raises(ValidationError):
        Scope(name="wrapped", inputs_model=Empty, outputs_model=Outputs)
