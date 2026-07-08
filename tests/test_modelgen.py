"""Unit tests for `shinobi.loaders._modelgen.dtype_to_type`'s dtype-string
parsing, in particular `Tuple[...]`/`Union[...]` support (previously both
silently fell back to `str` -- see cult-cargo's `wsclean-base.yml`, which
declares ~90 dtype fields including `Tuple[int, int]`, `Union[str,
List[str]]`, `List[Union[File, str]]`, `Union[str, Tuple[str, float]]`).
"""

from pathlib import Path

from shinobi.loaders._modelgen import dtype_to_type


def test_scalar_and_file_dtypes_unaffected():
    assert dtype_to_type("str") is str
    assert dtype_to_type("int") is int
    assert dtype_to_type("bool") is bool
    assert dtype_to_type("File") is Path
    assert dtype_to_type("totally-unknown") is str


def test_list_dtypes_unaffected():
    assert dtype_to_type("list:int") == list[int]
    assert dtype_to_type("List[str]") == list[str]


def test_tuple_dtype():
    assert dtype_to_type("Tuple[int, int]") == tuple[int, int]
    assert dtype_to_type("tuple[str, float]") == tuple[str, float]


def test_union_dtype():
    assert dtype_to_type("Union[str, int]") == str | int
    assert dtype_to_type("union[str, float]") == str | float


def test_union_of_list():
    assert dtype_to_type("Union[str, List[str]]") == str | list[str]


def test_list_of_union():
    assert dtype_to_type("List[Union[File, str]]") == list[Path | str]


def test_union_of_tuple():
    assert dtype_to_type("Union[str, Tuple[str, float]]") == str | tuple[str, float]


def test_union_containing_tuple_of_ints():
    assert dtype_to_type("Union[int, Tuple[int, int]]") == int | tuple[int, int]
