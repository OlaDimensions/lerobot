from typing import Any

import numpy as np
import pytest

from lerobot.websocket_inference.client import msgpack_numpy


def _check(expected, actual):
    if isinstance(expected, np.ndarray):
        assert expected.shape == actual.shape
        assert expected.dtype == actual.dtype
        assert np.array_equal(expected, actual, equal_nan=expected.dtype.kind == "f")
    else:
        assert expected == actual


def _map_structure_zip(fn, a: Any, b: Any) -> None:
    """Apply *fn* to corresponding leaves of two parallel structures."""
    if isinstance(a, dict):
        for k in a:
            _map_structure_zip(fn, a[k], b[k])
    elif isinstance(a, (list, tuple)):
        for x, y in zip(a, b):
            _map_structure_zip(fn, x, y)
    else:
        fn(a, b)


@pytest.mark.parametrize(
    "data",
    [
        1,  # int
        1.0,  # float
        "hello",  # string
        np.bool_(True),  # boolean scalar
        np.array([1, 2, 3])[0],  # int scalar
        np.str_("asdf"),  # string scalar
        [1, 2, 3],  # list
        {"key": "value"},  # dict
        {"key": [1, 2, 3]},  # nested dict
        np.array(1.0),  # 0D array
        np.array([1, 2, 3], dtype=np.int32),  # 1D integer array
        np.array(["asdf", "qwer"]),  # string array
        np.array([True, False]),  # boolean array
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),  # 2D float array
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.int16),  # 3D integer array
        np.array([np.nan, np.inf, -np.inf]),  # special float values
        {"arr": np.array([1, 2, 3]), "nested": {"arr": np.array([4, 5, 6])}},  # nested dict with arrays
        [np.array([1, 2]), np.array([3, 4])],  # list of arrays
        np.zeros((3, 4, 5), dtype=np.float32),  # 3D zeros
        np.ones((2, 3), dtype=np.float64),  # 2D ones with double precision
    ],
)
def test_pack_unpack(data):
    packed = msgpack_numpy.packb(data)
    unpacked = msgpack_numpy.unpackb(packed)
    _map_structure_zip(_check, data, unpacked)
