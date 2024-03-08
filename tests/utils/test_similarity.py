import numpy as np
import pytest

from rambla.utils.similarity import (
    BaseSimilarityModule,
    NumpyInnerProductModule,
    build_similarity_module,
)
from rambla.utils.types import ArrayLike

# flake8: noqa: N802


def test_BaseSimilarityModule_validate_arrays_arr0():
    arr0 = np.array([1, 2, 3])
    arr1 = np.array([1, 2, 3])

    with pytest.raises(ValueError) as exc_info:
        BaseSimilarityModule._validate_arrays(arr0=arr0, arr1=arr1)

    assert f"{arr0.ndim}" in str(exc_info.value)


def test_BaseSimilarityModule_validate_arrays_arr1():
    arr0 = np.array([[1, 2, 3]])
    arr1 = np.array([1, 2, 3])

    with pytest.raises(ValueError) as exc_info:
        BaseSimilarityModule._validate_arrays(arr0=arr0, arr1=arr1)

    assert f"{arr1.ndim}" in str(exc_info.value)


def test_BaseSimilarityModule_validate_arrays_shape_mismatch():
    arr0 = np.array([[1, 2, 3], [1, 2, 3]])
    arr1 = np.array([[1, 2, 3]])

    with pytest.raises(ValueError) as exc_info:
        BaseSimilarityModule._validate_arrays(arr0=arr0, arr1=arr1)

    assert f"{arr1.shape}" in str(exc_info.value)
    assert f"{arr0.shape}" in str(exc_info.value)


@pytest.mark.parametrize(
    "arr0, arr1",
    [
        (
            np.array([[1, 2, 3]]),
            np.array([[1, 2, 3]]),
        ),
        (
            np.array([[1, 2, 3]]).reshape(-1, 1),
            np.array([[1, 2, 3]]).reshape(-1, 1),
        ),
        (
            np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            ),
            np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            ),
        ),
    ],
)
def test_BaseSimilarityModule_validate_arrays_correct(
    arr0: np.ndarray, arr1: np.ndarray
):
    BaseSimilarityModule._validate_arrays(arr0=arr0, arr1=arr1)


def test_build_similarity_module():
    config = {"name": "numpy_inner_product"}
    #
    output = build_similarity_module(config)

    #
    assert isinstance(output, NumpyInnerProductModule)


def test_build_similarity_module_error():
    name = "__dummy__"
    config = {"name": name}
    #
    with pytest.raises(ValueError) as exc_info:
        build_similarity_module(config)

    #
    assert name in str(exc_info.value)


@pytest.mark.parametrize(
    "arr0, arr1, expected_output",
    [
        (
            np.array([[1, 2, 3]]),
            np.array([[1, 2, 3]]),
            np.array([14]),
        ),
        (
            np.array([[1, 2, 3]]).reshape(-1, 1),
            np.array([[1, 2, 3]]).reshape(-1, 1),
            np.array([1, 4, 9]),
        ),
        (
            np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            ),
            np.array(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ]
            ),
            np.array([14, 77]),
        ),
    ],
)
def test_NumpyInnerProductModule_run(
    arr0: ArrayLike, arr1: ArrayLike, expected_output: np.ndarray
):
    module = NumpyInnerProductModule()

    output = module.run(arr0=arr0, arr1=arr1)

    #
    assert np.allclose(output, expected_output)
