from __future__ import annotations

import abc
from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, validator

from rambla.utils.types import ArrayLike

ALLOWED_MODULES = ["numpy_inner_product"]


class SimilarityModuleConfig(BaseModel):
    name: str
    params: Optional[dict]

    @validator("name")
    @classmethod
    def validate_name(cls, name):
        assert (
            name in ALLOWED_MODULES
        ), f"{name=} not recognised. Try one of {ALLOWED_MODULES=}."
        return name


class BaseSimilarityModule(abc.ABC):
    @abc.abstractmethod
    def run(self, *, arr0: ArrayLike, arr1: ArrayLike) -> ArrayLike:
        ...

    @staticmethod
    def _validate_arrays(arr0: np.ndarray, arr1: np.ndarray):
        if arr0.ndim != 2:
            raise ValueError(f"Found array `arr0` with {arr0.ndim=} instead of 2.")
        if arr1.ndim != 2:
            raise ValueError(f"Found array `arr1` with {arr1.ndim=} instead of 2.")
        if arr0.shape != arr1.shape:
            raise ValueError(
                "Mismatch between shapes of arrays `arr0` "
                f"and `arr1`: {arr0.shape=} and {arr1.shape=}."
            )


class NumpyInnerProductModule(BaseSimilarityModule):
    def run(self, *, arr0: ArrayLike, arr1: ArrayLike) -> np.ndarray:
        arr0 = np.array(arr0)
        arr1 = np.array(arr1)

        BaseSimilarityModule._validate_arrays(arr0, arr1)

        inner_product = np.einsum("ij, ij->i", arr0, arr1)
        return inner_product


def build_similarity_module(
    config: Union[dict, SimilarityModuleConfig]
) -> BaseSimilarityModule:
    """Factory method for similarity modules."""
    if not isinstance(config, SimilarityModuleConfig):
        config = SimilarityModuleConfig.parse_obj(config)

    if config.name == "numpy_inner_product":
        return NumpyInnerProductModule()
    else:
        raise ValueError(
            f"{config.name=} not recognised. Try one of {ALLOWED_MODULES=}."
        )
