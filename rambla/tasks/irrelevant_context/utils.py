from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Union

import numpy as np
from datasets import Dataset
from pydantic import BaseModel, validator

from rambla.datasets.io import GenericDatasetConfig, prepare_dataset
from rambla.utils.task import BaseComponent


def shuffle_column(
    column: List[Any],
    seed: int,
) -> List[Any]:
    """Shuffles a column.

    Parameters
    ----------
    column : List[Any]
        Data to be shuffled.

    Returns
    -------
    List[Any]
        List with same data, but shuffled.
    """
    new_data = deepcopy(column)
    np.random.seed(seed)
    np.random.shuffle(new_data)
    return new_data


class ShufflingModuleConfig(BaseModel):
    field_name: str
    seed: int


class ShufflingModule(BaseComponent):
    def __init__(self, field_name: str, seed: int):
        """Shuffles a column of HF dataset."""
        self.field_name = field_name
        self.seed = seed

    @classmethod
    def from_config(cls, config: Union[dict, ShufflingModuleConfig]) -> ShufflingModule:
        if not isinstance(config, ShufflingModuleConfig):
            config = ShufflingModuleConfig.parse_obj(config)

        return cls(field_name=config.field_name, seed=config.seed)

    def run(self, dataset: Dataset) -> Dataset:
        shuffled_column = shuffle_column(
            column=dataset[self.field_name], seed=self.seed
        )
        dataset = dataset.rename_column(
            self.field_name, f"unshuffled_{self.field_name}"
        )
        dataset = dataset.add_column(self.field_name, shuffled_column)
        return dataset


class DatasetMixerConfig(BaseModel):
    # Config for that dataset that will provide the _new_ context
    source_dataset_config: GenericDatasetConfig
    source_field_name: str
    dest_field_name: str
    seed: int
    with_replacement: bool = True


class DatasetMixerModule(BaseComponent):
    def __init__(
        self,
        source_dataset: Dataset,
        source_field_name: str,
        dest_field_name: str,
        seed: int,
        with_replacement: bool = True,
    ):
        self.source_dataset = source_dataset
        self.source_field_name = source_field_name
        self.dest_field_name = dest_field_name
        self.seed = seed
        self.with_replacement = with_replacement

        if self.source_field_name not in source_dataset.features.keys():
            raise ValueError(
                f"{source_field_name=} not found in {source_dataset.features.keys()=}"
            )

    @classmethod
    def from_config(cls, config: Union[dict, DatasetMixerConfig]) -> DatasetMixerModule:
        if not isinstance(config, DatasetMixerConfig):
            config = DatasetMixerConfig.parse_obj(config)

        source_dataset = prepare_dataset(config.source_dataset_config.dict())

        return cls(
            source_dataset=source_dataset,
            source_field_name=config.source_field_name,
            dest_field_name=config.dest_field_name,
            seed=config.seed,
            with_replacement=config.with_replacement,
        )

    def run(self, dataset: Dataset) -> Dataset:
        n_rows_source = len(self.source_dataset)
        n_rows_dest = len(dataset)

        if self.dest_field_name in dataset.features.keys():
            dataset = dataset.rename_column(
                self.dest_field_name, f"original_{self.dest_field_name}"
            )

        column = self.source_dataset[self.source_field_name]

        np.random.seed(self.seed)

        if n_rows_source < n_rows_dest and not self.with_replacement:
            raise ValueError(
                f"Number of rows in source dataset ({n_rows_source}) is less than "
                f"number of rows in destination dataset ({n_rows_dest}). "
                "This is problematic when `with_replacement` is set to True."
            )

        draw = np.random.choice(column, n_rows_dest, replace=self.with_replacement)

        dataset = dataset.add_column(self.dest_field_name, draw)

        return dataset


class ContextOrderConfig(BaseModel):
    """Config for position of gold-standard context.

    For example,
    ```python
    context_order = ContextOrderConfig(total=3, position=0)
    ```
    This would put the gold-standard context first,
    followed by two irrelevant ones.

    Parameters
    ----------
    total: int
        Number of instances

    position: int
        Position of gold-standard

    Raises
    ------
    ValueError
        if `total` not positive
    ValueError
        if `position` not positive or is greater than `total`
    """

    total: int
    position: int

    @validator("total")
    @classmethod
    def validate_total(cls, total, values, **kwargs) -> int:
        if not 2 <= total:
            raise ValueError(f"`total` needs to be at least 2. Instead found: {total=}")
        return total

    @validator("position")
    @classmethod
    def validate_position(cls, position, values, **kwargs) -> int:
        if not 0 <= position < values["total"]:
            raise ValueError(
                "`position` needs to be greater than 0 and less than "
                f"`total` (={values['total']}). Instead found: {position=}"
            )
        return position


# TODO: Allow for position to be randomised
# i.e., different for each entry.
def create_shuffled_copies_of_column(
    data: List[str], n_contexts: int, position_of_original_context: int, seed: int
) -> List[List[str]]:
    """Example usage:

    n_contexts=3
    position_of_original_context=1
    --> In positions 0 and 2, we will have shuffled versions of the `data`
    --> In position 1 we will have the _original_ data.

    # NOTE: `position_of_original_context` starts from 0.

    Parameters
    ----------
    data : List[str]
        Original data to be copied and shuffled
    n_contexts : int
        How many contexts will be included in total.
    position_of_original_context : int
        The position of the original context. All other positions
        will be filled up with irrelevant context.
    seed : int
        Seed for shuffling

    Returns
    -------
    List[List[str]]
        Has the same length as `context_order.total`
        Each element of this is a `list` with length as `len(data)`
        The list at position `context_order.position` is _exactly_
        the same as `data`. All others are shuffled versions of it.
    """
    np.random.seed(seed)
    out = []

    shifting_draws = np.random.choice(
        a=range(1, len(data)),
        size=n_contexts,
        replace=False,
    )

    for ii, shifting_value in enumerate(shifting_draws):
        if ii == position_of_original_context:
            out.append(data)
        else:
            shifted_data = np.roll(deepcopy(data), shifting_value)
            out.append(list(shifted_data))
    return out


def merge_lists(data: List[List[str]], separator: str) -> List[str]:
    """Merges lists by joining them according to `separator`."""
    return list(map(lambda x: separator.join(x), zip(*data)))


class ContextAugmentingModuleConfig(BaseModel):
    n_contexts: int
    position_of_original_context: int
    field_name: str
    seed: int
    separator: str

    @validator("n_contexts")
    @classmethod
    def validate_total(cls, n_contexts, values, **kwargs) -> int:
        if not 2 <= n_contexts:
            raise ValueError(
                f"`n_contexts` needs to be at least 2. Instead found: {n_contexts=}"
            )
        return n_contexts

    @validator("position_of_original_context")
    @classmethod
    def validate_position(cls, position_of_original_context, values, **kwargs) -> int:
        if not 0 <= position_of_original_context < values["n_contexts"]:
            raise ValueError(
                "`position_of_original_context` needs to be greater "
                "than 0 and less than `total` "
                f"(={values['position_of_original_context']}). "
                f"Instead found: {position_of_original_context=}"
            )
        return position_of_original_context


class ContextAugmentingModule:
    def __init__(
        self,
        n_contexts: int,
        position_of_original_context: int,
        field_name: str,
        seed: int,
        separator: str,
    ):
        self.n_contexts = n_contexts
        self.position_of_original_context = position_of_original_context
        self.field_name = field_name
        self.seed = seed
        self.separator = separator

    @classmethod
    def from_config(cls, config: dict) -> "ContextAugmentingModule":
        config = ContextAugmentingModuleConfig.parse_obj(config)
        return cls(**config.dict())

    def run(self, dataset: Dataset) -> Dataset:
        shuffled_data = create_shuffled_copies_of_column(
            data=dataset[self.field_name],
            n_contexts=self.n_contexts,
            position_of_original_context=self.position_of_original_context,
            seed=self.seed,
        )

        merged_data = merge_lists(data=shuffled_data, separator=self.separator)

        new_dataset = dataset.rename_column(
            self.field_name, f"original_{self.field_name}"
        )
        new_dataset = new_dataset.add_column(self.field_name, merged_data)
        return new_dataset
