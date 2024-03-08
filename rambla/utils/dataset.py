from __future__ import annotations

from typing import Dict, List, Optional, Union

import pyarrow as pa
from datasets import Dataset
from datasets.arrow_dataset import Dataset as ArrowDataset
from datasets.arrow_writer import TypedSequence
from datasets.features import Features, Value
from datasets.info import DatasetInfo
from datasets.table import InMemoryTable
from pydantic import BaseModel, Extra, validator


def from_dict_to_dataset(mapping: Dict[str, Union[List[str], List[int]]]) -> Dataset:
    """Creates huggingface dataset from a dict.

    Parameters
    ----------
    mapping : Dict[str, Union[List[str], List[int]]]
        Dictionary of lists to be used to create the dataset

    Returns
    -------
    Dataset
        Huggingface dataset.
    """
    type_map = {
        str: Value("string"),
        int: Value("int32"),
    }
    _types = {key: type_map[type(values[0])] for key, values in mapping.items()}

    new_mapping = {
        k: pa.array(TypedSequence(v, type=_types[k])) for k, v in mapping.items()
    }
    pa_table = InMemoryTable.from_pydict(new_mapping)

    features = Features(
        {col: Value(str(array.type)) for col, array in new_mapping.items()}
    )

    info = DatasetInfo()
    info.features = features

    dataset = ArrowDataset(arrow_table=pa_table, info=info)
    return dataset


def add_fields_to_dataset(
    src_dataset: Dataset, dest_dataset: Dataset, field_names: Union[str, List[str]]
) -> Dataset:
    """Adds fields from a source dataset to a destination dataset

    Parameters
    ----------
    src_dataset : Dataset
        Dataset with fields to add to new dataset
    dest_dataset : Dataset
        Dataset to add new fields to
    field_names : Union[str, List[str]]
        Field name or list of field names to add from source dataset to destination
        dataset.

    Returns
    -------
    Dataset
        Destination dataset with added fields.

    Raises
    ------
    KeyError
        If fields not found in source dataset.
    """
    if isinstance(field_names, str):
        field_names = [field_names]

    for field in field_names:
        if field not in src_dataset.features.keys():
            raise KeyError(
                f"Field {field} not found in list of features: "
                f"{src_dataset.features.keys()}"
            )
        dest_dataset = dest_dataset.add_column(
            field, src_dataset[field]  # type: ignore
        )

    return dest_dataset


def slice_dataset(
    dataset: Dataset,
    start_slice: Optional[int] = None,
    stop_slice: Optional[int] = None,
) -> Dataset:
    """Takes a slice of a dataset

    Slice format equivalent to:
    dataset[`start_slice`:`stop_slice`]

    NOTE: If only `start_slice` provided then slices from
    `start_slice` to the end
    NOTE: If only `stop_slice` provided then slices from
    start to `stop_slice`

    Parameters
    ----------
    dataset : Dataset
        Dataset to slice
    start_slice : Optional[int], optional
        Index to slice from, by default None
    stop_slice : Optional[int], optional
        Index to slice to (not including), by default None

    Returns
    -------
    Dataset
        Sliced dataset

    Raises
    ------
    ValueError
        If slice variables missing
    ValueError
        If `stop_slice` goes beyond length of dataset
    """
    if not start_slice and not stop_slice:
        raise ValueError("Neither `start_slice` or `stop_slice` defined")

    if start_slice and not stop_slice:
        stop_slice = len(dataset)
    if stop_slice and not start_slice:
        start_slice = 0

    if stop_slice > len(dataset):
        raise ValueError(
            f"`stop_slice`: {stop_slice}"
            f"greater than number of rows in dataset: {len(dataset)}"
        )

    return dataset.select(range(start_slice, stop_slice))


class DatasetFilteringCondition(BaseModel):
    field_name: str
    field_values: list[str]
    # NOTE: If set to True then all entries from
    # the datasest where `field_name` matches any of
    # the `field_values` will be filtered out
    # If it is set to False then only entries
    # that match any of the field values will be included.
    filter_out: bool

    @validator("field_values", pre=True)
    @classmethod
    def validate_field_values(cls, field_values):
        if isinstance(field_values, str):
            field_values = [field_values]
        return field_values

    def __call__(self, x: dict) -> bool:
        if not self.filter_out:
            return x[self.field_name] in self.field_values
        else:
            return x[self.field_name] not in self.field_values

    class Config:  # noqa: D106
        extra = Extra.forbid


class DatasetFiltererConfig(BaseModel):
    filtering_conditions: list[DatasetFilteringCondition]

    class Config:  # noqa: D106
        extra = Extra.forbid

    @validator("filtering_conditions", pre=True)
    @classmethod
    def validate_filtering_conditions(cls, filtering_conditions):
        if isinstance(filtering_conditions, (dict, DatasetFilteringCondition)):
            filtering_conditions = [filtering_conditions]
        return filtering_conditions


class DatasetFilterer:
    def __init__(self, filtering_conditions: list[DatasetFilteringCondition]):
        self.filtering_conditions = filtering_conditions

    @classmethod
    def from_config(cls, config: Union[dict, DatasetFiltererConfig]) -> DatasetFilterer:
        if not isinstance(config, DatasetFiltererConfig):
            config = DatasetFiltererConfig.parse_obj(config)
        return cls(filtering_conditions=config.filtering_conditions)

    def run(self, dataset: Dataset) -> Dataset:
        for condition in self.filtering_conditions:
            if dataset:
                dataset = dataset.filter(condition)

        return dataset
