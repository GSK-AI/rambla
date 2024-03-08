from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from datasets import Dataset
from pydantic import BaseModel, root_validator

from rambla.preprocessing.base import BasePreprocessor


class ScalingFormatterConfig(BaseModel):
    label_field_name: str
    max_scale: Optional[float] = 1
    min_scale: Optional[float] = 0

    @root_validator()
    @classmethod
    def validate_scales(cls, values):
        if values["min_scale"] >= values["max_scale"]:
            raise ValueError(
                f"""The max_scale value ({values["min_scale"]})
                cannot be lower than the min_scale value
                ({values["max_scale"]})"""
            )
        return values


class ScalingFormatter(BasePreprocessor):
    def __init__(
        self,
        label_field_name: str,
        max_scale: Optional[float] = 1,
        min_scale: Optional[float] = 0,
    ) -> None:
        """Scales a column to within the desired range

        Parameters
        ----------
        label_field_name : str
            Field containing label data
        max_scale: float, optional
            Upper bound of scale range
        min_scale: float, optional
            Lower bound of scale range

        Example usecase
        ----------
        Scale labels before evaluation with binary responses
        """
        self.label_field_name = label_field_name
        self.max_scale = max_scale
        self.min_scale = min_scale

        if self.min_scale >= self.max_scale:
            raise ValueError(
                f"""The max_scale value ({self.max_scale})
                             cannot be lower than the min_scale value
                             ({self.min_scale})"""
            )

    @classmethod
    def from_config(
        cls, config: Union[dict, ScalingFormatterConfig]
    ) -> ScalingFormatter:
        if isinstance(config, dict):
            config = ScalingFormatterConfig.parse_obj(config)

        return cls(
            label_field_name=config.label_field_name,
            max_scale=config.max_scale,
            min_scale=config.min_scale,
        )

    def _process_column(self, label_column: List[str | int | float]) -> List[float]:
        # Convert to float and np array
        np_label_column = np.array(label_column, dtype=np.float64)
        # Get min and max
        min_label = np.min(np_label_column)
        max_label = np.max(np_label_column)
        # Scale responses
        np_label_column = np.array(label_column, dtype=np.float64)
        processed_column = (np_label_column - min_label) / (max_label - min_label)
        processed_column = processed_column * (self.max_scale - self.min_scale)
        processed_column = processed_column + self.min_scale

        return processed_column

    def format(self, dataset: Dataset) -> Dataset:
        # Get column
        label_column = dataset[self.label_field_name]
        # Process column
        processed_column = self._process_column(label_column)

        new_dataset = dataset.rename_column(
            self.label_field_name, f"unformatted_{self.label_field_name}"
        )

        new_dataset = new_dataset.add_column(
            self.label_field_name, processed_column  # type: ignore
        )

        return new_dataset
