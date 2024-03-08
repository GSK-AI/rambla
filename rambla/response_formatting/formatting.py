from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from datasets import Dataset
from pydantic import BaseModel, validator

from rambla.response_formatting.base import (
    STRING_FORMATTER_MAP,
    BaseResponseFormatter,
    process_string,
)
from rambla.utils.text_processing import extract_first_response_instance


class MCQAResponseFormatterConfig(BaseModel):
    response_field_name: str
    categories: List[str]
    string_formatter_name: str
    response_extractor_name: Optional[str]
    null_category: str = "null"


class MCQAResponseFormatter(BaseResponseFormatter):
    def __init__(
        self,
        response_field_name: str,
        categories: List[str],
        string_formatter: Callable[[str], str] = process_string,
        response_extractor: Callable[
            [str, List[str]], Union[str, None]
        ] = extract_first_response_instance,
        null_category: str = "null",
    ):
        """Formats and processes response column of a dataset.

        Parameters
        ----------
        response_field_name : str
            Name of response column.
        categories : List[str]
            Expected categories
        string_formatter : Callable[[str], str]
            Function to be used on a per-response for formatting.
        response_extractor : Callable[[str], str]
            Function to be used on a per-response for extracting responses.
        null_category : str, optional
            Predictions not present in `categories` will
            be mapped to this category, by default "null"
        """
        self.response_field_name = response_field_name
        self.categories = categories
        self.string_formatter = string_formatter
        self.response_extractor = response_extractor
        self.null_category = null_category

    @classmethod
    def from_config(cls, config: Union[dict, MCQAResponseFormatterConfig]):
        # NOTE: Maybe add a factory method for response extractor if multiple
        # methods start being in use
        if isinstance(config, dict):
            config = MCQAResponseFormatterConfig.parse_obj(config)

        return cls(
            response_field_name=config.response_field_name,
            categories=config.categories,
            string_formatter=STRING_FORMATTER_MAP[config.string_formatter_name],
            null_category=config.null_category,
        )

    def _format_column(self, column: List[str]) -> List[str]:
        """Applies the `self.string_formatter` to the whole column."""
        return list(map(self.string_formatter, column))

    def _process_column(self, column: List[str]) -> List[str]:
        """Any entry in `column` not found in `self.categories` will be mapped to `self.null_category`"""  # noqa: E501
        processed_column = []
        for response in column:
            extracted_response = self.response_extractor(response, self.categories)
            if extracted_response:
                processed_column.append(extracted_response)
            else:
                processed_column.append(self.null_category)

        return processed_column

    def format(self, dataset: Dataset) -> Dataset:
        """Formats the response column of a HF dataset."""
        response_column = dataset[self.response_field_name]
        formatted_column = self._format_column(response_column)
        processed_column = self._process_column(formatted_column)

        new_dataset = dataset.rename_column(
            self.response_field_name, f"unformatted_{self.response_field_name}"
        )
        new_dataset = new_dataset.add_column(self.response_field_name, processed_column)
        return new_dataset


class MappingResponseFormatterConfig(BaseModel):
    response_field_name: str
    renaming_map: Dict[str, Any]
    string_formatter_name: str
    response_extractor_name: Optional[str]
    null_category: str = "null"

    @validator("renaming_map")
    @classmethod
    def validate_renaming_map(cls, renaming_map):
        assert all(
            isinstance(item, (int, str, float)) for item in renaming_map.values()
        )
        return renaming_map


class MappingResponseFormatter(BaseResponseFormatter):
    def __init__(
        self,
        response_field_name: str,
        renaming_map: Dict[str, str],
        string_formatter: Callable[[str], str] = process_string,
        response_extractor: Callable[
            [str, List[str]], Union[str, None]
        ] = extract_first_response_instance,
        null_category: str = "null",
    ) -> None:
        """Formats and reverses negation of responses

        Parameters
        ----------
        response_field_name : str
            Field containing response data
        renaming_map : Dict[str, str]
            Dictionary mapping negated responses to non-negated response
        string_formatter : Callable[[str], str], optional
            Function to post-process raw responses, by default process_string
        response_extractor : Callable[ [str, List[str]], Union[str, None] ], optional
            Function to extract responses from responses, by default
            extract_first_response_instance
        null_category : str, optional
           Predictions not present in the `renaming_map` will be mapped to this
           category, by default "null"
        """
        self.response_field_name = response_field_name
        self.renaming_map = renaming_map
        self.string_formatter = string_formatter
        self.response_extractor = response_extractor
        self.null_category = null_category

    @property
    def in_going_categories(self) -> List[str]:
        return list(self.renaming_map.keys())

    @property
    def out_going_categories(self) -> List[str]:
        return list(self.renaming_map.values())

    @classmethod
    def from_config(
        cls, config: Union[dict, MappingResponseFormatterConfig]
    ) -> MappingResponseFormatter:
        if isinstance(config, dict):
            config = MappingResponseFormatterConfig.parse_obj(config)

        return cls(
            response_field_name=config.response_field_name,
            renaming_map=config.renaming_map,
            string_formatter=STRING_FORMATTER_MAP[config.string_formatter_name],
            null_category=config.null_category,
        )

    def _process_column(self, dataset: Dataset) -> List[str]:
        # Extracts standard multi-choice responses
        mcqa_response_formatter = MCQAResponseFormatter(
            response_field_name=self.response_field_name,
            categories=self.in_going_categories,
            string_formatter=self.string_formatter,
            response_extractor=self.response_extractor,
            null_category=self.null_category,
        )
        formatted_dataset = mcqa_response_formatter.format(dataset)

        return formatted_dataset[self.response_field_name]

    def _negate_column(self, column: List[str]) -> List[str]:
        negated_responses = []
        for response in column:
            if response in self.renaming_map.keys():
                negated_responses.append(self.renaming_map[response])
            else:
                negated_responses.append(self.null_category)

        return negated_responses

    def format(self, dataset: Dataset) -> Dataset:
        processed_column = self._process_column(dataset)
        renamed_column = self._negate_column(processed_column)

        new_dataset = dataset.rename_column(
            self.response_field_name, f"unformatted_{self.response_field_name}"
        )
        new_dataset = new_dataset.add_column(
            self.response_field_name, renamed_column  # type: ignore
        )

        return new_dataset
