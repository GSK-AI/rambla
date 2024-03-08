from __future__ import annotations

from typing import Dict, List, Optional, Union

from datasets import Dataset
from pydantic import BaseModel, validator

from rambla.prompts.base import BasePromptFormatter
from rambla.prompts.utils import find_field_placeholders


def format_template(template: str, var_map: Dict[str, str], entry: dict) -> str:
    """Formats an unformatted f-string template with data from an entry."""
    variables = {
        template_field: entry[dataset_field]
        for dataset_field, template_field in var_map.items()
    }
    return template.format(**variables)


def validate_dataset_and_template(
    placeholders: List[str],
    var_map: Dict[str, str],
    dataset: Dataset,
    prompt_field_name: str,
):
    """Validates template placeholders are in var_map and in dataset columns."""
    template_fields = list(var_map.values())
    dataset_fields = list(var_map.keys())

    if not all(field in dataset.features.keys() for field in dataset_fields):
        raise ValueError(
            "One or more entries from " f"{var_map.keys()=} not a column in `dataset`"
        )

    if not set(placeholders) == set(template_fields):
        raise ValueError(
            "Mismatch between found placeholders and `var_map.values()`. "
            f"Found placeholders in template: {placeholders}. "
            f"Found `var_map.values()`: {var_map.values()}."
        )

    if prompt_field_name in dataset.features.keys():
        raise ValueError(f"Field called {prompt_field_name} already exists in dataset")


class ColumnPromptFormatterConfig(BaseModel):
    var_map: Dict[str, str]
    index_field: str
    allow_duplicates: bool = False
    template: str

    @validator("template")
    @classmethod
    def validate_template(cls, template, values):
        allow_duplicates = values["allow_duplicates"]
        placeholders = find_field_placeholders(template)

        if not allow_duplicates and len(placeholders) != len(set(placeholders)):
            raise ValueError(f"Found duplicates in {placeholders=}")

        if not placeholders:
            raise ValueError(f"Found no placeholders in {template=}")

        return template


class ColumnPromptFormatter(BasePromptFormatter):
    def __init__(
        self,
        template: str,
        var_map: Dict[str, str],
        index_field: str,
        allow_duplicates: bool = False,
    ) -> None:
        """Formats template for a huggingface dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to base the new dataset on
        template : str
            The template to be used for creating the prompt (an unformatted f-string)
        index_field : str
            The index field that needs to be carried over.
        var_map : Dict[str, str]
            Maps field names in the `dataset` to placeholder fields in the `template.
            NOTE: The keys are the dataset columns and the values are the template
            fields.
            NOTE: an alternative is to have `var_map` be a `List[str]`, but this way
            we gain flexibility at the cost of some extra work.
        allow_duplicates: bool
            Whether to allow duplicates in the template

        Returns
        -------
        Dataset
            A huggingface dataset with prompt entry.
            NOTE: this dataset will have as entries:
            ('prompt', `index_field`, `carry_over_fields`)

        Raises
        ------
        ValueError
            If `var_map.values()` do not _exactly_ match the placeholder
            fields in `template.

        """
        super().__init__()
        self.template = template
        self.var_map = var_map
        self.index_field = index_field
        self.allow_duplicates = allow_duplicates

    @property
    def placeholders(self):
        return find_field_placeholders(self.template)

    @classmethod
    def from_config(
        cls, config: Union[dict, ColumnPromptFormatterConfig]
    ) -> ColumnPromptFormatter:
        if isinstance(config, dict):
            config = ColumnPromptFormatterConfig.parse_obj(config)

        return cls(
            template=config.template,
            var_map=config.var_map,
            index_field=config.index_field,
            allow_duplicates=config.allow_duplicates,
        )

    def format_instance(self, entry: dict) -> str:
        """Formats a single data instance."""
        return format_template(self.template, self.var_map, entry)

    def format(self, dataset: Dataset, prompt_field_name: str = "prompt") -> Dataset:
        """Formats a dataset into a dataset of prompts"""
        validate_dataset_and_template(
            self.placeholders, self.var_map, dataset, prompt_field_name
        )

        prompts = []
        for entry in dataset:
            prompts.append(self.format_instance(entry))

        dataset = dataset.add_column(prompt_field_name, prompts)

        return dataset


class ExamplesPromptFormatterConfig(BaseModel):
    var_map: Dict[str, str]
    index_field: str
    target_field: str
    examples_column_name: Optional[str] = "examples"
    allow_duplicates: bool = False
    intro_template: Optional[str] = ""
    examples_template: str
    final_question_template: str

    @validator("intro_template")
    @classmethod
    def validate_intro_template(cls, intro_template, values):
        allow_duplicates = values["allow_duplicates"]
        placeholders = find_field_placeholders(intro_template)

        if not allow_duplicates and len(placeholders) != len(set(placeholders)):
            raise ValueError(f"Found duplicates in {placeholders=}")

        if placeholders:
            raise ValueError(
                f"Found placeholders in {intro_template=} where there should be none."
            )

        return intro_template

    @validator("examples_template")
    @classmethod
    def validate_examples_template(cls, examples_template, values):
        allow_duplicates = values["allow_duplicates"]
        placeholders = find_field_placeholders(examples_template)

        if not allow_duplicates and len(placeholders) != len(set(placeholders)):
            raise ValueError(f"Found duplicates in {placeholders=}")

        if not placeholders:
            raise ValueError(f"Found no placeholders in {examples_template=}.")

        return examples_template

    @validator("final_question_template")
    @classmethod
    def validate_final_question_template(cls, final_question_template, values):
        allow_duplicates = values["allow_duplicates"]
        placeholders = find_field_placeholders(final_question_template)

        if not allow_duplicates and len(placeholders) != len(set(placeholders)):
            raise ValueError(f"Found duplicates in {placeholders=}")

        if not placeholders:
            raise ValueError(f"Found no placeholders in {final_question_template=}.")

        return final_question_template


class ExamplesPromptFormatter(BasePromptFormatter):
    def __init__(
        self,
        examples_template: str,
        final_question_template: str,
        var_map: Dict[str, str],
        index_field: str,
        target_field: str,
        intro_template: Optional[str] = "",
        allow_duplicates: bool = False,
        examples_column_name: Optional[str] = "examples",
    ) -> None:
        """Formats few-shot prompt template for a huggingface dataset.

        The final prompt is the union of: intro_template, examples_template
        repeated x times where x is the number of examples given by the user
        in the config, and final_question_template.

        Parameters
        ----------
        dataset : Dataset
            The dataset to base the new dataset on
        intro_template : str
            The template to be used for creating the intro to the prompt
            (an unformatted f-string)
        examples_template : str
            The template to be used for creating the examples in the prompt
            (an unformatted f-string)
        final_question_template : str
            The template to be used for creating the question to pose to the llm
            in the prompt (an unformatted f-string)
        index_field : str
            The index field that needs to be carried over.
        target_field : str
            The target field with the correct responses to the queries.
        var_map : Dict[str, str]
            Maps field names in the `dataset` to placeholder fields in the `template.
            NOTE: The keys are the dataset columns and the values are the template
            fields.
            NOTE: an alternative is to have `var_map` be a `List[str]`, but this way
            we gain flexibility at the cost of some extra work.
            NOTE: the var_map contains all fields used in the examples, with the
            assumption that the final_question is posed in the same way but without
            the "answer" field
        allow_duplicates: bool
            Whether to allow duplicates in the examples_template and the
            final_question_template. This condition applies to both templates but
            each one independently.

        Returns
        -------
        Dataset
            A huggingface dataset with prompt entry.
            NOTE: this dataset will have as entries:
            ('prompt', `index_field`, `carry_over_fields`)

        Raises
        ------
        ValueError
            If `var_map.values()` do not _exactly_ match the placeholder fields
            in `example_template` or if  `[var_map.values() not "answer"]` do not
            _exactly_ match the placeholder fields in `final_question_template.

        """
        super().__init__()
        self.intro_template = intro_template
        self.examples_template = examples_template
        self.final_question_template = final_question_template
        self.var_map = var_map
        self.index_field = index_field
        self.target_field = target_field
        self.allow_duplicates = allow_duplicates
        self.examples_column_name = examples_column_name

    @classmethod
    def from_config(
        cls, config: Union[dict, ExamplesPromptFormatterConfig]
    ) -> ExamplesPromptFormatter:
        if isinstance(config, dict):
            config = ExamplesPromptFormatterConfig.parse_obj(config)

        return cls(
            intro_template=config.intro_template,
            examples_template=config.examples_template,
            final_question_template=config.final_question_template,
            var_map=config.var_map,
            index_field=config.index_field,
            target_field=config.target_field,
            allow_duplicates=config.allow_duplicates,
            examples_column_name=config.examples_column_name,
        )

    @property
    def placeholders_examples(self) -> List[str]:
        return find_field_placeholders(self.examples_template)

    @property
    def placeholders_final_question(self) -> List[str]:
        return find_field_placeholders(self.final_question_template)

    @property
    def var_map_without_answer(self) -> dict:
        return {
            x: self.var_map[x] for x in self.var_map.keys() if x != self.target_field
        }

    def _extract_example_data(self, entry: dict, dataset: Dataset) -> dict:
        """Extracts a dataset of examples given the ordering of indices."""
        examples = None
        for k in entry[self.examples_column_name]:
            if not examples:
                examples = dataset.filter(lambda x: x[self.index_field] == k)
            else:
                examples = examples.add_item(
                    dataset.filter(lambda x: x[self.index_field] == k)[0]
                )

        entry[self.examples_column_name] = examples
        return entry

    def format_instance(self, entry: dict) -> str:
        """Formats a single data instance.

        This function creates a prompt for a single query. It is made up of the
        introductory template. Then, it appends a formatted example template in
        a loop for each example. Then, it appends the final question to be given
        to the LLM.
        """
        output = self.intro_template

        for example in entry[self.examples_column_name]:
            output += format_template(self.examples_template, self.var_map, example)

        output += format_template(
            self.final_question_template, self.var_map_without_answer, entry
        )

        return output

    def format(self, dataset: Dataset, prompt_field_name: str = "prompt") -> Dataset:
        """Formats a dataset into a dataset of prompts"""
        # validation of examples and final question templates
        validate_dataset_and_template(
            self.placeholders_examples, self.var_map, dataset, prompt_field_name
        )
        validate_dataset_and_template(
            self.placeholders_final_question,
            self.var_map_without_answer,
            dataset,
            prompt_field_name,
        )

        # prompt generation
        prompts = []
        for entry in dataset:
            entry = self._extract_example_data(entry, dataset)
            prompts.append(self.format_instance(entry))

        dataset = dataset.add_column(prompt_field_name, prompts)
        return dataset
