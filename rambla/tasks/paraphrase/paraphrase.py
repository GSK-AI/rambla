from __future__ import annotations

from typing import Union

from datasets import Dataset
from pydantic import Extra, root_validator

from rambla.datasets.io import MCQADatasetConfig, prepare_dataset
from rambla.evaluation.shortform import MCQAEvalComponent, MCQAEvalComponentConfig
from rambla.prompts.formatters import ColumnPromptFormatter, ColumnPromptFormatterConfig
from rambla.response_formatting.formatting import (
    MCQAResponseFormatter,
    MCQAResponseFormatterConfig,
)
from rambla.response_generation.response import (
    ResponseComponent,
    ResponseComponentConfig,
)
from rambla.tasks.base import BaseTask, BaseTaskConfig, LLMGenerator, RunTaskReturnType
from rambla.tasks.paraphrase.utils import RephrasingModule, RephrasingModuleConfig
from rambla.utils.config import validate_field_attribute_equality
from rambla.utils.misc import prepare_dicts_for_logging

# flake8: noqa: E501


class ParaphraseTaskConfig(BaseTaskConfig):
    dataset_config: MCQADatasetConfig
    rephrasing_module_config: RephrasingModuleConfig
    prompt_formatter_config: ColumnPromptFormatterConfig
    response_formatter_config: MCQAResponseFormatterConfig
    evaluator_config: MCQAEvalComponentConfig
    response_component_config: ResponseComponentConfig

    class Config:  # noqa: D106
        extra = Extra.forbid

    @root_validator(skip_on_failure=True)
    @classmethod
    def validate_target_field(cls, values):
        validate_field_attribute_equality(
            values["dataset_config"], values["evaluator_config"], "target_field"
        )
        return values

    @root_validator(skip_on_failure=True)
    @classmethod
    def validate_response_field(cls, values):
        response_formatter_config = values["response_formatter_config"]
        evaluator_config = values["evaluator_config"]
        validate_field_attribute_equality(
            response_formatter_config,
            evaluator_config,
            "response_field_name",
            "response_field",
        )
        return values

    @root_validator(skip_on_failure=True)
    @classmethod
    def validate_index_field(cls, values):
        validate_field_attribute_equality(
            values["dataset_config"], values["prompt_formatter_config"], "index_field"
        )
        validate_field_attribute_equality(
            values["dataset_config"],
            values["rephrasing_module_config"].prompt_formatter_config,
            "index_field",
        )
        return values

    @root_validator(skip_on_failure=True)
    @classmethod
    def validate_categories(cls, values):
        response_formatter_config = values["response_formatter_config"]
        evaluator_config = values["evaluator_config"]
        dataset_config = values["dataset_config"]

        dataset_categories = dataset_config.categories_to_keep
        response_formatter_categories = response_formatter_config.categories
        evaluator_categories = evaluator_config.categories

        if set(dataset_categories) != set(response_formatter_categories):
            raise ValueError(
                f"{dataset_config.categories_to_keep=} not the "
                f"same as {response_formatter_config.categories=}"
            )
        if set(dataset_categories + [response_formatter_config.null_category]) != set(
            evaluator_categories
        ):
            raise ValueError(
                f"{(dataset_categories + [response_formatter_config.null_category])=} not the "  # noqa: E501
                f"same as {evaluator_config.categories=}"  # noqa: E501
            )
        return values


class ParaphraseTask(BaseTask):
    """Rephrasing task

    Example uses:
    1. Rephrase the question field of a dataset and evaluate
    2. Negate the meaning of context of a dataset and evaluate
    NOTE: atm the code will not work for the negation task,
    but could do with minor tweaks.

    Steps:
    1. Rephrase column in dataset
    2. Create prompts for dataset (under `prompt`)
    3. Generate responses from `llm` (under `response`)
        - `llm` is an `LLMGenerator` provided in `.run_task`
    4. Post-process responses
    5. Run evaluation
    """

    def __init__(
        self,
        dataset: Dataset,
        rephrasing_module: RephrasingModule,
        prompt_formatter: ColumnPromptFormatter,
        response_formatter: MCQAResponseFormatter,
        evaluator: MCQAEvalComponent,
        response_component: ResponseComponent,
    ):
        self.dataset = dataset
        self.rephrasing_module = rephrasing_module
        self.prompt_formatter = prompt_formatter
        self.response_formatter = response_formatter
        self.evaluator = evaluator
        self.response_component = response_component

    @classmethod
    def from_config(cls, config: Union[dict, ParaphraseTaskConfig]) -> ParaphraseTask:
        config = ParaphraseTaskConfig.parse_obj(config)

        dataset = prepare_dataset(config.dataset_config.dict())
        rephrasing_module = RephrasingModule.from_config(
            config.rephrasing_module_config
        )
        prompt_formatter = ColumnPromptFormatter.from_config(
            config.prompt_formatter_config
        )
        response_formatter = MCQAResponseFormatter.from_config(
            config.response_formatter_config
        )
        evaluator = MCQAEvalComponent.from_config(config.evaluator_config)

        response_component = ResponseComponent.from_config(
            config.response_component_config
        )
        return cls(
            dataset=dataset,
            rephrasing_module=rephrasing_module,
            prompt_formatter=prompt_formatter,
            response_formatter=response_formatter,
            evaluator=evaluator,
            response_component=response_component,
        )

    def run_task(self, llm: LLMGenerator, verbose: bool = True) -> RunTaskReturnType:
        """Executes the steps defined by the task in sequence.

        Parameters
        ----------
        llm : LLMGenerator
            The model to be evaluated

        Returns
        -------
        RunTaskReturnType
        """
        # Generate rephrased dataset
        rephrased_dataset = self.rephrasing_module.run(self.dataset)

        # Form prompt dataset
        prompt_dataset = self.prompt_formatter.format(rephrased_dataset)

        # Generate responses
        response_dataset = self.response_component.batch_generate(
            model=llm,
            prompt_dataset=prompt_dataset,
            verbose=verbose,
        )

        # Post-process responses
        processed_response_dataset = self.response_formatter.format(response_dataset)

        # Evaluate
        eval_results = self.evaluator.evaluate(processed_response_dataset)

        # Preparing for logging
        to_log_as_metrics, to_log_as_dicts = prepare_dicts_for_logging(
            eval_results=eval_results,
        )

        return RunTaskReturnType(
            metrics=to_log_as_metrics,
            artifacts=eval_results,
            datasets={"final_dataset": processed_response_dataset},
            other=None,
            artifact_storing_format="json",
            plots=None,
            dictionaries=to_log_as_dicts,
        )
