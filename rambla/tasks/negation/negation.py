from __future__ import annotations

from typing import Optional, Union

from datasets import Dataset
from pydantic import BaseModel, root_validator

from rambla.datasets.io import MCQADatasetConfig, prepare_dataset
from rambla.evaluation.shortform import MCQAEvalComponent, MCQAEvalComponentConfig
from rambla.prompts.formatters import ColumnPromptFormatter, ColumnPromptFormatterConfig
from rambla.response_formatting.formatting import (
    MappingResponseFormatter,
    MappingResponseFormatterConfig,
)
from rambla.response_generation.response import (
    ResponseComponent,
    ResponseComponentConfig,
)
from rambla.tasks.base import BaseTask, LLMGenerator, RunTaskReturnType
from rambla.tasks.paraphrase.utils import RephrasingModule, RephrasingModuleConfig
from rambla.utils.config import validate_field_attribute_equality
from rambla.utils.dataset import slice_dataset
from rambla.utils.misc import prepare_dicts_for_logging


class NegationTaskConfig(BaseModel):
    dataset_config: MCQADatasetConfig
    rephrasing_module_config: RephrasingModuleConfig
    prompt_formatter_config: ColumnPromptFormatterConfig
    response_formatter_config: MappingResponseFormatterConfig
    evaluator_config: MCQAEvalComponentConfig
    response_component_config: ResponseComponentConfig
    subsample_size: Optional[int]

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
    def validate_negation_map(cls, values):
        dataset_categories = values["dataset_config"].categories_to_keep
        renaming_map = values["response_formatter_config"].renaming_map
        evaluator_categories = values["evaluator_config"].categories

        if set(dataset_categories) != set(renaming_map.keys()):
            raise ValueError(
                f"Negation map should have the following keys: {dataset_categories} "
                f"but instead has: {list(renaming_map.keys())}"
            )

        negation_categories = list(renaming_map.values()) + [
            values["response_formatter_config"].null_category
        ]
        if set(evaluator_categories) != set(negation_categories):
            raise ValueError(
                "`evaluator_config.categories` should have the following categories: "
                f"{negation_categories}, but has the following categories: "
                f"{evaluator_categories}"
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


class NegationTask(BaseTask):
    def __init__(
        self,
        dataset: Dataset,
        rephrasing_module: RephrasingModule,
        prompt_formatter: ColumnPromptFormatter,
        response_formatter: MappingResponseFormatter,
        evaluator: MCQAEvalComponent,
        response_component: ResponseComponent,
    ) -> None:
        self.dataset = dataset

        self.rephrasing_module = rephrasing_module
        self.prompt_formatter = prompt_formatter
        self.response_formatter = response_formatter
        self.evaluator = evaluator
        self.response_component = response_component

    @classmethod
    def from_config(cls, config: Union[NegationTaskConfig, dict]) -> NegationTask:
        if not isinstance(config, NegationTaskConfig):
            config = NegationTaskConfig.parse_obj(config)

        dataset = prepare_dataset(config.dataset_config.dict())
        if config.subsample_size:
            dataset = slice_dataset(dataset, stop_slice=config.subsample_size)

        rephrasing_module = RephrasingModule.from_config(
            config.rephrasing_module_config
        )
        prompt_formatter = ColumnPromptFormatter.from_config(
            config.prompt_formatter_config
        )
        response_formatter = MappingResponseFormatter.from_config(
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
            response_component=response_component,
            evaluator=evaluator,
        )

    def run_task(self, llm: LLMGenerator, verbose: bool = False) -> RunTaskReturnType:
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
