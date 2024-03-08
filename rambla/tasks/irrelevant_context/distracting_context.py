from __future__ import annotations

from typing import Union

from datasets import Dataset
from pydantic import root_validator

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
from rambla.tasks.irrelevant_context.utils import (
    ContextAugmentingModule,
    ContextAugmentingModuleConfig,
)
from rambla.utils.config import validate_field_attribute_equality
from rambla.utils.misc import prepare_dicts_for_logging


class DistractingContextConfig(BaseTaskConfig):
    dataset_config: MCQADatasetConfig
    context_augmenting_module_config: ContextAugmentingModuleConfig
    prompt_formatter_config: ColumnPromptFormatterConfig
    response_formatter_config: MCQAResponseFormatterConfig
    evaluator_config: MCQAEvalComponentConfig
    response_component_config: ResponseComponentConfig

    @root_validator()
    @classmethod
    def validate_target_field(cls, values):
        validate_field_attribute_equality(
            values["dataset_config"], values["evaluator_config"], "target_field"
        )
        return values

    @root_validator()
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

    @root_validator()
    @classmethod
    def validate_index_field(cls, values):
        validate_field_attribute_equality(
            values["dataset_config"], values["prompt_formatter_config"], "index_field"
        )

        return values


class DistractingContextTask(BaseTask):
    """Irrelevant context task.

    Steps:
    1. Creates a new `context` column that mixes the original
        `context` column AND shuffled versions of itself.
    2. Create prompts for dataset (under `prompt`)
    3. Generate responses from `llm` (under `response`)
        - `llm` is an `LLMGenerator` provided in `.run_task`
    4. Post-process responses
    5. Run evaluation
    """

    def __init__(
        self,
        dataset: Dataset,
        context_augmenting_module: ContextAugmentingModule,
        prompt_formatter: ColumnPromptFormatter,
        response_formatter: MCQAResponseFormatter,
        evaluator: MCQAEvalComponent,
        response_component: ResponseComponent,
    ):
        self.dataset = dataset
        self.context_augmenting_module = context_augmenting_module
        self.prompt_formatter = prompt_formatter
        self.response_formatter = response_formatter
        self.evaluator = evaluator
        self.response_component = response_component

    @classmethod
    def from_config(
        cls, config: Union[dict, DistractingContextConfig]
    ) -> DistractingContextTask:
        if not isinstance(config, DistractingContextConfig):
            config = DistractingContextConfig.parse_obj(config)

        dataset = prepare_dataset(config.dataset_config.dict())
        context_augmenting_module = ContextAugmentingModule.from_config(
            config.context_augmenting_module_config
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
            context_augmenting_module=context_augmenting_module,
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
        # Generate shuffled dataset
        mixed_dataset = self.context_augmenting_module.run(self.dataset)

        # Form prompt dataset
        prompt_dataset = self.prompt_formatter.format(mixed_dataset)

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
