from __future__ import annotations

from copy import deepcopy
from typing import Any, Union

from datasets import Dataset
from pydantic import Extra, root_validator, validator

from rambla.datasets.io import MCQADatasetConfig, prepare_dataset
from rambla.evaluation.shortform import MCQAEvalComponent, MCQAEvalComponentConfig
from rambla.prompts.formatters import (
    ExamplesPromptFormatter,
    ExamplesPromptFormatterConfig,
)
from rambla.response_formatting.formatting import (
    MCQAResponseFormatter,
    MCQAResponseFormatterConfig,
)
from rambla.response_generation.response import (
    ResponseComponent,
    ResponseComponentConfig,
)
from rambla.tasks.base import BaseTask, BaseTaskConfig, LLMGenerator, RunTaskReturnType
from rambla.tasks.few_shot_examples.utils import (
    ExamplesGeneratingModule,
    ExamplesGeneratingModuleConfig,
)
from rambla.utils.config import (
    validate_category_equality,
    validate_field_attribute_equality,
)
from rambla.utils.misc import (
    add_prefix_to_dict_keys,
    initialize_logger,
    list_of_dicts_to_dict_of_lists,
    merge_dicts,
    prepare_dicts_for_logging,
)

logger = initialize_logger(__name__)


class FewShotExamplesTaskConfig(BaseTaskConfig):
    dataset_config: MCQADatasetConfig
    examples_module_config: ExamplesGeneratingModuleConfig
    prompt_formatter_config: ExamplesPromptFormatterConfig
    response_formatter_config: MCQAResponseFormatterConfig
    evaluator_config: MCQAEvalComponentConfig
    response_component_config: ResponseComponentConfig

    class Config:  # noqa: D106
        extra = Extra.forbid

    @validator("prompt_formatter_config")
    @classmethod
    def validate_prompt_formatter_config(
        cls, prompt_formatter_config, values, **kwargs
    ):
        # target field
        examples_module_config = values["examples_module_config"]
        validate_field_attribute_equality(
            examples_module_config, prompt_formatter_config, "examples_column_name"
        )
        return prompt_formatter_config

    @validator("evaluator_config")
    @classmethod
    def validate_evaluator_config(cls, evaluator_config, values, **kwargs):
        # target field
        dataset_config = values["dataset_config"]
        validate_field_attribute_equality(
            dataset_config, evaluator_config, "target_field"
        )

        # response field
        response_formatter_config = values["response_formatter_config"]
        validate_field_attribute_equality(
            response_formatter_config,
            evaluator_config,
            "response_field_name",
            "response_field",
        )

        # categories
        dataset_categories = dataset_config.categories_to_keep
        response_formatter_categories = response_formatter_config.categories
        evaluator_categories = evaluator_config.categories

        validate_category_equality(dataset_categories, response_formatter_categories)
        validate_category_equality(
            (dataset_categories + [response_formatter_config.null_category]),
            evaluator_categories,
        )

        return evaluator_config

    @root_validator(skip_on_failure=True)
    @classmethod
    def validate_index_field(cls, values):
        validate_field_attribute_equality(
            values["dataset_config"], values["examples_module_config"], "index_field"
        )
        validate_field_attribute_equality(
            values["dataset_config"],
            values["prompt_formatter_config"],
            "index_field",
        )
        return values


class FewShotExamplesTask(BaseTask):
    """Few shot example task.

    Follows experiment as defined in https://arxiv.org/abs/2102.09690.
    Aim is to investigate the bias in the model response due to common
    and/or recent answers given in the examples (majority and recency bias).

    Steps:
    1. Sample example questions that follow a given order of answers,
        such as "yes", "yes", "no". (under 'examples')
    2. Create prompts for dataset which supply the examples and answers
        in order, followed by a test question. (under `prompt`)
    3. Generate responses from `llm` (under `response`)
        - `llm` is an `LLMGenerator` provided in `.run_task`
    4. Post-process responses.
    5. Run evaluation. In particular, calculate the bias
        in the responses towards the answer classes (e.g. "yes" or "no" biases)
    """

    def __init__(
        self,
        dataset: Dataset,
        examples_module: ExamplesGeneratingModule,
        prompt_formatter: ExamplesPromptFormatter,
        response_formatter: MCQAResponseFormatter,
        evaluator: MCQAEvalComponent,
        response_component: ResponseComponent,
    ):
        self.dataset = dataset
        self.examples_module = examples_module
        self.prompt_formatter = prompt_formatter
        self.response_formatter = response_formatter
        self.evaluator = evaluator
        self.response_component = response_component

    @classmethod
    def from_config(
        cls, config: Union[FewShotExamplesTaskConfig, dict]
    ) -> "FewShotExamplesTask":
        config = FewShotExamplesTaskConfig.parse_obj(config)

        dataset = prepare_dataset(config.dataset_config.dict())

        examples_module = ExamplesGeneratingModule.from_config(
            config.examples_module_config
        )

        prompt_formatter = ExamplesPromptFormatter.from_config(
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
            examples_module=examples_module,
            prompt_formatter=prompt_formatter,
            response_formatter=response_formatter,
            evaluator=evaluator,
            response_component=response_component,
        )

    def _add_biases(self, results, label_encoder_map) -> dict:
        label_encoder_map_copy = deepcopy(label_encoder_map)
        del label_encoder_map_copy[self.response_formatter.null_category]
        labels_no_null = list(set(label_encoder_map_copy.values()))

        conf_matrix = results["confusion_matrix"]
        conf_matrix_no_null = conf_matrix[labels_no_null][:, labels_no_null]

        for category, label in label_encoder_map_copy.items():
            bias = sum(conf_matrix[labels_no_null][:, label]) / sum(
                sum(conf_matrix_no_null)
            )
            results.update({"bias_for_" + str(category): bias})
        return results

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
        # Generate few-shot examples dataset
        examples_dataset = self.examples_module.run(self.dataset)

        # Form prompt dataset
        prompt_dataset = self.prompt_formatter.format(examples_dataset)

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
        label_encoder_map = eval_results["label_encoder"]
        eval_results["results"] = self._add_biases(
            eval_results["results"], label_encoder_map
        )

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


class ParentFewShotExamplesTaskConfig(BaseTaskConfig):
    """Executes the child few shot examples task in sequence for a list of orders.

    For example, if given the list of orders:
        1. "yes", "yes", "no"
        2. "no", "no", "yes"
        3. "yes", "no"

    this task will run the child class to evaluate the bias involved in
    each of these orders. It will evaluate all of them and combine the
    results to find patterns in the biases across a larger group of
    example orders (see Figure 4 in https://arxiv.org/abs/2102.09690 for
    an example of analysis across several orders).
    """

    seed: int
    orders: list[list[str]]
    child_task_config: FewShotExamplesTaskConfig

    @root_validator(pre=True)
    @classmethod
    def validate_all(cls, values):
        """Needed because these configs are added by hydra."""
        keys_to_transfer = ["response_component_config"]
        for key in keys_to_transfer:
            if key in values:
                values["child_task_config"][key] = values[key]

        # Keys to remove from _this_ config.
        keys_to_remove = ["dataset_config", "response_component_config"]
        for key in keys_to_remove:
            if key in values:
                values.pop(key)

        # Keys to remove from the `child_task_config`.
        keys_to_remove = ["class_key"]
        for key in keys_to_remove:
            if key in values["child_task_config"]:
                values["child_task_config"].pop(key)

        return values

    class Config:  # noqa: D106
        extra = Extra.forbid


class ParentFewShotExamplesTask(BaseTask):
    def __init__(
        self,
        seed: int,
        orders: list[list[str]],
        child_task_config: FewShotExamplesTaskConfig,
    ):
        self.seed = seed
        self.orders = orders
        self.child_task_config = child_task_config
        self.sep = "-"

    def order_to_string(self, order: list[str]) -> str:
        return self.sep.join(order)

    @classmethod
    def from_config(
        cls, config: Union[dict, ParentFewShotExamplesTaskConfig]
    ) -> ParentFewShotExamplesTask:
        if not isinstance(config, ParentFewShotExamplesTaskConfig):
            config = ParentFewShotExamplesTaskConfig.parse_obj(config)
        return cls(
            seed=config.seed,
            orders=config.orders,
            child_task_config=config.child_task_config,
        )

    @staticmethod
    def _format_artifacts(return_data: list[RunTaskReturnType]) -> dict[str, Any]:
        """Formats final artifacts dictionary"""
        artifacts = [i.artifacts for i in return_data]
        if None in artifacts:
            raise ValueError("All `return_data` must have a `artifacts` field")

        artifacts_dict = list_of_dicts_to_dict_of_lists(artifacts)  # type: ignore
        if "results" in artifacts_dict:
            artifacts_dict["results"] = list_of_dicts_to_dict_of_lists(  # type: ignore
                artifacts_dict["results"]
            )

        return artifacts_dict

    def _format_return_datasets(
        self, return_data: list[RunTaskReturnType]
    ) -> dict[str, Dataset]:
        """Formats list of datasets into dataset dictionary"""
        final_datasets = {}
        for idx, instance in enumerate(return_data):
            if not instance.datasets or "final_dataset" not in instance.datasets.keys():
                raise ValueError(f"Missing `final_dataset` for mutation round: {idx}")

            stringified_order = self.order_to_string(self.orders[idx])
            alias = f"final_dataset_{stringified_order}"
            final_datasets[alias] = instance.datasets["final_dataset"]

        return final_datasets

    def format_return_data(
        self, return_data: list[RunTaskReturnType]
    ) -> RunTaskReturnType:
        """Merges metrics for each mutation run into single return type"""
        return RunTaskReturnType(
            metrics=merge_dicts([item.metrics for item in return_data]),
            artifacts=self._format_artifacts(return_data),
            datasets=self._format_return_datasets(return_data),
            other=None,
            plots=None,
            dictionaries=None,
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
        outputs: list[RunTaskReturnType] = []
        for order in self.orders:
            logger.info(f"Working on {order=}")
            order_to_string = "-".join(order)
            config = deepcopy(self.child_task_config)
            config.examples_module_config.order = order
            config.examples_module_config.seed = self.seed
            child_task = FewShotExamplesTask.from_config(config)
            child_task_output = child_task.run_task(llm, verbose=verbose)
            child_task_output.metrics = add_prefix_to_dict_keys(
                child_task_output.metrics, f"order_{order_to_string}"
            )
            outputs.append(child_task_output)

        formatted_return_data = self.format_return_data(outputs)
        return formatted_return_data
