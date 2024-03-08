import copy
import numbers
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset
from pydantic import BaseModel, Extra, root_validator

from rambla import tasks
from rambla.tasks.base import BaseTask, BaseTaskConfig, LLMGenerator, RunTaskReturnType
from rambla.text_mutation import build_mutator
from rambla.text_mutation.base import BaseMutator
from rambla.utils.misc import (
    add_prefix_to_dict_keys,
    list_of_dicts_to_dict_of_lists,
    merge_dicts,
)


class TaskConfig(BaseModel):
    name: str
    config: dict

    class Config:  # noqa: D106
        extra = Extra.forbid


class PromptRobustnessConfig(BaseTaskConfig):
    subtask: TaskConfig
    mutator_config: dict
    mutation_schedule: List[int]
    field_to_mutate: str

    class Config:  # noqa: D106
        extra = Extra.forbid

    @root_validator(pre=True)
    @classmethod
    def validate_all(cls, values):
        """Needed because these configs are added by hydra."""
        keys_to_transfer = ["response_component_config"]
        for key in keys_to_transfer:
            if key in values:
                values["subtask"]["config"][key] = values[key]

        # Keys to remove from _this_ config.
        keys_to_remove = ["dataset_config", "response_component_config"]
        for key in keys_to_remove:
            if key in values:
                values.pop(key)

        # Keys to remove from the `subtask`.
        keys_to_remove = ["class_key"]
        for key in keys_to_remove:
            if key in values["subtask"]["config"]:
                values["subtask"]["config"].pop(key)

        return values


class PromptRobustness(BaseTask):
    def __init__(
        self,
        task: BaseTask,
        mutator: BaseMutator,
        mutation_schedule: List[int],
        field_to_mutate: str,
    ) -> None:
        """Tasks to assess the robustness of a model to variations in a prompt

        This task applies mutations to prompts passed to an LLM (e.g. spelling mistakes)
        to assess the impact these have on the final metrics.

        This task acts like a meta-task, taking an existing task and running it
        multiple times with an increasing number of mutations, before collating the
        results of every run.

        Parameters
        ----------
        task : BaseTask
            Original task to run with mutations
        mutator : BaseMutator
            Mutation class to use to apply mutations to prompts
        mutation_schedule : List[int]
            A list containing the number of mutations to apply at each run.
            For example:
                mutation_schedule = [1, 2, 5, 10]
            This will apply 1 mutation, then 2, then 5 ,...
        field_to_mutate : str
            Dataset field to apply the mutation to.
        """
        self.task = task
        self.mutator = mutator
        self.mutation_schedule = mutation_schedule
        self.field_to_mutate = field_to_mutate

    @classmethod
    def from_config(cls, config: dict | PromptRobustnessConfig) -> "PromptRobustness":
        if isinstance(config, dict):
            config = PromptRobustnessConfig.parse_obj(config)

        subtask = tasks.TASK_MAP[config.subtask.name].from_config(config.subtask.config)

        mutator = build_mutator(config.mutator_config)

        return cls(
            task=subtask,  # type: ignore
            mutator=mutator,
            mutation_schedule=config.mutation_schedule,
            field_to_mutate=config.field_to_mutate,
        )

    def _mutate_dataset(self, dataset: Dataset, num_mutations: int) -> Dataset:
        renamed_mutate_field = f"original_{self.field_to_mutate}"
        dataset = dataset.add_column(
            renamed_mutate_field,
            dataset[self.field_to_mutate],  # type: ignore
        )
        return dataset.map(
            lambda example: {
                self.field_to_mutate: self.mutator.mutate(
                    example[renamed_mutate_field], num_mutations
                )
            }
        )

    @staticmethod
    def _get_metric_relative_change(
        metrics_dict: Dict[str, List[float | int | str]]
    ) -> Dict[str, List[float]]:
        """Computes the change relative to first metric in list for each metric"""
        numerical_metric_names = [
            k for k, v in metrics_dict.items() if isinstance(v[0], numbers.Number)
        ]
        if not numerical_metric_names:
            UserWarning(
                f"No numeric metrics found in list: {list(metrics_dict.keys())}"
            )

        base_dict = {
            k: v[0] for k, v in metrics_dict.items() if k in numerical_metric_names
        }
        change_dict = {}
        for k, v in base_dict.items():
            change_dict[k] = (
                (np.array(metrics_dict[k][1:]) - v) / v  # type: ignore
            ).tolist()

        return change_dict

    def _format_artifacts(self, return_data: List[RunTaskReturnType]) -> Dict[str, Any]:
        """Formats final artifacts dictionary"""
        artifacts = [i.artifacts for i in return_data]
        if None in artifacts:
            raise ValueError("All `return_data` must have a `artifacts` field")

        artifacts_dict = list_of_dicts_to_dict_of_lists(artifacts)  # type: ignore
        if "results" in artifacts_dict:
            artifacts_dict["results"] = list_of_dicts_to_dict_of_lists(  # type: ignore
                artifacts_dict["results"]
            )
            change_metrics = self._get_metric_relative_change(artifacts_dict["results"])
            artifacts_dict["metrics_relative_change"] = change_metrics  # type: ignore

        return artifacts_dict

    def _format_return_datasets(
        self, return_data: List[RunTaskReturnType]
    ) -> Dict[str, Dataset]:
        """Formats list of datasets into dataset dictionary"""
        final_datasets = {}
        for idx, instance in enumerate(return_data):
            if not instance.datasets or "final_dataset" not in instance.datasets.keys():
                raise ValueError(f"Missing `final_dataset` for mutation round: {idx}")

            final_datasets[
                f"final_dataset_n{self.mutation_schedule[idx]}"
            ] = instance.datasets["final_dataset"]

        return final_datasets

    def format_return_data(
        self, return_data: List[RunTaskReturnType]
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

    @staticmethod
    def validate_outputs(
        formatted_return_data: RunTaskReturnType, num_iterations: int
    ) -> None:
        # Validates the results field if it exists
        if "results" in formatted_return_data.artifacts:  # type: ignore
            if not isinstance(
                formatted_return_data.artifacts["results"], dict  # type: ignore
            ):
                raise ValueError('"results" field should be of type: Dict[str, list]')
            elif not all(
                [
                    len(metric) == num_iterations
                    for _, metric in formatted_return_data.artifacts["results"].items()
                ]
            ):
                raise ValueError(
                    f"All results should contain lists of length: {num_iterations}"
                )

        # Validates that other artifacts contain the expected number of results
        for artifact, value in formatted_return_data.artifacts.items():  # type: ignore
            if isinstance(value, list):
                if len(value) != num_iterations:
                    raise ValueError(
                        f"Artifact: {artifact} should have {num_iterations} entries but"
                        f"only has {len(value)}."
                    )

        # Validates the datasets field contains expected number of results
        n_datasets = len(formatted_return_data.datasets.keys())
        if n_datasets != num_iterations:
            raise ValueError(
                f"Datasets should contain {num_iterations} datasets but only "
                f"contains {n_datasets}"
            )

    def run_task(self, llm: LLMGenerator) -> RunTaskReturnType:
        if not hasattr(self.task, "dataset"):
            raise AttributeError(
                "Task must have a `dataset` attribute to run dataset mutations"
            )

        original_dataset = copy.deepcopy(self.task.dataset)  # type: ignore
        return_data = []
        for mutation in self.mutation_schedule:
            mutated_dataset = self._mutate_dataset(original_dataset, mutation)
            self.task.dataset = mutated_dataset  # type: ignore
            results = self.task.run_task(llm)
            results.metrics = add_prefix_to_dict_keys(
                results.metrics, f"mut_{mutation}"
            )
            return_data.append(results)

        formatted_return_data = self.format_return_data(return_data)
        self.validate_outputs(formatted_return_data, len(self.mutation_schedule))

        return formatted_return_data
