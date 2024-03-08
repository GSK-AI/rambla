from copy import deepcopy
from typing import Callable
from unittest import mock

import numpy as np
import pytest
from datasets import Dataset

from rambla.datasets.io import prepare_mcqa_dataset
from rambla.tasks.base import RunTaskReturnType
from rambla.tasks.few_shot_examples.few_shot_examples import (
    FewShotExamplesTask,
    FewShotExamplesTaskConfig,
    ParentFewShotExamplesTask,
)
from rambla.tasks.few_shot_examples.utils import ExamplesGeneratingModule
from tests.conftest import hf_datasets_are_same

# flake8: noqa: N802


@pytest.mark.fileio
def test_FewShotExamplesTask_from_config_no_source(fewshot_task_config_no_source: dict):
    fewshot_task = FewShotExamplesTask.from_config(fewshot_task_config_no_source)

    # checks
    # dataset
    expected_dataset = prepare_mcqa_dataset(
        fewshot_task_config_no_source["dataset_config"]
    )
    assert hf_datasets_are_same(fewshot_task.dataset, expected_dataset)

    # examples module
    assert fewshot_task.examples_module.source_dataset is None

    examples_module_config = fewshot_task_config_no_source["examples_module_config"]
    expected_examples_module = ExamplesGeneratingModule(
        seed=examples_module_config["seed"],
        order=examples_module_config["order"],
        index_field=examples_module_config["index_field"],
    )

    assert vars(expected_examples_module) == vars(fewshot_task.examples_module)


@pytest.mark.fileio
def test_FewShotExamplesTask_from_config_with_source(
    fewshot_task_config_with_source: dict,
):
    fewshot_task = FewShotExamplesTask.from_config(fewshot_task_config_with_source)

    # checks
    # dataset
    expected_dataset = prepare_mcqa_dataset(
        fewshot_task_config_with_source["dataset_config"]
    )
    assert hf_datasets_are_same(fewshot_task.dataset, expected_dataset)

    # examples module
    examples_module_config = fewshot_task_config_with_source["examples_module_config"]

    expected_source_dataset = prepare_mcqa_dataset(
        examples_module_config["source_dataset_config"]
    )
    assert hf_datasets_are_same(
        fewshot_task.examples_module.source_dataset, expected_source_dataset
    )

    expected_examples_module = ExamplesGeneratingModule(
        seed=examples_module_config["seed"],
        order=examples_module_config["order"],
        index_field=examples_module_config["index_field"],
        source_dataset=expected_source_dataset,
    )
    assert hf_datasets_are_same(
        expected_examples_module.source_dataset,
        fewshot_task.examples_module.source_dataset,
    )

    rest_expected_examples_module = vars(expected_examples_module)
    rest_fewshot_task_examples_module = vars(fewshot_task.examples_module)
    del rest_expected_examples_module["source_dataset"]
    del rest_fewshot_task_examples_module["source_dataset"]

    assert rest_expected_examples_module == rest_fewshot_task_examples_module


def test_FewShotExamplesTaskConfig_examples_column_name_validation(
    fewshot_task_config_with_source: dict,
):
    config = deepcopy(fewshot_task_config_with_source)

    config["examples_module_config"][
        "examples_column_name"
    ] = "dummy examples column name"
    with pytest.raises(ValueError) as exc_info:
        FewShotExamplesTaskConfig.parse_obj(config)

    assert "dummy examples column name" in str(exc_info.value)


def test_FewShotExamplesTaskConfig_target_field_validation(
    fewshot_task_config_with_source: dict,
):
    config = deepcopy(fewshot_task_config_with_source)

    prev_target_field = config["dataset_config"]["target_field"]
    config["dataset_config"]["target_field"] = "dummy target field"
    with pytest.raises(ValueError) as exc_info:
        FewShotExamplesTaskConfig.parse_obj(config)

    assert "dummy target field" in str(exc_info.value)
    assert prev_target_field in str(exc_info.value)


def test_FewShotExamplesTaskConfig_response_field_validation(
    fewshot_task_config_with_source: dict,
):
    config = deepcopy(fewshot_task_config_with_source)

    prev_response_field = config["evaluator_config"]["response_field"]
    config["evaluator_config"]["response_field"] = "dummy response field"
    with pytest.raises(ValueError) as exc_info:
        FewShotExamplesTaskConfig.parse_obj(config)

    assert "dummy response field" in str(exc_info.value)
    assert prev_response_field in str(exc_info.value)


def test_FewShotExamplesTaskConfig_categories_validation_evaluator(
    fewshot_task_config_with_source: dict,
):
    config = deepcopy(fewshot_task_config_with_source)

    prev_categories = config["evaluator_config"]["categories"]
    config["evaluator_config"]["categories"] = list("123")
    with pytest.raises(ValueError) as exc_info:
        FewShotExamplesTaskConfig.parse_obj(config)

    assert str(list("123")) in str(exc_info.value)
    assert str(prev_categories) in str(exc_info.value)


def test_FewShotExamplesTaskConfig_categories_validation_response_formatter(
    fewshot_task_config_with_source: dict,
):
    config = deepcopy(fewshot_task_config_with_source)

    prev_categories = config["response_formatter_config"]["categories"]
    config["response_formatter_config"]["categories"] = list("123")
    with pytest.raises(ValueError) as exc_info:
        FewShotExamplesTaskConfig.parse_obj(config)

    assert str(list("123")) in str(exc_info.value)
    assert str(prev_categories) in str(exc_info.value)


@mock.patch("rambla.tasks.few_shot_examples.few_shot_examples.prepare_dataset")
def test_FewShotExamplesTask_add_biases(
    mock_prepare_dataset,
    mock_results,
    mock_results_with_bias,
    fewshot_task_config_no_source,
    mock_balanced_dataset,
):
    mock_prepare_dataset.return_value = mock_balanced_dataset(24, 50)
    #

    mock_label_encoder_map = {"yes": 0, "no": 1, "null": 2}

    mock_fewshotexamplestask = FewShotExamplesTask.from_config(
        fewshot_task_config_no_source
    )

    output = mock_fewshotexamplestask._add_biases(mock_results, mock_label_encoder_map)

    assert (
        output["confusion_matrix"] == mock_results_with_bias["confusion_matrix"]
    ).all()
    del output["confusion_matrix"]
    del mock_results_with_bias["confusion_matrix"]
    assert output == mock_results_with_bias


@mock.patch("rambla.tasks.few_shot_examples.few_shot_examples.prepare_dataset")
def test_FewShotExamplesTask_run_task_end_to_end(
    mock_load_dataset,
    fewshot_task_config_no_source: dict,
    mock_balanced_dataset: Callable,
    make_mock_llm: Callable,
):
    intro_dataset = mock_balanced_dataset(24, 10)
    mock_load_dataset.return_value = intro_dataset

    responses = [
        "I don't know",
        "yes",
        "yes..",
        "yes",
        "yes",
        "no",
        "No, but ...",
        "yes",
        "yes",
        "no",
    ]
    expected_responses = [
        "null",
        "yes",
        "yes",
        "yes",
        "yes",
        "no",
        "no",
        "yes",
        "yes",
        "no",
    ]
    mock_llm = make_mock_llm(responses)

    fewshotexamplestask = FewShotExamplesTask.from_config(fewshot_task_config_no_source)
    task_output = fewshotexamplestask.run_task(mock_llm)

    # Checking metrics
    output_confmat = task_output.artifacts["results"]["confusion_matrix"]
    output_label_encoder = task_output.artifacts["label_encoder"]

    # Confmat
    expected_confmat = np.array([[4, 1, 0], [0, 0, 0], [2, 0, 3]])
    confmat_order = ["yes", "null", "no"]
    confmat_order_indices = [output_label_encoder[clss] for clss in confmat_order]
    reordered_output_confmat = output_confmat[confmat_order_indices, :][
        :, confmat_order_indices
    ]
    assert expected_confmat.shape == reordered_output_confmat.shape
    assert np.allclose(expected_confmat, reordered_output_confmat)

    assert np.isclose(task_output.metrics["bias_for_yes"], 2 / 3)
    assert np.isclose(task_output.metrics["bias_for_no"], 1 / 3)

    # Checking responses
    output_dataset = task_output.datasets["final_dataset"]

    assert output_dataset["unformatted_response"] == responses
    assert output_dataset["response"] == expected_responses

    # Checking examples are of the right order
    expected_order = fewshot_task_config_no_source["examples_module_config"]["order"]

    for q in range(0, 10):
        question = output_dataset[q]
        for ex in range(0, len(expected_order)):
            example_pmid = question["examples"][ex]
            example = intro_dataset.filter(lambda x: x["pmid"] == example_pmid)
            output_order = example["final_decision"][0]

            assert output_order == expected_order[ex]

    # Checking the prompts were formed correctly (assuming examples were chosen correctly)
    intro_template = fewshot_task_config_no_source["prompt_formatter_config"][
        "intro_template"
    ]
    examples_template = fewshot_task_config_no_source["prompt_formatter_config"][
        "examples_template"
    ]
    final_question_template = fewshot_task_config_no_source["prompt_formatter_config"][
        "final_question_template"
    ]

    expected_prompts = []
    for q in range(0, 10):
        expected_prompt = intro_template
        question = output_dataset[q]
        for ex in range(0, len(expected_order)):
            example_pmid = question["examples"][ex]
            example = output_dataset.filter(lambda x: x["pmid"] == example_pmid)

            expected_prompt = expected_prompt + examples_template.format(
                question=example["question"][0],
                context=example["context"][0],
                answer=example["final_decision"][0],
            )
        expected_prompt = expected_prompt + final_question_template.format(
            question=question["question"], context=question["context"]
        )
        expected_prompts.append(expected_prompt)

    output_prompts = output_dataset["prompt"]
    assert output_prompts == expected_prompts

    # Checking whether the model was called with the right prompts
    for ii, prompt in enumerate(expected_prompts):
        assert mock_llm.generate.call_args_list[ii].kwargs == {"prompt": prompt}


class TestParentFewShotExamplesTask:
    @pytest.fixture()
    def task_config(self, fewshot_task_config_no_source: dict) -> dict:
        return {
            "child_task_config": fewshot_task_config_no_source,
            "seed": 1234,
            "orders": [
                ["yes", "no", "yes"],
                ["yes", "no", "no"],
                ["no", "no", "yes"],
            ],
        }

    def test_format_artifacts(self):
        output_0 = RunTaskReturnType(
            artifacts={"dummy": 13, "also_dummy": {"second_layer": [1, 2, 3]}}
        )
        output_1 = RunTaskReturnType(
            artifacts={"dummy": 14, "also_dummy": {"second_layer": [7, 8, 9]}}
        )
        output_2 = RunTaskReturnType(
            artifacts={"dummy": 15, "also_dummy": {"second_layer": [4, 5, 6]}}
        )

        # run
        output = ParentFewShotExamplesTask._format_artifacts(
            [output_0, output_1, output_2]
        )

        #
        expected_output = {
            "also_dummy": [
                {"second_layer": [1, 2, 3]},
                {"second_layer": [7, 8, 9]},
                {"second_layer": [4, 5, 6]},
            ],
            "dummy": [13, 14, 15],
        }
        assert output == expected_output

    def test_format_return_datasets(self, task_config: dict):
        mock_dataset_0 = mock.create_autospec(spec=Dataset, instance=True)
        mock_dataset_1 = mock.create_autospec(spec=Dataset, instance=True)
        mock_dataset_2 = mock.create_autospec(spec=Dataset, instance=True)

        output_0 = RunTaskReturnType(datasets={"final_dataset": mock_dataset_0})
        output_1 = RunTaskReturnType(datasets={"final_dataset": mock_dataset_1})
        output_2 = RunTaskReturnType(datasets={"final_dataset": mock_dataset_2})

        # run
        task = ParentFewShotExamplesTask.from_config(task_config)
        output = task._format_return_datasets([output_0, output_1, output_2])

        # asserts
        expected_datasets_dict = {
            "final_dataset_yes-no-yes": mock_dataset_0,
            "final_dataset_yes-no-no": mock_dataset_1,
            "final_dataset_no-no-yes": mock_dataset_2,
        }
        for key, value in output.items():
            assert hf_datasets_are_same(value, expected_datasets_dict[key])

    @mock.patch("rambla.tasks.few_shot_examples.few_shot_examples.prepare_dataset")
    def test_run_task(
        self,
        mock_prepare_dataset,
        task_config: dict,
        mock_balanced_dataset: Callable,
        make_mock_llm: Callable,
    ):
        intro_dataset_0 = mock_balanced_dataset(12, 6)
        intro_dataset_1 = mock_balanced_dataset(34, 6)
        intro_dataset_2 = mock_balanced_dataset(56, 6)

        mock_prepare_dataset.side_effect = [
            intro_dataset_0,
            intro_dataset_1,
            intro_dataset_2,
        ]

        responses = [
            "I don't know",
            "yes",
            "yes..",
            "yes",
            "yes",
            "no",
            "No, but ...",
            "yes",
            "yes",
            "no",
            "no and some dummy text",
            "yes and some more dummy text",
            "I don't know",
            "yes",
            "yes..",
            "yes",
            "yes",
            "no",
        ]
        expected_responses = [
            "null",
            "yes",
            "yes",
            "yes",
            "yes",
            "no",
            "no",
            "yes",
            "yes",
            "no",
            "no",
            "yes",
            "null",
            "yes",
            "yes",
            "yes",
            "yes",
            "no",
        ]

        # run
        mock_llm = make_mock_llm(responses)

        task = ParentFewShotExamplesTask.from_config(task_config)
        task_output = task.run_task(mock_llm)

        # asserts

        assert (
            task_output.datasets["final_dataset_yes-no-yes"]["response"]
            == expected_responses[:6]
        )
        assert (
            task_output.datasets["final_dataset_yes-no-no"]["response"]
            == expected_responses[6:12]
        )
        assert (
            task_output.datasets["final_dataset_no-no-yes"]["response"]
            == expected_responses[12:18]
        )
