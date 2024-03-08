from copy import deepcopy
from typing import Callable
from unittest import mock

import numpy as np
import pytest
from datasets import Dataset

from rambla.datasets.io import prepare_mcqa_dataset
from rambla.evaluation.longform import ResponseQualityEvalComponent
from rambla.evaluation.shortform import MCQAEvalComponent
from rambla.prompts.formatters import ColumnPromptFormatter
from rambla.response_formatting.base import STRING_FORMATTER_MAP
from rambla.response_formatting.formatting import MCQAResponseFormatter
from rambla.tasks.mcqabaseline.mcqabaseline import (
    MCQABaselineTask,
    MCQABaselineTaskConfig,
)
from tests.conftest import hf_datasets_are_same

# flake8: noqa: N802


@pytest.mark.fileio
def test_BaselineTask_from_config(baseline_task_config):
    baseline_task = MCQABaselineTask.from_config(baseline_task_config)

    # checks
    # dataset
    expected_dataset = prepare_mcqa_dataset(baseline_task_config["dataset_config"])
    assert hf_datasets_are_same(baseline_task.dataset, expected_dataset)

    # prompt formatter
    expected_prompt_formatter = ColumnPromptFormatter(
        **baseline_task_config["prompt_formatter_config"]
    )
    assert expected_prompt_formatter.__dict__ == baseline_task.prompt_formatter.__dict__

    # response formatter
    response_formatter_config = baseline_task_config["response_formatter_config"]
    expected_response_formatter = MCQAResponseFormatter(
        response_field_name=response_formatter_config["response_field_name"],
        categories=response_formatter_config["categories"],
        null_category=response_formatter_config["null_category"],
        string_formatter=STRING_FORMATTER_MAP[
            response_formatter_config["string_formatter_name"]
        ],
    )
    assert (
        expected_response_formatter.__dict__
        == baseline_task.response_formatter.__dict__
    )

    # evaluator component
    expected_evaluator = MCQAEvalComponent(**baseline_task_config["evaluator_config"])

    ev1 = expected_evaluator.__dict__
    ev2 = baseline_task.evaluator.__dict__
    for key, value in ev1.items():
        if key in ["metrics"]:
            assert value.keys() == ev2[key].keys(), f"{key=}"
        else:
            assert value == ev2[key], f"{key=}"

    # response quality evaluator
    expected_quality_evaluator = ResponseQualityEvalComponent(
        **baseline_task_config["response_quality_evaluator_config"]
    )
    assert (
        expected_quality_evaluator.__dict__
        == baseline_task.response_quality_evaluator.__dict__
    )


def test_BaselineTaskConfig_target_field_validation(baseline_task_config):
    config = deepcopy(baseline_task_config)

    prev_target_field = config["dataset_config"]["target_field"]
    config["dataset_config"]["target_field"] = "dummy target field"
    with pytest.raises(ValueError) as exc_info:
        MCQABaselineTaskConfig.parse_obj(config)

    assert "dummy target field" in str(exc_info.value)
    assert prev_target_field in str(exc_info.value)


def test_BaselineTaskConfig_response_field_validation(baseline_task_config):
    config = deepcopy(baseline_task_config)

    prev_response_field = config["evaluator_config"]["response_field"]
    config["evaluator_config"]["response_field"] = "dummy response field"
    with pytest.raises(ValueError) as exc_info:
        MCQABaselineTaskConfig.parse_obj(config)

    assert "dummy response field" in str(exc_info.value)
    assert prev_response_field in str(exc_info.value)


def test_BaselineTaskConfig_categories_validation_evaluator(baseline_task_config):
    config = deepcopy(baseline_task_config)

    prev_categories = config["evaluator_config"]["categories"]
    config["evaluator_config"]["categories"] = list("123")
    with pytest.raises(ValueError) as exc_info:
        MCQABaselineTaskConfig.parse_obj(config)

    assert str(list("123")) in str(exc_info.value)
    assert str(prev_categories) in str(exc_info.value)


def test_BaselineTaskConfig_categories_validation_response_formatter(
    baseline_task_config,
):
    config = deepcopy(baseline_task_config)

    prev_categories = config["response_formatter_config"]["categories"]
    config["response_formatter_config"]["categories"] = list("123")
    with pytest.raises(ValueError) as exc_info:
        MCQABaselineTaskConfig.parse_obj(config)

    assert str(list("123")) in str(exc_info.value)
    assert str(prev_categories) in str(exc_info.value)


@mock.patch("rambla.tasks.mcqabaseline.mcqabaseline.prepare_dataset")
def test_BaselineTask_run_task(
    mock_prepare_dataset,
    baseline_task_config: dict,
    mock_flat_pubmedqa_dataset: Dataset,
    make_mock_llm: Callable,
):
    mock_prepare_dataset.return_value = mock_flat_pubmedqa_dataset
    baseline_task = MCQABaselineTask.from_config(baseline_task_config)
    responses = [
        "Yes",
        "yes",
        "yes..",
        "yes",
        "yes",
        "no",
        "No, but ...",
        "no",
        "no",
        "no",
    ]

    mock_llm = make_mock_llm(responses)

    task_output = baseline_task.run_task(mock_llm)

    #
    expected_responses = [
        "yes",
        "yes",
        "yes",
        "yes",
        "yes",
        "no",
        "no",
        "no",
        "no",
        "no",
    ]

    output_dataset = task_output.datasets["final_dataset"]
    output_confmat = task_output.artifacts["results"]["confusion_matrix"]

    # Asserts

    # Label encoder
    output_label_encoder = task_output.artifacts["label_encoder"]
    assert output_label_encoder.keys() == set(["yes", "no", "maybe", "null"])
    assert set(output_label_encoder.values()) == set(range(4))

    # Confmat
    expected_confmat = np.array(
        [[0, 0, 0, 0], [0, 3, 0, 2], [0, 0, 0, 0], [0, 2, 0, 3]]
    )
    confmat_order = ["maybe", "no", "null", "yes"]
    confmat_order_indices = [output_label_encoder[clss] for clss in confmat_order]
    reordered_output_confmat = output_confmat[confmat_order_indices, :][
        :, confmat_order_indices
    ]
    assert expected_confmat.shape == reordered_output_confmat.shape
    assert np.allclose(expected_confmat, reordered_output_confmat)

    # Responses
    assert output_dataset["unformatted_response"] == responses
    assert output_dataset["response"] == expected_responses

    # Checking the prompts were formed correctly
    prompt_template = baseline_task_config["prompt_formatter_config"]["template"]
    expected_prompts = [
        prompt_template.format(question=question)
        for question in mock_flat_pubmedqa_dataset["question"]
    ]
    assert output_dataset["prompt"] == expected_prompts

    # Checking whether the model was called with the right prompts
    for ii, prompt in enumerate(expected_prompts):
        assert mock_llm.generate.call_args_list[ii].kwargs == {"prompt": prompt}
