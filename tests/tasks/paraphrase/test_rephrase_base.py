from typing import Callable
from unittest import mock

import numpy as np
import pytest
from datasets import Dataset

from rambla.tasks.paraphrase.paraphrase import ParaphraseTask, ParaphraseTaskConfig

# flake8: noqa: N802


@mock.patch("rambla.tasks.paraphrase.utils.build_llm")
@mock.patch("rambla.tasks.paraphrase.paraphrase.prepare_dataset")
def test_ParaphraseTask_run_task(
    mock_prepare_dataset,
    mock_build_llm,
    rephrase_task_config: dict,
    mock_flat_pubmedqa_dataset: Dataset,
    make_mock_llm: Callable,
):
    mock_prepare_dataset.return_value = mock_flat_pubmedqa_dataset
    rephrasing_responses = list(map(str, range(10)))
    rephrasing_llm = make_mock_llm(rephrasing_responses)
    mock_build_llm.return_value = rephrasing_llm

    task = ParaphraseTask.from_config(rephrase_task_config)
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

    #
    task_output = task.run_task(mock_llm)

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
    assert output_dataset["unformatted_response"] == responses
    assert output_dataset["response"] == expected_responses

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

    # Checking the prompts were formed correctly
    prompt_template = rephrase_task_config["prompt_formatter_config"]["template"]
    expected_prompts = [
        prompt_template.format(question=question) for question in rephrasing_responses
    ]
    assert output_dataset["prompt"] == expected_prompts
    assert output_dataset["original_question"] == mock_flat_pubmedqa_dataset["question"]

    # Checking whether the model was called with the right prompts
    for ii, prompt in enumerate(expected_prompts):
        assert mock_llm.generate.call_args_list[ii].kwargs == {"prompt": prompt}

    # Checking the rephrasing prompts were formed correctly
    rephrasing_prompt_template = rephrase_task_config["rephrasing_module_config"][
        "prompt_formatter_config"
    ]["template"]
    expected_rephrasing_prompts = [
        rephrasing_prompt_template.format(question=question)
        for question in mock_flat_pubmedqa_dataset["question"]
    ]
    assert output_dataset["rephrasing_prompt"] == expected_rephrasing_prompts

    # Checking whether the rephrasing model was called with the right prompts
    for ii, prompt in enumerate(expected_rephrasing_prompts):
        assert rephrasing_llm.generate.call_args_list[ii].kwargs == {"prompt": prompt}


def test_ParaphraseTaskConfig_target_field_validation(rephrase_task_config):
    prev_target_field = rephrase_task_config["dataset_config"]["target_field"]
    rephrase_task_config["dataset_config"]["target_field"] = "dummy target field"
    with pytest.raises(ValueError) as exc_info:
        ParaphraseTaskConfig.parse_obj(rephrase_task_config)

    assert "dummy target field" in str(exc_info.value)
    assert prev_target_field in str(exc_info.value)


def test_ParaphraseTaskConfig_response_field_validation(rephrase_task_config):
    prev_response_field = rephrase_task_config["evaluator_config"]["response_field"]
    rephrase_task_config["evaluator_config"]["response_field"] = "dummy response field"
    with pytest.raises(ValueError) as exc_info:
        ParaphraseTaskConfig.parse_obj(rephrase_task_config)

    assert "dummy response field" in str(exc_info.value)
    assert prev_response_field in str(exc_info.value)


def test_ParaphraseTaskConfig_categories_validation_evaluator(rephrase_task_config):
    prev_categories = rephrase_task_config["evaluator_config"]["categories"]
    rephrase_task_config["evaluator_config"]["categories"] = list("123")
    with pytest.raises(ValueError) as exc_info:
        ParaphraseTaskConfig.parse_obj(rephrase_task_config)

    assert str(list("123")) in str(exc_info.value)
    assert str(prev_categories) in str(exc_info.value)


def test_ParaphraseTaskConfig_categories_validation_response_formatter(
    rephrase_task_config,
):
    prev_categories = rephrase_task_config["response_formatter_config"]["categories"]
    rephrase_task_config["response_formatter_config"]["categories"] = list("123")
    with pytest.raises(ValueError) as exc_info:
        ParaphraseTaskConfig.parse_obj(rephrase_task_config)

    assert str(list("123")) in str(exc_info.value)
    assert str(prev_categories) in str(exc_info.value)
