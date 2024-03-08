from typing import Callable
from unittest import mock

import numpy as np
import pytest
from datasets import Dataset
from pydantic.error_wrappers import ValidationError

from rambla.tasks.negation.negation import NegationTask, NegationTaskConfig

# flake8: noqa: N802


@mock.patch("rambla.tasks.paraphrase.utils.build_llm")
@mock.patch("rambla.tasks.negation.negation.prepare_dataset")
def test_NegationTask_run_task(
    mock_prepare_dataset,
    mock_build_llm,
    negation_task_config: dict,
    mock_flat_pubmedqa_dataset: Dataset,
    make_mock_llm: Callable,
):
    mock_prepare_dataset.return_value = mock_flat_pubmedqa_dataset

    rephrasing_responses = list(map(str, range(10)))
    negation_llm = make_mock_llm(rephrasing_responses)
    mock_build_llm.return_value = negation_llm

    task = NegationTask.from_config(negation_task_config)
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
        "no",
        "no",
        "no",
        "no",
        "no",
        "yes",
        "yes",
        "yes",
        "yes",
        "yes",
    ]

    output_dataset = task_output.datasets["final_dataset"]
    output_confmat = task_output.artifacts["results"]["confusion_matrix"]

    # Assert
    assert output_dataset["unformatted_response"] == responses
    assert output_dataset["response"] == expected_responses

    # Label encoder
    output_label_encoder = task_output.artifacts["label_encoder"]
    assert output_label_encoder.keys() == set(["yes", "no", "maybe", "null"])
    assert set(output_label_encoder.values()) == set(range(4))

    # Confmat
    expected_confmat = np.array(
        [
            [0, 0, 0, 0],
            [0, 2, 0, 3],
            [0, 0, 0, 0],
            [0, 3, 0, 2],
        ]
    )
    confmat_order = ["maybe", "no", "null", "yes"]
    confmat_order_indices = [output_label_encoder[clss] for clss in confmat_order]
    reordered_output_confmat = output_confmat[confmat_order_indices, :][
        :, confmat_order_indices
    ]
    assert expected_confmat.shape == reordered_output_confmat.shape
    assert np.allclose(expected_confmat, reordered_output_confmat)

    # Checking the prompts were formed correctly
    prompt_template = negation_task_config["prompt_formatter_config"]["template"]
    expected_prompts = [
        prompt_template.format(question=question, context=context)
        for question, context in zip(
            mock_flat_pubmedqa_dataset["question"], rephrasing_responses
        )
    ]
    assert output_dataset["prompt"] == expected_prompts

    # Checking whether the model was called with the right prompts
    for ii, prompt in enumerate(expected_prompts):
        assert mock_llm.generate.call_args_list[ii].kwargs == {"prompt": prompt}

    # Checking the rephrasing prompts were formed correctly
    rephrasing_prompt_template = negation_task_config["rephrasing_module_config"][
        "prompt_formatter_config"
    ]["template"]
    expected_rephrasing_prompts = [
        rephrasing_prompt_template.format(context=context, question=question)
        for question, context in zip(
            mock_flat_pubmedqa_dataset["question"],
            mock_flat_pubmedqa_dataset["context"],
        )
    ]
    assert output_dataset["rephrasing_prompt"] == expected_rephrasing_prompts

    # Checking whether the negation model was called with the right prompts
    for ii, prompt in enumerate(expected_rephrasing_prompts):
        assert negation_llm.generate.call_args_list[ii].kwargs == {"prompt": prompt}


def test_negation_task_config(negation_task_config: dict) -> None:
    NegationTaskConfig.parse_obj(negation_task_config)
    assert True


def test_negation_task_config_invalid_target_field(negation_task_config: dict) -> None:
    negation_task_config["evaluator_config"]["target_field"] = "invalid_target"
    with pytest.raises(ValidationError) as exc_info:
        _ = NegationTaskConfig.parse_obj(negation_task_config)

    assert "target_field" in str(exc_info.value)
    assert "not the same as" in str(exc_info.value)


def test_negation_task_config_invalid_response_field(
    negation_task_config: dict,
) -> None:
    negation_task_config["response_formatter_config"][
        "response_field_name"
    ] = "invalid_response"
    with pytest.raises(ValidationError) as exc_info:
        _ = NegationTaskConfig.parse_obj(negation_task_config)

    assert "response_field" in str(exc_info.value)
    assert "not the same as" in str(exc_info.value)


def test_negation_task_config_invalid_evaluator_categories(
    negation_task_config: dict,
) -> None:
    # Checks that an error is raised if evaluator misses the null category
    negation_task_config["evaluator_config"]["categories"] = ["yes", "no"]

    with pytest.raises(ValidationError) as exc_info:
        _ = NegationTaskConfig.parse_obj(negation_task_config)

    assert "evaluator_config.categories" in str(exc_info.value)


def test_negation_task_config_invalid_renaming_map(negation_task_config: dict) -> None:
    negation_task_config["response_formatter_config"]["renaming_map"] = {"key": "value"}
    with pytest.raises(ValidationError) as exc_info:
        _ = NegationTaskConfig.parse_obj(negation_task_config)

    assert "Negation map" in str(exc_info.value)
