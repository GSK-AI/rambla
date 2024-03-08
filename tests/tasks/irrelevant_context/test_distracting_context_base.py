from typing import Callable
from unittest import mock

import numpy as np
import pytest
from datasets import Dataset

from rambla.tasks.irrelevant_context.distracting_context import DistractingContextTask

# flake8: noqa: N802


@pytest.fixture
def question_dataset() -> Dataset:
    question_column = list("ABCD")
    other_column = list("abcd")

    input_dataset = Dataset.from_dict(
        {
            "question": question_column,
            "other": other_column,
        }
    )
    return input_dataset


@mock.patch("rambla.tasks.irrelevant_context.distracting_context.prepare_dataset")
def test_DistractingContextTask_run_task(
    mock_prepare_dataset,
    distracting_context_task_config: dict,
    mock_flat_pubmedqa_dataset: Dataset,
    make_mock_llm: Callable,
):
    mock_prepare_dataset.return_value = mock_flat_pubmedqa_dataset
    task = DistractingContextTask.from_config(distracting_context_task_config)
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

    # Prompts
    for ii, prompt in enumerate(output_dataset["prompt"]):
        assert mock_flat_pubmedqa_dataset["question"][ii] in prompt
        assert mock_flat_pubmedqa_dataset["context"][ii] in prompt

    # Checking whether the model was called with the right prompts
    for ii, prompt in enumerate(output_dataset["prompt"]):
        assert mock_llm.generate.call_args_list[ii].kwargs == {"prompt": prompt}
