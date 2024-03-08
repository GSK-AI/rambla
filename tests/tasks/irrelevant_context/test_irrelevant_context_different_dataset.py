from typing import Callable
from unittest import mock

import numpy as np
from datasets import Dataset

from rambla.tasks.irrelevant_context import IrrelevantContextDiffDatasetTask

# flake8: noqa: N802


@mock.patch(
    "rambla.tasks.irrelevant_context.irrelevant_context_different_dataset.prepare_dataset"
)
@mock.patch("rambla.tasks.irrelevant_context.utils.prepare_dataset")
def test_IrrelevantContextDiffDatasetTask_run_task(
    mock_prepare_dataset,
    mock_prepare_dataset_mixer,
    irrelevant_context_different_dataset_task_config: dict,
    mock_flat_pubmedqa_dataset: Dataset,
    make_mock_llm: Callable,
):
    mock_prepare_dataset_mixer.return_value = mock_flat_pubmedqa_dataset
    mock_prepare_dataset.return_value = mock_flat_pubmedqa_dataset

    task = IrrelevantContextDiffDatasetTask.from_config(
        irrelevant_context_different_dataset_task_config
    )
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

    # Prompts
    for ii, prompt in enumerate(output_dataset["prompt"]):
        assert mock_flat_pubmedqa_dataset["question"][ii] in prompt
        assert mock_flat_pubmedqa_dataset["context"][ii] not in prompt

    # Checking whether the model was called with the right prompts
    for ii, prompt in enumerate(output_dataset["prompt"]):
        assert mock_llm.generate.call_args_list[ii].kwargs == {"prompt": prompt}
