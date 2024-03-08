from typing import Callable, Sequence
from unittest import mock

import numpy as np
import pytest
from datasets import Dataset

from rambla.tasks.base import RunTaskReturnType
from rambla.tasks.mcqabaseline import mcqabaseline
from rambla.tasks.prompt_robustness import PromptRobustness
from rambla.text_mutation.mutators import CharacterLevelMutator
from rambla.text_mutation.operators import SwapCharacterOperator
from rambla.text_mutation.utils import is_punctuation
from rambla.text_mutation.word_validation import WordValidator
from rambla.utils.dataset import from_dict_to_dataset


@pytest.fixture
def mock_dataset() -> Dataset:
    dataset_dict = {
        "index": list(range(10)),
        "question": ["What is the capital of France?"] * 10,
        "context": ["Some random context with for use in the question"] * 10,
        "final_decision": [
            "yes",
            "yes",
            "no",
            "no",
            "no",
            "yes",
            "no",
            "yes",
            "yes",
            "no",
        ],
    }

    return from_dict_to_dataset(dataset_dict)


@pytest.fixture
def mcqa_task(
    mock_dataset: Dataset, task_config: dict
) -> mcqabaseline.MCQABaselineTask:
    with mock.patch.object(
        mcqabaseline,
        "prepare_dataset",
        return_value=mock_dataset,
    ):
        task = mcqabaseline.MCQABaselineTask.from_config(task_config)

    return task


@pytest.fixture
def word_validators() -> Sequence[WordValidator]:
    return [
        WordValidator(validation_func=lambda x: not str.isspace(x), mode="all"),
        WordValidator(validation_func=lambda x: not is_punctuation(x), mode="any"),
    ]


def test_prompt_robustness(
    make_mock_llm: Callable,
    mcqa_task: mcqabaseline.MCQABaselineTask,
    word_validators: Sequence[WordValidator],
) -> None:
    mock_dataset = mcqa_task.dataset

    # Return values set so first has 0 errors, second 1 error, third 3
    mod_map = {"yes": "no", "no": "yes"}

    return_values1 = mock_dataset["final_decision"].copy()

    return_values2 = return_values1.copy()
    return_values2[0] = mod_map[return_values2[0]]

    return_values3 = return_values2.copy()
    return_values3[1] = mod_map[return_values3[1]]

    all_responses = return_values1 + return_values2 + return_values3
    mock_model = make_mock_llm(all_responses)

    # Expected return values
    expected_f1_scores = np.array([1.0, 0.9, 0.8])
    expected_f1_change = np.array([-0.1, -0.2])

    mutator = CharacterLevelMutator(
        mutation_operators=[SwapCharacterOperator()],
        word_validators=word_validators,
    )
    mutation_schedule = [2, 4, 6]
    prompt_robustess = PromptRobustness(
        mcqa_task, mutator, mutation_schedule, "question"
    )

    # Call
    task_output = prompt_robustess.run_task(mock_model)

    # Checks

    # Checking the generated prompts are all different
    prompt_call1 = task_output.datasets["final_dataset_n2"]["prompt"]
    prompt_call2 = task_output.datasets["final_dataset_n4"]["prompt"]
    prompt_call3 = task_output.datasets["final_dataset_n6"]["prompt"]

    assert prompt_call2 != prompt_call1
    assert prompt_call3 != prompt_call2
    assert prompt_call1 != prompt_call3

    # Checking whether the model was called with the right prompts
    prompts = prompt_call1 + prompt_call2 + prompt_call3

    for ii, prompt in enumerate(prompts):
        assert mock_model.generate.call_args_list[ii].kwargs == {"prompt": prompt}

    # Check metrics saved as correct type
    assert isinstance(task_output.artifacts["results"]["f1"], list)
    assert isinstance(task_output.artifacts["results"]["f1"][0], float)
    assert isinstance(task_output.artifacts["results"]["confusion_matrix"], list)
    assert isinstance(
        task_output.artifacts["results"]["confusion_matrix"][0], np.ndarray
    )

    # Checks other artifacts saved as list of dictionaries
    assert isinstance(task_output.artifacts["label_encoder"], list)
    assert isinstance(task_output.artifacts["label_encoder"][0], dict)

    # Checks metrics are broadly correct (accounting for floating point errors)
    true_f1 = np.array(task_output.artifacts["results"]["f1"])
    assert np.allclose(true_f1, expected_f1_scores, atol=0.03)

    true_f1_change = np.array(task_output.artifacts["metrics_relative_change"]["f1"])
    assert np.allclose(true_f1_change, expected_f1_change, atol=0.03)

    # Checks different responses for each mutation schedule
    response_mutation1 = task_output.datasets["final_dataset_n2"]["response"]
    response_mutation2 = task_output.datasets["final_dataset_n4"]["response"]
    response_mutation3 = task_output.datasets["final_dataset_n6"]["response"]

    assert response_mutation2 != response_mutation1
    assert response_mutation3 != response_mutation2
    assert response_mutation1 != response_mutation3


def test_prompt_robustness_from_config(prompt_robustness_config: dict) -> None:
    prompt_robustness_task = PromptRobustness.from_config(prompt_robustness_config)

    assert isinstance(prompt_robustness_task.task, mcqabaseline.MCQABaselineTask)
    assert isinstance(prompt_robustness_task.mutator, CharacterLevelMutator)

    assert (
        prompt_robustness_task.mutation_schedule
        == prompt_robustness_config["mutation_schedule"]
    )
    assert (
        prompt_robustness_task.field_to_mutate
        == prompt_robustness_config["field_to_mutate"]
    )


@pytest.mark.parametrize(
    "artifacts_dict",
    [
        # Results data not formatted into dict of lists
        {
            "results": [{"f1": 0.1}, {"f1": 0.3}],
            "label_encoder": [{"yes": 1, "no": 0}, {"yes": 1, "no": 0}],
        },
        # Incorrect number of entries in list
        {
            "results": {"f1": [0.1, 0.3]},
            "label_encoder": [{"yes": 1, "no": 0}],
        },
        {
            "results": {"f1": [0.1]},
            "label_encoder": [{"yes": 1, "no": 0}, {"yes": 1, "no": 0}],
        },
    ],
)
def test_prompt_robustness_validate_outputs_invalid_artifacts(
    artifacts_dict: dict,
    mock_dataset: Dataset,
) -> None:
    invalid_data = {
        "metrics": None,
        "artifacts": artifacts_dict,
        "datasets": {
            "final_dataset_n1": mock_dataset,
            "final_dataset_n2": mock_dataset,
        },
        "other": None,
        "plots": None,
    }
    invalid_return_data = RunTaskReturnType.parse_obj(invalid_data)

    with pytest.raises(ValueError):
        _ = PromptRobustness.validate_outputs(invalid_return_data, 2)


def test_prompt_robustness_validate_outputs_invalid_datasets(
    mock_dataset: Dataset,
) -> None:
    # Invalid number of datasets
    invalid_data = {
        "metrics": None,
        "artifacts": {
            "results": {"f1": [0.1, 0.3]},
        },
        "datasets": {
            "final_dataset_n1": mock_dataset,
        },
        "other": None,
        "plots": None,
    }
    invalid_return_data = RunTaskReturnType.parse_obj(invalid_data)

    with pytest.raises(ValueError):
        _ = PromptRobustness.validate_outputs(invalid_return_data, 2)
