from pathlib import Path
from typing import Callable
from unittest import mock

import numpy as np
import pytest
from datasets import Dataset

from rambla.evaluation.continuous import ContinuousEvalComponent
from rambla.evaluation.shortform import MCQAEvalComponent
from rambla.response_formatting.formatting import MappingResponseFormatter
from rambla.tasks.base import RunTaskReturnType
from rambla.text_to_text_tasks.text_to_text_similarity_eval import (
    TextToTextSimilarityEvaluation,
)
from tests.conftest import hf_datasets_are_same

# flake8: noqa: N802


@pytest.fixture
def continuous_evaluator_config(
    response_field_name: str,
    label_field: str,
) -> dict:
    return {
        "response_field": response_field_name,
        "metric_names": ["sklearn_roc_auc_score"],
        "target_field": label_field,
    }


@pytest.fixture
def task_config(
    dataset_config: dict,
    target_formatter_config: dict,
    continuous_evaluator_config: dict,
) -> dict:
    return {
        "dataset_config": dataset_config,
        "target_formatter_config": target_formatter_config,
        "evaluator_config": continuous_evaluator_config,
    }


@pytest.fixture
def text_to_text_evaluation_config(tmpdir) -> dict:
    # dataset="dummy"
    response = "response"
    response_categories = ["yes", "no", "null"]
    evaluator_categories = ["1", "0", "null"]
    final_decision = "response"

    dataset_config = {
        "name": "dummy",
        "params": {"path": "dummy", "subset": "dummy", "split": "dummy"},
    }

    renaming_map = {"yes": "1", "no": "0", "null": "null"}
    response_formatter_config = {
        "response_field_name": response,
        "renaming_map": renaming_map,
        "string_formatter_name": "basic",
        "null_category": "null",
        "categories": response_categories,
    }

    evaluator_config = {
        "name": "shortform",
        "params": {
            "categories": evaluator_categories,
            "response_field": response,
            "target_field": final_decision,
            "metric_names": ["recall", "f1", "precision"],
        },
    }

    config = {
        "dataset_config": dataset_config,
        "response_formatter_config": response_formatter_config,
        "evaluator_config": evaluator_config,
    }

    return config


@pytest.mark.fileio
@mock.patch("rambla.text_to_text_tasks.text_to_text_similarity_eval.prepare_dataset")
def test_TextToTextSimilarityEvaluation_from_config(
    mock_prepare_dataset,
    text_to_text_evaluation_config,
):
    mock_prepare_dataset.return_value = "dummy"
    text_to_text_similarity_evaluation_from_config = (
        TextToTextSimilarityEvaluation.from_config(text_to_text_evaluation_config)
    )

    # From init
    text_to_text_similarity_evaluation = TextToTextSimilarityEvaluation(
        dataset=mock_prepare_dataset.return_value,
        response_formatter=MappingResponseFormatter.from_config(
            text_to_text_evaluation_config["response_formatter_config"]
        ),
        evaluator=MCQAEvalComponent.from_config(
            text_to_text_evaluation_config["evaluator_config"]["params"]
        ),
    )

    # Assert
    assert text_to_text_similarity_evaluation_from_config.dataset == "dummy"
    assert (
        text_to_text_similarity_evaluation_from_config.dataset
        == text_to_text_similarity_evaluation.dataset
    )
    assert (
        text_to_text_similarity_evaluation_from_config.response_formatter.__dict__
        == text_to_text_similarity_evaluation.response_formatter.__dict__
    )

    evaluator_config_dict = (
        text_to_text_similarity_evaluation_from_config.evaluator.__dict__
    )
    evaluator_init_dict = text_to_text_similarity_evaluation.evaluator.__dict__

    for key in evaluator_config_dict:
        if key == "metrics":
            assert evaluator_config_dict[key].keys() == evaluator_init_dict[key].keys()
        else:
            assert evaluator_config_dict[key] == evaluator_init_dict[key]


def test_TextToTextSimilarityEvaluation_run(
    text_to_text_evaluation_config: dict,
):
    # Assign
    input_dataset = Dataset.from_dict(
        {
            "index": [0, 1],
            "text_1": ["ABCD", "EFGH"],
            "text_2": ["abcd", "efgh"],
        }
    )

    mock_t2tcomponent = mock.MagicMock()
    mock_t2tcomponent.run.return_value = input_dataset.add_column(
        "response", ["Yes abcd", "No efgh"]
    )

    final_dataset = input_dataset.add_column(
        "unformatted_response", ["Yes abcd", "No efgh"]
    ).add_column("response", ["1", "0"])

    final_artifacts = {
        "results": {"recall": 1.0, "f1": 1.0, "precision": 1.0},
        "label_encoder": {"1": 1, "0": 0, "null": 2},
    }

    response_formatter = MappingResponseFormatter.from_config(
        text_to_text_evaluation_config["response_formatter_config"]
    )

    evaluator = MCQAEvalComponent.from_config(
        text_to_text_evaluation_config["evaluator_config"]["params"]
    )

    # Run
    similarity_task = TextToTextSimilarityEvaluation(
        dataset=input_dataset,
        response_formatter=response_formatter,
        evaluator=evaluator,
    )

    output = similarity_task.run_task(mock_t2tcomponent)

    # Expected
    expected_runtask_output = RunTaskReturnType(
        metrics={
            "recall": 1.0,
            "f1": 1.0,
            "precision": 1.0,
        },
        artifacts=final_artifacts,
        datasets={"final_dataset": final_dataset},
        other=None,
        artifact_storing_format="json",
        plots=None,
        dictionaries={"label_encoder": final_artifacts["label_encoder"]},
    )

    # Assert
    fields = ["metrics", "artifacts", "other", "artifact_storing_format"]
    for field in fields:
        assert getattr(output, field) == getattr(expected_runtask_output, field)

    assert hf_datasets_are_same(
        output.datasets["final_dataset"],
        expected_runtask_output.datasets["final_dataset"],
    )


def test_TextToTextSimilarityEvaluation_run_task(
    task_config: dict,
    mock_text_to_text_dataset: Dataset,
    make_mock_embeddings_component: Callable,
):
    evaluator = ContinuousEvalComponent.from_config(task_config["evaluator_config"])
    formatter = MappingResponseFormatter.from_config(
        task_config["target_formatter_config"]
    )
    task = TextToTextSimilarityEvaluation(
        dataset=mock_text_to_text_dataset,
        evaluator=evaluator,
        response_formatter=formatter,
    )

    #
    return_values = np.array(range(40)).reshape(-1, 2)
    component = make_mock_embeddings_component(return_values)

    #
    task_output = task.run_task(component)

    #
    text_1_values, text_2_values = return_values[:10, :], return_values[10:, :]
    expected_responses = np.diag(text_1_values @ text_2_values.T).tolist()
    assert task_output.datasets["final_dataset"]["response"] == expected_responses
