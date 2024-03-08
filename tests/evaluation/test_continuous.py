from typing import Callable

import numpy as np
import pytest
from datasets import Dataset
from pydantic import ValidationError

from rambla.evaluation.continuous import ContinuousEvalComponent, run_metric

# flake8: noqa: N802


@pytest.fixture
def mock_evaluator_config(tmpdir) -> dict:
    config = {
        "metric_names": ["mse", "mae", "r_squared", "stats_pointbiserialr"],
        "response_field": "predictions",
        "target_field": "targets",
    }
    return config


@pytest.fixture
def mock_text_to_text_dataset() -> Callable:
    dataset = Dataset.from_dict(
        {
            "index": [0, 1, 2],
            "text_1": [
                "This is some dummy text",
                "This is some dummy text",
                "This is some dummy text",
            ],
            "text_2": [
                "This is some useless text",
                "This might be some dummy text",
                "This is not some dummy text",
            ],
            "predictions": [1, 1, 0],
            "targets": [1, 0.6, 0],
        }
    )

    return dataset


@pytest.fixture
def expected_output() -> dict:
    output = {
        "mse": 0.05333332697550475,
        "mae": 0.13333332538604736,
        "r_squared": 0.684,
        "stats_pointbiserialr_statistic": 0.9176629354822471,
        "stats_pointbiserialr_pvalue": 0.2601469382930058,
    }
    return output


def test_ContinuousEvalComponent_evaluate(
    mock_evaluator_config, mock_text_to_text_dataset, expected_output
):
    evaluator = ContinuousEvalComponent.from_config(mock_evaluator_config)

    output = evaluator.evaluate(mock_text_to_text_dataset)

    for key, value in output["results"].items():
        assert np.isclose(value, expected_output[key])


def test_run_metric_compute_huggingface_metric(
    mock_text_to_text_dataset, expected_output
):
    output = run_metric(
        metric_name="mse",
        predictions=mock_text_to_text_dataset["predictions"],
        targets=mock_text_to_text_dataset["targets"],
    )

    assert np.isclose(output["mse"], expected_output["mse"])


def test_run_metric_compute_scipy_metric(mock_text_to_text_dataset, expected_output):
    output = run_metric(
        metric_name="stats_pointbiserialr",
        predictions=mock_text_to_text_dataset["predictions"],
        targets=mock_text_to_text_dataset["targets"],
    )

    assert np.isclose(
        output["stats_pointbiserialr_statistic"],
        expected_output["stats_pointbiserialr_statistic"],
    )
    assert np.isclose(
        output["stats_pointbiserialr_pvalue"],
        expected_output["stats_pointbiserialr_pvalue"],
    )


def test_run_metric_not_scipy(
    mock_text_to_text_dataset,
):
    with pytest.raises(ValueError) as exc_info:
        _ = run_metric(
            metric_name="stats_dummy_metric",
            predictions=mock_text_to_text_dataset["predictions"],
            targets=mock_text_to_text_dataset["targets"],
        )

    assert "not in supported scipy stats" in str(exc_info.value)


def test_run_metric_not_huggingface(
    mock_text_to_text_dataset,
):
    with pytest.raises(ValueError) as exc_info:
        _ = run_metric(
            metric_name="dummy_metric",
            predictions=mock_text_to_text_dataset["predictions"],
            targets=mock_text_to_text_dataset["targets"],
        )

    assert "not in huggingface evaluate library" in str(exc_info.value)
