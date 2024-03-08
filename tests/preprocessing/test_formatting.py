from typing import List

import numpy as np
import pytest
from datasets import Dataset

from rambla.preprocessing.formatting import ScalingFormatter

# flake8: noqa: E501


@pytest.fixture
def mock_scaling_formater_config() -> dict:
    config = {
        "label_field_name": "labels",
        "max_scale": 2,
        "min_scale": -1,
    }
    return config


@pytest.fixture
def mock_scaling_formater_error_config() -> dict:
    config = {
        "label_field_name": "labels",
        "max_scale": 0,
        "min_scale": 1,
    }
    return config


@pytest.mark.parametrize(
    "index, labels",
    [
        (
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 20],
        ),
        (
            [0, 1, 2, 3, 4, 5, 6],
            ["0", "1", "2", "3", "4", "5", "20"],
        ),
    ],
)
def test_ScalingFormatter_format(
    mock_scaling_formater_config: dict,
    index: List[str],
    labels: List[str | int | float],
):
    scaler = ScalingFormatter.from_config(mock_scaling_formater_config)

    mock_continuous_dataset = Dataset.from_dict(
        {
            "index": index,
            "labels": labels,
        }
    )

    output_dataset = scaler.format(mock_continuous_dataset)

    expected_output = [-1.0, -0.85, -0.7, -0.55, -0.3999999999999999, -0.25, 2.0]

    for n in range(len(output_dataset)):
        assert np.isclose(output_dataset["labels"][n], expected_output[n])


@pytest.mark.parametrize(
    "index, labels",
    [
        (
            [0, 1, 2, 3, 4, 5, 6],
            ["0", "1", "2", "3", "4", "5", "dummy label"],
        ),
    ],
)
def test_ScalingFormatter_format_error(
    mock_scaling_formater_config: dict,
    index: List[str],
    labels: List[str | int | float],
):
    scaler = ScalingFormatter.from_config(mock_scaling_formater_config)

    mock_continuous_dataset = Dataset.from_dict(
        {
            "index": index,
            "labels": labels,
        }
    )

    with pytest.raises(ValueError) as exc_info:
        _ = scaler.format(mock_continuous_dataset)

        assert "could not convert string to float" in str(exc_info.value)
        assert "dummy label" in str(exc_info.value)


def test_error_ScalingFormatterConfig(mock_scaling_formater_error_config: dict):
    with pytest.raises(ValueError) as exc_info:
        _ = ScalingFormatter.from_config(mock_scaling_formater_error_config)

    assert "max_scale value" in str(exc_info.value)


def test_error_ScalingFormatter():
    with pytest.raises(ValueError) as exc_info:
        _ = ScalingFormatter(label_field_name="labels", max_scale=0, min_scale=1)

    assert "max_scale value" in str(exc_info.value)
