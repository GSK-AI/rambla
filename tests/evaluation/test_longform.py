from typing import Dict, List
from unittest import mock

import pytest
from datasets import Dataset

from rambla.evaluation.longform import (
    LongformQAEvalComponent,
    ResponseQualityEvalComponent,
    ResponseQualityEvalConfig,
)

# flake8: noqa: N802


@pytest.mark.parametrize(
    ("predictions, references, metric_names, metric_kwargs, expected_results"),
    [
        (
            ["no", "yes"],
            ["no", "yes"],
            ["rouge"],
            None,
            {"rouge1": 1.0, "rouge2": 0.0, "rougeL": 1.0, "rougeLsum": 1.0},
        ),
        (
            ["no", "yes"],
            ["no", "yes"],
            ["rouge"],
            {"rouge": {"use_aggregator": False}},
            {
                "rouge1": [1.0, 1.0],
                "rouge2": [0.0, 0.0],
                "rougeL": [1.0, 1.0],
                "rougeLsum": [1.0, 1.0],
            },
        ),
        (
            ["this is dummy text"],
            ["this is dummy text"],
            ["rouge"],
            None,
            {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0, "rougeLsum": 1.0},
        ),
        (
            ["this is dummy text", "also dummy"],
            ["this is dummy text", "also dummy"],
            ["rouge"],
            None,
            {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0, "rougeLsum": 1.0},
        ),
        (
            ["this is dummy text"],
            ["this is also some dummy text"],
            ["rouge"],
            None,
            {"rouge1": 0.8, "rouge2": 0.5, "rougeL": 0.8, "rougeLsum": 0.8},
        ),
    ],
)
def test_test_LongformQAEvalComponent_run(  # noqa: N802
    predictions: List[str],
    references: List[str],
    metric_names: List[str],
    metric_kwargs: Dict[str, dict],
    expected_results: Dict[str, float],
):
    eval_obj = LongformQAEvalComponent(
        metric_names=metric_names,
        metric_kwargs=metric_kwargs,
    )
    output = eval_obj.run(predictions=predictions, references=references)

    assert output == expected_results


@mock.patch("rambla.evaluation.longform.token_counter")
def test_ResponseQualityEvalComponent_compute_lengths_single(mock_token_counter):
    expected_return_values = [3, 5]
    mock_token_counter.side_effect = expected_return_values

    field_name = "response"
    data = ["hey", "hello"]
    dataset = Dataset.from_dict({field_name: data})
    config = {"field_names": field_name, "encoding": "text-davinci-003"}

    component = ResponseQualityEvalComponent.from_config(config)
    output = component._compute_lengths(dataset[field_name])

    #
    assert output == expected_return_values
    assert mock_token_counter.call_args_list[0][0][0] == data[0]
    assert mock_token_counter.call_args_list[1][0][0] == data[1]

    assert mock_token_counter.call_args_list[0][0][1] == config["encoding"]
    assert mock_token_counter.call_args_list[1][0][1] == config["encoding"]


@mock.patch("rambla.evaluation.longform.token_counter")
def test_ResponseQualityEvalComponent_compute_lengths_single_in_list(
    mock_token_counter,
):
    expected_return_values = [3, 5]
    mock_token_counter.side_effect = expected_return_values

    field_name = "response"
    data = ["hey", "hello"]
    dataset = Dataset.from_dict({field_name: data})
    config = {"field_names": [field_name], "encoding": "text-davinci-003"}

    component = ResponseQualityEvalComponent.from_config(config)
    output = component._compute_lengths(dataset[field_name])

    #
    assert output == expected_return_values
    assert mock_token_counter.call_args_list[0][0][0] == data[0]
    assert mock_token_counter.call_args_list[1][0][0] == data[1]

    assert mock_token_counter.call_args_list[0][0][1] == config["encoding"]
    assert mock_token_counter.call_args_list[1][0][1] == config["encoding"]


@mock.patch("rambla.evaluation.longform.token_counter")
def test_ResponseQualityEvalComponent_evaluate(mock_token_counter):
    expected_return_values = [3, 5]
    mock_token_counter.side_effect = expected_return_values

    field_name = "response"
    data = ["hey", "hello"]
    dataset = Dataset.from_dict({field_name: data})
    config = {"field_names": field_name, "encoding": "text-davinci-003"}
    component = ResponseQualityEvalComponent.from_config(config)

    output = component.evaluate(dataset)

    #
    expected_output = {field_name: {"mean": 4, "median": 4, "std": 1.0}}
    assert output == expected_output


@mock.patch("rambla.evaluation.longform.token_counter")
def test_ResponseQualityEvalComponent_evaluate_multiple(mock_token_counter):
    expected_return_values = [3, 5, 7, 9]
    mock_token_counter.side_effect = expected_return_values

    field_name_0 = "response"
    data_0 = ["hey", "hello"]

    field_name_1 = "response_1"
    data_1 = ["dummy", "another_dummy"]

    dataset = Dataset.from_dict({field_name_0: data_0, field_name_1: data_1})
    config = {
        "field_names": [field_name_0, field_name_1],
        "encoding": "text-davinci-003",
    }

    component = ResponseQualityEvalComponent.from_config(config)

    output = component.evaluate(dataset)

    #
    expected_output = {
        field_name_0: {"mean": 4, "median": 4, "std": 1.0},
        field_name_1: {"mean": 8, "median": 8, "std": 1.0},
    }
    assert output == expected_output


def test_ResponseQualityEvalConfig_single_field_name():
    config = {
        "field_names": "single",
        "encoding": "text-davinci-003",
    }

    basemodel = ResponseQualityEvalConfig.parse_obj(config)

    assert basemodel.field_names == [config["field_names"]]


def test_ResponseQualityEvalConfig_multiple_field_names():
    config = {
        "field_names": ["multiple_0", "multiple_1"],
        "encoding": "text-davinci-003",
    }

    basemodel = ResponseQualityEvalConfig.parse_obj(config)

    assert basemodel.field_names == config["field_names"]
