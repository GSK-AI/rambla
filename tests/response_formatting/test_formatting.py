from typing import Dict, List
from unittest import mock

import pytest
from datasets import Dataset

from rambla.response_formatting.formatting import (
    MappingResponseFormatter,
    MCQAResponseFormatter,
    MCQAResponseFormatterConfig,
    process_string,
)
from rambla.utils.dataset import from_dict_to_dataset

# flake8: noqa: E501


@pytest.mark.parametrize(
    "input_string, expected_output",
    [
        ("yes", "yes"),
        (" yes", "yes"),
        (" yes ", "yes"),
        (" yes   ", "yes"),
        (" yes  . ", "yes"),
        ("Yes", "yes"),
    ],
)
def test_process_string(input_string, expected_output):
    output = process_string(input_string)
    assert output == expected_output


@pytest.fixture
def mcqa_response_config_fixture():
    return MCQAResponseFormatterConfig(
        response_field_name="response",
        categories=["yes", "no", "maybe"],
        string_formatter_name="basic",
        null_category="null",
    )


def test_MCQAResponseFormatter_from_config(mcqa_response_config_fixture):
    response_formatter = MCQAResponseFormatter.from_config(mcqa_response_config_fixture)

    assert (
        response_formatter.response_field_name
        == mcqa_response_config_fixture.response_field_name
    )
    assert response_formatter.categories == mcqa_response_config_fixture.categories
    assert (
        response_formatter.null_category == mcqa_response_config_fixture.null_category
    )
    assert response_formatter.string_formatter == process_string


def test_MCQAResponseFormatter_format_column(mcqa_response_config_fixture):
    response_formatter = MCQAResponseFormatter.from_config(mcqa_response_config_fixture)
    mock_string_formatter = mock.MagicMock(side_effect=list("1234"))

    response_formatter.string_formatter = mock_string_formatter

    column = list("ABCD")
    output = response_formatter._format_column(column)

    assert output == list("1234")

    call_args = mock_string_formatter.call_args_list
    assert len(call_args) == 4
    assert [call_args[ii][0][0] for ii in range(4)] == column


@pytest.mark.parametrize(
    "input_list, expected_output",
    [
        (["hello"], ["null"]),
        (["yes", "hey"], ["yes", "null"]),
        (["yes", "no"], ["yes", "no"]),
    ],
)
def test_MCQAResponseFormatter_process_column(
    input_list, expected_output, mcqa_response_config_fixture
):
    response_formatter = MCQAResponseFormatter.from_config(mcqa_response_config_fixture)
    output = response_formatter._process_column(input_list)
    assert output == expected_output


def test_MCQAResponseFormatter_format(mcqa_response_config_fixture):
    response_formatter = MCQAResponseFormatter.from_config(mcqa_response_config_fixture)
    mock_format_column = mock.MagicMock()
    mock_format_column.return_value = list("1234")
    response_formatter._format_column = mock_format_column

    mock_process_column = mock.MagicMock()
    mock_process_column.return_value = list("abcd")
    response_formatter._process_column = mock_process_column

    response_data = list("ABCD")
    dataset = Dataset.from_dict({"response": response_data})

    #
    output_dataset = response_formatter.format(dataset)

    #
    mock_format_column.assert_called_once()
    mock_format_column.assert_called_with(response_data)

    mock_process_column.assert_called_once()
    mock_process_column.assert_called_with(list("1234"))

    assert output_dataset["unformatted_response"] == response_data
    assert output_dataset["response"] == list("abcd")

    assert output_dataset.features.keys() == set(["response", "unformatted_response"])


@pytest.mark.parametrize(
    "input_list, renaming_map, expected",
    [
        (
            ["yes", "no", "other", "maybe", "yes"],
            {"yes": "no", "no": "yes"},
            ["no", "yes", "null", "null", "no"],
        ),
        (
            ["yes", "no", "other", "maybe", "yes"],
            {"yes": "no", "no": "yes", "maybe": "maybe"},
            ["no", "yes", "null", "maybe", "no"],
        ),
    ],
)
def test_negation_response_formatter_negate_column(
    input_list: List[str],
    renaming_map: Dict[str, str],
    expected: List[str],
) -> None:
    null_category = "null"

    formatter = MappingResponseFormatter(
        response_field_name="",
        renaming_map=renaming_map,
        string_formatter=mock.MagicMock(),
        response_extractor=mock.MagicMock(),
        null_category=null_category,
    )

    assert formatter._negate_column(input_list) == expected


@pytest.fixture
def mock_negation_dataset() -> Dataset:
    dataset_dict = {
        "index": [0, 1, 2, 3, 4],
        "response": ["the answer is Yes!", " no", "I'm not sure", "maybe", "yes"],
    }

    return from_dict_to_dataset(dataset_dict)


def test_negation_response_formatter_format(
    mock_negation_dataset: Dataset,
) -> None:
    formatter = MappingResponseFormatter(
        response_field_name="response",
        renaming_map={"yes": "no", "no": "yes"},
        null_category="null",
    )
    processed_dataset = formatter.format(mock_negation_dataset)

    assert "response" in processed_dataset.features.keys()
    assert "unformatted_response" in processed_dataset.features.keys()

    assert processed_dataset["response"] == ["no", "yes", "null", "null", "no"]
