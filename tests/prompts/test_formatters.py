from typing import Dict, List
from unittest import mock

import pytest
from datasets import Dataset
from pydantic import ValidationError

from rambla.prompts.formatters import (
    ColumnPromptFormatter,
    ColumnPromptFormatterConfig,
    ExamplesPromptFormatter,
    ExamplesPromptFormatterConfig,
    format_template,
    validate_dataset_and_template,
)

# flake8: noqa: N802


def test_format_template(mock_dataset: Dataset):
    entry = mock_dataset[0]
    template = "Sample template with {question} and {context} and {answer} to fill in."
    var_map = {"question": "question", "context": "context", "final_decision": "answer"}
    output_prompt = format_template(template, var_map, entry)
    expected_prompt = template.format(
        question=entry["question"],
        context=entry["context"],
        answer=entry["final_decision"],
    )
    assert output_prompt == expected_prompt


def test_validate_dataset_and_template(
    placeholders: List[str], var_map: Dict[str, str], mock_dataset: Dataset
):
    validate_dataset_and_template(placeholders, var_map, mock_dataset, "prompt")


def test_validate_dataset_and_template_prompt_in_columns(
    placeholders: List[str], var_map: dict[str, str], mock_dataset: Dataset
):
    prompt_column_name = "prompt"
    dataset = mock_dataset.add_column(prompt_column_name, list("ABCD"))
    with pytest.raises(ValueError) as exc_info:
        validate_dataset_and_template(
            placeholders, var_map, dataset, prompt_column_name
        )
    assert f"Field called {prompt_column_name} already exists in dataset" in str(
        exc_info.value
    )


def test_validate_dataset_and_template_varmap_values(
    placeholders: List[str], var_map: dict[str, str], mock_dataset: Dataset
):
    prompt_column_name = "prompt"
    dataset = mock_dataset.rename_column("question", "new_question")
    with pytest.raises(ValueError) as exc_info:
        validate_dataset_and_template(
            placeholders, var_map, dataset, prompt_column_name
        )
    assert "One or more entries from" in str(exc_info.value)


def test_validate_dataset_and_template_varmap_keys(
    placeholders: List[str], var_map: dict[str, str], mock_dataset: Dataset
):
    prompt_column_name = "prompt"
    placeholders = placeholders[:1]
    with pytest.raises(ValueError) as exc_info:
        validate_dataset_and_template(
            placeholders, var_map, mock_dataset, prompt_column_name
        )
    assert "Mismatch between found placeholders" in str(exc_info.value)


def test_ColumnPromptFormatterConfig_no_placeholders(
    column_prompt_formatter_config: dict,
):
    template_with_no_placeholders = "template with no placeholders"
    column_prompt_formatter_config["template"] = template_with_no_placeholders
    with pytest.raises(ValidationError) as exc_info:
        ColumnPromptFormatterConfig.parse_obj(column_prompt_formatter_config)

    assert "Found no placeholders" in str(exc_info.value)


def test_ColumnPromptFormatterConfig_duplicates(column_prompt_formatter_config: dict):
    template_with_duplicates = "template with two {question} {question} {context}"
    column_prompt_formatter_config["template"] = template_with_duplicates
    with pytest.raises(ValidationError) as exc_info:
        ColumnPromptFormatterConfig.parse_obj(column_prompt_formatter_config)

    assert "Found duplicates" in str(exc_info.value)


def test_column_prompt_formatter_from_config() -> None:
    mock_config = {
        "template": "mock template {value}",
        "var_map": {"var": "value"},
        "index_field": "index",
    }
    config = ColumnPromptFormatterConfig.parse_obj(mock_config)
    formatter = ColumnPromptFormatter.from_config(config)

    assert formatter.template == mock_config["template"]
    assert formatter.var_map == mock_config["var_map"]
    assert formatter.index_field == mock_config["index_field"]
    assert not formatter.allow_duplicates


@mock.patch("rambla.prompts.formatters.format_template")
def test_ColumnPromptFormatter_format_instance(
    mock_format_template, column_prompt_formatter_config: dict, mock_dataset: Dataset
):
    formatter = ColumnPromptFormatter.from_config(column_prompt_formatter_config)
    template = column_prompt_formatter_config["template"]
    var_map = column_prompt_formatter_config["var_map"]
    entry = mock_dataset[0]
    mock_format_template.return_value = template.format(
        question=entry["question"], context=entry["context"]
    )

    output = formatter.format_instance(entry)
    assert mock_format_template.called_with(template, var_map, entry)


def test_ColumnPromptFormatter_format(
    column_prompt_formatter_config: dict, mock_dataset: Dataset
):
    formatter = ColumnPromptFormatter.from_config(column_prompt_formatter_config)
    template = column_prompt_formatter_config["template"]
    output_dataset = formatter.format(mock_dataset, "prompt")
    expected_output_prompts = [
        template.format(question=entry["question"], context=entry["context"])
        for entry in mock_dataset
    ]
    assert output_dataset["prompt"] == expected_output_prompts


@mock.patch("rambla.prompts.formatters.validate_dataset_and_template")
def test_ColumnPromptFormatter_format_calls(
    mock_validate_dataset_and_template,
    column_prompt_formatter_config: dict,
    mock_dataset: Dataset,
):
    with mock.patch.object(
        ColumnPromptFormatter, "format_instance", return_value="a string"
    ) as mock_format_instance:
        formatter = ColumnPromptFormatter.from_config(column_prompt_formatter_config)
        output_dataset = formatter.format(
            mock_dataset, prompt_field_name="prompt_field_name"
        )

    expected_prompt = ["a string"] * len(mock_dataset)

    assert output_dataset["prompt_field_name"] == expected_prompt

    # repeat for ExamplePromptFormatter


def test_ExamplesPromptFormatterConfig_no_placeholders_in_examples_template(
    examples_prompt_formatter_config: dict,
):
    template_with_no_placeholders = "template with no placeholders"
    examples_prompt_formatter_config[
        "examples_template"
    ] = template_with_no_placeholders
    with pytest.raises(ValidationError) as exc_info:
        ExamplesPromptFormatterConfig.parse_obj(examples_prompt_formatter_config)
    assert "Found no placeholders" in str(exc_info.value)


def test_ExamplesPromptFormatterConfig_no_placeholders_in_final_question_template(
    examples_prompt_formatter_config: dict,
):
    template_with_no_placeholders = "template with no placeholders"
    examples_prompt_formatter_config[
        "final_question_template"
    ] = template_with_no_placeholders
    with pytest.raises(ValidationError) as exc_info:
        ExamplesPromptFormatterConfig.parse_obj(examples_prompt_formatter_config)
    assert "Found no placeholders" in str(exc_info.value)


def test_ExamplesPromptFormatterConfig_placeholders_in_intro_template(
    examples_prompt_formatter_config: dict,
):
    template_with_placeholders = "template with {placeholders}"
    examples_prompt_formatter_config["intro_template"] = template_with_placeholders
    with pytest.raises(ValidationError) as exc_info:
        ExamplesPromptFormatterConfig.parse_obj(examples_prompt_formatter_config)
    assert "Found placeholders" in str(exc_info.value)


def test_ExamplesPromptFormatterConfig_duplicates_in_examples_template(
    examples_prompt_formatter_config: dict,
):
    template_with_duplicates = "template with two {question} {question} {context}"
    examples_prompt_formatter_config["examples_template"] = template_with_duplicates
    with pytest.raises(ValidationError) as exc_info:
        ExamplesPromptFormatterConfig.parse_obj(examples_prompt_formatter_config)

    assert "Found duplicates" in str(exc_info.value)


def test_ExamplesPromptFormatterConfig_duplicates_in_final_question_template(
    examples_prompt_formatter_config: dict,
):
    template_with_duplicates = "template with two {question} {question} {context}"
    examples_prompt_formatter_config[
        "final_question_template"
    ] = template_with_duplicates
    with pytest.raises(ValidationError) as exc_info:
        ExamplesPromptFormatterConfig.parse_obj(examples_prompt_formatter_config)

    assert "Found duplicates" in str(exc_info.value)


def test_examples_prompt_formatter_from_config(examples_prompt_formatter_config):
    mock_config = examples_prompt_formatter_config
    config = ExamplesPromptFormatterConfig.parse_obj(mock_config)
    formatter = ExamplesPromptFormatter.from_config(config)

    assert formatter.intro_template == mock_config["intro_template"]
    assert formatter.examples_template == mock_config["examples_template"]
    assert formatter.final_question_template == mock_config["final_question_template"]
    assert formatter.var_map == mock_config["var_map"]
    assert formatter.index_field == mock_config["index_field"]
    assert formatter.target_field == mock_config["target_field"]
    assert formatter.allow_duplicates == False


def test_examples_prompt_formatter_from_config_no_intro_template(
    examples_prompt_formatter_config,
):
    mock_config = examples_prompt_formatter_config
    del mock_config["intro_template"]
    config = ExamplesPromptFormatterConfig.parse_obj(mock_config)
    formatter = ExamplesPromptFormatter.from_config(config)

    assert formatter.intro_template == ""


def test_var_map_without_answer(examples_prompt_formatter_config):
    var_map = {"a": "A", "b": "B", "remove_this_field": "response"}
    examples_prompt_formatter_config["var_map"] = var_map
    examples_prompt_formatter_config["target_field"] = "remove_this_field"
    formatter = ExamplesPromptFormatter.from_config(examples_prompt_formatter_config)
    output_var_map = formatter.var_map_without_answer

    expected_output_var_map = {"a": "A", "b": "B"}

    assert output_var_map == expected_output_var_map


def test_extract_example_data_simple_case(
    examples_prompt_formatter_config, mock_dataset
):
    formatter = ExamplesPromptFormatter.from_config(examples_prompt_formatter_config)

    entry = mock_dataset[0]
    ouput_entry = formatter._extract_example_data(entry, mock_dataset)

    expected_entry = entry
    expected_entry[
        examples_prompt_formatter_config["examples_column_name"]
    ] = mock_dataset[:3]

    assert ouput_entry == expected_entry


@mock.patch(
    "rambla.prompts.formatters.format_template", side_effect=tuple(list("ABCD"))
)
def test_ExamplesPromptFormatter_format_instance(
    mock_format_template, examples_prompt_formatter_config: dict, mock_dataset: Dataset
):
    formatter = ExamplesPromptFormatter.from_config(examples_prompt_formatter_config)

    entry = mock_dataset[0]
    entry[
        examples_prompt_formatter_config["examples_column_name"]
    ] = mock_dataset.select(range(3))

    output = formatter.format_instance(entry)
    expected_output = examples_prompt_formatter_config["intro_template"] + "ABCD"

    assert output == expected_output


def test_ExamplesPromptFormatter_format_all_together(
    examples_prompt_formatter_config: dict, mock_dataset: Dataset
):
    formatter = ExamplesPromptFormatter.from_config(examples_prompt_formatter_config)
    intro_template = examples_prompt_formatter_config["intro_template"]
    examples_template = examples_prompt_formatter_config["examples_template"]
    final_question_template = examples_prompt_formatter_config[
        "final_question_template"
    ]

    output_dataset = formatter.format(mock_dataset, "prompt")

    first_prompt = (
        intro_template
        + "".join(
            [
                examples_template.format(
                    question=entry["question"],
                    context=entry["context"],
                    answer=entry["final_decision"],
                )
                for entry in mock_dataset.select([0, 1, 2])
            ]
        )
        + final_question_template.format(
            question=mock_dataset[0]["question"], context=mock_dataset[0]["context"]
        )
    )

    second_prompt = (
        intro_template
        + "".join(
            [
                examples_template.format(
                    question=entry["question"],
                    context=entry["context"],
                    answer=entry["final_decision"],
                )
                for entry in mock_dataset.select([3, 2, 1])
            ]
        )
        + final_question_template.format(
            question=mock_dataset[1]["question"], context=mock_dataset[1]["context"]
        )
    )

    third_prompt = (
        intro_template
        + "".join(
            [
                examples_template.format(
                    question=entry["question"],
                    context=entry["context"],
                    answer=entry["final_decision"],
                )
                for entry in mock_dataset.select([0, 2, 3])
            ]
        )
        + final_question_template.format(
            question=mock_dataset[2]["question"], context=mock_dataset[2]["context"]
        )
    )

    fourth_prompt = (
        intro_template
        + "".join(
            [
                examples_template.format(
                    question=entry["question"],
                    context=entry["context"],
                    answer=entry["final_decision"],
                )
                for entry in mock_dataset.select([3, 1, 2])
            ]
        )
        + final_question_template.format(
            question=mock_dataset[3]["question"], context=mock_dataset[3]["context"]
        )
    )

    assert output_dataset["prompt"] == [
        first_prompt,
        second_prompt,
        third_prompt,
        fourth_prompt,
    ]
