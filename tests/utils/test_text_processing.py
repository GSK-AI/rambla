from typing import List
from unittest import mock

import pytest
from datasets import Dataset

from rambla.utils.text_processing import (
    SentenceTrimmer,
    SentenceTrimmerConfig,
    extract_first_response_instance,
    trim_text_by_number_of_sentences,
)

# flake8: noqa: N802


@pytest.mark.parametrize(
    "input_text, n_sentences, expected_output",
    [
        ("Single sentence", 1, "Single sentence"),
        ("Single sentence", 3, "Single sentence"),
        ("Single sentence", -1, "Single sentence"),
        ("Single sentence with fullstop.", 1, "Single sentence with fullstop."),
        ("Single sentence with fullstop.", 3, "Single sentence with fullstop."),
        ("Single sentence with fullstop.", -1, "Single sentence with fullstop."),
        ("Two sentences. Second!", 1, "Two sentences."),
        ("Two sentences. Second!", 2, "Two sentences. Second!"),
        ("Two sentences. Second!", 3, "Two sentences. Second!"),
        ("Two sentences. Second!", -1, "Two sentences. Second!"),
    ],
)
def test_trim_text_by_number_of_sentences(input_text, n_sentences, expected_output):
    output = trim_text_by_number_of_sentences(input_text, n_sentences)
    assert output == expected_output


def test_SentenceTrimmer():
    field_name = "question"
    n_sentences = 3

    return_items = iter(list("1234"))

    dataset = Dataset.from_dict({field_name: list("ABCD")})
    sentence_trimmer = SentenceTrimmer(field_name=field_name, n_sentences=n_sentences)

    #
    with mock.patch(
        "rambla.utils.text_processing.trim_text_by_number_of_sentences",
        side_effect=return_items,
    ) as mock_trim_function:
        output_dataset = sentence_trimmer.run(dataset)

    #
    assert output_dataset[f"untrimmed_{field_name}"] == dataset[field_name]
    assert output_dataset[field_name] == list("1234")
    call_args = mock_trim_function.call_args_list

    assert tuple(call_args[0])[0] == ("A", 3)
    assert tuple(call_args[1])[0] == ("B", 3)
    assert tuple(call_args[2])[0] == ("C", 3)
    assert tuple(call_args[3])[0] == ("D", 3)


@pytest.mark.parametrize(
    "input_text, expected_output, allowed_categories",
    [
        ("the answer is yes, and no", "yes", ["yes", "no", "maybe"]),
        ("the answer is no, and maybe", "no", ["yes", "no", "maybe"]),
        ("the answer is maybe, and no", "maybe", ["yes", "no", "maybe"]),
        (
            "the answer is maybe",
            None,
            ["yes", "no"],
        ),  # Checks allowed_responses works
        (
            "the answer isyes",
            None,
            ["yes", "no", "maybe"],
        ),  # Checks only full words extracted
        ("the answer isno", None, ["yes", "no", "maybe"]),
        ("the answer is yes and yes", "yes", ["yes", "no", "maybe"]),
    ],
)
def test_extract_first_response_instance(
    input_text: str, expected_output: str, allowed_categories: List[str]
) -> None:
    assert (
        extract_first_response_instance(input_text, allowed_categories)
        == expected_output
    )


@pytest.fixture
def sentence_trimmer_config_dict() -> dict:
    return {
        "field_name": "context",
        "n_sentences": 3,
    }


@pytest.fixture
def sentence_trimmer_config_basemodel(
    sentence_trimmer_config_dict: dict,
) -> SentenceTrimmerConfig:
    return SentenceTrimmerConfig.parse_obj(sentence_trimmer_config_dict)


def test_SentenceTrimmer_from_config_dict(sentence_trimmer_config_dict: dict):
    module_from_config = SentenceTrimmer.from_config(sentence_trimmer_config_dict)
    module_from_init = SentenceTrimmer(**sentence_trimmer_config_dict)

    for key in sentence_trimmer_config_dict.keys():
        assert getattr(module_from_config, key) == getattr(module_from_init, key)


def test_SentenceTrimmer_from_basemodel(
    sentence_trimmer_config_basemodel: SentenceTrimmerConfig,
):
    module_from_config = SentenceTrimmer.from_config(sentence_trimmer_config_basemodel)
    module_from_init = SentenceTrimmer(**sentence_trimmer_config_basemodel.dict())

    for key in sentence_trimmer_config_basemodel.dict().keys():
        assert getattr(module_from_config, key) == getattr(module_from_init, key)
