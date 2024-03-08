import difflib
from typing import List

import pytest

from rambla.text_mutation import operators


def _get_extra_chars(input_text: str, output_text: str) -> List[str]:
    return [d for d in difflib.ndiff(input_text, output_text) if d[0] == "+"]


def test_swap_character_operator_all_lower() -> None:
    input_text = "sequence"

    operator = operators.SwapCharacterOperator(
        match_character_type=True,
        seed=123,
    )
    output = operator.transform(input_text)

    assert output != input
    assert output.islower()

    # Checks only one change made
    extra_chars = _get_extra_chars(input_text, output)
    assert len(extra_chars) == 1


def test_swap_character_operator_contains_upper() -> None:
    input_text = "Word"

    operator = operators.SwapCharacterOperator(
        match_character_type=True,
        seed=123,
    )
    output = operator.transform(input_text)

    assert output != input
    assert output[0].isupper()

    # Checks only one change made
    extra_chars = _get_extra_chars(input_text, output)
    assert len(extra_chars) == 1


def test_swap_character_operator_numeric() -> None:
    input_text = "3543"

    operator = operators.SwapCharacterOperator(
        match_character_type=True,
        seed=123,
    )
    output = operator.transform(input_text)

    assert output != input
    assert output.isdigit()

    # Checks only one change made
    extra_chars = _get_extra_chars(input_text, output)
    assert len(extra_chars) == 1


def test_swap_character_operator_same_seed() -> None:
    input_text = "random"

    base_operator = operators.SwapCharacterOperator(
        match_character_type=True,
        seed=123,
    )
    original_output = base_operator.transform(input_text)

    operator_same_seed = operators.SwapCharacterOperator(
        match_character_type=True,
        seed=123,
    )
    same_seed_output = operator_same_seed.transform(input_text)

    assert same_seed_output == original_output


def test_swap_character_operator_different_seed() -> None:
    input_text = "random"

    base_operator = operators.SwapCharacterOperator(
        match_character_type=True,
        seed=123,
    )
    original_output = base_operator.transform(input_text)

    operator_diff_seed = operators.SwapCharacterOperator(
        match_character_type=True,
        seed=42,
    )
    diff_seed_output = operator_diff_seed.transform(input_text)

    assert diff_seed_output != original_output


@pytest.mark.parametrize("input_text, expected", [("Word", "word"), ("tG434", "tg434")])
def test_switch_case_operator_mode_upper(input_text: str, expected: str) -> None:
    operator = operators.SwitchCaseOperator(case_mode="upper", seed=123)
    output_text = operator.transform(input_text)

    assert output_text == expected


@pytest.mark.parametrize("input_text", ["word", "rs1006737"])
def test_switch_case_operator_mode_lower(input_text: str) -> None:
    operator = operators.SwitchCaseOperator(case_mode="lower", seed=123)
    output_text = operator.transform(input_text)

    # Checks a single character is now uppercase
    uppercases = [letter.isupper() for letter in output_text]
    assert sum(uppercases) == 1


@pytest.mark.parametrize("input_text", ["Text", "text", "rs1006737", "mRNA"])
def test_switch_case_operator_mode_both(input_text: str) -> None:
    # Counts uppercase characters in input
    upper_input = [letter.isupper() for letter in input_text]

    operator = operators.SwitchCaseOperator(case_mode="lower", seed=123)
    output_text = operator.transform(input_text)

    # Counts uppercase characters in input
    upper_output = [letter.isupper() for letter in output_text]

    assert upper_input != upper_output

    # Checks only single character changed
    disagreement = [
        letter != upper_output[idx] for idx, letter in enumerate(upper_input)
    ]
    assert sum(disagreement) == 1


@pytest.mark.parametrize(
    "invalid_input_text, case_mode",
    [("text", "upper", "TEXT", "lower", "1234", "both")],
)
def switch_case_operator_invalid_input(invalid_input_text: str, case_mode: str) -> None:
    operator = operators.SwitchCaseOperator(
        case_mode=case_mode, seed=123  # type: ignore
    )

    with pytest.raises(ValueError):
        _ = operator.transform(invalid_input_text)


@pytest.mark.parametrize(
    "input_text, insert_character_opts",
    [
        ("Random piece of text", ["\n", "\t", " "]),
        ("Random piece\nof text", ["\n", "\t", " "]),
        ("Random piece of text", ["a", "b", "c", "d"]),
        ("Random piece\nof text", ["a", "b", "c", "d"]),
        ("Random piece of text", ["1", "2", "3", "4"]),
        ("Random piece\nof text", ["1", "2", "3", "4"]),
    ],
)
def test_insert_character_between_words_operator(
    input_text: str, insert_character_opts: List[str]
) -> None:
    operator = operators.InsertCharacterBetweenWordsOperator(
        insert_character_opts=insert_character_opts, seed=123
    )
    output_text = operator.transform(input_text)

    assert output_text != input_text

    # Checks only one extra character added
    extra_chars = _get_extra_chars(input_text, output_text)
    assert len(extra_chars) == 1
    assert extra_chars[0][2] in insert_character_opts

    # Checks whole words remain intact
    assert all([word in output_text for word in input_text.split(" ")])
