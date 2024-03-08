import string
from typing import List

import pytest

from rambla.text_mutation import utils


@pytest.mark.parametrize(
    "character, expected_response",
    [("±", True), ("£", True), ("®", True), ("a", False), ("G", False), ("7", False)],
)
def test_is_punctuation(character: str, expected_response: bool) -> None:
    assert utils.is_punctuation(character) == expected_response


def test_is_punctuation_invalid_input() -> None:
    with pytest.raises(ValueError) as exc_info:
        _ = utils.is_punctuation("multicharacter_string")

    assert "single character" in str(exc_info)


@pytest.mark.parametrize(
    "input_text, expected", [(" ", False), ("\n", False), ("A", True), ("A ", True)]
)
def test_is_not_space(input_text: str, expected: bool) -> None:
    pass


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("Random text", ["Random", " ", "text"]),
        ("Text with\nwhitespace", ["Text", " ", "with", "\n", "whitespace"]),
        ("Text with\twhitespace", ["Text", " ", "with", "\t", "whitespace"]),
        ("Text with\rwhitespace", ["Text", " ", "with", "\r", "whitespace"]),
        ("Text with\x0bwhitespace", ["Text", " ", "with", "\x0b", "whitespace"]),
        ("Text with\x0cwhitespace", ["Text", " ", "with", "\x0c", "whitespace"]),
    ],
)
def test_split_into_words_and_whitespace(input_text: str, expected: List[str]) -> None:
    assert utils.split_into_words_and_whitespace(input_text) == expected


@pytest.mark.parametrize(
    "input_char, expected_collection",
    [
        ("a", string.ascii_lowercase),
        ("A", string.ascii_uppercase),
        ("1", string.digits),
    ],
)
def test_get_character_collection(input_char: str, expected_collection: str) -> None:
    output_collection = utils.get_character_collection(input_char)

    assert output_collection == expected_collection


def test_get_character_collection_unsupported_character() -> None:
    with pytest.raises(ValueError):
        _ = utils.get_character_collection("®")
