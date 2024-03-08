import re
import string
import unicodedata
from typing import List

# All unicode categories regarded as punctuation
PUNCTUATION_CATEGORIES: List[str] = [
    "Pc",
    "Pd",
    "Pe",
    "Pf",
    "Pi",
    "Po",
    "Ps",
    "Sc",
    "Sk",
    "Sm",
    "So",
]


def is_punctuation(character: str) -> bool:
    """Checks whether a character is from a unicode punctuation category"""
    if len(character) > 1:
        raise ValueError(
            f"Invalid input: {character}. Input should be a single character"
        )
    return unicodedata.category(character) in PUNCTUATION_CATEGORIES


def is_not_space(text: str) -> bool:
    """Checks text is not entirely whitespace"""
    return not str.isspace(text)


def get_character_collection(char: str) -> str:
    """Returns the character collection a character is found in"""
    if char.isupper():
        return string.ascii_uppercase
    elif char.islower():
        return string.ascii_lowercase
    elif char.isnumeric():
        return string.digits
    else:
        raise ValueError(f"Character '{char}' is an unsupported character type")


def split_into_words_and_whitespace(text: str) -> List[str]:
    """Converts a string into list of individual words and whitespace characters"""
    return re.split(r"(\s+)", text)
