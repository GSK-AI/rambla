import pytest

from rambla.text_mutation import word_validation
from rambla.text_mutation.utils import is_punctuation


@pytest.mark.parametrize(
    "input, func, mode, expected_output",
    [
        ("Word", str.isupper, "any", True),
        ("Word", str.isupper, "all", False),
        ("1234", str.isalpha, "all", False),
        ("abcd", str.isalpha, "all", True),
        ("word", lambda x: not str.isspace(x), "all", True),
        (" word", lambda x: not str.isspace(x), "all", False),
        (" word", lambda x: not str.isspace(x), "any", True),
        ("\n", lambda x: not str.isspace(x), "any", False),
        ("Question?", lambda x: not is_punctuation(x), "any", True),
        # Checks empty strings return False
        ("", str.isspace, "all", False),
        ("", str.lower, "all", False),
    ],
)
def test_word_validator(
    input: str,
    func: word_validation.ValidationFunctionType,
    mode: str,
    expected_output: bool,
) -> None:
    validator = word_validation.WordValidator(
        validation_func=func, mode=mode  # type: ignore
    )
    output = validator.validate(input)

    assert output == expected_output


def test_word_validator_invalid_mode() -> None:
    invalid_mode = "invalid"
    with pytest.raises(ValueError):
        _ = word_validation.WordValidator(
            validation_func=str.isalpha, mode=invalid_mode  # type: ignore
        )
