import pytest

from rambla.prompts.utils import find_field_placeholders


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("no field present", []),
        ("single {field} present", ["field"]),
        ("two {fields} {present}", ["fields", "present"]),
        ("double {{brackets}} don't work", []),
    ],
)
def test_find_field_placeholders_no_duplicates(input_text, expected_output):
    output = find_field_placeholders(input_text)
    assert output == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("no field present", []),
        ("single {field} present", ["field"]),
        ("two {fields} {present}", ["fields", "present"]),
        ("double {{brackets}} don't work", []),
        ("two {fields} {fields}", ["fields", "fields"]),
        ("three {fields} {other} {fields}", ["fields", "other", "fields"]),
    ],
)
def test_find_field_placeholders_with_duplicates(input_text, expected_output):
    output = find_field_placeholders(input_text)
    assert output == expected_output
