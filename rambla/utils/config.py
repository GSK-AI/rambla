from typing import List, Optional, Union

from pydantic import BaseModel


def index_validator_helper(
    index_field: str, values: dict, fields_with_index_field: List[str]
) -> None:
    """Helper function for use in `index_field` pydantic validator"""
    for field in fields_with_index_field:
        if field in values:
            field_index = values[field].index_field
            err_msg = (
                f"{field}.index_field: '{field_index}' "
                f"does not agree with '{index_field}'"
            )
            assert field_index == index_field, err_msg


def _getattr_helper(obj, field):
    try:
        value = getattr(obj, field)
    except AttributeError:
        value = obj[field]
    return value


def validate_field_attribute_equality(
    field1: BaseModel,
    field2: BaseModel,
    attribute_name: str,
    attribute_name2: Optional[Union[str, None]] = None,
) -> None:
    """Checks for equality of attributes of 2 basemodel fields"""
    if not attribute_name2:
        attribute_name2 = attribute_name

    attr1 = _getattr_helper(field1, attribute_name)
    attr2 = _getattr_helper(field2, attribute_name2)

    if attr1 != attr2:
        field1_name = field1.__class__.__name__
        field2_name = field2.__class__.__name__
        raise ValueError(
            f"{field1_name}.{attribute_name}={attr1} not the same as "
            f"{field2_name}.{attribute_name2}={attr2}"
        )


def validate_category_equality(categories1: List[str], categories2: List[str]) -> None:
    """Validates that 2 lists of categories are equal"""
    if set(categories1) != set(categories2):
        raise ValueError(f"{categories1} not the same as {categories2}")
