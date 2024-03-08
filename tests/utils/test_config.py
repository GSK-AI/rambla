from __future__ import annotations

from typing import List
from unittest import mock

import pytest
from pydantic import BaseModel, error_wrappers, validator

from rambla.utils import config


# NOTE: Must be defined as global to allow config creator to access
class NestedConfig(BaseModel):
    index_field: str


@pytest.fixture()
def mock_config_creator():
    def create_mock_config(fields_with_index_field: List[str]) -> type:
        class MockConfig(BaseModel):
            nested_config1: NestedConfig
            nested_config2: NestedConfig
            index_field: str

            @validator("index_field", allow_reuse=True)
            @classmethod
            def validate_index_field(cls, v, values, **kwargs):
                config.index_validator_helper(v, values, fields_with_index_field)
                return v

        return MockConfig

    return create_mock_config


def test_index_validator_helper(mock_config_creator) -> None:
    config_dict = {
        "index_field": "index",
        "nested_config1": {
            "index_field": "index",
        },
        "nested_config2": {
            "index_field": "index",
        },
    }

    mock_config = mock_config_creator(["nested_config1", "nested_config2"])
    config = mock_config.parse_obj(config_dict)

    assert config.index_field == config_dict["index_field"]
    assert (
        config.nested_config1.index_field
        == config_dict["nested_config1"]["index_field"]
    )


@pytest.fixture
def config_dict() -> dict:
    return {
        "index_field": "index",
        "nested_config1": {
            "index_field": "index",
        },
        "nested_config2": {
            "index_field": "invalid_index",
        },
    }


def test_index_validator_helper_invalid_index(
    mock_config_creator, config_dict: dict
) -> None:
    mock_config = mock_config_creator(["nested_config1", "nested_config2"])
    with pytest.raises(error_wrappers.ValidationError):
        _ = mock_config.parse_obj(config_dict)


def test_index_validator_helper_invalid_index_ignore(
    mock_config_creator, config_dict: dict
) -> None:
    # Check validation passes if not checking for invalid field
    mock_config = mock_config_creator(["nested_config1"])
    config = mock_config.parse_obj(config_dict)

    assert config.index_field == config_dict["index_field"]
    assert (
        config.nested_config2.index_field
        == config_dict["nested_config2"]["index_field"]
    )


def test_validate_field_attribute_equality() -> None:
    mock_field1 = mock.MagicMock(spec=BaseModel)
    mock_field1.test_field = "value"

    mock_field2 = mock.MagicMock(spec=BaseModel)
    mock_field2.test_field = "value"

    config.validate_field_attribute_equality(mock_field1, mock_field2, "test_field")


def test_validate_field_attribute_equality_unequal_field() -> None:
    mock_field1 = mock.MagicMock(spec=BaseModel)
    mock_field1.test_field = "value"

    mock_field2 = mock.MagicMock(spec=BaseModel)
    mock_field2.test_field = "alt_value"

    with pytest.raises(ValueError):
        config.validate_field_attribute_equality(mock_field1, mock_field2, "test_field")


def test_validate_field_attribute_equality_different_field_names() -> None:
    mock_field1 = mock.MagicMock(spec=BaseModel)
    mock_field1.test_field = "value"

    mock_field2 = mock.MagicMock(spec=BaseModel)
    mock_field2.another_field = "value"

    config.validate_field_attribute_equality(
        mock_field1, mock_field2, "test_field", "another_field"
    )


def test_validate_category_equality() -> None:
    categories1 = ["yes", "no", "maybe"]
    categories2 = ["yes", "no", "maybe"]

    config.validate_category_equality(categories1, categories2)


def test_validate_category_equality_inequality() -> None:
    categories1 = ["yes", "no", "maybe"]
    categories2 = ["yes", "no"]

    with pytest.raises(ValueError):
        config.validate_category_equality(categories1, categories2)
