from typing import Dict, Optional
from unittest import mock

import pydantic
import pytest

from rambla import models


def test_validate_model_name(mock_model_map: Dict[str, mock.MagicMock]) -> None:
    with mock.patch.object(models, "MODEL_MAP", new=mock_model_map):
        _ = models._validate_model_name("mock_model1")


def test_validate_model_name_invalid_name(
    mock_model_map: Dict[str, mock.MagicMock]
) -> None:
    with mock.patch.object(models, "MODEL_MAP", new=mock_model_map):
        with pytest.raises(ValueError):
            _ = models._validate_model_name("invalid_name")


def test_build_llmodel(
    mock_model_map: Dict[str, mock.MagicMock], mock_decoding_strategy
) -> None:
    class MockParams(pydantic.BaseModel):
        param1: str
        param2: Optional[int]

    mock_config = {
        "name": "mock_model1",
        "params": {
            "param1": "value",
        },
    }

    with mock.patch.object(models, "MODEL_MAP", new=mock_model_map):
        _ = models.build_llm(mock_config)

    model_class = mock_model_map[mock_config["name"]]

    # Checks non-defined optional arg isn't passed as None
    assert "param2" not in model_class.from_config.call_args[0][0]
    assert model_class.from_config.call_args[0][0] == mock_config["params"]


def test_build_llmodel_missing_name() -> None:
    mock_config = {
        "param1": "value",
    }

    with pytest.raises(pydantic.ValidationError):
        _ = models.build_llm(mock_config)
