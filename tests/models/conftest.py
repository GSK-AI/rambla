from enum import Enum
from typing import Dict, Type
from unittest import mock

import pytest

from rambla.models.base_model import BaseLLM


@pytest.fixture
def mock_model_map() -> Dict[str, mock.MagicMock]:
    return {
        "mock_model1": mock.MagicMock(spec=BaseLLM),
        "mock_model2": mock.MagicMock(spec=BaseLLM),
    }


@pytest.fixture
def mock_decoding_strategy() -> Type[Enum]:
    class MockDecodingStrategy(str, Enum):
        GREEDY = "GREEDY"
        BEAM_SEARCH = "BEAM_SEARCH"

    return MockDecodingStrategy
