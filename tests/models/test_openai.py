import asyncio
from typing import Dict, Generator
from unittest import mock

import openai
import pydantic
import pytest

from rambla import models
from rambla.models.openai_models import (
    OpenaiChatCompletionModel,
    OpenaiCompletionModel,
    OpenaiEmbeddings,
    OpenAILLMParams,
)
from rambla.utils.misc import EnvCtxManager

# flake8: noqa: N802


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def make_mock_response_chat_completion(content):
    class MockMessage:
        def __init__(self, content):
            self.content = content

    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content=content)

    class MockResponse:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    return MockResponse(content=content)


async def async_make_mock_response_chat_completion(content):
    class MockMessage:
        def __init__(self, content):
            self.content = content

    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content=content)

    class MockResponse:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    return MockResponse(content=content)


def make_mock_response_completion(content):
    class MockChoice:
        def __init__(self, content):
            self.text = content

    class MockResponse:
        def __init__(self, content):
            self.choices = [MockChoice(content)]

    return MockResponse(content=content)


async def async_make_mock_response_completion(content):
    return make_mock_response_completion(content)


@EnvCtxManager(
    OPENAI_API_KEY="__dummy_openai_api_key__",
    OPENAI_API_BASE="__dummy_openai_api_base__",
    OPENAI_API_VERSION="__dummy_openai_api_version__",
)
def test_OpenaiChatCompletionModel_run():  # noqa: N802
    message = "dummy message"

    expected_content = "mock expected content"
    mock_response = make_mock_response_chat_completion(expected_content)

    model = OpenaiChatCompletionModel(temperature=0.0, engine="dummy engine")
    mock_client = mock.Mock()
    mock_client.chat.completions.create.return_value = mock_response
    model._client = mock_client
    out = model.generate(message)

    assert out == expected_content
    mock_create = mock_client.chat.completions.create
    assert mock_create.call_args.kwargs["messages"] == [
        {"role": "user", "content": "dummy message"}
    ]


@EnvCtxManager(
    OPENAI_API_KEY="__dummy_openai_api_key__",
    OPENAI_API_BASE="__dummy_openai_api_base__",
    OPENAI_API_VERSION="__dummy_openai_api_version__",
)
def test_OpenaiChatCompletionModel_arun(  # noqa: N802
    event_loop: asyncio.AbstractEventLoop,
):
    message = "dummy message"

    expected_content = "mock expected content"
    mock_response = make_mock_response_chat_completion(expected_content)

    model = OpenaiChatCompletionModel(
        temperature=0.0, engine="dummy engine", async_calls=True
    )
    mock_client = mock.Mock()
    mock_acreate = mock.AsyncMock()
    mock_acreate.return_value = mock_response
    mock_client.chat.completions.create = mock_acreate
    model._client = mock_client
    out = event_loop.run_until_complete(model.async_generate(message))

    assert out == expected_content

    mock_create = mock_client.chat.completions.create
    assert mock_create.call_args.kwargs["messages"] == [
        {"role": "user", "content": "dummy message"}
    ]


@EnvCtxManager(
    OPENAI_MODEL_NAME="openai_model_name",
    OPENAI_API_KEY="__dummy_openai_api_key__",
    OPENAI_API_BASE="__dummy_openai_api_base__",
    OPENAI_API_VERSION="__dummy_openai_api_version__",
)
def test_OpenaiCompletionModel_run():  # noqa: N802
    message = "dummy message"

    expected_content = "mock expected content"
    mock_response = make_mock_response_completion(expected_content)
    model = OpenaiCompletionModel(temperature=0.0)
    mock_client = mock.Mock()
    mock_client.completions.create.return_value = mock_response
    model._client = mock_client
    out = model.generate(message)

    assert out == expected_content
    mock_create = mock_client.completions.create
    assert mock_create.call_args.kwargs["prompt"] == message


class MockOpenaiInvalidRequestError(openai.BadRequestError):
    pass


def test_openaillm_config(mock_model_map: Dict[str, mock.MagicMock]) -> None:
    mock_config = {
        "name": "mock_model1",
        "params": {
            "temperature": 1.0,
            "engine": "mock-engine",
            "max_tokens": 5,
            "top_p": 0.5,
            "async_calls": False,
            "api_type": "azure",
        },
    }

    with mock.patch.object(models, "MODEL_MAP", new=mock_model_map):
        config = OpenAILLMParams.parse_obj(mock_config["params"])

    assert config.dict() == mock_config["params"]


def test_openaillm_config_no_optional_args(
    mock_model_map: Dict[str, mock.MagicMock]
) -> None:
    mock_config = {
        "name": "mock_model1",
        "params": {
            "temperature": 1.0,
            "engine": "mock-engine",
        },
    }

    with mock.patch.object(models, "MODEL_MAP", new=mock_model_map):
        config = OpenAILLMParams.parse_obj(mock_config["params"])

    assert config.temperature == mock_config["params"]["temperature"]
    assert not config.max_tokens
    assert not config.top_p
    assert config.async_calls is None  # Explicitly check for false as a boolean


def test_openaillm_config_invalid_name(
    mock_model_map: Dict[str, mock.MagicMock]
) -> None:
    mock_config = {
        "name": "invalid_name",
        "params": {"temperature": 1.0, "engine": "mock-engine"},
    }

    with mock.patch.object(models, "MODEL_MAP", new=mock_model_map):
        with pytest.raises(ValueError):
            models.build_llm(mock_config)


def test_openaillm_params_missing_args(
    mock_model_map: Dict[str, mock.MagicMock]
) -> None:
    mock_config = {
        "name": "mock_model1",
        "params": {"temperature": 1.0},
    }

    with mock.patch.object(models, "MODEL_MAP", new=mock_model_map):
        with pytest.raises(pydantic.ValidationError):
            _ = OpenAILLMParams.parse_obj(mock_config["params"])


@pytest.fixture
def openai_embeddings_config() -> dict:
    return {
        "engine": "text-embedding-ada-002",
        "async_calls": True,
    }


@EnvCtxManager(
    OPENAI_API_KEY="__dummy_openai_api_key__",
    OPENAI_API_BASE="__dummy_openai_api_base__",
)
def test_OpenaiEmbeddings_from_config(openai_embeddings_config: dict):
    openai_embeddings_config["api_type"] = None
    model = OpenaiEmbeddings.from_config(openai_embeddings_config)

    assert model.engine == openai_embeddings_config["engine"]
    assert model.async_calls == openai_embeddings_config["async_calls"]
    assert model._model_dict == {"engine": openai_embeddings_config["engine"]}


@EnvCtxManager(
    OPENAI_API_KEY="__dummy_openai_api_key__",
    OPENAI_API_BASE="__dummy_openai_api_base__",
    OPENAI_API_VERSION="__dummy_openai_api_version__",
)
def test_OpenaiEmbeddings_from_config_with_api_type(openai_embeddings_config: dict):
    model = OpenaiEmbeddings.from_config(openai_embeddings_config)

    assert model.engine == openai_embeddings_config["engine"]
    assert model.async_calls == openai_embeddings_config["async_calls"]
    assert model._model_dict == {"engine": openai_embeddings_config["engine"]}


@pytest.fixture
def mock_embedding_response() -> dict:
    mock_obj = mock.Mock()
    mock_embedding = mock.Mock()
    mock_embedding.embedding = [0.0, 1.0, 2.0]
    mock_obj.data = [mock_embedding]
    return mock_obj


@EnvCtxManager(
    OPENAI_API_KEY="__dummy_openai_api_key__",
    OPENAI_API_BASE="__dummy_openai_api_base__",
)
def test_OpenaiEmbeddings_generate(openai_embeddings_config, mock_embedding_response):
    openai_embeddings_config["api_type"] = None
    model = OpenaiEmbeddings.from_config(openai_embeddings_config)
    #

    mock_client = mock.Mock()
    mock_client.embeddings.create.return_value = mock_embedding_response
    model._client = mock_client

    input_text = "dummy input text"
    embedding = model.generate(input_text)

    #
    mock_openai_create = mock_client.embeddings.create
    mock_openai_create.assert_called_with(
        input=input_text, model=openai_embeddings_config["engine"]
    )
    assert embedding == [0.0, 1.0, 2.0]


@EnvCtxManager(
    OPENAI_API_KEY="__dummy_openai_api_key__",
    OPENAI_API_BASE="__dummy_openai_api_base__",
    OPENAI_API_VERSION="__dummy_openai_api_version__",
)
def test_OpenaiEmbeddings_generate_with_api_type(
    openai_embeddings_config, mock_embedding_response
):
    model = OpenaiEmbeddings.from_config(openai_embeddings_config)
    #

    mock_client = mock.Mock()
    mock_client.embeddings.create.return_value = mock_embedding_response
    model._client = mock_client

    input_text = "dummy input text"
    embedding = model.generate(input_text)

    #
    mock_openai_create = mock_client.embeddings.create
    mock_openai_create.assert_called_with(
        input=input_text, model=openai_embeddings_config["engine"]
    )
    assert embedding == [0.0, 1.0, 2.0]
