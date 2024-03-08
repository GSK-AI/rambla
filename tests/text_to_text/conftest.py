from typing import Callable
from unittest import mock

import pytest

from rambla.models.openai_models import OpenaiEmbeddings
from rambla.response_generation.response import ResponseComponent
from rambla.text_to_text_components.embeddings_component import (
    EmbeddingBasedTextToTextComponent,
)
from rambla.utils.similarity import build_similarity_module


@pytest.fixture
def dataset_config() -> dict:
    return {
        "name": "bigbio_dummy_dataset",
        "params": {
            "path": "bigbio/dummy",
            "subset": "dummy",
            "split": "train[:10]",
        },
    }


@pytest.fixture
def llm_config() -> dict:
    return {
        "name": "openai_chat",
        "params": {"temperature": 0.10, "engine": "gpt-4"},
    }


@pytest.fixture
def prompt_formatter_config(index_field: str) -> dict:
    return {
        "template": "mock template with {question}",
        "var_map": {"question": "question"},
        "index_field": index_field,
    }


@pytest.fixture
def parent_llm_component_config(
    llm_config: dict,
    prompt_formatter_config: dict,
    response_component_config: dict,
) -> dict:
    return {
        "name": "llm_component",
        "params": {
            "llm_config": llm_config,
            "prompt_formatter_config": prompt_formatter_config,
            "response_component_config": response_component_config,
            "response_field_name": "sim_response",
        },
    }


@pytest.fixture
def similarity_module_config() -> dict:
    return {"name": "numpy_inner_product", "params": {}}


@pytest.fixture
def embeddings_model_config() -> dict:
    return {
        "engine": "text-embedding-ada-002",
        "async_calls": True,
    }


@pytest.fixture
def cache_dir(tmpdir) -> str:
    return tmpdir


@pytest.fixture
def response_cache_fname() -> str:
    return "response"


@pytest.fixture
def response_field_name() -> str:
    return "response"


@pytest.fixture
def renaming_map() -> dict:
    return {"0": 0, "1": 1}


@pytest.fixture
def target_formatter_config(label_field: str, renaming_map: dict) -> dict:
    return {
        "response_field_name": label_field,
        "string_formatter_name": "basic",
        "renaming_map": renaming_map,
    }


@pytest.fixture
def continuous_evaluator_config(
    response_field_name: str,
    label_field: str,
) -> dict:
    return {
        "response_field": response_field_name,
        "metric_names": ["mse", "mae"],
        "target_field": label_field,
    }


@pytest.fixture
def embedding_based_component_config(
    embeddings_model_config: dict,
    similarity_module_config: dict,
    response_component_config: dict,
    text_field_1: str,
    text_field_2: str,
) -> dict:
    return {
        "embeddings_model_config": embeddings_model_config,
        "similarity_module_config": similarity_module_config,
        "response_component_config": response_component_config,
        "text_field_1": text_field_1,
        "text_field_2": text_field_2,
        "response_field_name": "response",
    }


@pytest.fixture
def make_mock_embeddings_model() -> Callable:
    def inner(outputs) -> OpenaiEmbeddings:
        mock_model = mock.create_autospec(spec=OpenaiEmbeddings, instance=True)
        mock_generate = mock.MagicMock(side_effect=outputs)
        mock_model.generate = mock_generate
        mock_model._model_dict = {"dummy": 1}
        mock_model.is_async = False
        return mock_model

    return inner


@pytest.fixture
def make_mock_embeddings_component(
    embedding_based_component_config: dict,
    make_mock_embeddings_model: Callable,
    response_component_config: dict,
) -> Callable:
    def inner(outputs) -> EmbeddingBasedTextToTextComponent:
        mock_model = make_mock_embeddings_model(outputs)
        similarity_module = build_similarity_module(
            embedding_based_component_config["similarity_module_config"]
        )
        response_component = ResponseComponent.from_config(response_component_config)

        component = EmbeddingBasedTextToTextComponent(
            model=mock_model,
            similarity_module=similarity_module,
            response_component=response_component,
            text_field_1=embedding_based_component_config["text_field_1"],
            text_field_2=embedding_based_component_config["text_field_2"],
        )
        return component

    return inner
