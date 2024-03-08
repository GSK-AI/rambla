from typing import Callable
from unittest import mock

import numpy as np
import pytest
from datasets import Dataset

from rambla.models.openai_models import OpenaiEmbeddings
from rambla.response_generation.response import ResponseComponent
from rambla.text_to_text_components.embeddings_component import (
    EmbeddingBasedTextToTextComponent,
)
from rambla.utils.similarity import build_similarity_module
from tests.conftest import hf_datasets_are_same

# flake8: noqa: N802


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


def test_EmbeddingBasedTextToTextComponent_run(
    make_mock_embeddings_model: Callable,
    embedding_based_component_config: dict,
    mock_text_to_text_dataset: Dataset,
):
    return_values = [np.random.random((10,)).tolist() for _ in range(20)]
    mock_model = make_mock_embeddings_model(return_values)
    similarity_module = build_similarity_module(
        embedding_based_component_config["similarity_module_config"]
    )
    response_component = ResponseComponent.from_config(
        embedding_based_component_config["response_component_config"]
    )

    component = EmbeddingBasedTextToTextComponent(
        model=mock_model,
        similarity_module=similarity_module,
        response_component=response_component,
        text_field_1=embedding_based_component_config["text_field_1"],
        text_field_2=embedding_based_component_config["text_field_2"],
    )

    #
    output_dataset = component.run(mock_text_to_text_dataset)

    #
    expected_output_dataset = mock_text_to_text_dataset
    expected_output_dataset = expected_output_dataset.add_column(
        "response_text_field_1", return_values[:10]
    )  # type: ignore
    expected_output_dataset = expected_output_dataset.add_column(
        "response_text_field_2", return_values[10:]
    )

    arr0 = np.array(return_values[:10])
    arr1 = np.array(return_values[10:])
    inner_product = similarity_module.run(arr0=arr0, arr1=arr1)
    expected_output_dataset = expected_output_dataset.add_column(
        "response", inner_product
    )
    assert hf_datasets_are_same(output_dataset, expected_output_dataset)

    # Checking whether the model was called with the right prompts
    for ii, prompt in enumerate(
        mock_text_to_text_dataset["text_1"] + mock_text_to_text_dataset["text_2"]
    ):
        assert mock_model.generate.call_args_list[ii].kwargs == {"prompt": prompt}
