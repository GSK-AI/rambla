from pathlib import Path
from typing import Callable
from unittest import mock

import pytest
from datasets import Dataset

from rambla.models.base_model import BaseLLM
from rambla.prompts.formatters import ColumnPromptFormatter
from rambla.response_generation.response import ResponseComponent
from rambla.tasks.base import LLMGenerator
from rambla.text_to_text_components.llm_similarity_component import (
    LLMTextToTextSimilarity,
)

# flake8: noqa: N802


@pytest.fixture
def text_to_text_config(tmpdir, response_component_config: dict) -> dict:
    template = "dummy template with {statement_1} and {statement_2}"
    index_field = "index"
    llm_config = {
        "name": "openai_chat",
        "params": {"strategy": "GREEDY"},
    }
    var_map = {
        "text_1": "statement_1",
        "text_2": "statement_2",
    }
    prompt_formatter_config = {
        "template": template,
        "var_map": var_map,
        "index_field": index_field,
    }
    config = {
        "llm_config": llm_config,
        "prompt_formatter_config": prompt_formatter_config,
        "response_field_name": "response",
        "response_component_config": response_component_config,
    }

    return config


@pytest.mark.fileio
@mock.patch("rambla.text_to_text_components.llm_similarity_component.build_llm")
def test_LLMTextToTextSimilarity_from_config(
    mock_build_llmodel,
    text_to_text_config: dict,
):
    mock_build_llmodel.return_value = "dummy"
    llm_component_similarity_from_config = LLMTextToTextSimilarity.from_config(
        text_to_text_config
    )
    mock_response_component = ResponseComponent.from_config(
        text_to_text_config["response_component_config"]
    )
    llm_component_similarity = LLMTextToTextSimilarity(
        llm=mock_build_llmodel.return_value,
        prompt_formatter=ColumnPromptFormatter.from_config(
            text_to_text_config["prompt_formatter_config"]
        ),
        response_component=mock_response_component,
        response_field_name=text_to_text_config["response_field_name"],
    )

    mock_build_llmodel.assert_called_with(text_to_text_config["llm_config"])
    assert llm_component_similarity_from_config.llm == "dummy"
    assert llm_component_similarity_from_config.llm == llm_component_similarity.llm
    assert (
        llm_component_similarity_from_config.prompt_formatter.__dict__
        == llm_component_similarity.prompt_formatter.__dict__
    )
    assert (
        llm_component_similarity_from_config.response_field_name
        == llm_component_similarity.response_field_name
    )


def test_LLMTextToTextSimilarity_run(
    make_mock_llm: Callable,
    mock_text_to_text_dataset: Dataset,
    text_to_text_config: dict,
):
    expected_responses = [
        "yes",
        "no",
        "yes",
        "yes",
        "yes",
        "no",
        "no",
        "no",
        "yes",
        "yes",
    ]
    mock_llm = make_mock_llm(expected_responses)

    #
    prompt_formatter = ColumnPromptFormatter.from_config(
        text_to_text_config["prompt_formatter_config"]
    )
    response_component = ResponseComponent.from_config(
        text_to_text_config["response_component_config"]
    )

    similarity_module = LLMTextToTextSimilarity(
        llm=mock_llm,
        prompt_formatter=prompt_formatter,
        response_component=response_component,
        response_field_name=text_to_text_config["response_field_name"],
    )

    output_dataset = similarity_module.run(mock_text_to_text_dataset)

    #
    assert output_dataset["response"] == expected_responses
    expected_prompts = [
        prompt_formatter.template.format(statement_1=text_1, statement_2=text_2)
        for text_1, text_2 in zip(
            mock_text_to_text_dataset["text_1"], mock_text_to_text_dataset["text_2"]
        )
    ]
    assert output_dataset["prompt"] == expected_prompts

    # Checking whether the model was called with the right prompts
    for ii, prompt in enumerate(expected_prompts):
        assert mock_llm.generate.call_args_list[ii].kwargs == {"prompt": prompt}
