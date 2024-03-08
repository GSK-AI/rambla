from typing import Callable

import pytest
from datasets import Dataset

from rambla.prompts.formatters import ColumnPromptFormatter
from rambla.response_generation.response import ResponseComponent
from rambla.tasks.paraphrase.utils import RephrasingModule

# flake8: noqa: N802


@pytest.fixture
def rephrasing_module_config(
    prompt_formatter_config: dict,
    response_component_config: dict,
) -> dict:
    llm_config = {
        "name": "openai_chat",
        "params": {"temperature": 1.0, "is_async": False},
    }

    return {
        "response_component_config": response_component_config,
        "prompt_formatter_config": prompt_formatter_config,
        "llm_config": llm_config,
        "field_rephrased": "question",
    }


def test_RephrasingModule_run(
    make_mock_llm: Callable,
    rephrasing_module_config: dict,
    mock_flat_pubmedqa_dataset: Dataset,
):
    responses = list(map(str, range(10)))
    mock_llm = make_mock_llm(responses)
    response_component = ResponseComponent.from_config(
        rephrasing_module_config["response_component_config"]
    )
    prompt_formatter = ColumnPromptFormatter.from_config(
        rephrasing_module_config["prompt_formatter_config"]
    )
    module = RephrasingModule(
        llm=mock_llm,
        prompt_formatter=prompt_formatter,
        response_component=response_component,
        field_rephrased="question",
    )

    #
    output_dataset = module.run(mock_flat_pubmedqa_dataset)

    #
    assert output_dataset["question"] == responses
    assert output_dataset["original_question"] == mock_flat_pubmedqa_dataset["question"]
