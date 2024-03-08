from typing import Callable
from unittest import mock

import numpy as np
import pytest
from datasets import Dataset

from rambla.models.base_model import BaseHuggingFaceModel
from rambla.models.huggingface import NLIModel
from rambla.prompts.formatters import ColumnPromptFormatter
from rambla.response_generation.response import ResponseComponent
from rambla.text_to_text_components.nli_strategies import build_combination_strategy
from rambla.text_to_text_components.nli_wrappers import (
    NLIBidirectional,
    NLIUnidirectional,
)

# flake8: noqa: N802


@pytest.fixture
def make_mock_nli():
    def inner(responses: list):
        mock_nli = mock.create_autospec(spec=BaseHuggingFaceModel, instance=True)
        mock_nli.generate = mock.MagicMock(side_effect=responses)
        mock_nli._model_dict = {"a": 1}
        return mock_nli

    return inner


@pytest.fixture
def mock_unidirectional_config() -> dict:
    config = {
        "model_config": {
            "name": "mock_model",
            "params": {
                "label_map": {
                    0: "entailment",
                    1: "neutral",
                    2: "contradiction",
                },
                "device": "cpu",
                "return_mode": "dict",
                "sequence_sep": "[SEP]",
            },
        },
        "prompt_formatter_config": {
            "template": "{text_1}[SEP]{text_2}",
            "var_map": {"text_1": "text_1", "text_2": "text_2"},
            "index_field": "id",
        },
        "response_component_config": {
            "cache_base_dir": None,
            "response_cache_fname": "response.json",
            "max_rate": 4,
            "run_async": False,
            "time_period": 60,
            "backoff_decorator_config": "DEFAULT",
        },
        "return_key": "entailment",
        "response_column_name": "response",
    }

    return config


@pytest.fixture
def mock_unidirectional_config_error() -> dict:
    config = {
        "model_config": {
            "name": "mock_model",
            "params": {
                "label_map": {
                    0: "entailment",
                    1: "neutral",
                    2: "contradiction",
                },
                "device": "cpu",
                "return_mode": "logits",
                "sequence_sep": "[SEP]",
            },
        },
        "prompt_formatter_config": {
            "template": "{text_1}[SEP]{text_2}",
            "var_map": {"text_1": "text_1", "text_2": "text_2"},
            "index_field": "id",
        },
        "response_component_config": {
            "cache_base_dir": None,
            "response_cache_fname": "response.json",
            "max_rate": 4,
            "run_async": False,
            "time_period": 60,
            "backoff_decorator_config": "DEFAULT",
        },
        "return_key": "entailment",
        "response_column_name": "response",
    }

    return config


@pytest.fixture
def mock_bidirectional_config(tmpdir) -> dict:
    config = {
        "model_config": {
            "name": "mock_model",
            "params": {
                "label_map": {
                    0: "entailment",
                    1: "neutral",
                    2: "contradiction",
                },
                "device": "cpu",
                "return_mode": "dict",
                "sequence_sep": "[SEP]",
            },
        },
        "prompt_formatter_config": {
            "template": "{text_1}[SEP]{text_2}",
            "var_map": {"text_1": "text_1", "text_2": "text_2"},
            "index_field": "id",
        },
        "flipped_prompt_formatter_config": {
            "template": "{text_1}[SEP]{text_2}",
            "var_map": {"text_1": "text_2", "text_2": "text_1"},
            "index_field": "id",
        },
        "response_component_config": {},
        "combination_strategy_config": {
            "name": "relaxed",
            "params": {"primary_key": "entailment", "secondary_key": "neutral"},
        },
        "response_column_name": "response",
    }

    return config


# Mocked because we don't want to build an actually NLI model
@mock.patch("rambla.models.huggingface.NLIModel.from_config")
def test_NLIUnidirectional_from_config(
    mock_NLIModel_from_config,
    mock_unidirectional_config: dict,
):
    mock_NLIModel_from_config.return_value = mock.create_autospec(
        spec=NLIModel, instance=True
    )

    mock_NLIUnidirectional_from_config = NLIUnidirectional.from_config(
        mock_unidirectional_config
    )

    mock_NLIUnidirectional = NLIUnidirectional(
        model=mock_NLIModel_from_config.return_value,
        prompt_formatter=ColumnPromptFormatter.from_config(
            mock_unidirectional_config["prompt_formatter_config"]
        ),
        response_component=ResponseComponent.from_config(
            mock_unidirectional_config["response_component_config"],
        ),
        response_column_name=mock_unidirectional_config["response_column_name"],
        return_key=mock_unidirectional_config["return_key"],
        model_return_mode=mock_unidirectional_config["model_config"]["params"][
            "return_mode"
        ],
    )

    mock_NLIModel_from_config.assert_called_with(
        mock_unidirectional_config["model_config"]
    )
    assert mock_NLIUnidirectional_from_config.model == mock_NLIUnidirectional.model
    assert (
        mock_NLIUnidirectional_from_config.prompt_formatter.__dict__
        == mock_NLIUnidirectional.prompt_formatter.__dict__
    )
    assert (
        mock_NLIUnidirectional_from_config.response_column_name
        == mock_NLIUnidirectional.response_column_name
    )
    assert (
        mock_NLIUnidirectional_from_config.return_key
        == mock_NLIUnidirectional.return_key
    )
    assert (
        mock_NLIUnidirectional_from_config.model_return_mode
        == mock_NLIUnidirectional.model_return_mode
    )


def test_NLIUnidirectional_from_config_error(
    mock_unidirectional_config: dict,
):
    mock_unidirectional_config["model_config"]["params"]["return_mode"] = "logits"

    with pytest.raises(ValueError) as exc_info:
        _ = NLIUnidirectional.from_config(mock_unidirectional_config)

    assert "NLIUnidirectional" in str(exc_info.value)


def test_NLIUnidirectional_run_label(
    mock_unidirectional_config: dict,
    mock_text_to_text_dataset: Dataset,
    make_mock_nli: Callable,
):
    return_mode = "label"

    expected_responses = [
        "entailment",
        "contradiction",
        "neutral",
        "entailment",
        "contradiction",
        "neutral",
        "entailment",
        "contradiction",
        "neutral",
        "entailment",
    ]

    mock_nli = make_mock_nli(expected_responses)

    prompt_formatter = ColumnPromptFormatter.from_config(
        mock_unidirectional_config["prompt_formatter_config"]
    )

    response_component = ResponseComponent.from_config(
        mock_unidirectional_config["response_component_config"],
    )

    mock_NLIUnidirectional = NLIUnidirectional(
        model=mock_nli,
        prompt_formatter=prompt_formatter,
        response_component=response_component,
        response_column_name=mock_unidirectional_config["response_column_name"],
        return_key=mock_unidirectional_config["return_key"],
        model_return_mode=return_mode,
    )

    output_dataset = mock_NLIUnidirectional.run(mock_text_to_text_dataset)

    assert output_dataset["response"] == expected_responses

    expected_prompts = [
        prompt_formatter.template.format(text_1=text_1, text_2=text_2)
        for text_1, text_2 in zip(
            mock_text_to_text_dataset["text_1"], mock_text_to_text_dataset["text_2"]
        )
    ]

    assert output_dataset["prompt"] == expected_prompts

    # Checking whether the model was called with the right prompts
    for ii, prompt in enumerate(expected_prompts):
        assert mock_nli.generate.call_args_list[ii].kwargs == {"prompt": prompt}


def test_NLIUnidirectional_run_dict(
    mock_unidirectional_config: dict,
    mock_text_to_text_dataset: Dataset,
    make_mock_nli: Callable,
):
    return_mode = "dict"

    intermediate_responses = [
        {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 1,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 0,
            "neutral": 1,
        },
        {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 1,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 0,
            "neutral": 1,
        },
        {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 1,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 0,
            "neutral": 1,
        },
        {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 0,
        },
    ]

    new_intermediate_responses = []
    for item in intermediate_responses:
        new_intermediate_responses.append({"response": item})

    # NOTE These values are the result of the softmax
    # NOTE They were obtained by running the intermediate_responses through
    final_expected_responses = [
        0.5761168847658291,
        0.21194155761708547,
        0.21194155761708547,
        0.5761168847658291,
        0.21194155761708547,
        0.21194155761708547,
        0.5761168847658291,
        0.21194155761708547,
        0.21194155761708547,
        0.5761168847658291,
    ]

    mock_nli = make_mock_nli(new_intermediate_responses)

    prompt_formatter = ColumnPromptFormatter.from_config(
        mock_unidirectional_config["prompt_formatter_config"]
    )

    response_component = ResponseComponent.from_config(
        mock_unidirectional_config["response_component_config"],
    )

    mock_NLIUnidirectional = NLIUnidirectional(
        model=mock_nli,
        prompt_formatter=prompt_formatter,
        response_component=response_component,
        response_column_name=mock_unidirectional_config["response_column_name"],
        return_key=mock_unidirectional_config["return_key"],
        model_return_mode=return_mode,
    )

    output_dataset = mock_NLIUnidirectional.run(mock_text_to_text_dataset)

    for n in range(len(output_dataset["response"])):
        assert np.isclose(output_dataset["response"][n], final_expected_responses[n])

    assert output_dataset["response"] == final_expected_responses

    expected_prompts = [
        prompt_formatter.template.format(text_1=text_1, text_2=text_2)
        for text_1, text_2 in zip(
            mock_text_to_text_dataset["text_1"], mock_text_to_text_dataset["text_2"]
        )
    ]

    assert output_dataset["prompt"] == expected_prompts

    # Checking whether the model was called with the right prompts
    for ii, prompt in enumerate(expected_prompts):
        assert mock_nli.generate.call_args_list[ii].kwargs == {"prompt": prompt}


# Mocked because we don't want to build an actually NLI model or combination_strategy
@mock.patch("rambla.models.huggingface.NLIModel.from_config")
@mock.patch("rambla.text_to_text_components.nli_wrappers.build_combination_strategy")
def test_NLIBidirectional_from_config(
    mock_build_combination_strategy,
    mock_NLIModel_from_config,
    mock_bidirectional_config: dict,
):
    mock_NLIModel_from_config.return_value = mock.create_autospec(
        spec=NLIModel, instance=True
    )
    mock_build_combination_strategy.return_value = mock.create_autospec(
        spec=build_combination_strategy, instance=True
    )

    mock_NLIBidirectional_from_config = NLIBidirectional.from_config(
        mock_bidirectional_config
    )

    mock_NLIBidirectional = NLIBidirectional(
        model=mock_NLIModel_from_config.return_value,
        prompt_formatter=ColumnPromptFormatter.from_config(
            mock_bidirectional_config["prompt_formatter_config"]
        ),
        flipped_prompt_formatter=ColumnPromptFormatter.from_config(
            mock_bidirectional_config["flipped_prompt_formatter_config"],
        ),
        response_component=ResponseComponent.from_config(
            mock_bidirectional_config["response_component_config"],
        ),
        response_column_name=mock_bidirectional_config["response_column_name"],
        combination_strategy=mock_build_combination_strategy.return_value,
    )

    mock_NLIModel_from_config.assert_called_with(
        mock_bidirectional_config["model_config"]
    )
    mock_build_combination_strategy.assert_called_with(
        mock_bidirectional_config["combination_strategy_config"]
    )

    assert mock_NLIBidirectional_from_config.model == mock_NLIBidirectional.model
    assert (
        mock_NLIBidirectional_from_config.prompt_formatter.__dict__
        == mock_NLIBidirectional.prompt_formatter.__dict__
    )
    assert (
        mock_NLIBidirectional_from_config.flipped_prompt_formatter.__dict__
        == mock_NLIBidirectional.flipped_prompt_formatter.__dict__
    )
    assert (
        mock_NLIBidirectional_from_config.response_column_name
        == mock_NLIBidirectional.response_column_name
    )
    assert (
        mock_NLIBidirectional_from_config.combination_strategy
        == mock_NLIBidirectional.combination_strategy
    )


def test_NLIBidirectional_from_config_error(
    mock_bidirectional_config: dict,
):
    mock_bidirectional_config["model_config"]["params"]["return_mode"] = "logits"

    with pytest.raises(ValueError) as exc_info:
        _ = NLIBidirectional.from_config(mock_bidirectional_config)

    assert "NLIBidirectional" in str(exc_info.value)


def test_NLIBidirectional_run_relaxed(
    mock_bidirectional_config: dict,
    mock_text_to_text_dataset: Dataset,
    make_mock_nli: Callable,
):
    combination_strategy_config = {
        "name": "relaxed",
        "params": {"primary_key": "entailment", "secondary_key": "neutral"},
    }

    intermediate_responses = [
        {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 1,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 0,
            "neutral": 1,
        },
        {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 1,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 0,
            "neutral": 1,
        },
        {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 1,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 0,
            "neutral": 1,
        },
        {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 0,
        },
    ]

    new_intermediate_responses = []
    for item in intermediate_responses:
        new_intermediate_responses.append({"response": item})

    # NOTE These values are the result of the combination_strategy
    # NOTE They were obtained by running the intermediate_responses through
    final_expected_responses = ["1", "0", "0", "1", "0"]

    mock_nli = make_mock_nli(new_intermediate_responses)

    prompt_formatter = ColumnPromptFormatter.from_config(
        mock_bidirectional_config["prompt_formatter_config"]
    )

    flipped_prompt_formatter = ColumnPromptFormatter.from_config(
        mock_bidirectional_config["flipped_prompt_formatter_config"],
    )

    response_component = ResponseComponent.from_config(
        mock_bidirectional_config["response_component_config"],
    )

    combination_strategy = build_combination_strategy(combination_strategy_config)

    mock_NLIBidirectional = NLIBidirectional(
        model=mock_nli,
        prompt_formatter=prompt_formatter,
        flipped_prompt_formatter=flipped_prompt_formatter,
        response_component=response_component,
        response_column_name=mock_bidirectional_config["response_column_name"],
        combination_strategy=combination_strategy,
    )

    output_dataset = mock_NLIBidirectional.run(
        mock_text_to_text_dataset.select(range(5))
    )

    assert output_dataset["response"] == final_expected_responses

    expected_prompts = [
        prompt_formatter.template.format(text_1=text_1, text_2=text_2)
        for text_1, text_2 in zip(
            mock_text_to_text_dataset.select(range(5))["text_1"],
            mock_text_to_text_dataset.select(range(5))["text_2"],
        )
    ]

    assert output_dataset["prompt"] == expected_prompts

    # Checking whether the model was called with the right prompts
    for ii, prompt in enumerate(expected_prompts):
        assert mock_nli.generate.call_args_list[ii].kwargs == {"prompt": prompt}


def test_NLIBidirectional_run_strict(
    mock_bidirectional_config: dict,
    mock_text_to_text_dataset: Dataset,
    make_mock_nli: Callable,
):
    combination_strategy_config = {
        "name": "strict",
        "params": {"positive_key": "entailment"},
    }

    intermediate_responses = [
        {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 1,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 0,
            "neutral": 1,
        },
        {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 1,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 0,
            "neutral": 1,
        },
        {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 1,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 0,
            "neutral": 1,
        },
        {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 0,
        },
    ]

    new_intermediate_responses = []
    for item in intermediate_responses:
        new_intermediate_responses.append({"response": item})

    # NOTE These values are the result of the combination_strategy
    # NOTE They were obtained by running the intermediate_responses through
    final_expected_responses = ["0", "0", "0", "0", "0"]

    mock_nli = make_mock_nli(new_intermediate_responses)

    prompt_formatter = ColumnPromptFormatter.from_config(
        mock_bidirectional_config["prompt_formatter_config"]
    )

    flipped_prompt_formatter = ColumnPromptFormatter.from_config(
        mock_bidirectional_config["flipped_prompt_formatter_config"],
    )

    response_component = ResponseComponent.from_config(
        mock_bidirectional_config["response_component_config"],
    )

    combination_strategy = build_combination_strategy(combination_strategy_config)

    mock_NLIBidirectional = NLIBidirectional(
        model=mock_nli,
        prompt_formatter=prompt_formatter,
        flipped_prompt_formatter=flipped_prompt_formatter,
        response_component=response_component,
        response_column_name=mock_bidirectional_config["response_column_name"],
        combination_strategy=combination_strategy,
    )

    output_dataset = mock_NLIBidirectional.run(
        mock_text_to_text_dataset.select(range(5))
    )

    assert output_dataset["response"] == final_expected_responses

    expected_prompts = [
        prompt_formatter.template.format(text_1=text_1, text_2=text_2)
        for text_1, text_2 in zip(
            mock_text_to_text_dataset.select(range(5))["text_1"],
            mock_text_to_text_dataset.select(range(5))["text_2"],
        )
    ]

    assert output_dataset["prompt"] == expected_prompts

    # Checking whether the model was called with the right prompts
    for ii, prompt in enumerate(expected_prompts):
        assert mock_nli.generate.call_args_list[ii].kwargs == {"prompt": prompt}


def test_NLIBidirectional_run_average(
    mock_bidirectional_config: dict,
    mock_text_to_text_dataset: Dataset,
    make_mock_nli: Callable,
):
    combination_strategy_config = {
        "name": "average",
        "params": {"positive_key": "entailment", "apply_softmax": True},
    }

    intermediate_responses = [
        {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 1,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 0,
            "neutral": 1,
        },
        {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 1,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 0,
            "neutral": 1,
        },
        {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 1,
            "neutral": 0,
        },
        {
            "entailment": 0,
            "contradiction": 0,
            "neutral": 1,
        },
        {
            "entailment": 1,
            "contradiction": 0,
            "neutral": 0,
        },
    ]

    new_intermediate_responses = []
    for item in intermediate_responses:
        new_intermediate_responses.append({"response": item})

    # NOTE These values are the result of the combination_strategy
    # NOTE They were obtained by running the intermediate_responses through
    final_expected_responses = [
        0.3940292211914573,
        0.3940292211914573,
        0.21194155761708547,
        0.3940292211914573,
        0.3940292211914573,
    ]

    mock_nli = make_mock_nli(new_intermediate_responses)

    prompt_formatter = ColumnPromptFormatter.from_config(
        mock_bidirectional_config["prompt_formatter_config"]
    )

    flipped_prompt_formatter = ColumnPromptFormatter.from_config(
        mock_bidirectional_config["flipped_prompt_formatter_config"],
    )

    response_component = ResponseComponent.from_config(
        mock_bidirectional_config["response_component_config"],
    )

    combination_strategy = build_combination_strategy(combination_strategy_config)

    mock_NLIBidirectional = NLIBidirectional(
        model=mock_nli,
        prompt_formatter=prompt_formatter,
        flipped_prompt_formatter=flipped_prompt_formatter,
        response_component=response_component,
        response_column_name=mock_bidirectional_config["response_column_name"],
        combination_strategy=combination_strategy,
    )

    output_dataset = mock_NLIBidirectional.run(
        mock_text_to_text_dataset.select(range(5))
    )

    for n in range(len(output_dataset["response"])):
        assert np.isclose(output_dataset["response"][n], final_expected_responses[n])

    assert output_dataset["response"] == final_expected_responses

    expected_prompts = [
        prompt_formatter.template.format(text_1=text_1, text_2=text_2)
        for text_1, text_2 in zip(
            mock_text_to_text_dataset.select(range(5))["text_1"],
            mock_text_to_text_dataset.select(range(5))["text_2"],
        )
    ]

    assert output_dataset["prompt"] == expected_prompts

    # Checking whether the model was called with the right prompts
    for ii, prompt in enumerate(expected_prompts):
        assert mock_nli.generate.call_args_list[ii].kwargs == {"prompt": prompt}
