import numpy as np
import pytest
from datasets import Dataset
from pydantic import ValidationError

from rambla.text_to_text_components.nlp_component import NgramTextToTextSimilarity

# flake8: noqa: N802


@pytest.fixture
def rouge_config() -> dict:
    config = {
        "metric_name": "rouge",
        "metric_kwargs": {"rouge_types": ["rougeL"], "use_aggregator": False},
        "predictions_field": "text_1",
        "references_field": "text_2",
        "column_name": "response",
    }
    return config


@pytest.fixture
def bleu_config() -> dict:
    config = {
        "metric_name": "bleu",
        "metric_kwargs": {"max_order": 4},
        "predictions_field": "text_1",
        "references_field": "text_2",
        "column_name": "response",
    }
    return config


@pytest.fixture
def rouge_not_defined_error_config() -> dict:
    config = {
        "metric_name": "rouge",
        "metric_kwargs": {"use_aggregator": False},
        "predictions_field": "text_1",
        "references_field": "text_2",
        "column_name": "response",
    }
    return config


@pytest.fixture
def rouge_len_error_config() -> dict:
    config = {
        "metric_name": "rouge",
        "metric_kwargs": {"rouge_types": ["rouge1", "rouge2"], "use_aggregator": False},
        "predictions_field": "text_1",
        "references_field": "text_2",
        "column_name": "response",
    }
    return config


@pytest.fixture
def NgramTextToTextSimilarity_error_config() -> dict:
    config = {
        "metric_name": "XXX",
        "metric_kwargs": {"rouge_types": ["rougeL"], "use_aggregator": False},
        "predictions_field": "text_1",
        "references_field": "text_2",
        "column_name": "response",
    }
    return config


def test_NgramTextToTextSimilarity_from_config(
    rouge_config: dict,
) -> None:
    nlp_component_from_config = NgramTextToTextSimilarity.from_config(rouge_config)
    assert nlp_component_from_config.metric.name == rouge_config["metric_name"]
    assert nlp_component_from_config.metric_kwargs == rouge_config["metric_kwargs"]
    assert (
        nlp_component_from_config.predictions_field == rouge_config["predictions_field"]
    )
    assert (
        nlp_component_from_config.references_field == rouge_config["references_field"]
    )
    assert nlp_component_from_config.column_name == rouge_config["column_name"]


def test_NgramTextToTextSimilarity_error_from_config(
    NgramTextToTextSimilarity_error_config: dict,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        _ = NgramTextToTextSimilarity.from_config(
            NgramTextToTextSimilarity_error_config
        )
    assert "unexpected value" in str(exc_info.value)
    assert NgramTextToTextSimilarity_error_config["metric_name"] in str(exc_info.value)


def test_NgramTextToTextSimilarity_rouge_not_defined_error_from_config(
    rouge_not_defined_error_config: dict,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        _ = NgramTextToTextSimilarity.from_config(rouge_not_defined_error_config)
    assert "not defined" in str(exc_info.value)


def test_NgramTextToTextSimilarity_rouge_len_error_from_config(
    rouge_len_error_config: dict,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        _ = NgramTextToTextSimilarity.from_config(rouge_len_error_config)
    assert "must be 1" in str(exc_info.value)


def test_NgramTextToTextSimilarity_rouge_run(
    rouge_config: dict,
) -> None:
    nlp_component = NgramTextToTextSimilarity.from_config(rouge_config)
    dummy_dataset_config = {
        "index": [0, 1],
        "text_1": ["This is some text", "This is some text"],
        "text_2": ["This is some dummy text", "Something is not quite right..."],
        "label": ["1", "0"],
    }
    expected_responses = [8 / 9, 2 / 9]
    dummy_dataset = Dataset.from_dict(dummy_dataset_config)
    output_dataset = nlp_component.run(dummy_dataset)
    for n in range(len(output_dataset[rouge_config["column_name"]])):
        assert np.isclose(
            output_dataset[rouge_config["column_name"]][n], expected_responses[n]
        )


def test_NgramTextToTextSimilarity_bleu_run(
    bleu_config: dict,
) -> None:
    nlp_component = NgramTextToTextSimilarity.from_config(bleu_config)
    dummy_dataset_config = {
        "index": [0, 1],
        "text_1": ["This is some text", "This is some text"],
        "text_2": ["This is some dummy text", "Something is not quite right..."],
        "label": ["1", "0"],
    }
    expected_responses = [0, 0]
    dummy_dataset = Dataset.from_dict(dummy_dataset_config)
    output_dataset = nlp_component.run(dummy_dataset)
    for n in range(len(output_dataset[bleu_config["column_name"]])):
        assert np.isclose(
            output_dataset[bleu_config["column_name"]][n], expected_responses[n]
        )
