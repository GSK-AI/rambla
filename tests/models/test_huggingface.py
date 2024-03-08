from typing import List, Union
from unittest import mock

import pydantic
import pytest
import torch
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer

from rambla.models import huggingface
from rambla.models.utils import MessageFormatter


@pytest.fixture
def mock_nli_model_config() -> dict:
    return {
        "name": "mock_model",
        "params": {
            "label_map": {0: "entailment", 1: "contradiction"},
            "device": "mock_device",
            "return_mode": "logits",
        },
    }


def test_nli_model_config_invalid_device(mock_nli_model_config: dict) -> None:
    invalid_device = "invalid_device"
    mock_nli_model_config["params"]["device"] = invalid_device

    with mock.patch.object(huggingface, "SUPPORTED_DEVICES", new=["mock_device"]):
        with pytest.raises(pydantic.error_wrappers.ValidationError):
            _ = huggingface.NLIModelConfig.parse_obj(mock_nli_model_config)


@mock.patch.object(huggingface.NLIModel, "__init__", return_value=None)
@pytest.mark.parametrize(
    "predictions, expected_label",
    [
        ([0.1, 0.5, 0.2], {"response": "neutral"}),
        ([0.5, 0.1, 0.2], {"response": "entailment"}),
    ],
)
def test_nli_model_generate_label(
    mock_init: mock.MagicMock, predictions: List[str], expected_label: str
) -> None:
    mock_tokens = mock.MagicMock()
    mock_tokens.to.return_value = mock.MagicMock()

    mock_tokenizer = mock.MagicMock(spec=PreTrainedTokenizer)
    mock_tokenizer.return_value = mock_tokens

    mock_model = mock.MagicMock(spec=PreTrainedModel)

    mock_predictions = mock.MagicMock()
    mock_predictions.logits.__getitem__.return_value = torch.Tensor(predictions)
    mock_model.return_value = mock_predictions

    mock_label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    model = huggingface.NLIModel(
        mock_tokenizer, mock_model, "mock_device", mock_label_map
    )
    model.tokenizer = mock_tokenizer
    model.model = mock_model
    model.device = "mock_device"
    model.label_map = mock_label_map
    model.return_mode = "label"
    model.sequence_sep = "[SEP]"

    assert model.generate("mock_string1[SEP]mock_string2") == expected_label


@mock.patch.object(huggingface.NLIModel, "__init__", return_value=None)
def test_nli_model_generate_logits(mock_init: mock.MagicMock) -> None:
    predictions = torch.Tensor([0.2, 0.7, 0.1])

    mock_tokens = mock.MagicMock()
    mock_tokens.to.return_value = mock.MagicMock()

    mock_tokenizer = mock.MagicMock(spec=PreTrainedTokenizer)
    mock_tokenizer.return_value = mock_tokens

    mock_model = mock.MagicMock(spec=PreTrainedModel)

    mock_predictions = mock.MagicMock()
    mock_predictions.logits.__getitem__.return_value = predictions
    mock_model.return_value = mock_predictions

    mock_label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    model = huggingface.NLIModel(
        mock_tokenizer, mock_model, "mock_device", mock_label_map
    )
    model.tokenizer = mock_tokenizer
    model.model = mock_model
    model.device = "mock_device"
    model.label_map = mock_label_map
    model.return_mode = "logits"
    model.sequence_sep = "[SEP]"

    assert model.generate("mock_string1[SEP]mock_string2") == {
        "response": predictions.tolist()
    }


@pytest.fixture
def mock_text_generation_model_config() -> dict:
    return {
        "name": "mock_model",
        "params": {
            "device": "mock_device",
            "max_tokens": 20,
            "temperature": 0.5,
            "top_p": 0.8,
        },
    }


@pytest.mark.parametrize(
    "invalid_param, invalid_value",
    [("max_tokens", -5), ("temperature", -0.2), ("top_p", 1.2)],
)
def test_text_generation_model_config_invalid_params(
    invalid_param: str,
    invalid_value: Union[int, float],
    mock_text_generation_model_config: dict,
) -> None:
    mock_text_generation_model_config["params"][invalid_param] = invalid_value
    with mock.patch.object(huggingface, "SUPPORTED_DEVICES", new=["mock_device"]):
        with pytest.raises(pydantic.error_wrappers.ValidationError):
            _ = huggingface.TextGenerationModelConfig.parse_obj(
                mock_text_generation_model_config
            )


@mock.patch.object(huggingface.TextGenerationModel, "__init__", return_value=None)
def test_text_generation_model_general_without_message_formatter(
    mock_init: mock.MagicMock,
) -> None:
    mock_tokens = mock.MagicMock()
    mock_tokens.to.return_value = mock.MagicMock()

    mock_tokenizer = mock.MagicMock(spec=PreTrainedTokenizer)
    mock_model = mock.MagicMock(spec=PreTrainedModel)
    mock_config = mock.Mock()
    mock_config.max_position_embeddings = 4096
    mock_model.config = mock_config

    mock_response = "__mock_response__"
    mock_decode = mock.MagicMock(return_value=mock_response)

    mock_tokenizer.return_value = mock_tokens
    mock_tokenizer.decode = mock_decode

    mock_pipeline = mock.MagicMock(
        spec=transformers.Pipeline, return_value=[{"generated_text": mock_response}]
    )

    model = huggingface.TextGenerationModel()  # type: ignore
    mock_count_tokens = mock.Mock(return_value=123)
    model._count_tokens = mock_count_tokens

    # Mock init
    model.tokenizer = mock_tokenizer
    model.pipeline = mock_pipeline
    model.model = mock_model
    model.device = "mock_device"
    model.generation_config = transformers.GenerationConfig(
        **{
            "temperature": 1.0,
            "top_p": 0.90,
        }
    )
    model.message_formatter = None
    mock_prompt = "mock prompt"
    response = model.generate(mock_prompt)

    assert response == mock_response
    expected_generation_config = transformers.GenerationConfig(
        max_new_tokens=3973, top_p=0.90
    )
    mock_pipeline.assert_called_once_with(
        mock_prompt,
        return_full_text=False,
        generation_config=expected_generation_config,
    )


@mock.patch.object(huggingface.TextGenerationModel, "__init__", return_value=None)
def test_text_generation_model_general_with_message_formatter(
    mock_init: mock.MagicMock,
) -> None:
    mock_tokens = mock.MagicMock()
    mock_tokens.to.return_value = mock.MagicMock()

    mock_tokenizer = mock.MagicMock(spec=PreTrainedTokenizer)
    mock_model = mock.MagicMock(spec=PreTrainedModel)
    mock_config = mock.Mock()
    mock_config.max_position_embeddings = 4096
    mock_model.config = mock_config

    mock_response = "__mock_response__"
    mock_decode = mock.MagicMock(return_value=mock_response)

    mock_tokenizer.return_value = mock_tokens
    mock_tokenizer.decode = mock_decode

    mock_pipeline = mock.MagicMock(
        spec=transformers.Pipeline, return_value=[{"generated_text": mock_response}]
    )

    model = huggingface.TextGenerationModel()  # type: ignore

    mock_count_tokens = mock.Mock(return_value=123)
    model._count_tokens = mock_count_tokens

    # Mock init
    model.tokenizer = mock_tokenizer
    model.pipeline = mock_pipeline
    model.model = mock_model
    model.device = "mock_device"
    model.generation_config = transformers.GenerationConfig(
        **{
            "temperature": 1.0,
            "top_p": 0.90,
        }
    )
    template = "dummy template {message}"
    model.message_formatter = MessageFormatter(template=template)
    mock_prompt = "This is a long long prompt. It includes a lot of dummy text."

    response = model.generate(mock_prompt)

    assert response == mock_response
    expected_generation_config = transformers.GenerationConfig(
        max_new_tokens=3973, top_p=0.90
    )
    mock_pipeline.assert_called_once_with(
        template.format(message=mock_prompt),
        return_full_text=False,
        generation_config=expected_generation_config,
    )


@mock.patch.object(huggingface.TextGenerationModel, "__init__", return_value=None)
def test_text_generation_config_max_new_tokens_greater_than_allowed(
    mock_init: mock.MagicMock,
) -> None:
    mock_tokens = mock.MagicMock()
    mock_tokens.to.return_value = mock.MagicMock()

    mock_tokenizer = mock.MagicMock(spec=PreTrainedTokenizer)
    mock_model = mock.MagicMock(spec=PreTrainedModel)
    mock_config = mock.Mock()
    mock_config.max_position_embeddings = 4096
    mock_model.config = mock_config

    mock_response = "__mock_response__"
    mock_decode = mock.MagicMock(return_value=mock_response)

    mock_tokenizer.return_value = mock_tokens
    mock_tokenizer.decode = mock_decode

    mock_pipeline = mock.MagicMock(
        spec=transformers.Pipeline, return_value=[{"generated_text": mock_response}]
    )

    model = huggingface.TextGenerationModel()  # type: ignore

    mock_count_tokens = mock.Mock(return_value=123)
    model._count_tokens = mock_count_tokens

    # Mock init
    model.tokenizer = mock_tokenizer
    model.pipeline = mock_pipeline
    model.model = mock_model
    model.device = "mock_device"
    model.generation_config = transformers.GenerationConfig(
        **{
            "max_new_tokens": 5096,
            "temperature": 1.0,
            "top_p": 0.90,
        }
    )
    template = "dummy template {message}"
    model.message_formatter = MessageFormatter(template=template)
    mock_prompt = "This is a long long prompt. It includes a lot of dummy text."

    response = model.generate(mock_prompt)

    assert response == mock_response
    expected_generation_config = transformers.GenerationConfig(
        max_new_tokens=3973, top_p=0.90
    )
    mock_pipeline.assert_called_once_with(
        template.format(message=mock_prompt),
        return_full_text=False,
        generation_config=expected_generation_config,
    )


@mock.patch.object(huggingface.TextGenerationModel, "__init__", return_value=None)
def test_text_generation_context_overflow(
    mock_init: mock.MagicMock,
) -> None:
    mock_tokens = mock.MagicMock()
    mock_tokens.to.return_value = mock.MagicMock()

    mock_tokenizer = mock.MagicMock(spec=PreTrainedTokenizer)
    mock_model = mock.MagicMock(spec=PreTrainedModel)
    mock_config = mock.Mock()
    mock_config.max_position_embeddings = 4096
    mock_model.config = mock_config

    mock_response = "__mock_response__"
    mock_decode = mock.MagicMock(return_value=mock_response)

    mock_tokenizer.return_value = mock_tokens
    mock_tokenizer.decode = mock_decode

    mock_pipeline = mock.MagicMock(
        spec=transformers.Pipeline, return_value=[{"generated_text": mock_response}]
    )

    model = huggingface.TextGenerationModel()  # type: ignore

    mock_count_tokens = mock.Mock(return_value=5096)
    model._count_tokens = mock_count_tokens

    # Mock init
    model.tokenizer = mock_tokenizer
    model.pipeline = mock_pipeline
    model.model = mock_model
    model.device = "mock_device"
    model.generation_config = transformers.GenerationConfig(
        **{
            "temperature": 1.0,
            "top_p": 0.90,
        }
    )
    template = "dummy template {message}"
    model.message_formatter = MessageFormatter(template=template)
    mock_prompt = "This is a long long prompt. It includes a lot of dummy text."

    with pytest.raises(huggingface.MaximumContextLengthExceededError):
        model.generate(mock_prompt)
