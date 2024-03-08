from typing import Type, Union

from pydantic import BaseModel, validator

from rambla.models.base_model import BaseAPIModel
from rambla.models.huggingface import TextGenerationModel
from rambla.models.openai_models import (
    OpenaiChatCompletionModel,
    OpenaiCompletionModel,
    OpenaiEmbeddings,
)

MODEL_MAP = {
    "openai_chat": OpenaiChatCompletionModel,
    "openai_completion": OpenaiCompletionModel,
    "openai_embeddings": OpenaiEmbeddings,
    "huggingface_llm": TextGenerationModel,
    "openai_35_chat": OpenaiChatCompletionModel,
}


def _validate_model_name(model_name: str):
    """Checks provided name is in `MODEL_MAP`"""
    if model_name not in MODEL_MAP.keys():
        raise ValueError(
            f"Invalid name: {model_name}. Name must be one of {list(MODEL_MAP.keys())}"
        )


class LLMConfig(BaseModel):
    name: str
    params: dict

    @validator("name")
    @classmethod
    def validate_name(cls, v):
        _validate_model_name(v)
        return v


def build_llm(config: Union[dict, LLMConfig]) -> BaseAPIModel:
    """Prepares LLM based on config."""
    if not isinstance(config, LLMConfig):
        config = LLMConfig.parse_obj(config)
    model_class: Type[BaseAPIModel] = MODEL_MAP[config.name]
    return model_class.from_config(config.params)
