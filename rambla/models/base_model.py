from __future__ import annotations

import abc
from typing import Any, Dict, Hashable, List, Optional, Union

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from rambla.utils.pytorch import switch_model_device


class InvalidResponseError(Exception):
    pass


class BaseAPIModel(abc.ABC):
    @abc.abstractmethod
    def __init__(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def is_async(self) -> bool:
        ...

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config) -> BaseAPIModel:  # noqa: N805
        ...

    @property
    @abc.abstractmethod
    def _model_dict(self) -> Dict[str, Hashable]:
        """Returns a config dictionary for hashing purposes

        Property should return a dictionary of hashable model parameters
        used in hashing a model config for caching purposes.
        """
        ...


class BaseLLM(BaseAPIModel):
    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        ...

    @abc.abstractmethod
    async def async_generate(self, prompt: str) -> str:
        ...


class BaseHuggingFaceModel(abc.ABC):
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        model: PreTrainedModel,
        device: Optional[str] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.device = device
        self.model = model
        # Ensures model loaded to correct device
        if self.device:
            self.model = switch_model_device(self.model, self.device)

    @property
    @abc.abstractmethod
    def _model_dict(self) -> Dict[str, Hashable]:
        """Returns a config dictionary for hashing purposes

        Property should return a dictionary of hashable model parameters
        used in hashing a model config for caching purposes.
        """
        ...

    async def async_generate(self, prompt: str) -> Any:
        """Created for compatibility purposes with response component protocol"""
        raise NotImplementedError("Cannot run huggingface model in async mode.")

    @abc.abstractmethod
    def generate(self, text: str) -> Any:
        ...


class BaseAPIEmbeddingsModel(BaseAPIModel):
    @abc.abstractmethod
    def generate(self, prompt: str) -> List[float]:
        ...

    @abc.abstractmethod
    async def async_generate(self, prompt: str) -> List[float]:
        ...
