from __future__ import annotations

import os
from typing import Dict, Hashable, List, Optional, Union

import openai
from pydantic import BaseModel, validator

from rambla.models.base_model import (
    BaseAPIEmbeddingsModel,
    BaseLLM,
    InvalidResponseError,
)


class OpenAIInvalidResponseError(InvalidResponseError):
    pass


class OpenAILLMParams(BaseModel):
    temperature: float
    engine: str
    max_tokens: Optional[int]
    top_p: Optional[float]
    async_calls: Optional[bool]
    api_type: Optional[str] = "azure"


class OpenaiBaseModel(BaseLLM):
    def __init__(
        self,
        temperature: float,
        engine: str,
        max_tokens: int = 2000,
        top_p: float = 0.95,
        async_calls: bool = False,
        api_type: Optional[str] = "azure",
    ):
        self.async_calls = async_calls

        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        if api_type == "azure":
            _client_class = (
                openai.AsyncAzureOpenAI if self.async_calls else openai.AzureOpenAI
            )
            self._client = _client_class(
                api_key=os.environ["OPENAI_API_KEY"],
                azure_endpoint=os.environ["OPENAI_API_BASE"],
                api_version=os.environ["OPENAI_API_VERSION"],
            )
        else:
            _client_class = openai.AsyncOpenAI if self.async_calls else openai.OpenAI
            self._client = _client_class(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ["OPENAI_API_BASE"],
            )

    @classmethod
    def from_config(cls, config: Union[dict, OpenAILLMParams]) -> OpenaiBaseModel:
        if isinstance(config, dict):
            config = OpenAILLMParams.parse_obj(config)
        return cls(**config.dict(exclude_unset=True))

    @property
    def is_async(self) -> bool:
        return self.async_calls

    @staticmethod
    def validate_response(content: str):  # noqa: D102
        if not isinstance(content, str):
            raise OpenAIInvalidResponseError(
                f"Model returned invalid response: {content}"
            )

    def _get_model_params(self, exclude_engine: bool = False) -> dict:
        params = dict(
            engine=self.engine,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        if exclude_engine:
            params.pop("engine")
        return params

    @property
    def _model_dict(self) -> dict:
        """Keyword arguments for model."""
        return self._get_model_params(exclude_engine=False)


class OpenaiChatCompletionModel(OpenaiBaseModel):
    def __init__(
        self,
        temperature: float,
        engine: str,
        max_tokens: int = 800,
        top_p: float = 0.95,
        async_calls: bool = False,
    ):
        super().__init__(
            temperature=temperature,
            engine=engine,
            max_tokens=max_tokens,
            top_p=top_p,
            async_calls=async_calls,
        )

    @classmethod
    def from_config(cls, config: Union[dict, OpenAILLMParams]):
        if isinstance(config, dict):
            config = OpenAILLMParams.parse_obj(config)
        return cls(**config.dict(exclude_unset=True))

    def generate(self, prompt: str) -> str:
        """Call to openai chatcompletion model."""
        messages = [{"role": "user", "content": prompt}]
        response = self._client.chat.completions.create(
            model=self.engine,
            messages=messages,
            **self._get_model_params(exclude_engine=True),
        )
        content = response.choices[0].message.content
        OpenaiChatCompletionModel.validate_response(content)
        return content

    async def async_generate(self, prompt: str) -> str:
        """Async call to openai chatcompletion model."""
        messages = [{"role": "user", "content": prompt}]
        response = await self._client.chat.completions.create(
            model=self.engine,
            messages=messages,
            **self._get_model_params(exclude_engine=True),
        )
        content = response.choices[0].message.content
        OpenaiChatCompletionModel.validate_response(content)

        return content


class OpenaiCompletionModel(OpenaiBaseModel):
    def __init__(
        self,
        temperature: float,
        engine: str = "text-davinci-003",
        max_tokens: int = 800,
        top_p: float = 0.95,
        async_calls: bool = False,
    ):
        super().__init__(
            temperature=temperature,
            engine=engine,
            max_tokens=max_tokens,
            top_p=top_p,
            async_calls=async_calls,
        )

    @classmethod
    def from_config(cls, config: Union[dict, OpenAILLMParams]):
        if isinstance(config, dict):
            config = OpenAILLMParams.parse_obj(config)
        return cls(**config.dict(exclude_unset=True))

    def generate(self, prompt: str) -> str:
        """Call to openai chatcompletion model."""
        response = self._client.completions.create(
            model=self.engine,
            prompt=prompt,
            **self._get_model_params(exclude_engine=True),
        )
        content = response.choices[0].text
        OpenaiCompletionModel.validate_response(content)
        return content

    async def async_generate(self, prompt: str) -> str:
        raise NotImplementedError("Not supported in openai v1")


class OpenaiEmbeddingsConfig(BaseModel):
    engine: str
    api_type: Optional[str] = "azure"
    async_calls: bool = True

    @validator("api_type")
    @classmethod
    def validate_api_type(cls, api_type):
        if isinstance(api_type, str):
            assert api_type == "azure"
        else:
            assert isinstance(api_type, type(None))
        return api_type


class OpenaiEmbeddings(BaseAPIEmbeddingsModel):
    """Example usage:

    ```python
        model = OpenaiEmbeddings()

        response_module = ResponseComponent(
            model=model,
            run_async=True,
            response_cache_fname="response.pkl",
            )
        dataset = Dataset.from_dict({
            "index": [0, 1],
            "texts": ["what is NASH?", "What is 2+2?"]
        })
        response = response_module.batch_generate(
            dataset, index_field="index", prompt_field="texts"
        )

        --> Dataset({
        -->     features: ['index', 'texts', 'response'],
        -->     num_rows: 2
        -->})
    ```

    """

    def __init__(
        self,
        engine: str = "text-embedding-ada-002",
        async_calls: bool = True,
        api_type: str = "azure",
    ):
        self.engine = engine
        self.async_calls = async_calls

        if isinstance(api_type, str):
            if api_type != "azure":
                raise ValueError

        if api_type == "azure":
            self._client = openai.AzureOpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                azure_endpoint=os.environ["OPENAI_API_BASE"],
                api_version=os.environ["OPENAI_API_VERSION"],
            )
        else:
            self._client = openai.OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ["OPENAI_API_BASE"],
            )

    @property
    def is_async(self) -> bool:
        return self.async_calls

    @property
    def _model_dict(self) -> Dict[str, Hashable]:
        return {"engine": self.engine}

    @classmethod
    def from_config(
        cls, config: Union[dict, OpenaiEmbeddingsConfig]
    ) -> OpenaiEmbeddings:
        if not isinstance(config, OpenaiEmbeddingsConfig):
            config = OpenaiEmbeddingsConfig.parse_obj(config)
        return cls(
            engine=config.engine,
            async_calls=config.async_calls,
            api_type=config.api_type,
        )

    def generate(self, prompt: str) -> List[float]:
        """Call to openai embedding api."""
        prompt = prompt.replace("\n", " ")
        response = self._client.embeddings.create(input=prompt, model=self.engine)
        return response.data[0].embedding

    async def async_generate(self, prompt: str) -> List[float]:
        raise NotImplementedError("Not supported yet.")
