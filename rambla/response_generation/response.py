from __future__ import annotations

import asyncio
import functools
import hashlib
import json
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

import aiolimiter
from datasets import Dataset
from pydantic import BaseModel, Extra, root_validator
from tqdm import tqdm
from tqdm.asyncio import tqdm as asyncio_tqdm

from rambla.utils.caching import async_file_cache, file_cache
from rambla.utils.misc import initialize_logger
from rambla.utils.requests import (
    ACCEPTED_MODEL_TYPES_FOR_BACKOFF,
    BackoffConfig,
    DecoratorType,
    SkipDecoratorConfig,
    backoff_decorator_factory,
    make_default_backoff_decorator,
    skip_decorator_factory,
)

logger = initialize_logger(__name__)


# flake8: noqa: E501


class RequestFuncProtocol(Protocol):
    """Defines the expected IO signature for `model.generate`"""

    def __call__(self, prompt: str) -> Any:
        ...


class ResponseType(BaseModel):
    response: Any
    extra: Optional[Any] = None

    class Config:  # noqa: D106
        extra = Extra.forbid

    @classmethod
    def parse_general(cls, response: Any) -> ResponseType:
        if not isinstance(response, dict):
            response = {"response": response}
        return cls.parse_obj(response)


@runtime_checkable
class ModelProtocol(Protocol):
    def generate(self, prompt: str) -> Any:
        ...

    async def async_generate(self, prompt: str) -> Any:
        ...

    @property
    def _model_dict(self) -> Dict[str, Hashable]:
        ...


class ResponseComponentConfig(BaseModel):
    cache_base_dir: Optional[Union[str, Path]]
    response_cache_fname: str = "response.json"
    max_rate: Union[int, float] = 4
    run_async: bool = False
    time_period: int = 60

    # NOTE: it's important to ensure that there is no
    # overlap between the exceptions that are retried
    # and the exceptions that will be skipped.
    backoff_decorator_config: Optional[Union[str, BackoffConfig]] = "DEFAULT"

    # TODO: Set up `skip_decorator_config` in a similar
    # way to `backoff_decorator_config` so that a decorator
    # is built according to model type.
    skip_decorator_config: Optional[SkipDecoratorConfig]

    class Config:  # noqa: D106
        extra = Extra.forbid

    @root_validator()
    @classmethod
    def validate_decorator_configs(cls, values):
        skip_decorator_config = values["skip_decorator_config"]
        backoff_decorator_config = values["backoff_decorator_config"]
        if (
            skip_decorator_config
            and backoff_decorator_config
            and isinstance(backoff_decorator_config, BackoffConfig)
        ):
            assert not set(skip_decorator_config.exception).intersection(
                set(backoff_decorator_config.exception)
            )
        return values


class ResponseComponent:
    def __init__(
        self,
        cache_base_dir: Optional[Union[str, Path]] = None,
        response_cache_fname: str = "response.json",
        run_async: Optional[bool] = False,
        max_rate: Union[int, float] = 4,
        time_period: int = 60,
        backoff_decorator: Optional[Union[Literal["DEFAULT"], BackoffConfig]] = None,
        skip_decorator: Optional[DecoratorType] = None,
    ) -> None:
        """Generates responses to a prompt dataset by quering an LLM API

        Parameters
        ----------
        cache_base_dir : Optional[Union[str, Path]], optional
            Base directory to store cached responses in. Will create a subdirectory
                based on the model's hash. If none provided then
                creates a 'data' directory in the package root
                directory. By default None
        response_cache_fname : str, optional
            Name to save each cached response, by default "response.json"
        run_async : Optional[bool], optional
            If true then makes requests to the API asynchronously, by default False
        max_rate : Union[int, float], optional
            Max number of requests to make per time period, only valid if
                run_async is True, by default 4
        time_period : int, optional
            Time period to apply max_rate over (in seconds). For example
                to make 4 requests per minute, then: max_rate=4 and
                time_period=60. By default 60 (per minute).
        backoff_decorator : Union[Literal["DEFAULT"], DecoratorType], optional
            Argument for backoff decorator. If "DEFAULT" is passed, then we
            build a constant backoff decorator
        """
        self.max_rate = max_rate
        self.time_period = time_period
        self.run_async = run_async
        self.cache_base_dir = cache_base_dir
        self.response_cache_fname = response_cache_fname

        # Sets up a rate limiter to ensure requests don't exceed limits
        # NOTE: this will get get used if `self.run_async` is set to `True`.
        self._limiter = aiolimiter.AsyncLimiter(
            max_rate=max_rate, time_period=time_period
        )

        self.backoff_decorator = backoff_decorator
        self.skip_decorator = skip_decorator

        if self.skip_decorator and self.run_async:
            logger.info(
                "A skip_decorator is incompotible with `run_async=True`. "
                "The skip_decorator will not be applied."
            )

    @property
    def cache_responses(self) -> bool:
        return self.cache_base_dir is not None

    @classmethod
    def from_config(
        cls, config: Union[dict, ResponseComponentConfig]
    ) -> ResponseComponent:
        if not isinstance(config, ResponseComponentConfig):
            config = ResponseComponentConfig.parse_obj(config)

        backoff_decorator = config.backoff_decorator_config
        if isinstance(backoff_decorator, BackoffConfig):
            backoff_decorator = backoff_decorator_factory(
                config.backoff_decorator_config
            )

        skip_decorator = config.skip_decorator_config
        if isinstance(skip_decorator, SkipDecoratorConfig):
            skip_decorator = skip_decorator_factory(config.skip_decorator_config)

        return cls(
            cache_base_dir=config.cache_base_dir,
            response_cache_fname=config.response_cache_fname,
            run_async=config.run_async,
            max_rate=config.max_rate,
            time_period=config.time_period,
            backoff_decorator=backoff_decorator,
            skip_decorator=skip_decorator,
        )

    def _build_backoff_decorator(self, model: ModelProtocol) -> DecoratorType:
        if (
            isinstance(self.backoff_decorator, str)
            and self.backoff_decorator == "DEFAULT"
        ):
            backoff_decorator = make_default_backoff_decorator(model)
        else:
            backoff_decorator = self.backoff_decorator

        return backoff_decorator

    @staticmethod
    def _model_hash(model_dict: Dict[str, Hashable]) -> str:
        """Creates a hash for a model config to use for caching responses"""
        hash_key = hashlib.md5()
        hash_key.update(json.dumps(model_dict, sort_keys=True, default=vars).encode())
        return hash_key.hexdigest()

    def _configure_cache_dir(
        self,
        model_dict: Dict[str, Hashable],
    ):
        if not self.cache_base_dir:
            raise ValueError(
                "This method cannot be used if `self.cache_base_dir` is not set."
            )

        cache_base_dir = Path(self.cache_base_dir)

        logger.info(f"Using {cache_base_dir} to cache responses")

        cache_dir = cache_base_dir / ResponseComponent._model_hash(model_dict)

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir

    def _make_request_func(self, model: ModelProtocol) -> Callable:
        """Builds the function to make requests to the model.

        Step 1.: Selects the right request method (sync/async)
        Step 2.: If a backoff decorator is provided, then it's used
        Step 3.: If caching is requested, then we decorate with the caching decorator.
        Step 4.: If skipping is requested, then we decorate with the skip decorator.
        """
        if self.run_async:
            request_func = model.async_generate
            cache_decorator_func = async_file_cache
        else:
            request_func = model.generate
            cache_decorator_func = file_cache

        if self.backoff_decorator:
            if isinstance(model, ACCEPTED_MODEL_TYPES_FOR_BACKOFF):
                backoff_decorator = self._build_backoff_decorator(model)
                request_func = functools.wraps(request_func)(backoff_decorator)(
                    request_func
                )
            else:
                logger.warn(
                    f"Backoff decorator provided but model is of type: {type(model)}. "
                    f"Only {ACCEPTED_MODEL_TYPES_FOR_BACKOFF} are supported. "
                    "Backoff will not be applied."
                )

        if self.cache_responses:
            cache_decorator = cache_decorator_func(
                self.cache_dir, self.response_cache_fname
            )
            request_func = functools.wraps(request_func)(cache_decorator)(request_func)

        if self.skip_decorator and not self.run_async:
            request_func = self.skip_decorator(request_func)

        return request_func

    async def _async_generate(
        self, request_func: RequestFuncProtocol, prompt: str
    ) -> ResponseType:
        """Generates responses and caches if self.cache_responses is True"""
        async with self._limiter:
            response = await request_func(prompt=prompt)

        parsed_response = ResponseType.parse_general(response)
        return parsed_response

    def _generate(self, request_func: RequestFuncProtocol, prompt: str) -> ResponseType:
        """Generates responses and caches if self.cache_responses is True"""
        response = request_func(prompt=prompt)
        parsed_response = ResponseType.parse_general(response)
        return parsed_response

    async def _batch_generate_async(
        self,
        request_func: RequestFuncProtocol,
        prompts: List[str],
        verbose: bool = True,
    ) -> List[ResponseType]:
        """Batch generates responses asynchronously"""
        tasks = [
            asyncio.ensure_future(self._async_generate(request_func, prompt))
            for prompt in prompts
        ]

        if verbose:
            responses = await asyncio_tqdm.gather(*tasks)
        else:
            responses = await asyncio.gather(*tasks)

        return responses

    def _batch_generate_sync(
        self,
        request_func: RequestFuncProtocol,
        prompts: List[str],
        verbose: bool = True,
    ) -> List[ResponseType]:
        if verbose:
            prompts = tqdm(prompts)  # type: ignore

        responses: List[ResponseType] = []
        for prompt in prompts:
            response = self._generate(request_func, prompt)
            responses.append(response)
        return responses

    @staticmethod
    def _augment_dataset_with_responses(
        dataset: Dataset,
        responses: List[ResponseType],
        *,
        response_field: str = "response",
        extra_field: str = "extra",
    ) -> Dataset:
        """NOTE:

        `datasets.Dataset` and `pyarrow.Table` cannot handle columns of mixed types
        When the inputs are struct (e.g., `dict`) then they need to follow the same schema.

        For example, here are some input-output pairs:

        Example 1.:
            Input:  [{"dummy": 1}, {"dummy": 3}, {"dummy": 2}]
            Output: [{"dummy": 1}, {"dummy": 3}, {"dummy": 2}]

        Example 2.: Can handle struct + none
            Input:  [{"dummy": 1}, {"dummy": 3}, None]
            Output: [{"dummy": 1}, {"dummy": 3}, None]

        Example 3.: The schema changes to match the rest
            Input:  [{"dummy": 1}, {"dummy": 3}, {}]
            Output: [{"dummy": 1}, {"dummy": 3}, {"dummy": None}]

        Example 4.: Same as above
            Input:  [{"dummy": 1}, {"dummy": 3}, {"also_dummy": 4}],
            Output: [{"dummy": 1, "also_dummy": None}, {"dummy": 3, "also_dummy": None}, {"dummy": None, "also_dummy": 4}],

        Example 5.: This would raise an error: `pyarrow.lib.ArrowInvalid: cannot mix struct and non-struct, non-null values`
            Input:  [{"dummy": 1}, {"dummy": 3}, "dummy"]

        """
        responses_main = [item.response for item in responses]

        response_dataset: Dataset = dataset.add_column(response_field, responses_main)  # type: ignore

        responses_extra = [item.extra for item in responses]
        if any(responses_extra):
            response_dataset = response_dataset.add_column(extra_field, responses_extra)  # type: ignore

        return response_dataset

    def batch_generate(
        self,
        model: ModelProtocol,
        prompt_dataset: Dataset,
        *,
        prompt_field: str = "prompt",
        response_field: str = "response",
        extra_field: str = "extra",
        verbose: bool = True,
    ) -> Dataset:
        """Generates responses to every prompt in a dataset

        Parameters
        ----------
        model : ModelProtocol
            Instance of ModelProtocol. Class should wrap calls to an API. Currently
            supports text generation and embedding generation models.
        prompt_dataset : Dataset
            Huggingface dataset of prompts containing the following colums:
                "index": An index column
                "prompt": The prompt to pass to the model.
        prompt_field : str
            Name of dataset field containing prompts, by default "prompt"
        response_field : str
            Name of dataset field containing responses, by default "response"
        verbose : bool
            Whether or not to display a status bar, by default True.

        Returns
        -------
        Dataset
            A dataset of responses to each prompt in the input dataset
        """
        # Setup
        if not isinstance(model, ModelProtocol):
            raise TypeError(
                "Input `model` does not conform to the `ModelProtocol` protocol."
            )

        if self.cache_responses:
            self._configure_cache_dir(model._model_dict)

        # NOTE: This will return a decorated function of `model.generate`
        # This might include:
        # 1. A caching decorator
        # 2. A backoff-retry decorator
        # 3. both
        request_func = self._make_request_func(model)

        # Call
        prompts = prompt_dataset[prompt_field]

        if self.run_async:
            responses = asyncio.run(
                self._batch_generate_async(request_func, prompts, verbose)
            )
        else:
            responses = self._batch_generate_sync(request_func, prompts, verbose)

        response_dataset = ResponseComponent._augment_dataset_with_responses(
            dataset=prompt_dataset,
            responses=responses,
            response_field=response_field,
            extra_field=extra_field,
        )
        return response_dataset
