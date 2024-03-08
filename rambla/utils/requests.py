import functools
import logging
import os
from typing import Callable, Union

import backoff
import openai
from pydantic import BaseModel, Field, validator

from rambla.models.base_model import BaseLLM
from rambla.models.huggingface import MaximumContextLengthExceededError
from rambla.models.openai_models import OpenaiBaseModel, OpenaiEmbeddings
from rambla.utils.types import DecoratorType, FlexibleExceptionType

logger = logging.getLogger(__file__)


ACCEPTED_MODEL_TYPES_FOR_BACKOFF = (OpenaiBaseModel, OpenaiEmbeddings)

# NOTE: more info here: https://platform.openai.com/docs/guides/error-codes
OPENAI_API_RETRY_EXCEPTIONS = [
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
]


class SkipDecoratorConfig(BaseModel):
    exception: FlexibleExceptionType = (
        MaximumContextLengthExceededError,
        openai.BadRequestError,
    )

    # NOTE: This might become restrictive at some point.
    # It needs to have the same type as the rest of the responses
    # Otherwise, `datasets.Dataset.add_column` will throw an error.
    null_response: str = "-1"

    @validator("exception", pre=True)
    @classmethod
    def validate_exception(cls, exception):
        """The purpose of this validator is to parse any strings to Exceptions."""
        if isinstance(exception, tuple):
            exception = list(exception)
        if isinstance(exception, (str, list)):
            if isinstance(exception, str):
                exception = [exception]
            for ii, item in enumerate(exception):
                if isinstance(item, str):
                    exception[ii] = eval(item)
            exception = tuple(exception)
        return exception


class SkipOperator:
    """To be used as a skip decorator.

    Class that can be used with LLMs to skip any requests that raise certain errors.
    """

    def __init__(
        self,
        func: Callable,
        exceptions_to_skip: FlexibleExceptionType,
        null_response: Union[int, str] = "-1",
    ):
        """_summary_

        Parameters
        ----------
        func : Callable
            The function that was decorated
        exceptions_to_skip : FlexibleExceptionType
            If any of these exceptions is raised then
            the null response will be returned.
        null_response : Union[int, str], optional
            What will be returned if one of `exceptions_to_skip`
            is raised by `func`, by default -1
        """
        self.func = func
        self.exceptions_to_skip = exceptions_to_skip
        self.null_response = null_response

    def __call__(self, *args, **kwargs):
        try:
            response = self.func(*args, **kwargs)
        except self.exceptions_to_skip:
            response = self.null_response
        return response


def skip_decorator(
    exceptions_to_skip: FlexibleExceptionType,
    null_response: Union[int, str] = -1,
) -> Callable:
    """Builds skip decorator."""

    def _skip(func: Callable) -> Callable:
        skip_op = SkipOperator(
            func=func,
            exceptions_to_skip=exceptions_to_skip,
            null_response=null_response,
        )
        decorator = functools.wraps(func)(skip_op)
        return decorator

    return _skip


def skip_decorator_factory(
    config: Union[dict, SkipDecoratorConfig]
) -> DecoratorType:  # noqa: D103
    if not isinstance(config, SkipDecoratorConfig):
        config = SkipDecoratorConfig.parse_obj(config)
    return skip_decorator(
        exceptions_to_skip=config.exception, null_response=config.null_response
    )


class BackoffConfig(BaseModel):
    """Config for `backoff` decorator.

    Examples
    --------
    ```python
    constant_backoff_config = BackoffConfig(
        exception=Exception,
        wait_gen=backoff.constant,
        max_tries=3,
        raise_on_giveup=3,
        params={"internal": 30},
    )

    exp_backoff_config = BackoffConfig(
        exception=Exception,
        wait_gen=backoff.expo,
        max_tries=3,
        raise_on_giveup=3,
        params={"base": 2, "factor": 1, "max_value": None},
    )
    ```
    """

    exception: FlexibleExceptionType = Exception
    wait_gen: backoff._typing._WaitGenerator = backoff.constant
    max_tries: int = 3
    raise_on_giveup: bool = True
    params: dict = Field(default={})

    @validator("wait_gen", pre=True)
    @classmethod
    def validate_wait_gen(cls, wait_gen):
        if isinstance(wait_gen, str):
            wait_gen = wait_gen.lower()

            supported_wait_gens = ["constant", "fibo", "expo"]
            error_message = (
                f"Received input {wait_gen=} not supported. "
                f"Try one of {supported_wait_gens=}."
            )
            assert wait_gen in supported_wait_gens, error_message
            wait_gen = getattr(backoff, wait_gen)
        return wait_gen

    @validator("exception", pre=True)
    @classmethod
    def validate_exception(cls, exception):
        if isinstance(exception, tuple):
            exception = list(exception)
        if isinstance(exception, (str, list)):
            if isinstance(exception, str):
                exception = [exception]
            for ii, item in enumerate(exception):
                if isinstance(item, str):
                    exception[ii] = eval(item)
            exception = tuple(exception)
        return exception


def backoff_decorator_factory(config: BackoffConfig) -> DecoratorType:
    """Builds backoff decorator."""
    return backoff.on_exception(
        wait_gen=config.wait_gen,
        exception=config.exception,
        max_tries=config.max_tries,
        raise_on_giveup=config.raise_on_giveup,
        jitter=None,
        **config.params,
    )


def make_default_backoff_decorator(llm: BaseLLM) -> DecoratorType:  # noqa: D103
    if isinstance(llm, (OpenaiBaseModel, OpenaiEmbeddings)):
        exception = tuple(OPENAI_API_RETRY_EXCEPTIONS)
    else:
        raise ValueError(f"Unsupported model type: {type(llm)}.")

    constant_backoff_config = BackoffConfig(
        exception=exception,
        wait_gen=backoff.constant,
        max_tries=int(os.getenv("BACKOFF_MAX_TRIES", 3)),  # type: ignore
        raise_on_giveup=True,
        params={"interval": int(os.getenv("BACKOFF_INTERVAL", 40))},
    )

    return backoff_decorator_factory(constant_backoff_config)
