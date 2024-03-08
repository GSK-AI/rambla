import os
from unittest import mock

import backoff
import pytest

from rambla.models.openai_models import OpenaiBaseModel
from rambla.utils.requests import (
    OPENAI_API_RETRY_EXCEPTIONS,
    BackoffConfig,
    SkipDecoratorConfig,
    SkipOperator,
    backoff_decorator_factory,
    make_default_backoff_decorator,
    skip_decorator,
)

# flake8: noqa: N802


class MockError(Exception):
    pass


@pytest.fixture
def expo_backoff_config_fixture_factory():
    def expo_backoff_config_fixture(max_retries: int, base: float):
        return BackoffConfig(
            exception=Exception,
            wait_gen=backoff.expo,
            max_tries=max_retries,
            raise_on_giveup=True,
            params={"base": base},
        )

    return expo_backoff_config_fixture


@pytest.fixture
def constant_backoff_config_fixture_factory():
    def constant_backoff_config_fixture(max_retries: int, interval: float):
        return BackoffConfig(
            exception=Exception,
            wait_gen=backoff.constant,
            max_tries=max_retries,
            raise_on_giveup=True,
            params={"interval": interval},
        )

    return constant_backoff_config_fixture


def test_backoff_decorator_factory_constant(constant_backoff_config_fixture_factory):
    max_retries = 5
    n_retries = 0

    config = constant_backoff_config_fixture_factory(
        max_retries=max_retries, interval=1e-5
    )
    decorator = backoff_decorator_factory(config)

    def mock_api_call(prompt: str) -> str:
        nonlocal n_retries

        if n_retries < (max_retries - 1):
            n_retries += 1
            raise MockError("Mock error")
        else:
            return "mock_response"

    response = decorator(mock_api_call)("mock_prompt")

    assert response == "mock_response"
    assert n_retries == (max_retries - 1)


def test_backoff_decorator_factory_expo(expo_backoff_config_fixture_factory):
    max_retries = 5
    n_retries = 0

    config = expo_backoff_config_fixture_factory(max_retries=max_retries, base=1e-5)
    decorator = backoff_decorator_factory(config)

    def mock_api_call(prompt: str) -> str:
        nonlocal n_retries

        if n_retries < (max_retries - 1):
            n_retries += 1
            raise MockError("Mock error")
        else:
            return "mock_response"

    response = decorator(mock_api_call)("mock_prompt")

    assert response == "mock_response"
    assert n_retries == (max_retries - 1)


@mock.patch("rambla.utils.requests.os")
@mock.patch("rambla.utils.requests.backoff_decorator_factory")
def test_make_default_backoff_decorator_openai_model(
    mock_backoff_decorator_factory, mock_os
):
    mock_backoff_decorator_factory.return_value = "dummy return value"
    expected_exceptions = tuple(OPENAI_API_RETRY_EXCEPTIONS)

    mock_getenv = mock.Mock(side_effect=("3", "40"))
    mock_os.getenv = mock_getenv

    excepted_max_tries = 3
    expected_params = {"interval": 40}
    expected_backoff_config = BackoffConfig(
        exception=expected_exceptions,
        wait_gen=backoff.constant,
        max_tries=excepted_max_tries,  # type: ignore
        raise_on_giveup=True,
        params=expected_params,
    )

    llm = mock.MagicMock(spec=OpenaiBaseModel)
    #
    output = make_default_backoff_decorator(llm)

    #
    assert output == "dummy return value"
    mock_backoff_decorator_factory.assert_called_with(expected_backoff_config)


def test_SkipDecoratorConfig_single():
    skip_decorator_config = {
        "exception": "ValueError",
        "null_response": -1,
    }

    config = SkipDecoratorConfig.parse_obj(skip_decorator_config)

    assert config.exception == (ValueError,)


def test_SkipDecoratorConfig_single_tuple():
    skip_decorator_config = {
        "exception": ("ValueError",),
        "null_response": -1,
    }

    config = SkipDecoratorConfig.parse_obj(skip_decorator_config)

    assert config.exception == (ValueError,)


def test_SkipDecoratorConfig_tuple():
    skip_decorator_config = {
        "exception": (
            "ValueError",
            "TypeError",
        ),
        "null_response": -1,
    }

    config = SkipDecoratorConfig.parse_obj(skip_decorator_config)

    assert config.exception == (ValueError, TypeError)


def test_SkipOperator_no_args_no_kwargs_no_error():
    def mock_func():
        return 3

    decorated_func = SkipOperator(mock_func, ValueError)

    # asserts
    output = decorated_func()
    assert output == 3


def test_SkipOperator_with_args_no_error():
    def mock_func(first_arg):
        return first_arg

    decorated_func = SkipOperator(mock_func, ValueError)

    # asserts
    input_arg = "__dummy__"
    output = decorated_func(input_arg)
    assert output == input_arg


def test_SkipOperator_with_args_and_kwargs_no_error():
    def mock_func(first_arg, second_arg):
        return first_arg + second_arg

    decorated_func = SkipOperator(mock_func, ValueError)

    # asserts
    first_arg = "__dummy__"
    second_arg = "__also_dummy__"
    output = decorated_func(first_arg, second_arg=second_arg)
    assert output == first_arg + second_arg


def test_SkipOperator_raises_different_error():
    def mock_func():
        raise TypeError

    decorated_func = SkipOperator(mock_func, ValueError)

    # asserts
    with pytest.raises(TypeError):
        decorated_func()


def test_SkipOperator_catches_error():
    def mock_func():
        raise ValueError

    null_response = -1
    decorated_func = SkipOperator(mock_func, ValueError, null_response)

    # asserts
    output = decorated_func()
    assert output == null_response


def test_skip_decorator_no_args_no_kwargs_no_error():
    def mock_func():
        return 3

    decorated_func = skip_decorator(ValueError)(mock_func)

    # asserts
    output = decorated_func()
    assert output == 3


def test_skip_decorator_with_args_no_error():
    def mock_func(first_arg):
        return first_arg

    decorated_func = skip_decorator(ValueError)(mock_func)

    # asserts
    input_arg = "__dummy__"
    output = decorated_func(input_arg)
    assert output == input_arg


def test_skip_decorator_with_args_and_kwargs_no_error():
    def mock_func(first_arg, second_arg):
        return first_arg + second_arg

    decorated_func = skip_decorator(ValueError)(mock_func)

    # asserts
    first_arg = "__dummy__"
    second_arg = "__also_dummy__"
    output = decorated_func(first_arg, second_arg=second_arg)
    assert output == first_arg + second_arg


def test_skip_decorator_raises_different_error():
    def mock_func():
        raise TypeError

    decorated_func = skip_decorator(ValueError)(mock_func)

    # asserts
    with pytest.raises(TypeError):
        decorated_func()


def test_skip_decorator_catches_error():
    def mock_func():
        raise ValueError

    null_response = -1
    decorated_func = skip_decorator(ValueError, null_response)(mock_func)

    # asserts
    output = decorated_func()
    assert output == null_response
