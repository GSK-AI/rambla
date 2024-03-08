import os
import random
from pathlib import Path
from typing import Any, Callable, Optional

import aiolimiter
import pytest
from datasets import Dataset
from pydantic import ValidationError

from rambla.response_generation import response
from rambla.response_generation.response import (
    ResponseComponent,
    ResponseComponentConfig,
    ResponseType,
)

# flake8: noqa: N802


@pytest.fixture
def responsetype_factory() -> Callable:
    def inner(
        responses: list[str], extras: Optional[list[str]] = None
    ) -> list[ResponseType]:
        out = []
        if extras:
            assert len(responses) == len(extras)
            for response, extra in zip(responses, extras):
                out.append(ResponseType(response=response, extra=extra))
        else:
            for response in responses:
                out.append(ResponseType(response=response))
        return out

    return inner


@pytest.fixture
def mock_response_component(response_component_config: dict) -> ResponseComponent:
    return ResponseComponent.from_config(response_component_config)


@pytest.mark.parametrize(
    "inputs, expected_output",
    [
        (
            "__dummy_response__",
            response.ResponseType(response="__dummy_response__"),
        ),
        (
            {"response": "__dummy_response__"},
            response.ResponseType(response="__dummy_response__"),
        ),
        (
            {"response": "__dummy_response__", "extra": "also_dummy"},
            response.ResponseType(response="__dummy_response__", extra="also_dummy"),
        ),
        (
            {"response": "__dummy_response__", "extra": {"a dict": 1}},
            response.ResponseType(response="__dummy_response__", extra={"a dict": 1}),
        ),
    ],
)
def test_ResponseType_parse_general(inputs, expected_output):
    output = response.ResponseType.parse_general(inputs)
    assert output == expected_output


def test_ResponseComponent_augment_dataset_with_responses_no_extras(
    mock_prompt_dataset: Dataset,
    responsetype_factory: Callable,
):
    responses = list("ABC")
    response_types = responsetype_factory(responses=responses)

    #
    output_dataset = ResponseComponent._augment_dataset_with_responses(
        dataset=mock_prompt_dataset,
        responses=response_types,
        response_field="response",
        extra_field="extra",
    )

    #
    assert output_dataset["response"] == responses
    assert "extra" not in output_dataset


@pytest.mark.parametrize(
    "extras, expected_extras",
    [
        (
            [{"dummy": 1}, {"dummy": 3}, {"dummy": 2}],
            [{"dummy": 1}, {"dummy": 3}, {"dummy": 2}],
        ),
        (
            [{"dummy": 1}, {"dummy": 3}, None],
            [{"dummy": 1}, {"dummy": 3}, None],
        ),
        (
            [{"dummy": 1}, {"dummy": 3}, {}],
            [{"dummy": 1}, {"dummy": 3}, {"dummy": None}],
        ),
        (
            [{"dummy": 1}, {"dummy": 3}, {"also_dummy": 4}],
            [
                {"dummy": 1, "also_dummy": None},
                {"dummy": 3, "also_dummy": None},
                {"dummy": None, "also_dummy": 4},
            ],
        ),
        # NOTE: this would raise a pyarrow error:
        # pyarrow.lib.ArrowInvalid: cannot mix struct and non-struct, non-null values
        # (
        #     [{"dummy": 1}, {"dummy": 3}, "dummy"],
        #     [{"dummy": 1}, {"dummy": 3}, None],
        # ),
    ],
)
def test_ResponseComponent_augment_dataset_with_responses_with_extras_of_different_schema(
    mock_prompt_dataset: Dataset,
    responsetype_factory: Callable,
    extras: list[Any],
    expected_extras: list[Any],
):
    responses = list("ABC")
    response_types = responsetype_factory(responses=responses, extras=extras)

    #
    output_dataset = ResponseComponent._augment_dataset_with_responses(
        dataset=mock_prompt_dataset,
        responses=response_types,
        response_field="response",
        extra_field="extra",
    )

    #
    assert output_dataset["extra"] == expected_extras


def test_model_hash_modify_param(mock_model_dict: dict) -> None:
    original_hash = ResponseComponent._model_hash(mock_model_dict)

    modified_model_dict = mock_model_dict.copy()
    modified_model_dict["top_p"] = 0.85

    new_hash = ResponseComponent._model_hash(modified_model_dict)

    assert original_hash != new_hash


def test_model_hash_key_order(mock_model_dict: dict) -> None:
    original_hash = ResponseComponent._model_hash(mock_model_dict)

    # Checks order of keys doesn't affect hash
    ds_keys = list(mock_model_dict.keys())
    random.shuffle(ds_keys)

    shuffled_dict = {key: mock_model_dict[key] for key in ds_keys}
    shuffle_hash = ResponseComponent._model_hash(shuffled_dict)

    assert shuffle_hash == original_hash


@pytest.mark.fileio
def test_without_backoff_without_cache(
    response_component_config: dict,
    make_mock_llm: Callable,
    mock_prompt_dataset: Dataset,
):
    response_component_config["cache_base_dir"] = None
    response_component_config["backoff_decorator_config"] = None

    responses = list("ABC")
    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    _ = response_component.batch_generate(
        model=mock_llm, prompt_dataset=mock_prompt_dataset
    )

    #
    assert not response_component.cache_responses
    assert not response_component.backoff_decorator


@pytest.mark.fileio
def test_without_backoff_with_cache(
    response_component_config: dict,
    make_mock_llm: Callable,
    mock_prompt_dataset: Dataset,
    tmpdir,
):
    response_component_config["cache_base_dir"] = tmpdir
    response_component_config["backoff_decorator_config"] = None

    responses = list("ABC")
    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    _ = response_component.batch_generate(
        model=mock_llm, prompt_dataset=mock_prompt_dataset
    )

    #
    assert response_component.cache_responses
    assert not response_component.backoff_decorator

    cache_dir = Path(response_component_config["cache_base_dir"])
    assert len(os.listdir(cache_dir)) == 1

    model_cache_dir = cache_dir / ResponseComponent._model_hash(mock_llm._model_dict)
    assert model_cache_dir.is_dir()
    assert len(os.listdir(model_cache_dir)) == 3


@pytest.mark.fileio
def test_with_default_backoff_without_cache(
    response_component_config: dict,
    make_mock_llm: Callable,
    mock_prompt_dataset: Dataset,
):
    response_component_config["cache_base_dir"] = None
    response_component_config["backoff_decorator_config"] = "DEFAULT"

    responses = list("ABC")
    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    _ = response_component.batch_generate(
        model=mock_llm, prompt_dataset=mock_prompt_dataset
    )

    #
    assert not response_component.cache_responses
    assert response_component.backoff_decorator


@pytest.mark.fileio
def test_with_default_backoff_with_cache(
    response_component_config: dict,
    make_mock_llm: Callable,
    mock_prompt_dataset: Dataset,
    tmpdir,
):
    response_component_config["cache_base_dir"] = tmpdir
    response_component_config["backoff_decorator_config"] = "DEFAULT"

    responses = list("ABC")
    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    _ = response_component.batch_generate(
        model=mock_llm, prompt_dataset=mock_prompt_dataset
    )

    #
    assert response_component.cache_responses
    assert response_component.backoff_decorator

    cache_dir = Path(response_component_config["cache_base_dir"])
    assert len(os.listdir(cache_dir)) == 1

    model_cache_dir = cache_dir / ResponseComponent._model_hash(mock_llm._model_dict)
    assert model_cache_dir.is_dir()
    assert len(os.listdir(model_cache_dir)) == 3


def test_make_request_func_with_backoff_without_cache(
    response_component_config: dict,
    make_mock_llm: Callable,
):
    response_component_config["cache_base_dir"] = None
    response_component_config["backoff_decorator_config"] = None

    responses = list("ABC")
    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    request_func = response_component._make_request_func(model=mock_llm)

    #
    assert request_func == mock_llm.generate
    assert not response_component.cache_responses
    assert not response_component.backoff_decorator


def test_make_request_func_with_default_backoff_without_cache(
    response_component_config: dict,
    make_mock_llm: Callable,
):
    response_component_config["cache_base_dir"] = None
    response_component_config["backoff_decorator_config"] = "DEFAULT"

    responses = list("ABC")
    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    request_func = response_component._make_request_func(model=mock_llm)

    #
    assert request_func.__wrapped__ == mock_llm.generate
    assert not response_component.cache_responses
    assert response_component.backoff_decorator


def test_make_request_func_without_backoff_with_cache(
    response_component_config: dict,
    make_mock_llm: Callable,
    tmpdir,
):
    response_component_config["cache_base_dir"] = tmpdir
    response_component_config["backoff_decorator_config"] = None

    responses = list("ABC")
    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    response_component._configure_cache_dir(mock_llm._model_dict)
    request_func = response_component._make_request_func(model=mock_llm)

    #
    assert request_func.fname == response_component_config["response_cache_fname"]
    assert request_func.cache_dir == response_component.cache_dir

    assert request_func.__wrapped__ == mock_llm.generate
    assert response_component.cache_responses
    assert not response_component.backoff_decorator


def test_make_request_func_with_default_backoff_with_cache(
    response_component_config: dict,
    make_mock_llm: Callable,
    tmpdir,
):
    response_component_config["cache_base_dir"] = tmpdir
    response_component_config["backoff_decorator_config"] = "DEFAULT"

    responses = list("ABC")
    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    response_component._configure_cache_dir(mock_llm._model_dict)
    request_func = response_component._make_request_func(model=mock_llm)

    #
    assert request_func.function.__name__ == "retry"
    assert request_func.fname == response_component_config["response_cache_fname"]
    assert request_func.cache_dir == response_component.cache_dir
    assert request_func.__wrapped__.__wrapped__ == mock_llm.generate
    assert response_component.cache_responses
    assert response_component.backoff_decorator


@pytest.mark.fileio
def test_without_backoff_without_cache_async(
    response_component_config: dict,
    make_mock_llm: Callable,
    mock_prompt_dataset: Dataset,
):
    response_component_config["run_async"] = True
    response_component_config["cache_base_dir"] = None
    response_component_config["backoff_decorator_config"] = None

    responses = list("ABC")
    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    _ = response_component.batch_generate(
        model=mock_llm, prompt_dataset=mock_prompt_dataset
    )

    #
    assert not response_component.cache_responses
    assert not response_component.backoff_decorator


@pytest.mark.fileio
def test_without_backoff_with_cache_async(
    response_component_config: dict,
    make_mock_llm: Callable,
    mock_prompt_dataset: Dataset,
    tmpdir,
):
    response_component_config["cache_base_dir"] = tmpdir
    response_component_config["run_async"] = True
    response_component_config["backoff_decorator_config"] = None

    responses = list("ABC")
    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    _ = response_component.batch_generate(
        model=mock_llm, prompt_dataset=mock_prompt_dataset
    )

    #
    assert response_component.cache_responses
    assert not response_component.backoff_decorator

    cache_dir = Path(response_component_config["cache_base_dir"])
    assert len(os.listdir(cache_dir)) == 1

    model_cache_dir = cache_dir / ResponseComponent._model_hash(mock_llm._model_dict)
    assert model_cache_dir.is_dir()
    assert len(os.listdir(model_cache_dir)) == 3


@pytest.mark.fileio
def test_with_default_backoff_without_cache_async(
    response_component_config: dict,
    make_mock_llm: Callable,
    mock_prompt_dataset: Dataset,
):
    response_component_config["run_async"] = True
    response_component_config["cache_base_dir"] = None
    response_component_config["backoff_decorator_config"] = "DEFAULT"

    responses = list("ABC")
    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    _ = response_component.batch_generate(
        model=mock_llm, prompt_dataset=mock_prompt_dataset
    )

    #
    assert not response_component.cache_responses
    assert response_component.backoff_decorator


@pytest.mark.fileio
def test_with_default_backoff_with_cache_async(
    response_component_config: dict,
    make_mock_llm: Callable,
    mock_prompt_dataset: Dataset,
    tmpdir,
):
    response_component_config["cache_base_dir"] = tmpdir
    response_component_config["run_async"] = True
    response_component_config["backoff_decorator_config"] = "DEFAULT"

    responses = list("ABC")
    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    _ = response_component.batch_generate(
        model=mock_llm, prompt_dataset=mock_prompt_dataset
    )

    #
    assert response_component.cache_responses
    assert response_component.backoff_decorator

    cache_dir = Path(response_component_config["cache_base_dir"])
    assert len(os.listdir(cache_dir)) == 1

    model_cache_dir = cache_dir / ResponseComponent._model_hash(mock_llm._model_dict)
    assert model_cache_dir.is_dir()
    assert len(os.listdir(model_cache_dir)) == 3


def test_make_request_func_with_backoff_without_cache_async(
    response_component_config: dict,
    make_mock_llm: Callable,
):
    response_component_config["run_async"] = True
    response_component_config["cache_base_dir"] = None
    response_component_config["backoff_decorator_config"] = None

    responses = list("ABC")
    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    request_func = response_component._make_request_func(model=mock_llm)

    #
    assert request_func == mock_llm.async_generate
    assert not response_component.cache_responses
    assert not response_component.backoff_decorator


def test_make_request_func_with_default_backoff_without_cache_async(
    response_component_config: dict,
    make_mock_llm: Callable,
):
    response_component_config["run_async"] = True
    response_component_config["cache_base_dir"] = None
    response_component_config["backoff_decorator_config"] = "DEFAULT"

    responses = list("ABC")
    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    request_func = response_component._make_request_func(model=mock_llm)

    #
    assert request_func.__wrapped__ == mock_llm.async_generate
    assert not response_component.cache_responses
    assert response_component.backoff_decorator


def test_make_request_func_without_backoff_with_cache_async(
    response_component_config: dict,
    make_mock_llm: Callable,
    tmpdir,
):
    response_component_config["cache_base_dir"] = tmpdir
    response_component_config["run_async"] = True
    response_component_config["backoff_decorator_config"] = None

    responses = list("ABC")
    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    response_component._configure_cache_dir(mock_llm._model_dict)
    request_func = response_component._make_request_func(model=mock_llm)

    #
    assert request_func.fname == response_component_config["response_cache_fname"]
    assert request_func.cache_dir == response_component.cache_dir

    assert request_func.__wrapped__ == mock_llm.async_generate
    assert response_component.cache_responses
    assert not response_component.backoff_decorator


def test_make_request_func_with_default_backoff_with_cache_async(
    response_component_config: dict, make_mock_llm: Callable, tmpdir
):
    response_component_config["cache_base_dir"] = tmpdir
    response_component_config["run_async"] = True
    response_component_config["backoff_decorator_config"] = "DEFAULT"

    responses = list("ABC")
    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    response_component._configure_cache_dir(mock_llm._model_dict)
    request_func = response_component._make_request_func(model=mock_llm)

    #
    assert request_func.fname == response_component_config["response_cache_fname"]
    assert request_func.cache_dir == response_component.cache_dir
    assert request_func.__wrapped__.__wrapped__ == mock_llm.async_generate
    assert response_component.cache_responses
    assert response_component.backoff_decorator


@pytest.mark.fileio
def test_batch_generate_model_outputs_string_sync(
    response_component_config: dict,
    make_mock_llm: Callable,
    mock_prompt_dataset: Dataset,
):
    response_component_config["backoff_decorator_config"] = None

    responses = list("ABC")
    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    output_dataset = response_component.batch_generate(
        model=mock_llm, prompt_dataset=mock_prompt_dataset
    )

    #
    assert output_dataset["response"] == responses


@pytest.mark.fileio
def test_batch_generate_model_outputs_dict_sync(
    response_component_config: dict,
    make_mock_llm: Callable,
    mock_prompt_dataset: Dataset,
):
    response_component_config["backoff_decorator_config"] = None

    responses = [
        {"response": "A", "extra": 1},
        {"response": "B", "extra": 2},
        {"response": "C", "extra": 3},
    ]

    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    output_dataset = response_component.batch_generate(
        model=mock_llm, prompt_dataset=mock_prompt_dataset
    )

    #
    assert output_dataset["response"] == list("ABC")
    assert output_dataset["extra"] == [1, 2, 3]


@pytest.mark.fileio
def test_batch_generate_model_outputs_dict_async(
    response_component_config: dict,
    make_mock_llm: Callable,
    mock_prompt_dataset: Dataset,
):
    response_component_config["run_async"] = True
    response_component_config["backoff_decorator_config"] = None

    responses = [
        {"response": "A", "extra": 1},
        {"response": "B", "extra": 2},
        {"response": "C", "extra": 3},
    ]

    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    output_dataset = response_component.batch_generate(
        model=mock_llm, prompt_dataset=mock_prompt_dataset
    )

    #
    assert output_dataset["response"] == list("ABC")
    assert output_dataset["extra"] == [1, 2, 3]


def test_batch_generate_sync_with_skip_no_error(
    response_component_config: dict,
    make_mock_llm: Callable,
    mock_prompt_dataset: Dataset,
):
    response_component_config["cache_base_dir"] = None
    response_component_config["backoff_decorator_config"] = None
    response_component_config["skip_decorator_config"] = {
        "exception": "ValueError",
        "null_response": -1,
    }

    responses = [
        {"response": "A", "extra": 1},
        {"response": "B", "extra": 2},
        {"response": "C", "extra": 3},
    ]

    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    output_dataset = response_component.batch_generate(
        model=mock_llm, prompt_dataset=mock_prompt_dataset
    )

    #
    assert output_dataset["response"] == list("ABC")
    assert output_dataset["extra"] == [1, 2, 3]


def test_batch_generate_sync_with_skip_with_error(
    response_component_config: dict,
    make_mock_llm: Callable,
    mock_prompt_dataset: Dataset,
):
    response_component_config["cache_base_dir"] = None
    response_component_config["backoff_decorator_config"] = None
    response_component_config["skip_decorator_config"] = {
        "exception": "ValueError",
        "null_response": "-1",
    }

    responses = [
        {"response": "A", "extra": 1},
        {"response": "B", "extra": 2},
        ValueError,
    ]

    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    output_dataset = response_component.batch_generate(
        model=mock_llm, prompt_dataset=mock_prompt_dataset
    )

    #
    assert output_dataset["response"] == ["A", "B", "-1"]
    assert output_dataset["extra"] == [1, 2, None]


def test_batch_generate_sync_with_skip_with_cache_with_error(
    response_component_config: dict,
    make_mock_llm: Callable,
    mock_prompt_dataset: Dataset,
    tmpdir,
):
    response_component_config["cache_base_dir"] = Path(tmpdir)
    response_component_config["backoff_decorator_config"] = None
    response_component_config["skip_decorator_config"] = {
        "exception": "ValueError",
        "null_response": "-1",
    }

    responses = [
        {"response": "A", "extra": 1},
        {"response": "B", "extra": 2},
        ValueError,
    ]

    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)
    output_dataset = response_component.batch_generate(
        model=mock_llm, prompt_dataset=mock_prompt_dataset
    )

    #
    assert output_dataset["response"] == ["A", "B", "-1"]
    assert output_dataset["extra"] == [1, 2, None]

    path = os.listdir(Path(tmpdir))[0]
    assert len(os.listdir(Path(tmpdir) / path)) == 2


def test_batch_generate_sync_with_skip_with_different_error(
    response_component_config: dict,
    make_mock_llm: Callable,
    mock_prompt_dataset: Dataset,
):
    response_component_config["cache_base_dir"] = None
    response_component_config["backoff_decorator_config"] = None
    response_component_config["skip_decorator_config"] = {
        "exception": "ValueError",
        "null_response": "-1",
    }

    responses = [
        {"response": "A", "extra": 1},
        {"response": "B", "extra": 2},
        TypeError,
    ]

    mock_llm = make_mock_llm(responses)

    #
    response_component = ResponseComponent.from_config(response_component_config)

    with pytest.raises(TypeError):
        response_component.batch_generate(
            model=mock_llm, prompt_dataset=mock_prompt_dataset
        )


def test_ResponseComponentConfig_singles():
    config = {
        "backoff_decorator_config": {
            "exception": "ValueError",
        },
        "skip_decorator_config": {
            "exception": "ValueError",
        },
    }

    with pytest.raises(ValidationError):
        ResponseComponentConfig.parse_obj(config)


def test_ResponseComponentConfig_tuples():
    config = {
        "backoff_decorator_config": {
            "exception": (
                "ValueError",
                "TypeError",
            )
        },
        "skip_decorator_config": {
            "exception": (
                "ValueError",
                "KeyError",
            )
        },
    }

    with pytest.raises(ValidationError):
        ResponseComponentConfig.parse_obj(config)
