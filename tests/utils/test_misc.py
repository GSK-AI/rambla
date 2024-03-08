import os
from pathlib import Path
from typing import Any, Dict, List
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from rambla.utils import misc
from rambla.utils.io import dump, load
from rambla.utils.misc import (
    compute_class_counts_from_confmat,
    prepare_dicts_for_logging,
)

# flake8: noqa: N802


def compare_dicts(d0: dict, d1: dict) -> bool:
    if d0.keys() != d1.keys():
        return False

    for k, v in d0.items():
        if isinstance(v, (int, float, list)):
            if v != d1[k]:
                return False
        elif isinstance(v, dict):
            if not compare_dicts(v, d1[k]):
                return False
        elif isinstance(v, np.ndarray):
            if not np.allclose(v, d1[k]):
                return False
        else:
            raise TypeError
    return True


@mock.patch("rambla.utils.misc.get_available_task_yaml_files")
def test_validate_tasks(mock_get_available_task_yaml_files):
    mock_get_available_task_yaml_files.return_value = ["first", "second", "third"]
    tasks = ["first", "second"]
    misc.validate_tasks(tasks)


@mock.patch("rambla.utils.misc.get_available_task_yaml_files")
def test_validate_tasks_error(mock_get_available_task_yaml_files):
    mock_get_available_task_yaml_files.return_value = ["first", "second", "third"]
    tasks = ["first", "fourth"]

    with pytest.raises(RuntimeError):
        misc.validate_tasks(tasks)


@mock.patch("rambla.utils.misc.get_available_model_yaml_files")
def test_validate_models(mock_get_available_model_yaml_files):
    mock_get_available_model_yaml_files.return_value = ["first", "second", "third"]
    models = ["first", "second"]
    misc.validate_models(models)


@mock.patch("rambla.utils.misc.get_available_model_yaml_files")
def test_validate_models_error(mock_get_available_model_yaml_files):
    mock_get_available_model_yaml_files.return_value = ["first", "second", "third"]
    models = ["first", "fourth"]

    with pytest.raises(RuntimeError):
        misc.validate_models(models)


@pytest.mark.fileio
@mock.patch("rambla.utils.misc.get_hydra_conf_dir")
def test_get_available_model_yaml_files(mock_get_hydra_conf_dir, tmpdir):
    mock_get_hydra_conf_dir.return_value = Path(tmpdir)
    path = Path(tmpdir) / "model"
    path.mkdir()
    filenames = ["first.yaml", "second.yaml", "third.yaml"]
    for filename in filenames:
        dump("dummy", path / filename)

    output = misc.get_available_model_yaml_files()
    assert set(output) == set(["first", "second", "third"])


@pytest.mark.fileio
@mock.patch("rambla.utils.misc.get_hydra_conf_dir")
def test_get_available_task_yaml_files(mock_get_hydra_conf_dir, tmpdir):
    mock_get_hydra_conf_dir.return_value = Path(tmpdir)
    path = Path(tmpdir) / "task"
    path.mkdir()
    filenames = ["first.yaml", "second.yaml", "third.yaml"]
    for filename in filenames:
        dump("dummy", path / filename)

    output = misc.get_available_task_yaml_files()
    assert set(output) == set(["first", "second", "third"])


def test_EnvCtxManager_env_variables_not_previously_set():
    kwargs = {"dummy": "hello", "also_dummy": "1"}
    with misc.EnvCtxManager(**kwargs):
        # Checking that the env variables were correctly set.
        for k, v in kwargs.items():
            assert os.environ[k] == v

    # Checking that the env variables were unset
    for k, v in kwargs.items():
        assert k not in os.environ


def test_EnvCtxManager_env_variables_previously_set():
    env_kwargs = {"dummy": "hi", "also_dummy": "2"}
    for k, v in env_kwargs.items():
        os.environ[k] = v

    kwargs = {"dummy": "hello", "also_dummy": "1"}
    with misc.EnvCtxManager(**kwargs):
        # Checking that the env variables were correctly set.
        for k, v in kwargs.items():
            assert os.environ[k] == v

    # Checking that the env variables were unset
    for k, v in env_kwargs.items():
        assert os.environ[k] == v

    ## Unsetting env variables
    for k, v in env_kwargs.items():
        del os.environ[k]


def test_EnvCtxManager_as_decorator_with_env_variables_not_previously_set():
    kwargs = {"dummy": "hello", "also_dummy": "1"}
    decorator = misc.EnvCtxManager(**kwargs)
    mock_func = mock.Mock()

    def outer_func():
        input_kwargs = {}
        for k in kwargs.keys():
            input_kwargs[k] = os.environ[k]

        mock_func(**input_kwargs)

    decorated_func = decorator(outer_func)

    # Call
    decorated_func()

    # Checks
    assert mock_func.call_args.kwargs == kwargs

    # Checking that the env variables were unset
    for k, v in kwargs.items():
        assert k not in os.environ


def test_EnvCtxManager_as_decorator_with_env_variables_previously_set():
    env_kwargs = {"dummy": "hi", "also_dummy": "2"}
    for k, v in env_kwargs.items():
        os.environ[k] = v

    kwargs = {"dummy": "hello", "also_dummy": "1"}

    decorator = misc.EnvCtxManager(**kwargs)
    mock_func = mock.Mock()

    def outter_func():
        input_kwargs = {}
        for k in kwargs.keys():
            input_kwargs[k] = os.environ[k]

        mock_func(**input_kwargs)

    decorated_func = decorator(outter_func)

    # Call
    decorated_func()

    # Checks
    assert mock_func.call_args.kwargs == kwargs

    # Checking that the env variables were unset
    for k, v in env_kwargs.items():
        assert os.environ[k] == v

    ## Unsetting env variables
    for k, v in env_kwargs.items():
        del os.environ[k]


@pytest.mark.parametrize(
    "input_dict, expected_class",
    [
        ({"dummy": 1}, "dummy"),
        ({"dummy": 1, "also_dummy": 0.9}, "dummy"),
    ],
)
def test_dict_argmax(input_dict: dict, expected_class: str):
    assert misc.dict_argmax(input_dict) == expected_class


@pytest.mark.parametrize(
    "input_dict, expected_dict",
    [
        ({"dummy": 1}, {"dummy": 1}),
        ({"a": 1, "b": 1}, {"a": 0.50, "b": 0.50}),
    ],
)
def test_dict_softmax(input_dict: dict, expected_dict: str):
    assert misc.dict_softmax(input_dict) == expected_dict


def test_make_json_serialisable():
    numpy_array = np.array([1, 2, 3]).astype(np.int64)

    path_string = "dummy/path/for/testing"
    second_path_string = "second/path/string"
    third_path_string = "second/path/string"

    input_obj = {
        "numpy_array": numpy_array,
        "path": Path(path_string),
        "list": [
            Path(second_path_string),
            Path(third_path_string),
        ],
        "numpy_int": np.int64(3),
        "none_value": None,
    }
    output_obj = misc.make_json_serialisable(input_obj)

    expected_output = {
        "numpy_array": [1, 2, 3],
        "path": path_string,
        "list": [
            second_path_string,
            third_path_string,
        ],
        "numpy_int": 3,
        "none_value": None,
    }

    assert output_obj == expected_output


def test_make_json_serialisable_and_dump(tmpdir):
    numpy_array = np.array([1, 2, 3]).astype(np.int64)

    path_string = "dummy/path/for/testing"
    second_path_string = "second/path/string"
    third_path_string = "second/path/string"

    input_obj = {
        "numpy_array": numpy_array,
        "path": Path(path_string),
        "list": [
            Path(second_path_string),
            Path(third_path_string),
        ],
        "numpy_int": np.int64(3),
        "none_value": None,
    }

    expected_output = {
        "numpy_array": [1, 2, 3],
        "path": path_string,
        "list": [
            second_path_string,
            third_path_string,
        ],
        "numpy_int": 3,
        "none_value": None,
    }
    #
    output_obj = misc.make_json_serialisable(input_obj)
    dump(output_obj, Path(tmpdir) / "metrics.json")

    #
    loaded_obj = load(Path(tmpdir) / "metrics.json")

    assert loaded_obj == expected_output


@pytest.mark.parametrize(
    "input_text, sep, expected_output_text",
    [
        (
            "I like ice cream[SEP]I love ice cream",
            "[SEP]",
            ["I like ice cream", "I love ice cream"],
        ),
        (
            "I like ice cream[SEP]I love ice cream",
            "like",
            ["I ", " ice cream[SEP]I love ice cream"],
        ),
        (
            "A man is smiling[SEP]The man is happy",
            "[SEP]",
            ["A man is smiling", "The man is happy"],
        ),
        (
            "Sequence1[SEPARATOR]Sequence2[SEPARATOR]Sequence3",
            "[SEPARATOR]",
            ["Sequence1", "Sequence2", "Sequence3"],
        ),
    ],
)
def test_split_text_on_sep(
    input_text: str,
    sep: str,
    expected_output_text: tuple[str],
) -> None:
    assert misc.split_text_on_sep(input_text, sep) == expected_output_text


def test_split_text_on_sep_invalid_sequences() -> None:
    # Checks an error raised if more sequences that expected found
    with pytest.raises(ValueError) as exc_info:
        _ = misc.split_text_on_sep(
            "A text sequence[SEP]with multiple[SEP]input sequences.", "[SEP]", 2
        )

    assert "Input must contain 2 sequences" in str(exc_info)


def test_split_text_on_sep_no_sep_found() -> None:
    with pytest.raises(ValueError) as exc_info:
        _ = misc.split_text_on_sep("A sequence missing a separator", "[SEP]")

    assert "No separator" in str(exc_info)


@pytest.mark.parametrize(
    "input_dict,expected_output",
    [
        (
            [{"a": 1, "b": "text"}, {"a": 2, "b": "text"}, {"a": 3, "b": "random"}],
            {"a": [1, 2, 3], "b": ["text", "text", "random"]},
        ),
        (
            [{"a": {"nested": "value"}}, {"a": {"nested": "value2"}}],
            {"a": [{"nested": "value"}, {"nested": "value2"}]},
        ),
        (
            [{"a": [{"b": 2}, {"c": 3}]}, {"a": [{"b": 2}, {"c": 3}]}],
            {"a": [[{"b": 2}, {"c": 3}], [{"b": 2}, {"c": 3}]]},
        ),
    ],
)
def test_list_of_dicts_to_dict_of_lists(
    input_dict: List[Dict[str, Any]], expected_output: Dict[str, list]
) -> None:
    assert misc.list_of_dicts_to_dict_of_lists(input_dict) == expected_output


def test_list_of_dicts_to_dict_of_lists_non_matching_keys() -> None:
    input_dict = [{"a": "value", "b": "another_value"}, {"a": "value"}]

    with pytest.raises(KeyError):
        _ = misc.list_of_dicts_to_dict_of_lists(input_dict)


def test_convert_dict_key_to_str():
    dummy_dict = {0: 0, 1: {0: 0, 1: 1}}
    parsed_dummy_dict = misc.convert_dict_key_to_str(dummy_dict)
    expected_output = {"0": 0, "1": {"0": 0, "1": 1}}

    assert parsed_dummy_dict == expected_output


def test_prepare_dicts_for_logging_without_label_encoder_without_quality_eval():
    eval_results = {
        "results": {
            "integer": 3,
            "dict": {"dummy": 10, "also_dummy": [1, 2, 3]},
            "list": [1, 2, 3],
            "numpy": np.array([1, 2, 3]),
            "pandas_series": pd.Series([1, 2, 3]),
        }
    }

    to_log_as_metrics, to_log_as_dicts = prepare_dicts_for_logging(
        eval_results=eval_results
    )

    expected_output = {
        "integer": 3,
        "dict": {"dummy": 10, "also_dummy": [1, 2, 3]},
        "list": [1, 2, 3],
        "numpy": np.array([1, 2, 3]),
    }

    assert not to_log_as_dicts
    assert compare_dicts(to_log_as_metrics, expected_output)


def test_prepare_dicts_for_logging_with_label_encoder_without_quality_eval():
    eval_results = {
        "results": {
            "integer": 3,
            "dict": {"dummy": 10, "also_dummy": [1, 2, 3]},
            "list": [1, 2, 3],
            "numpy": np.array([1, 2, 3]),
            "pandas_series": pd.Series([1, 2, 3]),
        },
        "label_encoder": {
            "yes": 1,
            "no": 0,
            "maybe": 3,
        },
    }

    to_log_as_metrics, to_log_as_dicts = prepare_dicts_for_logging(
        eval_results=eval_results
    )

    expected_output = {
        "integer": 3,
        "dict": {"dummy": 10, "also_dummy": [1, 2, 3]},
        "list": [1, 2, 3],
        "numpy": np.array([1, 2, 3]),
    }

    expected_log_as_dict_output = {
        "label_encoder": {
            "yes": 1,
            "no": 0,
            "maybe": 3,
        },
    }
    assert to_log_as_dicts == expected_log_as_dict_output
    assert compare_dicts(to_log_as_metrics, expected_output)


def test_prepare_dicts_for_logging_with_label_encoder_with_quality_eval():
    eval_results = {
        "results": {
            "integer": 3,
            "dict": {"dummy": 10, "also_dummy": [1, 2, 3]},
            "list": [1, 2, 3],
            "numpy": np.array([1, 2, 3]),
            "pandas_series": pd.Series([1, 2, 3]),
        },
        "label_encoder": {
            "yes": 1,
            "no": 0,
            "maybe": 3,
        },
    }

    quality_eval = {
        "median": 3.16,
        "mean": 4.12,
    }
    to_log_as_metrics, to_log_as_dicts = prepare_dicts_for_logging(
        eval_results=eval_results, quality_eval=quality_eval
    )

    expected_output = {
        "integer": 3,
        "dict": {"dummy": 10, "also_dummy": [1, 2, 3]},
        "list": [1, 2, 3],
        "numpy": np.array([1, 2, 3]),
        "quality_eval/median": 3.16,
        "quality_eval/mean": 4.12,
    }

    expected_log_as_dict_output = {
        "label_encoder": {
            "yes": 1,
            "no": 0,
            "maybe": 3,
        },
    }

    assert compare_dicts(to_log_as_metrics, expected_output)
    assert to_log_as_dicts == expected_log_as_dict_output


def test_compute_class_counts_from_confmat():
    confmat = np.array(
        [
            [0, 10, 20],
            [5, 1, 3],
            [6, 13, 200],
        ]
    )
    label_encoder = {
        "yes": 0,
        "no": 2,
        "null": 1,
    }

    # run
    output = compute_class_counts_from_confmat(confmat, label_encoder)

    # checks
    expected_output = {
        "n_pred_yes": 30,
        "n_pred_no": 219,
        "n_pred_null": 9,
        "n_target_yes": 11,
        "n_target_no": 223,
        "n_target_null": 24,
        #
        "prop_pred_yes": 30 / confmat.sum(),
        "prop_pred_no": 219 / confmat.sum(),
        "prop_pred_null": 9 / confmat.sum(),
        "prop_target_yes": 11 / confmat.sum(),
        "prop_target_no": 223 / confmat.sum(),
        "prop_target_null": 24 / confmat.sum(),
    }

    assert output == expected_output
