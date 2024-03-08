from typing import Type

import numpy as np
import pytest

from rambla.text_to_text_components.nli_strategies import (
    AverageNLIOutputCombinationStrategy,
    BaseNLIOutputCombinationStrategy,
    CombinationStrategyConfig,
    RelaxedNLIOutputCombinationStrategy,
    StrictNLIOutputCombinationStrategy,
    build_combination_strategy,
)

# flake8: noqa: N802


@pytest.mark.parametrize(
    "response0, response1",
    [
        ({"a": 1, "b": 1}, {"a": 3, "b": 10}),
    ],
)
def test_validate_input_dicts_correct(response0: dict, response1: dict):
    BaseNLIOutputCombinationStrategy._validate_input_dicts(response0, response1)


@pytest.mark.parametrize(
    "response0, response1",
    [
        ({"a": 1}, {}),
        ({}, {"a": 1}),
        ({"a": 1, "b": 1}, {"a": 3}),
    ],
)
def test_validate_input_dicts_error_different_keys(response0: dict, response1: dict):
    with pytest.raises(ValueError) as exc_info:
        BaseNLIOutputCombinationStrategy._validate_input_dicts(response0, response1)

    assert "different sets of keys" in str(exc_info.value)


@pytest.mark.parametrize(
    "response0, response1",
    [
        ({"a": 1}, "dummy"),
        (1, {"a": 1}),
    ],
)
def test_validate_input_dicts_error_not_dicts(response0: dict, response1: dict):
    with pytest.raises(TypeError) as exc_info:
        BaseNLIOutputCombinationStrategy._validate_input_dicts(response0, response1)

    assert "Inputs should be of type `dict`" in str(exc_info.value)


@pytest.mark.parametrize(
    "response0, response1",
    [
        ({"a": 1}, {"a": "hey"}),
    ],
)
def test_validate_input_dicts_error_wrong_type(response0: dict, response1: dict):
    with pytest.raises(TypeError) as exc_info:
        BaseNLIOutputCombinationStrategy._validate_input_dicts(response0, response1)

    assert "Values need to have as type one" in str(exc_info.value)


@pytest.mark.parametrize(
    "response0, response1, positive_key, expected",
    [
        (
            {"entail": 1, "neutral": 2, "contra": 3},
            {"entail": 1, "neutral": 2, "contra": 3},
            "entail",
            "0",
        ),
        (
            {"entail": 1, "neutral": 2, "contra": 3},
            {"entail": 10, "neutral": 2, "contra": 3},
            "entail",
            "0",
        ),
        (
            {"entail": 10, "neutral": 2, "contra": 3},
            {"entail": 10, "neutral": 2, "contra": 3},
            "entail",
            "1",
        ),
    ],
)
def test_StrictCombinationStrategy_run(
    response0: dict, response1: dict, positive_key: str, expected: int
):
    strategy = StrictNLIOutputCombinationStrategy(positive_key=positive_key)
    output = strategy.run(response0, response1)
    assert output == expected


@pytest.mark.parametrize(
    "response0, response1, primary_key, secondary_key, expected",
    [
        (
            {"entail": 1, "neutral": 2, "contra": 3},
            {"entail": 1, "neutral": 2, "contra": 3},
            "entail",
            "neutral",
            "0",
        ),
        (
            {"entail": 1, "neutral": 2, "contra": 3},
            {"entail": 10, "neutral": 2, "contra": 3},
            "entail",
            "neutral",
            "0",
        ),
        (
            {"entail": 10, "neutral": 2, "contra": 3},
            {"entail": 10, "neutral": 2, "contra": 3},
            "entail",
            "neutral",
            "1",
        ),
        (
            {"entail": 10, "neutral": 2, "contra": 3},
            {"entail": 1, "neutral": 10, "contra": 3},
            "entail",
            "neutral",
            "1",
        ),
        (
            {"entail": 1, "neutral": 10, "contra": 3},
            {"entail": 10, "neutral": 2, "contra": 3},
            "entail",
            "neutral",
            "1",
        ),
        (
            {"entail": 1, "neutral": 10, "contra": 3},
            {"entail": 10, "neutral": 20, "contra": 3},
            "entail",
            "neutral",
            "0",
        ),
    ],
)
def test_RelaxedCombinationStrategy_run(
    response0: dict,
    response1: dict,
    primary_key: str,
    secondary_key: str,
    expected: int,
):
    strategy = RelaxedNLIOutputCombinationStrategy(
        primary_key=primary_key, secondary_key=secondary_key
    )
    output = strategy.run(response0, response1)
    assert output == expected


@pytest.mark.parametrize(
    "response0, response1, positive_key, apply_softmax, expected",
    [
        (
            {"entail": 1, "neutral": 2, "contra": 3},
            {"entail": 3, "neutral": 2, "contra": 3},
            "entail",
            False,
            2,
        ),
        (
            {"entail": 1, "neutral": 2, "contra": 3},
            {"entail": 3, "neutral": 4, "contra": 3},
            "neutral",
            False,
            3,
        ),
        (
            {"entail": 1, "neutral": 2, "contra": 3},
            {"entail": 2, "neutral": 2, "contra": 3},
            "entail",
            True,
            0.150986,
        ),
    ],
)
def test_ContinuousCombinationStrategy_run(
    response0: dict,
    response1: dict,
    positive_key: str,
    apply_softmax: bool,
    expected: float,
):
    strategy = AverageNLIOutputCombinationStrategy(
        positive_key=positive_key, apply_softmax=apply_softmax
    )
    output = strategy.run(response0, response1)
    assert np.isclose(output, expected)


@pytest.mark.parametrize(
    "config, expected_class",
    [
        (
            {"name": "strict", "params": {"positive_key": "entail"}},
            StrictNLIOutputCombinationStrategy,
        ),
        (
            {
                "name": "relaxed",
                "params": {"primary_key": "entail", "secondary_key": "neutral"},
            },
            RelaxedNLIOutputCombinationStrategy,
        ),
        (
            {"name": "average", "params": {"positive_key": "entail"}},
            AverageNLIOutputCombinationStrategy,
        ),
    ],
)
def test_build_combination_strategy(
    config: dict, expected_class: Type[BaseNLIOutputCombinationStrategy]
):
    obj = build_combination_strategy(config)
    assert isinstance(obj, expected_class)


@pytest.mark.parametrize(
    "config",
    [
        ({"name": "strict", "params": {"positive_key": "entail"}}),
        (
            {
                "name": "relaxed",
                "params": {"primary_key": "entail", "secondary_key": "neutral"},
            }
        ),
        ({"name": "average", "params": {"positive_key": "entail"}}),
    ],
)
def test_CombinationStrategyConfig(config: dict):
    parsed_config = CombinationStrategyConfig.parse_obj(config)
    assert parsed_config.name == config["name"]
    assert parsed_config.params == config["params"]


def test_CombinationStrategyConfig_error():
    dummy_name = "__dummy_name__"
    config = {"name": dummy_name, "params": {}}

    with pytest.raises(ValueError) as exc_info:
        CombinationStrategyConfig.parse_obj(config)

    assert dummy_name in str(exc_info.value)


@pytest.mark.parametrize(
    "response0, response1, positive_key, error_on",
    [
        (
            {"entail": 1, "neutral": 2, "contra": 3},
            {"entail": 1, "neutral": 2, "contra": 3},
            "__entail__",
            "response0",
        ),
    ],
)
def test_StrictNLIOutputCombinationStrategy_validate_input_dicts_against_class_keys_error(
    response0: dict, response1: dict, positive_key: str, error_on: str
):
    strategy = StrictNLIOutputCombinationStrategy(positive_key=positive_key)
    with pytest.raises(ValueError) as exc_info:
        strategy.run(response0, response1)

    assert (
        f"key='positive_key' with value='{positive_key}' not found in `{error_on}`."
        in str(exc_info.value)
    )


@pytest.mark.parametrize(
    "response0, response1, primary_key, secondary_key, error_on, which_key",
    [
        (
            {"entail": 1, "neutral": 2, "contra": 3},
            {"entail": 1, "neutral": 2, "contra": 3},
            "__entail__",
            "neutral",
            "response0",
            "primary_key",
        ),
    ],
)
def test_RelaxedNLIOutputCombinationStrategy_validate_input_dicts_against_class_keys_error_primary_key(
    response0: dict,
    response1: dict,
    primary_key: str,
    secondary_key: str,
    error_on: str,
    which_key: str,
):
    strategy = RelaxedNLIOutputCombinationStrategy(
        primary_key=primary_key, secondary_key=secondary_key
    )
    with pytest.raises(ValueError) as exc_info:
        strategy.run(response0, response1)

    assert (
        f"key='{which_key}' with value='{primary_key}' not found in `{error_on}`."
        in str(exc_info.value)
    )


@pytest.mark.parametrize(
    "response0, response1, primary_key, secondary_key, error_on, which_key",
    [
        (
            {"entail": 1, "neutral": 2, "contra": 3},
            {"entail": 1, "neutral": 2, "contra": 3},
            "entail",
            "__neutral__",
            "response0",
            "secondary_key",
        ),
    ],
)
def test_RelaxedNLIOutputCombinationStrategy_validate_input_dicts_against_class_keys_error_secondary_key(
    response0: dict,
    response1: dict,
    primary_key: str,
    secondary_key: str,
    error_on: str,
    which_key: str,
):
    strategy = RelaxedNLIOutputCombinationStrategy(
        primary_key=primary_key, secondary_key=secondary_key
    )
    with pytest.raises(ValueError) as exc_info:
        strategy.run(response0, response1)

    assert (
        f"key='{which_key}' with value='{secondary_key}' not found in `{error_on}`."
        in str(exc_info.value)
    )


@pytest.mark.parametrize(
    "response0, response1, positive_key, error_on",
    [
        (
            {"entail": 1, "neutral": 2, "contra": 3},
            {"entail": 1, "neutral": 2, "contra": 3},
            "__entail__",
            "response0",
        ),
    ],
)
def test_StrictNLIOutputCombinationStrategy_validate_input_dicts_against_class_keys_error(
    response0: dict, response1: dict, positive_key: str, error_on: str
):
    strategy = AverageNLIOutputCombinationStrategy(positive_key=positive_key)
    with pytest.raises(ValueError) as exc_info:
        strategy.run(response0, response1)

    assert (
        f"key='positive_key' with value='{positive_key}' not found in `{error_on}`."
        in str(exc_info.value)
    )
