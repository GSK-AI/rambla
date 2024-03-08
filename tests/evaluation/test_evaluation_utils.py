from unittest import mock

import numpy as np
import pytest

from rambla.evaluation.utils import (
    compute_huggingface_metric,
    compute_scipy_metric,
    compute_sklearn_metric,
    run_metric,
)


@pytest.fixture
def mock_categorical_targets() -> list[int]:
    return [0, 1, 1, 1, 0, 0, 1, 0]


@pytest.fixture
def mock_float_predictions() -> list[float]:
    return [0.10, 0.20, 0.30, 0.70, 0.80, 0.40, 0.25, 0.50]


@pytest.fixture
def mock_float_targets() -> list[float]:
    return [0.10, 0.20, 0.30, 0.70, 0.80, 0.40, 0.25, 0.50]


# NOTE: mocking to be generic
@mock.patch("rambla.evaluation.utils.cached_load_module")
def test_compute_huggingface_metric_without_kwargs(mock_cached_load_module):
    mock_metric = mock.Mock()
    mock_metric.compute = mock.Mock()
    expected_output = {"value": 13}
    mock_metric.compute.return_value = expected_output
    mock_cached_load_module.return_value = mock_metric

    metric_name = "__dummy_metric_name__"
    predictions = [1, 2, 3]
    targets = [1, 2, 3]

    # call
    output = compute_huggingface_metric(
        metric_name=metric_name,
        predictions=predictions,
        targets=targets,
    )

    # asserts
    assert output == expected_output
    call_kwargs = mock_metric.compute.call_args.kwargs
    assert np.allclose(
        np.array(predictions, dtype=np.float64), call_kwargs["predictions"]
    )
    assert np.allclose(np.array(targets, dtype=np.float64), call_kwargs["references"])


# NOTE: mocking to be generic
@mock.patch("rambla.evaluation.utils.cached_load_module")
def test_compute_huggingface_metric_with_kwargs(mock_cached_load_module):
    mock_metric = mock.Mock()
    mock_metric.compute = mock.Mock()
    expected_output = {"value": 13}
    mock_metric.compute.return_value = expected_output
    mock_cached_load_module.return_value = mock_metric

    metric_name = "__dummy_metric_name__"
    predictions = [1, 2, 3]
    targets = [1, 2, 3]
    kwargs = {"dummy": 3, "another_dummy": "hello"}

    # call
    output = compute_huggingface_metric(
        metric_name=metric_name, predictions=predictions, targets=targets, kwargs=kwargs
    )

    # asserts
    assert output == expected_output
    call_kwargs = mock_metric.compute.call_args.kwargs
    assert np.allclose(
        np.array(predictions, dtype=np.float64), call_kwargs["predictions"]
    )
    assert np.allclose(np.array(targets, dtype=np.float64), call_kwargs["references"])
    assert call_kwargs["dummy"] == kwargs["dummy"]
    assert call_kwargs["another_dummy"] == kwargs["another_dummy"]


@pytest.mark.parametrize(
    "metric_name, predictions, targets, expected_output",
    [
        (
            "roc_auc_score",
            [0.10, 0.20, 0.50, 0.80, 0.25, 0.90, 0.15, 0.85],
            [0, 0, 1, 0, 1, 0, 0, 0],
            {"roc_auc_score": 0.50},
        ),
        (
            "reverse_roc_auc_score",
            [0, 0, 1, 0, 1, 0, 0, 0],
            [0.10, 0.20, 0.50, 0.80, 0.25, 0.90, 0.15, 0.85],
            {"reverse_roc_auc_score": 0.50},
        ),
    ],
)
def test_compute_sklearn_metric_roc_auc_score(
    metric_name: str,
    predictions: list[float],
    targets: list[float],
    expected_output: float,
):
    output = compute_sklearn_metric(
        metric_name=metric_name, predictions=predictions, targets=targets
    )

    # asserts
    assert np.isclose(output[metric_name], expected_output[metric_name])


@pytest.mark.parametrize(
    "metric_name, predictions, targets, kwargs, expected_output",
    [
        (
            "roc_curve",
            [0.10, 0.20, 0.50, 0.80],
            [0, 0, 1, 0],
            {"pos_label": 1},
            {
                "roc_curve": {
                    "fpr": np.array([0.0, 1 / 3, 1 / 3, 1.0]),
                    "tpr": np.array([0.0, 0.0, 1.0, 1.0]),
                    "thresholds": np.array([np.inf, 0.8, 0.5, 0.1]),
                }
            },
        ),
        (
            "roc_curve",
            [0.10, 0.20, 0.50, 0.80],
            [0, 0, 1, 0],
            {"pos_label": 0},
            {
                "roc_curve": {
                    "fpr": np.array([0.0, 0.0, 1.0, 1.0]),
                    "tpr": np.array([0.0, 1 / 3, 1 / 3, 1.0]),
                    "thresholds": np.array([np.inf, 0.8, 0.5, 0.1]),
                }
            },
        ),
        (
            "roc_curve",
            [0.10, 0.20, 0.50, 0.80, 0.05, 0.90],
            [0, 0, 1, 0, 1, 1],
            {"pos_label": 1},
            {
                "roc_curve": {
                    "fpr": np.array([0.0, 0.0, 1 / 3, 1 / 3, 1.0, 1.0]),
                    "tpr": np.array([0.0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1.0]),
                    "thresholds": np.array([np.inf, 0.9, 0.8, 0.5, 0.1, 0.05]),
                }
            },
        ),
        (
            "roc_curve",
            [0.10, 0.20, 0.50, 0.80, 0.05, 0.90],
            [0, 0, 1, 0, 1, 1],
            {"pos_label": 0},
            {
                "roc_curve": {
                    "fpr": np.array([0.0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1.0]),
                    "tpr": np.array([0.0, 0.0, 1 / 3, 1 / 3, 1.0, 1.0]),
                    "thresholds": np.array([np.inf, 0.9, 0.8, 0.5, 0.1, 0.05]),
                }
            },
        ),
        (
            "reverse_roc_curve",
            [0, 0, 1, 0],
            [0.10, 0.20, 0.50, 0.80],
            {"pos_label": 1},
            {
                "reverse_roc_curve": {
                    "fpr": np.array([0.0, 1 / 3, 1 / 3, 1.0]),
                    "tpr": np.array([0.0, 0.0, 1.0, 1.0]),
                    "thresholds": np.array([np.inf, 0.8, 0.5, 0.1]),
                }
            },
        ),
        (
            "reverse_roc_curve",
            [0, 0, 1, 0],
            [0.10, 0.20, 0.50, 0.80],
            {"pos_label": 0},
            {
                "reverse_roc_curve": {
                    "fpr": np.array([0.0, 0.0, 1.0, 1.0]),
                    "tpr": np.array([0.0, 1 / 3, 1 / 3, 1.0]),
                    "thresholds": np.array([np.inf, 0.8, 0.5, 0.1]),
                }
            },
        ),
        (
            "reverse_roc_curve",
            [0, 0, 1, 0, 1, 1],
            [0.10, 0.20, 0.50, 0.80, 0.05, 0.90],
            {"pos_label": 1},
            {
                "reverse_roc_curve": {
                    "fpr": np.array([0.0, 0.0, 1 / 3, 1 / 3, 1.0, 1.0]),
                    "tpr": np.array([0.0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1.0]),
                    "thresholds": np.array([np.inf, 0.9, 0.8, 0.5, 0.1, 0.05]),
                }
            },
        ),
        (
            "reverse_roc_curve",
            [0, 0, 1, 0, 1, 1],
            [0.10, 0.20, 0.50, 0.80, 0.05, 0.90],
            {"pos_label": 0},
            {
                "reverse_roc_curve": {
                    "fpr": np.array([0.0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1.0]),
                    "tpr": np.array([0.0, 0.0, 1 / 3, 1 / 3, 1.0, 1.0]),
                    "thresholds": np.array([np.inf, 0.9, 0.8, 0.5, 0.1, 0.05]),
                }
            },
        ),
    ],
)
def test_compute_sklearn_metric_roc_curve(
    metric_name: str,
    predictions: list[float],
    targets: list[float],
    kwargs: dict,
    expected_output: tuple[np.ndarray, np.ndarray],
):
    output = compute_sklearn_metric(
        metric_name=metric_name, predictions=predictions, targets=targets, kwargs=kwargs
    )

    # asserts
    assert np.allclose(output[metric_name]["tpr"], expected_output[metric_name]["tpr"])
    assert np.allclose(output[metric_name]["fpr"], expected_output[metric_name]["fpr"])
    assert np.allclose(
        output[metric_name]["thresholds"], expected_output[metric_name]["thresholds"]
    )


@pytest.mark.parametrize(
    "metric_name, expected_statistic, expected_pvalue",
    [
        ("pearsonr", 0.9999, 0),
        ("spearmanr", 1.0, 0),
        ("pointbiserialr", 0.9999, 0),
        ("mannwhitneyu", 32.0, 1.0),
        ("kendalltau", 0.9999, 4.96e-05),
    ],
)
def test_compute_scipy_metric(
    metric_name: str,
    expected_statistic: float,
    expected_pvalue: float,
    mock_float_predictions: list[float],
    mock_float_targets: list[float],
):
    output_statistic, output_pvalue = compute_scipy_metric(
        metric_name=metric_name,
        predictions=mock_float_predictions,
        targets=mock_float_targets,
    )

    # asserts
    assert np.isclose(output_statistic, expected_statistic, rtol=1e-4)
    assert np.isclose(output_pvalue, expected_pvalue, rtol=1e-4)


@mock.patch("rambla.evaluation.utils.compute_scipy_metric")
def test_run_metric_scipy_with_kwargs(
    mock_compute_scipy_metric,
    mock_float_predictions: list[float],
    mock_float_targets: list[float],
):
    # setup
    mock_output = (0.93, 0.01)
    mock_compute_scipy_metric.return_value = mock_output
    kwargs = {"dummy": 3, "also_dummy": "hey"}
    metric_name = "pearsonr"
    expected_output = {
        f"stats_{metric_name}_statistic": mock_output[0],
        f"stats_{metric_name}_pvalue": mock_output[1],
    }

    # Run
    output = run_metric(
        metric_name=f"stats_{metric_name}",
        predictions=mock_float_predictions,
        targets=mock_float_targets,
        kwargs_dict=kwargs,
    )

    # checks
    assert output == expected_output
    call_kwargs = mock_compute_scipy_metric.call_args.kwargs
    assert call_kwargs["metric_name"] == metric_name
    assert call_kwargs["kwargs"] == kwargs
    assert np.allclose(call_kwargs["predictions"], mock_float_predictions)
    assert np.allclose(call_kwargs["targets"], mock_float_targets)


@mock.patch("rambla.evaluation.utils.compute_scipy_metric")
def test_run_metric_scipy_without_kwargs(
    mock_compute_scipy_metric,
    mock_float_predictions: list[float],
    mock_float_targets: list[float],
):
    # setup
    mock_output = (0.93, 0.01)
    mock_compute_scipy_metric.return_value = mock_output
    metric_name = "pearsonr"
    expected_output = {
        f"stats_{metric_name}_statistic": mock_output[0],
        f"stats_{metric_name}_pvalue": mock_output[1],
    }

    # Run
    output = run_metric(
        metric_name=f"stats_{metric_name}",
        predictions=mock_float_predictions,
        targets=mock_float_targets,
    )

    # checks
    assert output == expected_output
    call_kwargs = mock_compute_scipy_metric.call_args.kwargs
    assert call_kwargs["metric_name"] == metric_name
    assert call_kwargs["kwargs"] == {}
    assert np.allclose(call_kwargs["predictions"], mock_float_predictions)
    assert np.allclose(call_kwargs["targets"], mock_float_targets)


@mock.patch("rambla.evaluation.utils.compute_sklearn_metric")
def test_run_metric_sklearn_with_kwargs(
    mock_compute_scipy_metric,
    mock_float_predictions: list[float],
    mock_float_targets: list[float],
):
    # setup
    mock_output = (0.93, 0.01)
    mock_compute_scipy_metric.return_value = mock_output
    kwargs = {"dummy": 3, "also_dummy": "hey"}
    metric_name = "roc_auc_score"

    # Run
    output = run_metric(
        metric_name=f"sklearn_{metric_name}",
        predictions=mock_float_predictions,
        targets=mock_float_targets,
        kwargs_dict=kwargs,
    )

    # checks
    assert output == mock_output
    call_kwargs = mock_compute_scipy_metric.call_args.kwargs
    assert call_kwargs["metric_name"] == metric_name
    assert call_kwargs["kwargs"] == kwargs
    assert np.allclose(call_kwargs["predictions"], mock_float_predictions)
    assert np.allclose(call_kwargs["targets"], mock_float_targets)


@mock.patch("rambla.evaluation.utils.compute_sklearn_metric")
def test_run_metric_sklearn_without_kwargs(
    mock_compute_scipy_metric,
    mock_float_predictions: list[float],
    mock_float_targets: list[float],
):
    # setup
    mock_output = (0.93, 0.01)
    mock_compute_scipy_metric.return_value = mock_output
    metric_name = "roc_auc_score"

    # Run
    output = run_metric(
        metric_name=f"sklearn_{metric_name}",
        predictions=mock_float_predictions,
        targets=mock_float_targets,
    )

    # checks
    assert output == mock_output
    call_kwargs = mock_compute_scipy_metric.call_args.kwargs
    assert call_kwargs["metric_name"] == metric_name
    assert call_kwargs["kwargs"] == {}
    assert np.allclose(call_kwargs["predictions"], mock_float_predictions)
    assert np.allclose(call_kwargs["targets"], mock_float_targets)


@mock.patch("rambla.evaluation.utils.compute_huggingface_metric")
def test_run_metric_huggingface_with_kwargs(
    mock_compute_scipy_metric,
    mock_float_predictions: list[float],
    mock_float_targets: list[float],
):
    # setup
    mock_output = (0.93, 0.01)
    mock_compute_scipy_metric.return_value = mock_output
    kwargs = {"dummy": 3, "also_dummy": "hey"}
    metric_name = "recall"

    # Run
    _ = run_metric(
        metric_name=metric_name,
        predictions=mock_float_predictions,
        targets=mock_float_targets,
        kwargs_dict=kwargs,
    )

    # checks
    call_kwargs = mock_compute_scipy_metric.call_args.kwargs
    assert call_kwargs["metric_name"] == metric_name
    assert call_kwargs["kwargs"] == kwargs
    assert np.allclose(call_kwargs["predictions"], mock_float_predictions)
    assert np.allclose(call_kwargs["targets"], mock_float_targets)


@mock.patch("rambla.evaluation.utils.compute_huggingface_metric")
def test_run_metric_huggingface_without_kwargs(
    mock_compute_scipy_metric,
    mock_float_predictions: list[float],
    mock_float_targets: list[float],
):
    # setup
    mock_output = (0.93, 0.01)
    mock_compute_scipy_metric.return_value = mock_output
    metric_name = "recall"

    # Run
    _ = run_metric(
        metric_name=metric_name,
        predictions=mock_float_predictions,
        targets=mock_float_targets,
    )

    # checks
    call_kwargs = mock_compute_scipy_metric.call_args.kwargs
    assert call_kwargs["metric_name"] == metric_name
    assert call_kwargs["kwargs"] == {}
    assert np.allclose(call_kwargs["predictions"], mock_float_predictions)
    assert np.allclose(call_kwargs["targets"], mock_float_targets)
