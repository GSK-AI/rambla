import numpy as np
import pytest

from rambla.utils.metrics import (
    compute_class_counts_from_confmat,
    compute_f1_from_confmat,
    compute_metrics_from_confmat,
    compute_precision_from_confmat,
    compute_recall_from_confmat,
    filter_confmat,
    get_metrics_helper,
)


@pytest.mark.parametrize(
    "confusion_matrix, expected_output",
    [
        (
            np.array([[1, 2, 5], [2, 1, 2], [1, 1, 3]]),
            {
                "per_class": np.array([0.125, 0.2, 0.6]),
                "micro": np.array(5 / 18),
                "macro": np.array(0.925 / 3),
            },
        ),
        (
            np.array([[0, 0, 0], [2, 1, 2], [1, 1, 3]]),
            {
                "per_class": np.array([1, 0.2, 0.6]),
                "micro": np.array(0.40),
                "macro": np.array(0.60),
            },
        ),
        (
            np.array([[0, 2, 4], [0, 2, 2], [0, 1, 3]]),
            {
                "per_class": np.array([0.0, 0.5, 0.75]),
                "micro": np.array(5 / 14),
                "macro": np.array(1.25 / 3),
            },
        ),
        (
            np.array([[0, 2, 5], [2, 0, 2], [1, 1, 0]]),
            {
                "per_class": np.array([0.0, 0, 0]),
                "micro": np.array(0.0),
                "macro": np.array(0.0),
            },
        ),
        (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            {
                "per_class": np.array([1.0, 1.0, 1.0]),
                "micro": np.array(1.0),
                "macro": np.array(1.0),
            },
        ),
    ],
)
def test_compute_recall_from_confmat(confusion_matrix, expected_output):
    output = compute_recall_from_confmat(confusion_matrix)
    assert np.allclose(
        output["per_class"], expected_output["per_class"], equal_nan=True
    )
    assert np.allclose(output["micro"], expected_output["micro"], equal_nan=True)
    assert np.allclose(output["macro"], expected_output["macro"], equal_nan=True)


@pytest.mark.parametrize(
    "confusion_matrix, expected_output",
    [
        (
            np.array([[1, 2, 5], [2, 1, 2], [1, 1, 3]]),
            {
                "per_class": np.array([0.25, 0.25, 0.30]),
                "micro": np.array(5 / 18),
                "macro": np.array([0.25, 0.25, 0.30]).sum() / 3,
            },
        ),
        (
            np.array([[0, 0, 0], [2, 1, 2], [1, 1, 3]]),
            {
                "per_class": np.array([0.0, 0.50, 0.60]),
                "micro": np.array(0.40),
                "macro": np.array(1.1 / 3),
            },
        ),
        (
            np.array([[0, 2, 4], [0, 2, 2], [0, 1, 3]]),
            {
                "per_class": np.array([1, 0.40, 0.3333]),
                "micro": np.array(5 / 14),
                "macro": np.array(26 / 45),
            },
        ),
        (
            np.array([[0, 2, 5], [2, 0, 2], [1, 1, 0]]),
            {
                "per_class": np.array([0.0, 0, 0]),
                "micro": np.array(0.0),
                "macro": np.array(0.0),
            },
        ),
        (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            {
                "per_class": np.array([1.0, 1.0, 1.0]),
                "micro": np.array(1.0),
                "macro": np.array(1.0),
            },
        ),
    ],
)
def test_compute_precision_from_confmat(confusion_matrix, expected_output):
    output = compute_precision_from_confmat(confusion_matrix)
    assert np.allclose(
        output["per_class"], expected_output["per_class"], equal_nan=True, atol=1e-4
    )
    assert np.allclose(output["micro"], expected_output["micro"], equal_nan=True)
    assert np.allclose(output["macro"], expected_output["macro"], equal_nan=True)


@pytest.mark.parametrize(
    "confusion_matrix, expected_output",
    [
        (
            np.array([[1, 2, 5], [2, 1, 2], [1, 1, 3]]),
            {
                "per_class": 2 * np.array([0.03125 / 0.375, 0.1111, 0.20]),
                "micro": np.array(2 * (5 / 18 * 5 / 18) / (10 / 18)),
                "macro": 2 / 3 * np.array([0.03125 / 0.375, 0.1111, 0.20]).sum(),
            },
        ),
        (
            np.array([[0, 0, 0], [2, 1, 2], [1, 1, 3]]),
            {
                "per_class": 2 * np.array([0, 1 / 7, 0.30]),
                "micro": np.array(0.40),
                "macro": np.array(0.2952),
            },
        ),
        (
            np.array([[0, 2, 4], [0, 2, 2], [0, 1, 3]]),
            {
                "per_class": 2 * np.array([0, 0.2222, (0.75 * 0.3333) / 1.0833]),
                "micro": np.array(5 / 14),
                "macro": np.array(0.3020),
            },
        ),
        (
            np.array([[0, 2, 5], [2, 0, 2], [1, 1, 0]]),
            {
                "per_class": np.array(
                    [
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    ]
                ),
                "micro": np.array(float("nan")),
                "macro": np.array(float("nan")),
            },
        ),
        (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            {
                "per_class": np.array([1.0, 1.0, 1.0]),
                "micro": np.array(1.0),
                "macro": np.array(1.0),
            },
        ),
    ],
)
def test_compute_f1_from_confmat(confusion_matrix, expected_output):
    output = compute_f1_from_confmat(confusion_matrix)
    assert np.allclose(
        output["per_class"], expected_output["per_class"], equal_nan=True, atol=1e-4
    )
    assert np.allclose(
        output["micro"], expected_output["micro"], equal_nan=True, atol=1e-4
    )
    assert np.allclose(
        output["macro"], expected_output["macro"], equal_nan=True, atol=1e-4
    )


def test_filter_confmat():
    confmat = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    label_encoder = {"yes": 2, "no": 0, "null": 1}
    to_exclude = ["null"]

    #
    output_confmat, output_label_encoder = filter_confmat(
        confmat, label_encoder, to_exclude
    )

    #
    expected_confmat = confmat[[0, 2], :][:, [0, 2]]
    expected_label_encoder = {"no": 0, "yes": 1}

    assert output_confmat.shape == expected_confmat.shape
    assert np.allclose(output_confmat, expected_confmat)
    assert output_label_encoder == expected_label_encoder


def test_filter_confmat_label_not_present():
    confmat = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    label_encoder = {"yes": 2, "no": 0, "null": 1}
    to_exclude = ["dummy"]

    #
    output_confmat, output_label_encoder = filter_confmat(
        confmat, label_encoder, to_exclude
    )

    #
    expected_confmat = confmat
    expected_label_encoder = {"no": 0, "null": 1, "yes": 2}

    assert output_confmat.shape == expected_confmat.shape
    assert np.allclose(output_confmat, expected_confmat)
    assert output_label_encoder == expected_label_encoder


def test_filter_confmat_label_not_present_strct():
    confmat = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    label_encoder = {"yes": 2, "no": 0, "null": 1}
    to_exclude = ["dummy"]

    #
    with pytest.raises(KeyError):
        filter_confmat(confmat, label_encoder, to_exclude, True)


def test_get_metrics_helper():
    confmat = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    label_encoder = {"yes": 2, "no": 0, "null": 1}
    to_exclude = ["null"]

    #
    output_metrics, output_label_encoder = get_metrics_helper(
        confmat, label_encoder, to_exclude
    )

    reduced_confmat = confmat[[0, 2], :][:, [0, 2]]
    expected_metrics = compute_metrics_from_confmat(reduced_confmat)

    #
    assert output_label_encoder == {"no": 0, "yes": 1}
    for key, value in output_metrics.items():
        metric_name, metric_type = key.split("/")
        expected_value = expected_metrics[metric_name][metric_type]
        if isinstance(value, np.ndarray):
            assert np.allclose(value, expected_value)
        else:
            assert value == expected_value


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
        "null": 1,
        "no": 2,
    }

    # run
    output = compute_class_counts_from_confmat(confmat, label_encoder)

    # checks
    # NOTE: confmat.sum() -> 258
    expected_output = {
        "n_pred_yes": 11,
        "n_pred_null": 24,
        "n_pred_no": 223,
        "n_target_yes": 30,
        "n_target_null": 9,
        "n_target_no": 219,
        #
        "prop_pred_yes": 11 / 258,
        "prop_pred_null": 24 / 258,
        "prop_pred_no": 223 / 258,
        "prop_target_yes": 30 / 258,
        "prop_target_null": 9 / 258,
        "prop_target_no": 219 / 258,
    }

    assert output == expected_output
