import math
from typing import Dict, List
from unittest import mock
from unittest.mock import patch

import numpy as np
import pytest
from datasets import Dataset

from rambla.evaluation.shortform import MCQAEvalComponent, MCQAStratifiedEvalComponent

# flake8: noqa: N802


@pytest.mark.parametrize(
    "categories",
    [
        (["yes", "no"]),
        (["yes"]),
    ],
)
def test_MCQAEvalComponent_unseen_category_in_predictions(  # noqa: E501
    categories,
):
    predictions = [" yes"]
    references = ["yes"]

    eval_obj = MCQAEvalComponent(categories=categories)
    with pytest.raises(ValueError):
        eval_obj.run(predictions=predictions, references=references)


@pytest.mark.parametrize(
    "categories",
    [
        (["yes", "no"]),
        (["yes"]),
    ],
)
def test_MCQAEvalComponent_unseen_category_in_references(categories):
    predictions = ["yes"]
    references = [" yes"]

    eval_obj = MCQAEvalComponent(categories=categories)
    with pytest.raises(ValueError):
        eval_obj.run(predictions=predictions, references=references)


@pytest.mark.parametrize(
    (
        "predictions, references, categories, metric_names, "
        "metric_kwargs, expected_results, expected_transform_map"
    ),
    [
        (
            ["no"],
            ["no"],
            ["no"],
            ["recall", "precision"],
            {"recall": {"pos_label": 0}, "precision": {"pos_label": 0}},
            {"recall": 1.0, "precision": 1.0},
            {"no": 0},
        ),
        (
            ["no", "yes"],
            ["no", "yes"],
            ["no", "yes"],
            ["recall", "precision"],
            {"recall": {"pos_label": 1}, "precision": {"pos_label": 1}},
            {"recall": 1.0, "precision": 1.0},
            {"no": 0, "yes": 1},
        ),
        (
            ["no", "yes", "yes"],
            ["no", "yes", "yes"],
            ["no", "yes", "null"],  # Checks it works with null category
            ["recall", "precision"],
            {"recall": {"pos_label": 1}, "precision": {"pos_label": 1}},
            {"recall": 1.0, "precision": 1.0},
            {"no": 0, "null": 1, "yes": 2},
        ),
    ],
)
def test_MCQAEvalComponent_run(
    predictions: List[str],
    references: List[str],
    categories: List[str],
    metric_names: List[str],
    metric_kwargs: Dict[str, dict],
    expected_results: Dict[str, float],
    expected_transform_map: Dict[str, int],
):
    eval_obj = MCQAEvalComponent(
        categories=categories, metric_names=metric_names, metric_kwargs=metric_kwargs
    )
    output, le = eval_obj.run(predictions=predictions, references=references)

    assert output == expected_results
    assert le == expected_transform_map


@pytest.mark.parametrize(
    "predictions, references, categories, average, expected_results",
    [
        ([1, 1, 1, 1], [0, 1, 1, 1], [0, 1], "binary", {"recall": 1.0}),
        ([1, 1, 1, 1], [0, 1, 1, 1], [0, 1], "weighted", {"recall": 0.75}),
        # Checks works with additional category (e.g. null category)
        ([1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 2], "weighted", {"recall": 0.75}),
    ],
)
def test_MCQAEvalComponent_run_set_average(
    predictions: List[int],
    references: List[int],
    categories: List[int],
    average: str,
    expected_results: Dict[str, float],
):
    eval_obj = MCQAEvalComponent(
        categories=categories, metric_names=["recall"], average=average  # type: ignore
    )
    output, _ = eval_obj.run(predictions=predictions, references=references)  # type: ignore

    assert output == expected_results


def test_MCQAEvalComponent_average_is_configurable():
    config = dict(categories=["0", "1"], average="binary", metric_names=["recall"])

    predictions = ["1", "1", "1", "1"]
    references = ["0", "1", "1", "1"]

    eval_obj = MCQAEvalComponent.from_config(config)
    output, _ = eval_obj.run(predictions=predictions, references=references)

    assert output == {"recall": 1.0}  # Recall is 0.75 in default weighted average


@pytest.mark.parametrize(
    "input_kwargs, expected_kwargs",
    [
        (None, {"recall": {"average": "binary"}}),
        ({"recall": {"average": "weighted"}}, {"recall": {"average": "weighted"}}),
    ],
)
def test_MCQAEvalComponent_override_kwargs(input_kwargs: dict, expected_kwargs) -> None:
    eval_obj = MCQAEvalComponent(
        categories=["yes", "no"],
        metric_names=["recall"],
        average="binary",
        metric_kwargs=input_kwargs,
    )

    assert eval_obj.metric_kwargs == expected_kwargs


def test_MCQAEvalComponent_evaluate_mock():
    eval_obj = MCQAEvalComponent(
        categories=["yes", "no"],
        response_field="response",
        target_field="final_decision",
    )
    predictions = list("ABCD")
    targets = list("1234")

    dataset = Dataset.from_dict({"response": predictions, "final_decision": targets})

    mock_run = mock.MagicMock()
    mock_run.return_value = ("mock_results", "mock_label_encoder")

    eval_obj.run = mock_run

    #
    output = eval_obj.evaluate(dataset)

    #
    assert output == {"results": "mock_results", "label_encoder": "mock_label_encoder"}
    mock_run.assert_called_with(predictions=predictions, references=targets)


def test_MCQAEvalComponent_evaluate():
    predictions = ["yes", "yes", "no", "null", "no", "yes"]
    targets = ["no", "yes", "yes", "null", "null", "yes"]

    dataset = Dataset.from_dict({"response": predictions, "final_decision": targets})
    categories = ["no", "yes", "null"]

    eval_obj = MCQAEvalComponent(
        categories=categories,
        metric_names=["BucketHeadP65/confusion_matrix"],
        metric_kwargs={"BucketHeadP65/confusion_matrix": {"labels": [0, 1, 2]}},
    )

    #
    output_dict = eval_obj.evaluate(dataset)

    #
    expected_label_encoder = {"no": 0, "null": 1, "yes": 2}
    expected_confmat = np.array(
        [
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 2],
        ]
    )
    expected_results = {
        "confusion_matrix": expected_confmat,
    }

    assert output_dict["label_encoder"] == expected_label_encoder

    for key, value in output_dict["results"].items():
        if isinstance(value, np.ndarray):
            assert np.allclose(value, expected_results[key]), f"{key=}"
        else:
            assert value == expected_results[key], f"{key=}"


def test_MCQAEvalComponent_evaluate_with_classes_to_exclude():
    predictions = ["yes", "yes", "no", "null", "no", "yes", "yes", "no", "yes", "no"]
    targets = ["no", "yes", "yes", "no", "yes", "yes", "no", "yes", "no", "no"]

    dataset = Dataset.from_dict({"response": predictions, "final_decision": targets})
    categories = ["no", "yes", "null"]

    eval_obj = MCQAEvalComponent(
        categories=categories,
        metric_names=["BucketHeadP65/confusion_matrix"],
        metric_kwargs={"BucketHeadP65/confusion_matrix": {"labels": [0, 1, 2]}},
        classes_to_exclude=["null"],
    )

    #
    output_dict = eval_obj.evaluate(dataset)

    #
    expected_label_encoder = {"no": 0, "null": 1, "yes": 2}

    expected_results = {
        "confusion_matrix": np.array([[1, 1, 3], [0, 0, 0], [3, 0, 2]]),
        "excl/recall/per_class": np.array([0.25, 0.4]),
        "excl/recall/macro": 0.325,
        "excl/recall/micro": 1 / 3,
        "excl/precision/per_class": np.array([0.25, 0.4]),
        "excl/precision/macro": 0.325,
        "excl/precision/micro": 1 / 3,
        "excl/f1/micro": 1 / 3,
        "excl/f1/per_class": np.array([0.25, 0.4]),
        "excl/f1/macro": 0.325,
        "excl/label_encoder": {"no": 0, "yes": 1},
    }

    assert output_dict["label_encoder"] == expected_label_encoder

    for key, value in output_dict["results"].items():
        if isinstance(value, np.ndarray):
            assert np.allclose(value, expected_results[key], equal_nan=True), f"{key=}"
        elif isinstance(value, dict):
            assert value == expected_results[key]
        else:
            assert math.isclose(value, expected_results[key]), f"{key=}"


class TestMCQAStratifiedEvalComponent:
    @pytest.fixture(scope="class")
    def config(self):
        return dict(
            stratify_field="difficulty",
            categories=["yes", "no"],
            response_field="response",
            target_field="final_decision",
            metric_names=["recall", "precision"],
            average="binary",
        )

    @pytest.fixture
    def stratified_eval_obj(self, config):
        return MCQAStratifiedEvalComponent(**config)

    @pytest.fixture
    def stratified_eval_obj_from_config(self, config):
        return MCQAStratifiedEvalComponent.from_config(config)

    @pytest.mark.parametrize(
        "fixture",
        [
            "stratified_eval_obj",
            "stratified_eval_obj_from_config",
        ],
    )
    def test_raises_if_not_enough_stratify_groups(self, request, fixture):
        # arrange
        stratified_eval_obj = request.getfixturevalue(fixture)

        predictions = ["yes", "no", "yes", "no"]
        targets = ["yes", "yes", "yes", "yes"]
        difficulties = ["easy", "easy", "easy", "easy"]

        dataset_with_only_one_difficulty = Dataset.from_dict(
            {
                "response": predictions,
                "final_decision": targets,
                "difficulty": difficulties,
            }
        )

        # act & assert
        with pytest.raises(ValueError):
            stratified_eval_obj.evaluate(
                dataset_with_only_one_difficulty,
            )

    @pytest.mark.parametrize(
        "fixture",
        [
            "stratified_eval_obj",
            "stratified_eval_obj_from_config",
        ],
    )
    def test_produces_metrics_per_stratify_group(self, request, fixture):
        # arrange
        stratified_eval_obj = request.getfixturevalue(fixture)

        predictions = ["yes", "no", "yes", "yes", "no", "yes"]
        targets = ["yes", "no", "yes", "no", "yes", "no"]
        difficulties = ["easy", "easy", "medium", "medium", "hard", "hard"]

        dataset_with_multiple_difficulties = Dataset.from_dict(
            {
                "response": predictions,
                "final_decision": targets,
                "difficulty": difficulties,
            }
        )

        # act
        with patch.object(
            stratified_eval_obj,
            "run",
            wraps=stratified_eval_obj.run,
        ) as spied_eval_obj:
            actual_output = stratified_eval_obj.evaluate(
                dataset_with_multiple_difficulties,
            )

            # assert
            assert spied_eval_obj.call_count == 3

            expected_output = {
                "easy": ({"recall": 1.0, "precision": 1.0}, {"yes": 1, "no": 0}),
                "medium": ({"recall": 1.0, "precision": 0.5}, {"yes": 1, "no": 0}),
                "hard": ({"recall": 0.0, "precision": 0.0}, {"yes": 1, "no": 0}),
            }

            assert actual_output == expected_output


def test_MCQAEvalComponent_evaluate_with_compute_class_counts_from_confmat():
    predictions = ["yes", "yes", "no", "null", "no", "yes"]
    targets = ["no", "yes", "yes", "null", "null", "yes"]

    dataset = Dataset.from_dict({"response": predictions, "final_decision": targets})
    categories = ["no", "yes", "null"]

    eval_obj = MCQAEvalComponent(
        categories=categories,
        metric_names=["BucketHeadP65/confusion_matrix"],
        metric_kwargs={"BucketHeadP65/confusion_matrix": {"labels": [0, 1, 2]}},
        compute_class_counts_from_confmat=True,
    )

    #
    output_dict = eval_obj.evaluate(dataset)

    #
    expected_label_encoder = {"no": 0, "null": 1, "yes": 2}
    expected_confmat = np.array(
        [
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 2],
        ]
    )

    # NOTE: expected_confmat.sum() -> 6
    expected_results = {
        "class_counts/n_pred_yes": 3,
        "class_counts/n_pred_no": 2,
        "class_counts/n_pred_null": 1,
        "class_counts/n_target_yes": 3,
        "class_counts/n_target_no": 1,
        "class_counts/n_target_null": 2,
        #
        "class_counts/prop_pred_yes": 3 / 6,
        "class_counts/prop_pred_no": 2 / 6,
        "class_counts/prop_pred_null": 1 / 6,
        "class_counts/prop_target_yes": 3 / 6,
        "class_counts/prop_target_no": 1 / 6,
        "class_counts/prop_target_null": 2 / 6,
        #
        "confusion_matrix": expected_confmat,
    }

    assert output_dict["label_encoder"] == expected_label_encoder

    for key, value in output_dict["results"].items():
        if isinstance(value, np.ndarray):
            assert np.allclose(value, expected_results[key]), f"{key=}"
        else:
            assert value == expected_results[key], f"{key=}"
