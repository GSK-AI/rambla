from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from datasets import Dataset
from pydantic import BaseModel, Extra, root_validator, validator
from sklearn.preprocessing import LabelEncoder

from rambla.evaluation.base import BaseTargetReferenceEvalComponent
from rambla.evaluation.utils import run_metric
from rambla.utils.metrics import compute_class_counts_from_confmat, get_metrics_helper
from rambla.utils.misc import add_prefix_to_dict_keys


class MCQAEvalComponentConfig(BaseModel):
    categories: List[str]
    response_field: str = "response"
    target_field: str = "final_decision"
    metric_names: Optional[List[str]] = None
    metric_kwargs: Dict[str, Dict[str, Any]] = None
    average: Literal["binary", "micro", "macro", "weighted", "samples"] = "weighted"
    classes_to_exclude: Optional[str | list[str]]
    compute_class_counts_from_confmat: bool = False

    class Config:  # noqa: D106
        extra = Extra.forbid

    @validator("classes_to_exclude", pre=True)
    @classmethod
    def validate_classes_to_exclude(cls, classes_to_exclude):
        if classes_to_exclude is not None and isinstance(classes_to_exclude, str):
            classes_to_exclude = [classes_to_exclude]
        return classes_to_exclude

    @root_validator()
    @classmethod
    def validate_confmat_present(cls, values):
        if values["classes_to_exclude"] is not None:
            error_msg = (
                "If `classes_to_exclude` requires `BucketHeadP65/confusion_matrix` "
                "to be one of the values in `metric_names`."
            )
            assert isinstance(values["metric_names"], list)
            assert "BucketHeadP65/confusion_matrix" in values["metric_names"], error_msg
        return values

    @root_validator()
    @classmethod
    def validate_compute_class_counts_from_confmat(cls, values):
        if values["compute_class_counts_from_confmat"]:
            error_msg = (
                "If `compute_class_counts_from_confmat` requires "
                "`BucketHeadP65/confusion_matrix` to be one of "
                "the values in `metric_names`."
            )
            assert isinstance(values["metric_names"], list)
            assert "BucketHeadP65/confusion_matrix" in values["metric_names"], error_msg
        return values


class MCQAEvalComponent(BaseTargetReferenceEvalComponent):
    """Evaluation component focused on multiple-choice (/shortform) questions.

    # NOTE: preprocessing should happen outside this class
    """

    default_metric_names: List[str] = [
        "recall",
        "f1",
        "precision",
        "BucketHeadP65/confusion_matrix",
    ]

    @property
    def name(self):
        return "MCQAEvalComponent"

    def __init__(
        self,
        categories: List[str],
        response_field: str = "response",
        target_field: str = "final_decision",
        metric_names: List[str] = None,
        metric_kwargs: Dict[str, Dict[str, Any]] = None,
        average: Literal[
            "binary", "micro", "macro", "weighted", "samples"
        ] = "weighted",
        classes_to_exclude: Optional[list[str]] = None,
        compute_class_counts_from_confmat: bool = False,
    ) -> None:
        """Prepares evaluation metrics.

        Parameters
        ----------
        categories : List[str]
            Expected categories to check `predictions` and `references`
            against in `.run` method
        metric_names : List[str], optional
            What metrics to load from `evaluate`. If none is provided
            we default to ["recall", "f1", "precision"], by default None
        metric_kwargs: Dict[str, Dict[str, Any]], optional
            Option for providing kwargs to be passed on when computing metrics
        average : Literal["binary", "micro", "macro", "weighted", "samples"], optional
            Method for computing metrics when doing mutliclass classification. By
            default, "weighted", which calculates metrics for each label and takes
            average weighted by support.
            See "Inputs" of: https://huggingface.co/spaces/evaluate-metric/recall for
            full decsription of options.
        """
        self.categories = categories
        self.response_field = response_field
        self.target_field = target_field
        self.classes_to_exclude = classes_to_exclude

        self.average = average

        if not metric_names:
            metric_names = self.default_metric_names
            metric_kwargs = {
                "BucketHeadP65/confusion_matrix": {
                    "labels": list(range(len(categories)))
                }
            }
        self.metric_names = metric_names
        self.metric_kwargs = metric_kwargs or {}

        for metric_name in metric_names:
            if metric_name in ["recall", "f1", "precision"]:
                if metric_name not in self.metric_kwargs:
                    self.metric_kwargs[metric_name] = {"average": self.average}
                else:
                    self.metric_kwargs[metric_name].setdefault("average", self.average)
        self.compute_class_counts_from_confmat = compute_class_counts_from_confmat

    @classmethod
    def from_config(
        cls, config: Union[dict, MCQAEvalComponentConfig]
    ) -> "MCQAEvalComponent":
        if isinstance(config, dict):
            config = MCQAEvalComponentConfig.parse_obj(config)
        return cls(**config.dict())

    def _validate_list(self, lst: List[str]) -> bool:
        return all(item in self.categories for item in lst)

    def evaluate(
        self,
        dataset: Dataset,
    ) -> dict:
        predictions = dataset[self.response_field]
        targets = dataset[self.target_field]

        results, label_encoder = self.run(predictions=predictions, references=targets)

        # Computing stats
        if self.classes_to_exclude:
            extra_metrics, new_label_encoder = get_metrics_helper(
                confmat=np.array(results["confusion_matrix"]),
                label_encoder=deepcopy(label_encoder),
                to_exclude=self.classes_to_exclude,
            )
            extra_metrics = add_prefix_to_dict_keys(extra_metrics, "excl", "/")
            results.update(extra_metrics)
            results["excl/label_encoder"] = new_label_encoder

        # Computing stats
        if self.compute_class_counts_from_confmat:
            confmat = np.array(results["confusion_matrix"])
            stats_dict = compute_class_counts_from_confmat(confmat, label_encoder)
            stats_dict = add_prefix_to_dict_keys(stats_dict, "class_counts")
            results.update(stats_dict)

        return {"results": results, "label_encoder": label_encoder}

    def run(
        self,
        *,
        predictions: List[str],
        references: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        """Carries out evaluation on predictions against references (/targets).

        Parameters
        ----------
        predictions : List[str]
        references : List[str]

        Returns
        -------
        Tuple[Dict[str, float], Dict[str, int]]
            results dict and label encoding map used

        Raises
        ------
        ValueError
            If any of the predictions not found in expected categories.
            These categories are either the ones provided at `__init__`
            or ones extracted from provided `references`.
        ValueError
            If `categories` are provided at `__init__` then we check
            the provided `references` against them and raise ValueError
            if we find unseen entries.
        """
        if not self._validate_list(predictions):
            raise ValueError(
                "One or more entries from `predictions` "
                f"not found in {self.categories=}."
            )

        if not self._validate_list(references):
            raise ValueError(
                "One or more entries from `references` "
                f"not found in {self.categories=}."
            )

        label_encoder = LabelEncoder().fit(self.categories)
        encoded_references = label_encoder.transform(references).tolist()
        encoded_predictions = label_encoder.transform(predictions).tolist()

        results = {}
        for metric_name in self.metric_names:
            kwargs_dict = self.metric_kwargs.get(metric_name, {})

            result = run_metric(
                metric_name=metric_name,
                predictions=encoded_predictions,
                targets=encoded_references,
                kwargs_dict=kwargs_dict,
            )
            results.update(result)

        label_encoder_map = dict(
            zip(self.categories, label_encoder.transform(self.categories))
        )
        return results, label_encoder_map


class MCQAStratifiedEvalComponentConfig(MCQAEvalComponentConfig):
    stratify_field: str


class MCQAStratifiedEvalComponent(MCQAEvalComponent):
    """Subclass of MCQAEvalComponent which enables stratified evaluation.

    NOTE: preprocessing should happen outside this class
    """

    def __init__(self, stratify_field: str, *args, **kwargs):
        """Constructor method.

        Parameters
        ----------
        stratify_field : str
            Field name over which the stratification is applied.

        *args
            Positional arguments to the constructor of the superclass MCQAEvalComponent.

        **kwargs
            Keyword arguments to the constructor of the superclass MCQAEvalComponent.
        """
        self.stratify_field = stratify_field
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return "MCQAStratifiedEvalComponent"

    @classmethod
    def from_config(
        cls, config: Union[dict, MCQAStratifiedEvalComponentConfig]
    ) -> "MCQAStratifiedEvalComponent":
        if isinstance(config, dict):
            config = MCQAStratifiedEvalComponentConfig.parse_obj(config)
        return cls(**config.dict())

    def evaluate(
        self,
        dataset: Dataset,
    ) -> Dict[str, Tuple[Dict[str, float], Dict[str, int]]]:
        """Run stratified evaluation given a dataset.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Dict[str, Tuple[Dict[str, float], Dict[str, int]]]
            Results dict and label encoding map used per stratified group.
            e.g. {
                "easy": ({"recall": 1.0, "precision": 1.0}, {"yes": 1, "no": 0}),
                "medium": ({"recall": 0.5, "precision": 0.25}, {"yes": 1, "no": 0}),
                "hard": ({"recall": 0.0, "precision": 0.0}, {"yes": 1, "no": 0}),
            }

        Raises
        ------
        ValueError
            If the provided strafitication only has one or less groups.
        """
        stratify_groups = set(dataset[self.stratify_field])

        if len(stratify_groups) <= 1:
            raise ValueError(
                "More than one groups required to stratify, but only found"
                f" {stratify_groups} - increase the number of groups or use"
                "MCQAEvalComponent.evaluate() to run non-stratified evaluation."
            )

        all_results: Dict[str, Tuple[Dict[str, float], Dict[str, int]]] = {}

        for group in stratify_groups:
            subset = dataset.filter(lambda row: row[self.stratify_field] == group)
            results = self.run(
                predictions=subset[self.response_field],
                references=subset[self.target_field],
            )
            all_results[group] = results

        return all_results
