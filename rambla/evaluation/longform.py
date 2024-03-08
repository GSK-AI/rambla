from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
from datasets import Dataset
from pydantic import BaseModel, validator

from rambla.evaluation.base import BaseEvalComponent, BaseTargetReferenceEvalComponent
from rambla.utils.text_processing import ALLOWED_ENCODING_NAMES, token_counter


class LongformQAEvalComponent(BaseTargetReferenceEvalComponent):
    """Eval class for long form question-answer evaluation.

    # NOTE: preprocessing should happen outside this class
    """

    default_metric_names: List[str] = ["rouge", "bleu"]

    def __init__(
        self,
        metric_names: List[str] = None,
        metric_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Prepares evaluation metrics.

        Parameters
        ----------
        metric_names : List[str], optional
            What metrics to load from `evaluate`. If none is provided
            we default to ["recall", "f1", "precision"], by default None
        metric_kwargs: Dict[str, Dict[str, Any]], optional
            Option for providing kwargs to be passed on when computing metrics
        """
        self.metric_kwargs = metric_kwargs
        if not self.metric_kwargs:
            self.metric_kwargs = {}

        if not metric_names:
            metric_names = self.default_metric_names

        self.metrics = {
            metric_name: evaluate.load(metric_name) for metric_name in metric_names
        }

    def evaluate(self):
        raise NotImplementedError

    def run(self, *, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Computes metrics.

        Parameters
        ----------
        predictions : List[str]
        references : List[str]

        Returns
        -------
        Dict[str, float]
            results
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Found predictions with length: {len(predictions)} "
                f"and references with length: {len(references)}"
            )

        results = {}
        for metric_name, metric in self.metrics.items():
            kwargs = self.metric_kwargs.get(metric_name, {})

            results.update(
                metric.compute(predictions=predictions, references=references, **kwargs)
            )
        return results


class ResponseQualityEvalConfig(BaseModel):
    # NOTE: add option for functions to apply
    field_names: List[str]
    # What encoding is used by model.
    encoding: str

    @validator("field_names", pre=True)
    @classmethod
    def validate_field_names(cls, field_names):
        if isinstance(field_names, str):
            return [field_names]
        return field_names

    @validator("encoding")
    @classmethod
    def validate_encoding(cls, v):
        assert v in ALLOWED_ENCODING_NAMES
        return v


class ResponseQualityEvalComponent(BaseEvalComponent):
    """Evaluates the quality of the LLM response by measuring its length."""

    def __init__(self, field_names: List[str], encoding: str):
        """_summary_

        Parameters
        ----------
        field_names : List[str]
            Which field to look at.
        encoding : str
            What encoding was used. Can be either a model's name or an encoding's name.
            For example, "text-davinci-003" or "p50k_base".
        """
        if isinstance(field_names, str):
            field_names = [field_names]
        self.field_names = field_names
        self.encoding = encoding

    @classmethod
    def from_config(
        cls, config: Union[dict, ResponseQualityEvalConfig]
    ) -> "ResponseQualityEvalComponent":
        if isinstance(config, dict):
            config = ResponseQualityEvalConfig.parse_obj(config)
        return cls(**config.dict())

    def _compute_lengths(self, column: List[str]) -> List[int]:
        func = lambda x: token_counter(x, self.encoding)  # noqa: E731
        return list(map(func, column))

    def evaluate(self, dataset: Dataset) -> Dict[str, Dict[str, Any]]:
        out = {}
        for field_name in self.field_names:
            lengths = self._compute_lengths(dataset[field_name])
            out[field_name] = {
                "mean": np.mean(lengths),
                "median": np.median(lengths),
                "std": np.std(lengths),
            }
        return out
