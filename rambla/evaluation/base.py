import abc
from typing import Any, List

from datasets import Dataset


class BaseEvalComponent(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, dataset: Dataset) -> Any:
        """Operates on a HF dataset to calculate metrics."""
        ...


class BaseTargetReferenceEvalComponent(BaseEvalComponent):
    @abc.abstractmethod
    def run(self, *, predictions: List[str], references: List[str]):
        """Operates on a list of predictions and a list of targets to calculate metrics."""  # noqa: E501
        ...
