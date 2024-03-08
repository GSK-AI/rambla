import abc

from datasets import Dataset


class BasePreprocessor(abc.ABC):
    @abc.abstractmethod
    def format(self, dataset: Dataset) -> Dataset:
        """Processes and formats a column of a HF dataset."""
        ...
