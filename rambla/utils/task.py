import abc

from datasets import Dataset


class BaseComponent(abc.ABC):
    @abc.abstractmethod
    def run(self, dataset: Dataset) -> Dataset:
        ...
