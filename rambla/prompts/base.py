import abc

from datasets import Dataset


class BasePromptFormatter(abc.ABC):
    """Base class for all prompt formatters

    Instances of this class format a dataset into a dataset of prompts
    to pass to an LLM.
    """

    @abc.abstractmethod
    def format(self, dataset: Dataset, prompt_field_name: str = "prompt") -> Dataset:
        ...
