from typing import Protocol, runtime_checkable

from datasets import Dataset

from rambla.utils.task import BaseComponent


@runtime_checkable
class TextToTextSimilarityComponent(Protocol):
    # NOTE Could reconsider protocol and remove
    """A generic type that generates responses.

    This could either be an LLM itself,
    or a more complex object that encapsulates an LLM.

    Responses indicate the degree of similarity between
    the two pieces of provided text. Responses can be
    binary, discrete or continuous.
    """

    def run(self, dataset: Dataset) -> Dataset:
        ...


class BaseTextToTextSimilarityComponent(BaseComponent):
    pass
