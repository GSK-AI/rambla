import abc

from datasets import Dataset


def process_string(text: str) -> str:
    """Removes spaces, fullstops and makes the text lowercase."""
    return text.replace(".", "").strip().lower()


STRING_FORMATTER_MAP = {
    "basic": process_string,
}


class BaseResponseFormatter(abc.ABC):
    @abc.abstractmethod
    def format(self, dataset: Dataset) -> Dataset:
        """Formats a column of responses in a HF dataset."""
        ...
