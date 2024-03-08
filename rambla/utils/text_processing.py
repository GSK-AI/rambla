import re
from functools import lru_cache
from typing import List, Union

import tiktoken
from datasets import Dataset
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel
from tiktoken.model import MODEL_TO_ENCODING

from rambla.utils.task import BaseComponent

models = list(MODEL_TO_ENCODING.keys())
encodings = list(MODEL_TO_ENCODING.values())
ALLOWED_ENCODING_NAMES = tuple(models + encodings)


@lru_cache(1)
def get_tokenizer(encoding_type: ALLOWED_ENCODING_NAMES) -> tiktoken.Encoding:
    """Loads tiktoken tokenizer from either encoding name or encoding model."""
    if "k_base" in encoding_type:
        return tiktoken.get_encoding(encoding_type)
    else:
        return tiktoken.encoding_for_model(encoding_type)


def token_counter(text: str, encoding: ALLOWED_ENCODING_NAMES) -> int:
    """Number of tokens in text."""
    tokenizer = get_tokenizer(encoding)
    return len(tokenizer.encode(text))


class BaseTrimmer(BaseComponent):
    pass


def trim_text_by_number_of_sentences(text: str, n_sentences: int) -> str:
    """Trims a piece of text based on number of sentences."""
    sentences = sent_tokenize(text)
    if n_sentences != -1:
        sentences = sentences[:n_sentences]
    return " ".join(sentences)


class SentenceTrimmerConfig(BaseModel):
    field_name: str
    n_sentences: int


class SentenceTrimmer(BaseTrimmer):
    def __init__(self, field_name: str, n_sentences: int):
        self.field_name = field_name
        self.n_sentences = n_sentences

    @classmethod
    def from_config(
        cls, config: Union[dict, SentenceTrimmerConfig]
    ) -> "SentenceTrimmer":
        if isinstance(config, dict):
            config = SentenceTrimmerConfig.parse_obj(config)
        return cls(field_name=config.field_name, n_sentences=config.n_sentences)

    def run(self, dataset: Dataset) -> Dataset:
        new_field_name = "__temporary_field_name__"

        def trimming_helper(entry: dict) -> dict:
            entry[new_field_name] = trim_text_by_number_of_sentences(
                entry[self.field_name], self.n_sentences
            )
            return entry

        dataset = dataset.map(trimming_helper)

        dataset = dataset.rename_column(self.field_name, f"untrimmed_{self.field_name}")
        dataset = dataset.rename_column(new_field_name, self.field_name)

        return dataset


def extract_first_response_instance(
    response_str: str,
    allowed_categories: List[str],
) -> Union[str, None]:
    """Extracts the first instance of an allowed response category from an LLM response

    Parameters
    ----------
    response_str : str
        LLM response string
    allowed_categories : List[str]
        List of acceptable categories

    Returns
    -------
    Union[str, None]
        First instance of `allowed_categories` if found, else None.
    """
    regexp = rf"\b({'|'.join(allowed_categories)})\b"
    search_result = re.search(regexp, response_str)

    if search_result:
        return response_str[search_result.start() : search_result.end()]
    else:
        return None
