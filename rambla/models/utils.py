import abc
import functools
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from rambla.prompts.utils import find_field_placeholders

# flake8: noqa: E501


@functools.lru_cache(1)
def load_pretrained_tokenizer(tokenizer_path: str):
    """Loads HF tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        device_map="auto",
        local_files_only=True,
        pad_token="<|PAD|>",
        return_special_tokens_mask=True,
        cache_dir=".",
    )
    return tokenizer


def count_tokens(text: str, tokenizer_path: str) -> int:
    """Counds tokens for a given text based on a pretrained tokenizer."""
    tokenizer = load_pretrained_tokenizer(tokenizer_path)
    return len(tokenizer.tokenize(text))


def tokenizer_based_message_formatter(
    tokenizer: PreTrainedTokenizerFast, message: str
) -> str:
    """Message formatter for the mistral/llama model"""
    message = message.strip()
    return tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": message}],
        tokenize=False,
    )  # type: ignore


class MessageFormatter:
    """Formats the prompt message to suit what the model expects.

    If a template is provided (an unformatted f-string with `message` as a placeholder)
    then it will be used.
    Example templates:
        - `<s>[INST] {message} [/INST]"`
        - "<|prompter|>{message}</s><|assistant|>"

    If a template is not provided then a tokenizer needs to provided which will be used
    through `tokenizer_based_message_formatter`.
    """

    def __init__(
        self,
        *,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        template: Optional[str] = None,
    ):
        self.template = template
        self.tokenizer = tokenizer

        if self.template:
            self._validate_template(self.template)
        else:
            if self.tokenizer is None:
                raise ValueError(
                    "At least one of tokenizer or template needs to be provided."
                )

    def _validate_template(self, template: str):
        placeholders = find_field_placeholders(template)
        if not set(placeholders) == set(["message"]):
            raise ValueError("Template needs to have one placeholder: `message`.")

    def _format_with_tokenizer(self, message: str) -> str:
        if self.tokenizer is None:
            raise ValueError("No tokenizer provided.")
        return tokenizer_based_message_formatter(self.tokenizer, message)

    def _format_with_template(self, message: str) -> str:
        if self.template is None:
            raise ValueError("No template provided.")
        return self.template.format(message=message)

    def format(self, message: str) -> str:
        if self.template:
            return self._format_with_template(message)
        else:
            return self._format_with_tokenizer(message)
