from unittest import mock

import pytest
from transformers import PreTrainedTokenizerFast

from rambla.models.utils import MessageFormatter


class TestMessageFormatter:
    @pytest.fixture()
    def mock_tokenizer(self):
        return mock.create_autospec(spec=PreTrainedTokenizerFast, instance=True)

    def test_no_argument_provided(self):
        with pytest.raises(ValueError) as exc_info:
            MessageFormatter()

        assert "At least one of" in str(exc_info.value)

    def test_template_with_no_placeholder(self):
        template = "no placeholder provided"
        with pytest.raises(ValueError) as exc_info:
            MessageFormatter(template=template)

        assert "Template needs to have one placeholder" in str(exc_info.value)

    def test_template_with_wrong_placeholder(self):
        template = "wrong {placeholder} provided"
        with pytest.raises(ValueError) as exc_info:
            MessageFormatter(template=template)

        assert "Template needs to have one placeholder" in str(exc_info.value)

    def test_template(self):
        template = "correct placeholder {message} provided"
        message_formatter = MessageFormatter(template=template)

        # run
        prompt = "this is a dummy input"
        expected_output = template.format(message=prompt)
        output = message_formatter.format(prompt)

        # asserts
        assert output == expected_output

    def test_mistral_tokenizer(self, mock_tokenizer):
        mock_tokenizer.name_or_path = "mistral_instruct"
        mock_tokenizer.apply_chat_template.return_value = "mock_return_value"

        message_formatter = MessageFormatter(tokenizer=mock_tokenizer)

        # run
        prompt = "this is a dummy input"
        output = message_formatter.format(prompt)

        #
        assert output == "mock_return_value"
        expected_call_kwargs = {
            "conversation": [{"role": "user", "content": prompt}],
            "tokenize": False,
        }

        mock_tokenizer.apply_chat_template.assert_called_once_with(
            **expected_call_kwargs
        )

    def test_template_and_tokenizer_provided(self, mock_tokenizer):
        mock_tokenizer.name_or_path = "mistral_instruct"
        template = "correct placeholder {message} provided"

        message_formatter = MessageFormatter(
            template=template,
            tokenizer=mock_tokenizer,
        )

        # run
        prompt = "this is a dummy input"
        expected_output = template.format(message=prompt)
        output = message_formatter.format(prompt)

        # asserts
        assert output == expected_output
        mock_tokenizer.apply_chat_template.assert_not_called()
