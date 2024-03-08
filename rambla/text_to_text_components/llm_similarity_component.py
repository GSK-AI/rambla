from __future__ import annotations

from typing import Optional, Union

from datasets import Dataset
from pydantic import BaseModel, Extra

from rambla.models import LLMConfig, build_llm
from rambla.models.base_model import BaseLLM
from rambla.prompts.formatters import ColumnPromptFormatter, ColumnPromptFormatterConfig
from rambla.response_generation.response import (
    ResponseComponent,
    ResponseComponentConfig,
)
from rambla.text_to_text_components.base import BaseTextToTextSimilarityComponent


class LLMTextToTextSimilarityConfig(BaseModel):
    llm_config: LLMConfig
    prompt_formatter_config: ColumnPromptFormatterConfig
    response_component_config: ResponseComponentConfig
    response_field_name: str

    class Config:  # noqa: D106
        extra = Extra.forbid


class LLMTextToTextSimilarity(BaseTextToTextSimilarityComponent):
    """LLM Similarity task

    Example uses:
    1. Use an LLM to generate responses on the
    semantic similarity for examples in a text to text dataset

    Steps:
    1. Accepts text to text dataset
    2. Create prompts for dataset (under `prompt`)
    3. Generate responses from `llm` (under `response`)
    """

    def __init__(
        self,
        llm: BaseLLM,
        prompt_formatter: ColumnPromptFormatter,
        response_component: ResponseComponent,
        response_field_name: str = "similarity_response",
    ):
        self.llm = llm
        self.response_component = response_component
        self.prompt_formatter = prompt_formatter
        self.response_field_name = response_field_name

    @classmethod
    def from_config(
        cls, config: Union[dict, LLMTextToTextSimilarityConfig]
    ) -> LLMTextToTextSimilarity:
        if not isinstance(config, LLMTextToTextSimilarityConfig):
            config = LLMTextToTextSimilarityConfig.parse_obj(config)

        llm = build_llm(config.llm_config.dict())

        prompt_formatter = ColumnPromptFormatter.from_config(
            config.prompt_formatter_config
        )

        response_component = ResponseComponent.from_config(
            config.response_component_config
        )

        return cls(
            llm=llm,
            prompt_formatter=prompt_formatter,
            response_component=response_component,
            response_field_name=config.response_field_name,
        )

    def run(
        self, dataset: Dataset, prompt_field_name: Optional[str] = "prompt"
    ) -> Dataset:
        """Executes the steps defined by the task in sequence.

        Parameters
        ----------
        dataset : Dataset
            The text to text dataset to be used
            (must contain two columns of text and
            ideally a label column defining their
            semantic similarity)

        Returns
        -------
        Dataset
        """
        # Form prompt dataset
        prompt_dataset = self.prompt_formatter.format(
            dataset, prompt_field_name=prompt_field_name
        )

        # Generate responses
        response_dataset = self.response_component.batch_generate(
            model=self.llm,
            prompt_dataset=prompt_dataset,
            prompt_field=prompt_field_name,
            response_field=self.response_field_name,
        )

        return response_dataset
