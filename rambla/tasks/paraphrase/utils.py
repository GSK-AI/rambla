from __future__ import annotations

from pathlib import Path
from typing import Union

from datasets import Dataset
from pydantic import BaseModel, Extra

from rambla.models import LLMConfig, build_llm
from rambla.models.base_model import BaseLLM
from rambla.prompts.formatters import ColumnPromptFormatter, ColumnPromptFormatterConfig
from rambla.response_generation.response import (
    ResponseComponent,
    ResponseComponentConfig,
)
from rambla.utils.task import BaseComponent

# flake8: noqa: E501


class RephrasingModuleConfig(BaseModel):
    llm_config: LLMConfig
    prompt_formatter_config: ColumnPromptFormatterConfig
    response_component_config: ResponseComponentConfig
    field_rephrased: str

    class Config:  # noqa: D106
        extra = Extra.forbid


class RephrasingModule(BaseComponent):
    """Class to be used for rephrasing a field in a dataset.

    NOTE: can be used for paraphrasing the question field in a dataset
    such that the meaning remains the same.

    NOTE: can be used for negating the meaning of a context
    """

    def __init__(
        self,
        llm: BaseLLM,
        prompt_formatter: ColumnPromptFormatter,
        response_component: ResponseComponent,
        field_rephrased: str,
    ):
        self.llm = llm
        self.prompt_formatter = prompt_formatter
        self.field_rephrased = field_rephrased
        self.response_component = response_component

    @classmethod
    def from_config(
        cls, config: Union[dict, RephrasingModuleConfig]
    ) -> RephrasingModule:
        if not isinstance(config, RephrasingModuleConfig):
            config = RephrasingModuleConfig.parse_obj(config)

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
            field_rephrased=config.field_rephrased,
        )

    def run(self, dataset: Dataset, verbose: bool = True) -> Dataset:
        """Runs all the steps necessary in sequence.

        Parameters
        ----------
        dataset : Dataset
            Huggingface dataset whose column will be rephrased.

        Returns
        -------
        Dataset
            Huggingface dataset with column rephrased.

            NOTE: the resulting dataset will have the rephrased column
            under `self.field_rephrased` and the old column
            will be moved to `f"original_{self.field_rephrased}"`.
        """
        # Form prompt dataset
        prompt_dataset = self.prompt_formatter.format(dataset, "rephrasing_prompt")

        # Generate responses
        response_dataset = self.response_component.batch_generate(
            model=self.llm,
            prompt_dataset=prompt_dataset,
            prompt_field="rephrasing_prompt",
            verbose=verbose,
        )

        response_dataset = response_dataset.rename_column(
            self.field_rephrased,
            f"original_{self.field_rephrased}",
        )
        response_dataset = response_dataset.rename_column(
            "response", self.field_rephrased
        )

        return response_dataset
