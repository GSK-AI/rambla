from typing import Union

from datasets import Dataset
from pydantic import BaseModel, validator

from rambla.models.huggingface import NLIModel, NLIModelConfig
from rambla.prompts.formatters import ColumnPromptFormatter, ColumnPromptFormatterConfig
from rambla.response_generation.response import (
    ResponseComponent,
    ResponseComponentConfig,
)
from rambla.text_to_text_components.base import BaseTextToTextSimilarityComponent
from rambla.text_to_text_components.nli_strategies import (
    BaseNLIOutputCombinationStrategy,
    CombinationStrategyConfig,
    build_combination_strategy,
)
from rambla.utils.misc import dict_softmax


class NLIUnidirectionalConfig(BaseModel):
    model_config: NLIModelConfig
    prompt_formatter_config: ColumnPromptFormatterConfig
    response_component_config: ResponseComponentConfig
    response_column_name: str = "response"
    # return_key determines which label (entailment, neutral
    # or contradiction) predictiona are calculate for
    return_key: str = "entailment"

    @validator("model_config")
    @classmethod
    def validate_model_return_mode(cls, model_config):
        allowed_list = ["dict", "label"]
        assert (
            model_config.params.return_mode in allowed_list
        ), f"""
        {model_config.params.return_mode=} not supported
        with NLIUnidirectional try one of {allowed_list}."""
        return model_config


class NLIUnidirectional(BaseTextToTextSimilarityComponent):
    """Component to run huggingface NLI models in a unidirectional way.

    Steps:
    1. Form prompt dataset
    2. Obtain dataset with result from NLI
    3. Determine the overall result
    """

    def __init__(
        self,
        model: NLIModel,
        prompt_formatter: ColumnPromptFormatter,
        response_component: ResponseComponent,
        response_column_name: str = "response",
        model_return_mode: dict = "dict",
        return_key: str = "entailment",
    ):
        self.model = model
        self.prompt_formatter = prompt_formatter
        self.response_component = response_component
        self.response_column_name = response_column_name
        self.model_return_mode = model_return_mode
        self.return_key = return_key

    @classmethod
    def from_config(
        cls, config: Union[dict, NLIUnidirectionalConfig]
    ) -> "NLIUnidirectional":
        if isinstance(config, dict):
            config = NLIUnidirectionalConfig.parse_obj(config)

        prompt_formatter = ColumnPromptFormatter.from_config(
            config.prompt_formatter_config
        )

        response_component = ResponseComponent.from_config(
            config.response_component_config
        )

        model = NLIModel.from_config(config.model_config)

        return cls(
            model=model,
            prompt_formatter=prompt_formatter,
            response_component=response_component,
            response_column_name=config.response_column_name,
            model_return_mode=config.model_config.params.return_mode,
            return_key=config.return_key,
        )

    def run(self, dataset: Dataset) -> Dataset:
        """
        Runs an NLI model unidirectionally.

        Parameters
        ----------
        dataset : Dataset
            The text to text dataset to be used.
            Must contain the input sequences separated by separator string.

        Returns
        -------
        Dataset
        """
        # Form prompt dataset
        prompt_dataset = self.prompt_formatter.format(dataset)

        # Get results
        response_dataset = self.response_component.batch_generate(
            model=self.model,
            prompt_dataset=prompt_dataset,
            response_field=self.response_column_name,
        )

        # Determine final predictions
        if self.model_return_mode == "dict":
            # NOTE This applies softmax in the future we may
            # want to add more functionality for continuous evaluation
            def new_column(
                entry: dict,
            ) -> dict:
                entry[self.response_column_name] = dict_softmax(
                    entry[self.response_column_name]
                )[self.return_key]
                return entry

            response_dataset = response_dataset.map(new_column)

        return response_dataset


class NLIBidirectionalConfig(BaseModel):
    model_config: NLIModelConfig
    prompt_formatter_config: ColumnPromptFormatterConfig
    flipped_prompt_formatter_config: ColumnPromptFormatterConfig
    response_component_config: ResponseComponentConfig
    # Combination_strategy determines how the final predictions are calculated
    combination_strategy_config: CombinationStrategyConfig
    response_column_name: str = "response"

    @validator("model_config")
    @classmethod
    def validate_model_return_mode(cls, model_config):
        allowed_list = ["dict"]
        assert (
            model_config.params.return_mode in allowed_list
        ), f"""
        {model_config.params.return_mode=} not supported
        with NLIBidirectional try one of {allowed_list}."""
        return model_config


class NLIBidirectional(BaseTextToTextSimilarityComponent):
    """Component to run huggingface NLI models in a bidirectional way.

    Steps:
    1. Obtain dataset with result from NLI
    2. Flip the two sequences of text
    3. Obtain dataset with result from NLI on flipped sequences
    4. Determine the overall bidirectional result
    """

    def __init__(
        self,
        model: NLIModel,
        prompt_formatter: ColumnPromptFormatter,
        flipped_prompt_formatter: ColumnPromptFormatter,
        response_component: ResponseComponent,
        combination_strategy: BaseNLIOutputCombinationStrategy,
        response_column_name: str = "response",
    ):
        self.model = model
        self.prompt_formatter = prompt_formatter
        self.flipped_prompt_formatter = flipped_prompt_formatter
        self.response_component = response_component
        self.combination_strategy = combination_strategy
        self.response_column_name = response_column_name

    @classmethod
    def from_config(
        cls, config: Union[dict, NLIBidirectionalConfig]
    ) -> "NLIBidirectional":
        if isinstance(config, dict):
            config = NLIBidirectionalConfig.parse_obj(config)

        prompt_formatter = ColumnPromptFormatter.from_config(
            config.prompt_formatter_config
        )

        flipped_prompt_formatter = ColumnPromptFormatter.from_config(
            config.flipped_prompt_formatter_config
        )

        response_component = ResponseComponent.from_config(
            config.response_component_config
        )

        combination_strategy = build_combination_strategy(
            config.combination_strategy_config
        )

        model = NLIModel.from_config(config.model_config)

        return cls(
            model=model,
            prompt_formatter=prompt_formatter,
            flipped_prompt_formatter=flipped_prompt_formatter,
            response_component=response_component,
            combination_strategy=combination_strategy,
            response_column_name=config.response_column_name,
        )

    def run(self, dataset: Dataset) -> Dataset:
        """
        Runs an NLI model twice bidirectionally.

        Parameters
        ----------
        dataset : Dataset
            The text to text dataset to be used.
            Must contain the input sequences separated by separator string.

        Returns
        -------
        Dataset
        """
        # Form prompt dataset
        prompt_dataset = self.prompt_formatter.format(dataset)

        # Get results in first direction
        response_dataset = self.response_component.batch_generate(
            model=self.model,
            prompt_dataset=prompt_dataset,
            response_field=f"forward_{self.response_column_name}",
        )

        # Flip text
        prompt_dataset = self.flipped_prompt_formatter.format(
            response_dataset, prompt_field_name="flipped_prompt"
        )

        # Get results in second direction
        response_dataset = self.response_component.batch_generate(
            model=self.model,
            prompt_dataset=prompt_dataset,
            prompt_field="flipped_prompt",
            response_field=f"backward_{self.response_column_name}",
        )

        # Determine final predictions
        def new_column(entry: dict) -> dict:
            entry[self.response_column_name] = self.combination_strategy.run(
                entry[f"forward_{self.response_column_name}"],
                entry[f"backward_{self.response_column_name}"],
            )
            return entry

        response_dataset = response_dataset.map(new_column)

        return response_dataset
