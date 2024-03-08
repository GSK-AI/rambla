from typing import Optional, Union

from datasets import Dataset
from pydantic import Extra, root_validator

from rambla.datasets.io import GenericDatasetConfig, prepare_dataset
from rambla.evaluation import EvalComponentConfig, build_eval_component
from rambla.evaluation.base import BaseEvalComponent
from rambla.preprocessing.formatting import ScalingFormatter, ScalingFormatterConfig
from rambla.response_formatting.formatting import (
    MappingResponseFormatter,
    MappingResponseFormatterConfig,
)
from rambla.tasks.base import RunTaskReturnType
from rambla.text_to_text_components.base import TextToTextSimilarityComponent
from rambla.text_to_text_tasks.base import BaseTextToTextTask, BaseTextToTextTaskConfig
from rambla.utils.config import (
    validate_category_equality,
    validate_field_attribute_equality,
)
from rambla.utils.misc import prepare_dicts_for_logging


class TextToTextSimilarityEvaluationConfig(BaseTextToTextTaskConfig):
    dataset_config: GenericDatasetConfig
    preprocessor_config: Optional[ScalingFormatterConfig]
    response_formatter_config: Optional[MappingResponseFormatterConfig]
    evaluator_config: EvalComponentConfig

    class Config:  # noqa: D106
        extra = Extra.forbid

    @root_validator()
    @classmethod
    def validate_response_field(cls, values):
        if not values["response_formatter_config"]:
            return values

        response_formatter_config = values["response_formatter_config"]
        evaluator_config = values["evaluator_config"]

        validate_field_attribute_equality(
            response_formatter_config,
            evaluator_config.params,
            "response_field_name",
            "response_field",
        )
        return values

    @root_validator()
    @classmethod
    def validate_categories(cls, values):
        if not values["response_formatter_config"]:
            return values

        response_formatter_config = values["response_formatter_config"]
        evaluator_config = values["evaluator_config"]
        if evaluator_config.name not in ["shortform"]:
            return values

        # NOTE: Comparing out going categories
        # NOTE: This approach assumes the response formatter
        # negation map provided is exhaustive
        response_formatter_categories = list(
            response_formatter_config.renaming_map.values()
        )
        evaluator_categories = evaluator_config.params["categories"]

        validate_category_equality(response_formatter_categories, evaluator_categories)
        return values


class TextToTextSimilarityEvaluation(BaseTextToTextTask):
    """Text to Text Similarity Evaluation Module

    Example uses:
    1. Evaluate the perfomance of a model in identifying
    semantic similarity with text to text datasets

    Steps:
    1. Obtain dataset with responses from llm from the text_to_text_component
    2. Post-process responses
    3. Run evaluation
    """

    def __init__(
        self,
        dataset: Dataset,
        evaluator: BaseEvalComponent,
        response_formatter: Optional[MappingResponseFormatter] = None,
    ):
        self.dataset = dataset
        self.response_formatter = response_formatter
        self.evaluator = evaluator

    @classmethod
    def from_config(
        cls, config: Union[dict, TextToTextSimilarityEvaluationConfig]
    ) -> "TextToTextSimilarityEvaluation":
        if not isinstance(config, TextToTextSimilarityEvaluationConfig):
            config = TextToTextSimilarityEvaluationConfig.parse_obj(config)

        dataset = prepare_dataset(config.dataset_config.dict())

        preprocessor = None
        if config.preprocessor_config:
            preprocessor = ScalingFormatter.from_config(config.preprocessor_config)
            dataset = preprocessor.format(dataset)

        response_formatter = None
        if config.response_formatter_config:
            response_formatter = MappingResponseFormatter.from_config(
                config.response_formatter_config
            )

        evaluator = build_eval_component(config.evaluator_config.dict())

        return cls(
            dataset=dataset,
            response_formatter=response_formatter,
            evaluator=evaluator,
        )

    def run_task(
        self, text_to_text_component: TextToTextSimilarityComponent
    ) -> RunTaskReturnType:
        """Executes the steps defined by the task in sequence.

        Parameters
        ----------
        text_to_text_component: BaseComponent
        The component used to generate responses from text to text pairs


        Returns
        -------
        RunTaskReturnType
        """
        dataset_size = len(self.dataset)

        # Generate responses from text to text pairs
        response_dataset = text_to_text_component.run(self.dataset)

        # Post-process responses
        if self.response_formatter:
            response_dataset = self.response_formatter.format(response_dataset)

        # Evaluate
        eval_results = self.evaluator.evaluate(response_dataset)

        # Preparing for logging
        to_log_as_metrics, to_log_as_dicts = prepare_dicts_for_logging(
            eval_results=eval_results,
        )
        to_log_as_dicts["dataset_stats"] = {"dataset_size": dataset_size}

        return RunTaskReturnType(
            metrics=to_log_as_metrics,
            artifacts=eval_results,
            datasets={"final_dataset": response_dataset},
            other=None,
            artifact_storing_format="json",
            plots=None,
            dictionaries=to_log_as_dicts,
        )
