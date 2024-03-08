from __future__ import annotations

from typing import Optional, Union

from datasets import Dataset
from pydantic import BaseModel, Extra, root_validator

from rambla.datasets.io import GenericDatasetConfig, MCQADatasetConfig, prepare_dataset
from rambla.evaluation import EvalComponentConfig, build_eval_component
from rambla.evaluation.base import BaseEvalComponent
from rambla.evaluation.longform import (
    ResponseQualityEvalComponent,
    ResponseQualityEvalConfig,
)
from rambla.evaluation.shortform import MCQAEvalComponent, MCQAEvalComponentConfig
from rambla.models import LLMConfig, build_llm
from rambla.models.base_model import BaseLLM
from rambla.prompts.base import BasePromptFormatter
from rambla.prompts.formatters import ColumnPromptFormatter, ColumnPromptFormatterConfig
from rambla.response_formatting.formatting import (
    MappingResponseFormatter,
    MappingResponseFormatterConfig,
    MCQAResponseFormatter,
    MCQAResponseFormatterConfig,
)
from rambla.response_generation.response import (
    ResponseComponent,
    ResponseComponentConfig,
)
from rambla.tasks.base import BaseTask, LLMGenerator, RunTaskReturnType
from rambla.text_to_text_components import (
    ParentTextToTextModuleConfig,
    build_text_to_text_module,
)
from rambla.text_to_text_components.base import TextToTextSimilarityComponent
from rambla.utils import config as config_utils
from rambla.utils.config import (
    validate_category_equality,
    validate_field_attribute_equality,
)
from rambla.utils.dataset import DatasetFilterer, DatasetFiltererConfig, slice_dataset
from rambla.utils.misc import initialize_logger, prepare_dicts_for_logging

logger = initialize_logger(__name__)


class MCQALongFormTaskConfig(BaseModel):
    dataset_config: MCQADatasetConfig
    longform_prompt_formatter_config: ColumnPromptFormatterConfig
    scoring_model_config: LLMConfig
    question_response_formatter_config: ColumnPromptFormatterConfig
    response_formatter_config: MCQAResponseFormatterConfig
    evaluator_config: MCQAEvalComponentConfig
    question_field: str
    target_field: str

    response_component_config: ResponseComponentConfig
    # TODO: Validation of subsample size based on size of `dataset_config`
    subsample_size: Optional[int]

    class Config:  # noqa: D106
        extra = Extra.forbid

    @root_validator()
    @classmethod
    def validate_target_field(cls, values):
        config_utils.validate_field_attribute_equality(
            values["dataset_config"], values["evaluator_config"], "target_field"
        )
        return values

    @root_validator()
    @classmethod
    def validate_response_field(cls, values):
        response_formatter_config = values["response_formatter_config"]
        evaluator_config = values["evaluator_config"]
        config_utils.validate_field_attribute_equality(
            response_formatter_config,
            evaluator_config,
            "response_field_name",
            "response_field",
        )
        return values

    @root_validator()
    @classmethod
    def validate_index_field(cls, values):
        config_utils.validate_field_attribute_equality(
            values["longform_prompt_formatter_config"],
            values["question_response_formatter_config"],
            "index_field",
        )

        config_utils.validate_field_attribute_equality(
            values["longform_prompt_formatter_config"],
            values["dataset_config"],
            "index_field",
        )
        return values

    @root_validator()
    @classmethod
    def validate_categories(cls, values):
        # categories
        response_formatter_categories = values["response_formatter_config"].categories
        evaluator_categories = values["evaluator_config"].categories

        config_utils.validate_category_equality(
            (
                response_formatter_categories
                + [values["response_formatter_config"].null_category]
            ),
            evaluator_categories,
        )
        return values


class MCQALongFormTask(BaseTask):
    def __init__(
        self,
        dataset: Dataset,
        longform_prompt_formatter: BasePromptFormatter,
        scoring_model: BaseLLM,
        question_response_formatter: BasePromptFormatter,
        response_formatter: MCQAResponseFormatter,
        evaluator: MCQAEvalComponent,
        response_component: ResponseComponent,
        *,
        question_field: str,
        target_field: str,
    ) -> None:
        """Task for using a scoring LLM to evaluate longform responses

        This task will use another LLM (typically a more powerful LLM) to score
        longform responses to prompts by comparing LLM-generated longform responses
        to a ground-truth MCQA dataset.

        Example Application
        -------------------
        Evaluation of conclusion generation using PubMedQA:
        1. Create prompts instructing the model to generate a conclusion for the
        pubmedqa context field
        2. Use the original pubmedqa question and the LLM-generated conclusions to
        format prompts to pass to the scoring LLM, instructing it to give a shortform
        answer to the question using the LLM-generated conclusion as a new context
        3. Compare the responses of the scoring LLM to the ground-truth answers
        to score the longform conclusions

        Steps
        -----
        1. Create prompts for dataset (under `prompt`)
        2. Generate responses from `llm` (under `response`) (`llm` is an `LLMGenerator`
        provided in `.run_task`)
        3. Create scoring prompt (under `score_prompt`) formatting using the generated
        responses
        4. Generate response from `scoring_model`
        5. Post-process responses
        6. Evaluate responses by comparing to the ground-truth `target_field`

        Parameters
        ----------
        dataset : Dataset
            Dataset to format prompts with.
        longform_prompt_formatter : ColumnPromptFormatter
            Prompt formatter for formatting prompts for longform response generation
        scoring_model : BaseLLM
            LLM to score longform summaries with (typically a more powerful model than
            the LLM under evaluation)
        question_response_formatter : ColumnPromptFormatter
            Prompt formatter for formatting prompts for scoring LLM response generation
        response_formatter : MCQAResponseFormatter
            Formatter to post-process scoring model responses
        evaluator : MCQAEvalComponent
            Class for evaluating multiple-choice responses from scoring LLM
        cache_dir : Path
            Directory to cache any LLM responses
        question_field : str
            Dataset field containing original multiple-choice question
        target_field : str
            Dataset field containing the target values
        index_field : str
            Dataset field containing the index column
        """
        self.dataset = dataset

        self.longform_prompt_formatter = longform_prompt_formatter
        self.scoring_model = scoring_model
        self.question_response_formatter = question_response_formatter
        self.response_formatter = response_formatter
        self.evaluator = evaluator

        self.question_field = question_field
        self.target_field = target_field
        self.scoring_prompt_field_name = "score_prompt"
        self.scored_response_field_name = "scored_response"
        self.response_component = response_component

    @classmethod
    def from_config(
        cls, config: Union[dict, MCQALongFormTaskConfig]
    ) -> MCQALongFormTask:
        config = MCQALongFormTaskConfig.parse_obj(config)

        dataset = prepare_dataset(config.dataset_config.dict())
        if config.subsample_size:
            dataset = slice_dataset(dataset, stop_slice=config.subsample_size)

        longform_prompt_formatter = ColumnPromptFormatter.from_config(
            config.longform_prompt_formatter_config
        )

        scoring_model = build_llm(config.scoring_model_config.dict())

        question_response_formatter = ColumnPromptFormatter.from_config(
            config.question_response_formatter_config
        )

        response_formatter = MCQAResponseFormatter.from_config(
            config.response_formatter_config
        )

        evaluator = MCQAEvalComponent.from_config(config.evaluator_config)

        response_component = ResponseComponent.from_config(
            config.response_component_config
        )

        return cls(
            dataset=dataset,
            longform_prompt_formatter=longform_prompt_formatter,
            scoring_model=scoring_model,
            question_response_formatter=question_response_formatter,
            response_formatter=response_formatter,
            evaluator=evaluator,
            response_component=response_component,
            question_field=config.question_field,
            target_field=config.target_field,
        )

    def run_task(self, llm: LLMGenerator, verbose: bool = True) -> RunTaskReturnType:
        """Runs evaluation of model

        Parameters
        ----------
        llm : LLMGenerator
            Model to be evaluated

        Returns
        -------
        Dict[str, float]: Dictionary of results
        Dict[str, int]]: Label encoder
        """
        # Creates initial prompt dataset
        prompt_dataset = self.longform_prompt_formatter.format(self.dataset)

        # Generates longform responses from LLM
        response_dataset = self.response_component.batch_generate(
            model=llm,
            prompt_dataset=prompt_dataset,
        )

        # Creates dataset of prompts for scoring
        scoring_prompts = self.question_response_formatter.format(
            response_dataset, self.scoring_prompt_field_name
        )

        # Scores responses using another LLM
        scored_response_dataset = self.response_component.batch_generate(
            model=self.scoring_model,
            prompt_dataset=scoring_prompts,
            prompt_field=self.scoring_prompt_field_name,
            response_field=self.scored_response_field_name,
            verbose=verbose,
        )

        # Processes responses for evaluation
        processed_response_dataset = self.response_formatter.format(
            scored_response_dataset
        )

        eval_results = self.evaluator.evaluate(processed_response_dataset)

        # Preparing for logging
        to_log_as_metrics, to_log_as_dicts = prepare_dicts_for_logging(
            eval_results=eval_results,
        )

        return RunTaskReturnType(
            metrics=to_log_as_metrics,
            artifacts=eval_results,
            datasets={"final_dataset": processed_response_dataset},
            other=None,
            artifact_storing_format="json",
            plots=None,
            dictionaries=to_log_as_dicts,
        )


class SimilarityBasedLongFormTaskConfig(BaseModel):
    dataset_config: GenericDatasetConfig
    prompt_formatter_config: ColumnPromptFormatterConfig
    response_component_config: ResponseComponentConfig
    dataset_filterer_config: Optional[DatasetFiltererConfig]
    text_to_text_component_config: ParentTextToTextModuleConfig
    evaluator_config: EvalComponentConfig
    response_field_name: str
    response_formatter_config: Optional[MappingResponseFormatterConfig]
    response_quality_evaluator_config: ResponseQualityEvalConfig

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


class SimilarityBasedLongFormTask(BaseTask):
    def __init__(
        self,
        dataset: Dataset,
        prompt_formatter: ColumnPromptFormatter,
        response_component: ResponseComponent,
        dataset_filterer: Optional[DatasetFilterer],
        evaluator: BaseEvalComponent,
        response_field_name: str,
        response_quality_evaluator: ResponseQualityEvalComponent,
        text_to_text_component: TextToTextSimilarityComponent,
        response_formatter: Optional[MappingResponseFormatter] = None,
    ) -> None:
        """Task for evaluating longform responses via semantic similarity.

        This task will use a text_to_text_component to evaluate an LLM's
        longform responses to questions with long form ground-truth answers.

        Steps:
        1. Create prompts for dataset
        2. Filter dataset as necessary
        3. Generate responses from LLM
        4. Generate response from text_to_text_component
        5. Post-process responses
        5. Run evaluation

        Example uses:
        1. Evaluate a LLM in creating a question from an answer/ context
        2. Evaluate the perfomance of a LLM in long form Q&A
        3. Evaluate the perfomance of a LLM in rephasing statements

        Parameters
        ----------
        dataset : Dataset
            Dataset to use in the task
        prompt_formatter : ColumnPromptFormatter
            Class to generate prompts
        response_component : ResponseComponent
            Class to generate responses from prompts using the LLM
        dataset_filterer: DatasetFilterer
            Class to skip entries whos prompt length excedes the context length
        text_to_text_component : TextToTextSimilarityComponent
            Class to identify semantic similarity from text pairs
        response_formatter : BaseResponseFormatter
            Class to post-process text_to_text_component responses
        evaluator : BaseEvalComponent
            Class for evaluating responses
        response_quality_evaluator: ResponseQualityEvalComponent
            Class for evaluating the token length of responses
        response_field_name : str
            Dataset field containing the target values
        """
        self.dataset = dataset
        self.prompt_formatter = prompt_formatter
        self.response_component = response_component
        self.dataset_filterer = dataset_filterer
        self.text_to_text_component = text_to_text_component
        self.response_formatter = response_formatter
        self.evaluator = evaluator
        self.response_field_name = response_field_name
        self.response_quality_evaluator = response_quality_evaluator

    @classmethod
    def from_config(
        cls, config: Union[dict, SimilarityBasedLongFormTaskConfig]
    ) -> SimilarityBasedLongFormTask:
        if not isinstance(config, SimilarityBasedLongFormTaskConfig):
            config = SimilarityBasedLongFormTaskConfig.parse_obj(config)

        dataset = prepare_dataset(config.dataset_config.dict())

        prompt_formatter = ColumnPromptFormatter.from_config(
            config.prompt_formatter_config
        )

        response_component = ResponseComponent.from_config(
            config.response_component_config
        )

        dataset_filterer = None
        if config.dataset_filterer_config:
            dataset_filterer = DatasetFilterer.from_config(
                config.dataset_filterer_config
            )

        text_to_text_component = build_text_to_text_module(
            config.text_to_text_component_config
        )

        response_formatter = None
        if config.response_formatter_config:
            response_formatter = MappingResponseFormatter.from_config(
                config.response_formatter_config
            )

        evaluator = build_eval_component(config.evaluator_config)

        response_quality_evaluator = ResponseQualityEvalComponent.from_config(
            config.response_quality_evaluator_config
        )

        return cls(
            dataset=dataset,
            prompt_formatter=prompt_formatter,
            response_component=response_component,
            dataset_filterer=dataset_filterer,
            text_to_text_component=text_to_text_component,
            response_formatter=response_formatter,
            evaluator=evaluator,
            response_field_name=config.response_field_name,
            response_quality_evaluator=response_quality_evaluator,
        )

    def run_task(self, llm: LLMGenerator, verbose: bool = True) -> RunTaskReturnType:
        """Runs evaluation of model

        Parameters
        ----------
        llm : LLMGenerator
            Model to be evaluated

        Returns
        -------
        RunTaskReturnType
        """
        # Creates initial prompt dataset
        prompt_dataset = self.prompt_formatter.format(self.dataset)

        # Generates longform responses from LLM
        response_dataset = self.response_component.batch_generate(
            model=llm,
            prompt_dataset=prompt_dataset,
            response_field=self.response_field_name,
            verbose=verbose,
        )

        to_log_as_dict_extras = None
        # Filter dataset and log lengths
        # (excludes those prompts that exceeded the LLM context length)
        if self.dataset_filterer:
            to_log_as_dict_extras = {}
            original_dataset_len = len(response_dataset)
            response_dataset = self.dataset_filterer.run(response_dataset)
            if len(response_dataset) == 0:
                raise ValueError("All rows have been filtered out of the dataset")
            filtered_dataset_len = len(response_dataset)
            logger.info(
                f"""Started with {original_dataset_len} entries, after filtering
                    the dataset contained {filtered_dataset_len} entries"""
            )
            # Record dataset len pre and post filter
            to_log_as_dict_extras["dataset_len"] = {}
            to_log_as_dict_extras["dataset_len"][
                "original_dataset_len"
            ] = original_dataset_len
            to_log_as_dict_extras["dataset_len"][
                "filtered_dataset_len"
            ] = filtered_dataset_len

        # Generate responses from text to text component
        response_dataset = self.text_to_text_component.run(
            dataset=response_dataset, prompt_field_name="evaluation_prompt"
        )

        # Post-process responses
        if self.response_formatter:
            response_dataset = self.response_formatter.format(response_dataset)

        # Evaluate
        eval_output = self.evaluator.evaluate(response_dataset)

        # Evaluate lengths
        quality_eval = self.response_quality_evaluator.evaluate(response_dataset)

        # Preparing for logging
        to_log_as_metrics, to_log_as_dicts = prepare_dicts_for_logging(
            eval_results=eval_output,
            quality_eval=quality_eval,
        )
        if to_log_as_dict_extras is not None:
            to_log_as_dicts.update(to_log_as_dict_extras)

        return RunTaskReturnType(
            metrics=to_log_as_metrics,
            artifacts={
                "response_quality_eval": quality_eval,
                **eval_output,
            },
            datasets={"final_dataset": response_dataset},
            other=None,
            artifact_storing_format="json",
            plots=None,
            dictionaries=to_log_as_dicts,
        )
