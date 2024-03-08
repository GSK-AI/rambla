from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

import evaluate
from datasets import Dataset
from pydantic import BaseModel, root_validator, validator

from rambla.text_to_text_components.base import BaseTextToTextSimilarityComponent
from rambla.utils.misc import initialize_logger
from rambla.utils.text_processing import ALLOWED_ENCODING_NAMES, get_tokenizer

logger = initialize_logger(__name__)

ACCEPTED_METRICS = ["rouge", "bleu"]


class NgramTextToTextSimilarityConfig(BaseModel):
    metric_name: Literal["rouge", "bleu"] = "rouge"
    metric_kwargs: Dict[str, Any]
    predictions_field: str = "predictions"
    references_field: str = "references"
    column_name: str = "similarity_response"
    encoding: Optional[str]

    @validator("encoding")
    @classmethod
    def validate_encoding(cls, encoding):
        if encoding is not None:
            error_msg = f"""{encoding} encoding is not implemented.
                Available encodings are {ALLOWED_ENCODING_NAMES}"""
            assert encoding in ALLOWED_ENCODING_NAMES, error_msg
        return encoding

    @root_validator
    @classmethod
    def validate_rouge_kwargs(cls, values):
        metric_name, metric_kwargs = values.get("metric_name"), values.get(
            "metric_kwargs"
        )
        if metric_name == "rouge":
            if "rouge_types" not in metric_kwargs.keys():
                raise ValueError(
                    """"rouge_types" not defined. Please specify one of
                ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
                using rouge_type in the metric_kwargs."""
                )
            elif len(metric_kwargs["rouge_types"]) == 1:
                pass
            else:
                raise ValueError(
                    f"""The length of "rouge_types" was
                 {len(metric_kwargs["rouge_types"])}.
                It must be 1 to allow for subsequent evalaution.
                Please specify one of ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
                using rouge_types in the metric_kwargs."""
                )
        return values


class NgramTextToTextSimilarity(BaseTextToTextSimilarityComponent):
    """Component that uses BLEU or ROUGE to evaluate semantic similarity.

    Steps:
    1. Accepts text to text dataset
    2. Evaluates similarity of text pairs based on ROUGE or BLEU metric
    """

    def __init__(
        self,
        metric: evaluate.EvaluationModule,
        metric_kwargs: Dict[str, Any],
        predictions_field: str,
        references_field: str,
        column_name: str,
    ):
        self.metric = metric
        self.metric_kwargs = metric_kwargs
        self.predictions_field = predictions_field
        self.references_field = references_field
        self.column_name = column_name

    @classmethod
    def from_config(
        cls, config: Union[dict, NgramTextToTextSimilarityConfig]
    ) -> NgramTextToTextSimilarity:
        if not isinstance(config, NgramTextToTextSimilarityConfig):
            config = NgramTextToTextSimilarityConfig.parse_obj(config)

        if config.metric_name in ACCEPTED_METRICS:
            metric = evaluate.load(config.metric_name)
        else:
            raise ValueError(
                f"{config.metric_name} not recognised. "
                f"Available metrics are: {ACCEPTED_METRICS}."
            )

        if config.metric_kwargs:
            metric_kwargs = config.metric_kwargs
        else:
            metric_kwargs = {}

        if config.encoding:
            tokenizer = get_tokenizer(config.encoding)
            if "tokenizer" in metric_kwargs.keys():
                logger.warn(
                    """tokenizer already defined in metric_kwargs.
                              This will be overwritten by the tokenizer
                              defined by the encoding"""
                )
            metric_kwargs["tokenizer"] = lambda x: tokenizer.encode(x)

        return cls(
            metric=metric,
            metric_kwargs=metric_kwargs,
            predictions_field=config.predictions_field,
            references_field=config.references_field,
            column_name=config.column_name,
        )

    def _process_bleu(
        self, predictions: List[str], references: List[str]
    ) -> List[float]:
        """Returns results for a dataset when bleu score is used"""
        # Convert to nested lists to return result for each sample
        predictions = [[i] for i in predictions]
        references = [[i] for i in references]
        results_list = []
        for n in range(len(predictions)):
            result = self.metric.compute(
                predictions=predictions[n],
                references=references[n],
                **self.metric_kwargs,
            )
            result = result["bleu"]
            results_list.append(result)
        return results_list

    def _process_rouge(
        self, predictions: List[str], references: List[str]
    ) -> List[float]:
        """Returns results for a dataset when rouge score is used"""
        result = self.metric.compute(
            predictions=predictions, references=references, **self.metric_kwargs
        )
        results_list = list(result.values())[0]
        return results_list

    def run(
        self,
        dataset: Dataset,
    ) -> Dataset:
        """Obtains metrics for text pairs in dataset.

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
        predictions = dataset[self.predictions_field]
        references = dataset[self.references_field]

        if self.metric.name == "bleu":
            results_list = self._process_bleu(predictions, references)

        elif self.metric.name == "rouge":
            results_list = self._process_rouge(predictions, references)

        response_dataset = dataset.add_column(self.column_name, results_list)

        return response_dataset
