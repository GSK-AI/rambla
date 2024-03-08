from typing import Any, Dict, List, Optional, Union

from datasets import Dataset
from pydantic import BaseModel

from rambla.evaluation.base import BaseTargetReferenceEvalComponent
from rambla.evaluation.utils import run_metric

"""
Example usage:
if __name__ == "__main__":
    config = {
        "metric_names": [
            "mse",
            "mae",
            "r_squared",
            "stats_pointbiserialr",
        ]
    }

    evaluator = ContinuousEvalComponent.from_config(config)

    output_results = evaluator.run(predictions=[0, 0, 1], targets=[0, 0.5, 0.8])
"""


class ContinuousEvalComponentConfig(BaseModel):
    metric_names: Optional[List[str]] = None
    metric_kwargs: Optional[Dict[str, Dict[str, Any]]] = None
    response_field: str = "response"
    target_field: str = "targets"


class ContinuousEvalComponent(BaseTargetReferenceEvalComponent):
    """Eval class for continuous prediction evaluation.

    # NOTE: preprocessing should happen outside this class
    """

    default_metric_names: List[str] = ["mse", "mae", "r_squared"]

    def __init__(
        self,
        metric_names: Optional[List[str]] = None,
        metric_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        response_field: str = "response",
        target_field: str = "targets",
    ) -> None:
        """Prepares evaluation metrics.

        Parameters
        ----------
        metric_names : List[str], optional
            What metrics to load from `evaluate`. If none is provided
            we default to ["mse", "mae", "r_squared"], by default None
        metric_kwargs: Dict[str, Dict[str, Any]], optional
            Option for providing kwargs to be passed on when computing metrics
        """
        self.response_field = response_field
        self.target_field = target_field
        if not metric_kwargs:
            metric_kwargs = {}
        self.metric_kwargs = metric_kwargs

        if not metric_names:
            metric_names = self.default_metric_names
        self.metric_names = metric_names

    @classmethod
    def from_config(
        cls, config: Union[dict, ContinuousEvalComponentConfig]
    ) -> "ContinuousEvalComponent":
        if isinstance(config, dict):
            config = ContinuousEvalComponentConfig.parse_obj(config)
        return cls(**config.dict())

    def evaluate(
        self,
        dataset: Dataset,
    ) -> dict:
        predictions = dataset[self.response_field]
        targets = dataset[self.target_field]
        return {"results": self.run(predictions=predictions, targets=targets)}

    def run(
        self,
        *,
        predictions: List[float],
        targets: List[float],
    ) -> Dict[str, float]:
        """Computes metrics.

        Parameters
        ----------
        predictions : List[float]
        targets : List[float]


        Returns
        -------
        Dict[str, float]
            results
        """
        # Check lengths are equal
        if len(predictions) != len(targets):
            raise ValueError(
                f"Found predictions with length: {len(predictions)} "
                f"and targets with length: {len(targets)}"
            )
        # Get results
        results = {}
        for metric in self.metric_names:
            kwargs_dict = self.metric_kwargs.get(metric, {})
            result = run_metric(
                metric_name=metric,
                predictions=predictions,
                targets=targets,
                kwargs_dict=kwargs_dict,
            )
            results.update(result)
        return results
