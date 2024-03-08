import copy
import functools
from typing import Dict, List, Optional, Union

import evaluate
import numpy as np
import sklearn.metrics
from scipy import stats

from rambla.utils.misc import initialize_logger

logger = initialize_logger(__name__)


SUPPORTED_SCIPY_METRICS = [
    "pearsonr",
    "spearmanr",
    "pointbiserialr",
    "mannwhitneyu",
    "kendalltau",
]

SUPPORTED_SKLEARN_METRICS = [
    "roc_curve",
    "roc_auc_score",
    # Reverse roc can be used as a metric in the case that the model is
    # predicting a binary output but the labels are continous.
    # It is calaculated as a normal auroc but the labels are
    # considered as predictions and the predictions are considered as labels.
    # This is needed as a metric because more traditional regression
    # metrics such as mse aren ot appropriate in these scenarios.
    "reverse_roc_curve",
    "reverse_roc_auc_score",
]


@functools.lru_cache(10)
def cached_load_module(metric_name: str) -> evaluate.EvaluationModule:
    """Caches huggingface loading module.

    Parameters
    ----------
    metric_names : str,
        Metric to load
    """
    return evaluate.load(metric_name)


def compute_huggingface_metric(
    metric_name: str,
    *,
    predictions: List[float],
    targets: List[float],
    kwargs: Optional[dict] = None,
) -> Union[dict, float]:
    """Loads and computes huggingface evaluation metrics.

    Parameters
    ----------
    metric_names : str,
        Metric to load and compute
    predictions : List[str]
    targets : List[str]
    metric_kwargs: Dict[str, Dict[str, Any]], optional
        Option for providing kwargs to be passed on when computing metrics
    """
    if not kwargs:
        kwargs = {}
    metric = cached_load_module(metric_name)
    # Convert to float
    predictions = np.array(predictions, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    result = metric.compute(predictions=predictions, references=targets, **kwargs)
    return result


def compute_sklearn_metric(
    metric_name: str,
    *,
    predictions: List[float],
    targets: List[float],
    kwargs: Optional[dict] = None,
) -> Union[dict, float]:
    """Loads and computes sklearn evaluation metrics.

    Parameters
    ----------
    metric_names : str,
        Metric to load and compute
    predictions : List[str]
    targets : List[str]
    metric_kwargs: Dict[str, Dict[str, Any]], optional
        Option for providing kwargs to be passed on when computing metrics
    """
    if not kwargs:
        kwargs = {}
    full_metric_name = copy.copy(metric_name)
    if metric_name in ["reverse_roc_curve", "reverse_roc_auc_score"]:
        # Need to swap predictions and targets here to make auroc work
        predictions, targets = targets, predictions
        # Strip reverse out of name
        metric_name = metric_name.split("_", 1)
        metric_name = metric_name[1]

    metric = getattr(sklearn.metrics, metric_name)

    result = metric(y_score=predictions, y_true=list(map(int, targets)), **kwargs)

    if metric_name == "roc_curve":
        output = {full_metric_name: dict(zip(["fpr", "tpr", "thresholds"], result))}
    elif metric_name == "roc_auc_score":
        output = {full_metric_name: result}
    else:
        raise ValueError(
            f"""{metric_name=} not supported.
            Try one of {SUPPORTED_SKLEARN_METRICS}."""
        )
    return output


def compute_scipy_metric(
    metric_name: str,
    *,
    predictions: List[float],
    targets: List[float],
    kwargs: Optional[dict] = None,
) -> tuple[float, float]:
    """Loads and computes scipy stats evaluation metrics.

    Parameters
    ----------
    metric_names : str,
        Metric to load and compute
    predictions : List[str]
    targets : List[str]
    """
    if not kwargs:
        kwargs = {}
    metric = getattr(stats, metric_name)
    # Convert to float
    predictions = np.array(predictions, dtype=np.float64)
    targets = np.array(targets, dtype=np.float64)
    statistic, pvalue = metric(predictions, targets, **kwargs)
    return statistic, pvalue


def run_metric(
    metric_name: str,
    predictions: List[float],
    targets: List[float],
    kwargs_dict: Optional[dict] = None,
) -> Dict[str, float]:
    """Obtains a result for a given evaluation metric.

    Parameters
    ----------
    metric_names : str,
        Metrics to load and compute. Metrics are loaded from sklearn,
        scipy or huggingface.
        NOTE: if the name starts with "stats_" we look for the
        metric within the `scipy.stats` libary.
        NOTE: if the name starts with "sklearn_" we look for the
        metric within the `sklearn.metrics` package.
    predictions : List[str]
    targets : List[str]
    metric_kwargs: Dict[str, Dict[str, Any]], optional
        Option for providing kwargs to be passed on when computing metrics
    """
    if not kwargs_dict:
        kwargs_dict = {}

    # Check if metric in scipy stats
    if metric_name.startswith("stats_"):
        metric_name = metric_name.split("_", 1)[1]
        if metric_name in SUPPORTED_SCIPY_METRICS:
            statistic, pvalue = compute_scipy_metric(
                metric_name=metric_name,
                predictions=predictions,
                targets=targets,
                kwargs=kwargs_dict,
            )
            return {
                f"stats_{metric_name}_statistic": statistic,
                f"stats_{metric_name}_pvalue": pvalue,
            }
        else:
            raise ValueError(
                f"""{metric_name} not in supported scipy stats list.
                             Try one of {SUPPORTED_SCIPY_METRICS=}"""
            )
    elif metric_name.startswith("sklearn_"):
        metric_name = metric_name.split("_", 1)[1]
        if metric_name in SUPPORTED_SKLEARN_METRICS:
            result = compute_sklearn_metric(
                metric_name=metric_name,
                predictions=predictions,
                targets=targets,
                kwargs=kwargs_dict,
            )
            return result
        else:
            raise ValueError(f"{metric_name=} not in {SUPPORTED_SKLEARN_METRICS=}.")
    # Check if metric in huggingface evaluate
    else:
        try:
            cached_load_module(metric_name)
        except FileNotFoundError as err:
            logger.info(err)
            raise ValueError(
                f"{metric_name} not in huggingface evaluate library. "
                "If you want a scipy stats metric add the prefix "
                "'stats_' to the metric name string and if you want "
                "an sklearn metric add the prefix 'sklearn_'."
            )
        else:
            if metric_name == "r_squared":
                result = compute_huggingface_metric(
                    metric_name=metric_name,
                    predictions=predictions,
                    targets=targets,
                    kwargs=kwargs_dict,
                )
                return {metric_name: result}
            else:
                # Only need to return normal result for other hugging face
                # metrics as they allready return a dict
                result = compute_huggingface_metric(
                    metric_name=metric_name,
                    predictions=predictions,
                    targets=targets,
                    kwargs=kwargs_dict,
                )
                return result
