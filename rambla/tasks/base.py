import abc
import json
from typing import Any, Dict, Optional, Protocol, Union, runtime_checkable

import datasets
import numpy as np
from pydantic import BaseModel, Extra, validator

from rambla.utils.misc import make_json_serialisable

# flake8: noqa: E501, N805


def is_jsonable(obj: Any) -> bool:
    """Checks if object can be stored as a json."""
    try:
        json.dumps(obj)
    except TypeError:
        return False
    else:
        return True


# NOTE: Everything will be stored, but not everything will be logged
class RunTaskReturnType(BaseModel):
    # TODO: Do we need `str`?
    # Will be logged with `gsk.ai.metrics.loggers.mlflow_logger.MlflowLogger.log_metrics`, needs to be flat
    metrics: Optional[Dict[str, Union[float, int, Any]]]

    # Will be logged with `gsk.ai.metrics.loggers.mlflow_logger.MlflowLogger.log_artifacts`, can be anything
    artifacts: Optional[Dict[str, Any]]

    # Will be logged with `gsk.ai.metrics.loggers.mlflow_logger.MlflowLogger.log_dict`
    # NOTE: Each dict needs to be json-able!
    dictionaries: Optional[Dict[str, dict]]

    # Will be stored in json format
    datasets: Optional[Dict[str, datasets.Dataset]]

    # Will be pickled
    other: Optional[Dict[str, Any]]

    # Will be stored as .png
    plots: Optional[Dict[str, Any]]

    # NOTE: if set to `json` then artifacts will be parsed to
    # a format that's json serialisable
    # using `rambla.utils.misc.make_json_serialisable`.
    artifact_storing_format: str = "json"

    class Config:  # noqa: D106
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @validator("dictionaries")
    @classmethod
    def validate_dictionaries(cls, dictionaries):
        if not dictionaries:
            return
        for k, v in dictionaries.items():
            assert is_jsonable(v), f"{dictionaries[k]=} not json-able"
        return dictionaries

    @validator("metrics")
    @classmethod
    def validate_metrics(cls, metrics):
        if not metrics:
            return
        for k, v in metrics.items():
            if not isinstance(v, (int, float)):
                v = make_json_serialisable(v)
                assert is_jsonable(v), f"{metrics[k]=} not json-able"
                metrics[k] = v
        return metrics


@runtime_checkable
class LLMGenerator(Protocol):
    """A generic type that generates responses.

    This could either be an LLM itself, or a more complex object that encapsulates an LLM.
    """

    def generate(self, text: str) -> str:
        ...


class BaseTask(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Union[dict, BaseModel]):
        """Class constructor from a config.

        NOTE: the current plan is to have two constructors:
        1. `__init__` -> Build class from components.
        2. `.from_config` -> Build class by first building components from config.
        3. Another option is to have a flexible constructor that for each component in the
        config checks whether what's provided is the config or the actual component. If
        it is the config we build the component.
        """
        pass

    @abc.abstractmethod
    def run_task(self, llm: LLMGenerator) -> RunTaskReturnType:
        """Runs all components in sequence."""
        pass


class BaseTaskConfig(BaseModel):
    """Leave blank. Only used for typechecking."""

    @property
    @classmethod
    def name(cls):
        return cls.__name__
