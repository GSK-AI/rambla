import abc
from typing import Union

from pydantic import BaseModel

from rambla.tasks.base import RunTaskReturnType


class BaseTextToTextTask(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Union[dict, BaseModel]):  # noqa: N805
        """Class constructor from a config.

        NOTE: the current plan is to have two constructors:
        1. `__init__` -> Build class from components.
        2. `.from_config` -> Build class by first building components from config.
        3. Another option is to have a flexible constructor that for each component
        in the config checks whether what's provided is the config or the actual
        component. If it is the config we build the component.
        """
        pass

    @abc.abstractmethod
    def run_task(self, text_to_text_component) -> RunTaskReturnType:
        """Runs all components in sequence."""
        pass


class BaseTextToTextTaskConfig(BaseModel):
    """Leave blank. Only used for typechecking."""

    @property
    @classmethod
    def name(cls):
        return cls.__name__
