from __future__ import annotations

import abc
from typing import Optional, Tuple, Type, Union

from pydantic import BaseModel, validator

from rambla.utils.misc import dict_argmax, dict_softmax
from rambla.utils.types import BinaryInt, BinaryIntString, LabelDictType

DEFAULT_ENTAILMENT_KEY = "entailment"
DEFAULT_NEUTRAL_KEY = "neutral"


class BaseNLIOutputCombinationStrategy(abc.ABC):
    ALLOWED_NUMERIC_TYPES: Tuple[Type, ...] = (int, float)

    @classmethod
    def _validate_input_dicts(cls, response0: LabelDictType, response1: LabelDictType):
        # Validating type
        if not (isinstance(response0, dict) and isinstance(response1, dict)):
            raise TypeError(
                "Inputs should be of type `dict`. "
                f"Found inputs with types: {type(response0)=} and {type(response1)=}."
            )

        # Validating the dictionaries have the same keys
        if not response0.keys() == response1.keys():
            raise ValueError(
                "Input dicts have different sets of keys, "
                f"{response0.keys()=} and {response1.keys()=}."
            )

        # Validating the values are of numeric type.
        for key, value in response0.items():
            if not isinstance(value, cls.ALLOWED_NUMERIC_TYPES):
                raise TypeError(
                    f"Found entry {key=} with {type(value)=} in `response0`. "
                    f"Values need to have as type one of {cls.ALLOWED_NUMERIC_TYPES=}."
                )

        for key, value in response1.items():
            if not isinstance(value, cls.ALLOWED_NUMERIC_TYPES):
                raise TypeError(
                    f"Found entry {key=} with {type(value)=} in `response1`. "
                    f"Values need to have as type one of {cls.ALLOWED_NUMERIC_TYPES=}."
                )

    @classmethod
    def _validate_input_dicts_against_class_keys(
        cls, response0: LabelDictType, response1: LabelDictType, **class_key_kwargs
    ):
        for key, value in class_key_kwargs.items():
            if value not in response0:
                raise ValueError(f"{key=} with {value=} not found in `response0`.")
            if value not in response1:
                raise ValueError(f"{key=} with {value=} not found in `response1`.")

    @abc.abstractmethod
    def run(
        self, response0: LabelDictType, response1: LabelDictType
    ) -> Union[BinaryInt, float]:
        ...


class StrictNLIOutputCombinationStrategy(BaseNLIOutputCombinationStrategy):
    """Summary:

    Combines the two NLI responses in such that a positive outcome is only achieved
    if _both_ of them have `self.positive_key` as the key with the corresponding
    value being the largest value.
    """

    def __init__(self, positive_key: str = DEFAULT_ENTAILMENT_KEY):
        self.positive_key = positive_key

    def run(
        self, response0: LabelDictType, response1: LabelDictType
    ) -> BinaryIntString:
        """Accepts the two NLI response dicts and combines them."""
        self._validate_input_dicts(response0, response1)
        self._validate_input_dicts_against_class_keys(
            response0, response1, positive_key=self.positive_key
        )

        if dict_argmax(response0) == dict_argmax(response1) == self.positive_key:
            return "1"
        return "0"


class RelaxedNLIOutputCombinationStrategy(BaseNLIOutputCombinationStrategy):
    """Summary:

    Combines the two NLI responses in such that a positive outcome is only achieved
    if either:
        1. _both_ of them have `self.primary_key` as the key with the corresponding
        value being the largest value.
        2. at least one of them has `self.primary_key` as the key with the corresponding
        value being the largest value, and the other response has `self.secondary_key`
        as the key with the largest corresponding largest value.
    """

    def __init__(
        self,
        *,
        primary_key: str = DEFAULT_ENTAILMENT_KEY,
        secondary_key: str = DEFAULT_NEUTRAL_KEY,
    ):
        """Summary

        The `primary_key` needs to _definetely_ be one of the argmax values.
        The `secondary_key` can optinally be one of the two argmax values,
        _given_ that the other key is the `primary_key`.

        The following pairs would constitute a positive response:
            1. (`primary_key`, `primary_key`)
            2. (`primary_key`, `secondary_key`)
            3. (`secondary_key`, `primary_key`)
        NOTE: the above values refer to the keys that correspond to the argmax value
        of each dict.
        """
        self.primary_key = primary_key
        self.secondary_key = secondary_key

        if primary_key == secondary_key:
            raise ValueError(
                f"{primary_key=} should not be the same as the {secondary_key=}."
            )

    def run(
        self, response0: LabelDictType, response1: LabelDictType
    ) -> BinaryIntString:
        """Accepts the two NLI response dicts and combines them."""
        self._validate_input_dicts(response0, response1)
        self._validate_input_dicts_against_class_keys(
            response0,
            response1,
            primary_key=self.primary_key,
            secondary_key=self.secondary_key,
        )

        keys = set([dict_argmax(response0), dict_argmax(response1)])

        if keys == set([self.primary_key]) or keys == set(
            [self.primary_key, self.secondary_key]
        ):
            return "1"
        return "0"


class AverageNLIOutputCombinationStrategy(BaseNLIOutputCombinationStrategy):
    """Summary:

    Combines the two NLI responses by averaging the values of the dict value
    corresponding to `self.positive_key`.
    """

    def __init__(
        self, positive_key: str = DEFAULT_ENTAILMENT_KEY, apply_softmax: bool = False
    ):
        self.apply_softmax = apply_softmax
        self.positive_key = positive_key

    def run(self, response0: LabelDictType, response1: LabelDictType) -> float:
        """Accepts the two NLI response dicts and combines them."""
        self._validate_input_dicts(response0, response1)
        self._validate_input_dicts_against_class_keys(
            response0, response1, positive_key=self.positive_key
        )

        if self.apply_softmax:
            v0 = dict_softmax(response0)[self.positive_key]
            v1 = dict_softmax(response1)[self.positive_key]
        else:
            v0 = response0[self.positive_key]
            v1 = response1[self.positive_key]
        return (v0 + v1) / 2


ALLOWED_COMBINATION_STRATEGIES_MAP = {
    "strict": StrictNLIOutputCombinationStrategy,
    "relaxed": RelaxedNLIOutputCombinationStrategy,
    "average": AverageNLIOutputCombinationStrategy,
}


class CombinationStrategyConfig(BaseModel):
    name: str
    params: Optional[dict] = {}

    @validator("name")
    @classmethod
    def validate_name(cls, name):
        assert (
            name in ALLOWED_COMBINATION_STRATEGIES_MAP.keys()
        ), f"{name=} not in {ALLOWED_COMBINATION_STRATEGIES_MAP.keys()=}."
        return name


def build_combination_strategy(
    config: Union[dict, CombinationStrategyConfig]
) -> BaseNLIOutputCombinationStrategy:
    """Factory method for NLI strategies."""
    if not isinstance(config, CombinationStrategyConfig):
        config = CombinationStrategyConfig.parse_obj(config)

    return ALLOWED_COMBINATION_STRATEGIES_MAP[config.name](**config.params)
