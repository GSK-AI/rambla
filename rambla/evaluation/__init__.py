from typing import Type, Union

from pydantic import BaseModel, validator

from rambla.evaluation.base import BaseEvalComponent
from rambla.evaluation.continuous import ContinuousEvalComponent
from rambla.evaluation.longform import LongformQAEvalComponent
from rambla.evaluation.shortform import MCQAEvalComponent, MCQAStratifiedEvalComponent

EVAL_COMPONENT_MAP = {
    "shortform": MCQAEvalComponent,
    "shortform_stratified": MCQAStratifiedEvalComponent,
    "longform": LongformQAEvalComponent,
    "continuous": ContinuousEvalComponent,
}


def _validate_eval_component_name(eval_component_name: str):
    """Checks provided name is in `EVAL_COMPONENT_MAP`"""
    if eval_component_name not in EVAL_COMPONENT_MAP.keys():
        raise ValueError(
            f"""Invalid name: {eval_component_name}.
            Name must be one of {list(EVAL_COMPONENT_MAP.keys())}"""
        )


class EvalComponentConfig(BaseModel):
    name: str
    params: dict

    @validator("name")
    @classmethod
    def validate_name(cls, v):
        _validate_eval_component_name(v)
        return v


def build_eval_component(config: Union[dict, EvalComponentConfig]) -> BaseEvalComponent:
    """Prepares eval component based on config."""
    if not isinstance(config, EvalComponentConfig):
        config = EvalComponentConfig.parse_obj(config)
    evaluator: Type[BaseEvalComponent] = EVAL_COMPONENT_MAP[config.name]
    return evaluator.from_config(config.params)
