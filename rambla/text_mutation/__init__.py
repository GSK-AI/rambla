from typing import Dict, List, Literal, Optional, Type

from pydantic import BaseModel, Extra, validator

from rambla.text_mutation import base, mutators, operators
from rambla.text_mutation.utils import is_not_space, is_punctuation
from rambla.text_mutation.word_validation import WordValidator

MUTATION_OPERATOR_MAP: Dict[str, Type[base.BaseMutationOperator]] = {
    "swap_character": operators.SwapCharacterOperator,
    "insert_character_between_words": operators.InsertCharacterBetweenWordsOperator,
    "switch_case": operators.SwitchCaseOperator,
}


DEFAULT_WORD_VALIDATORS: List[WordValidator] = [
    WordValidator(validation_func=is_not_space, mode="all"),
]


def _validate_config_name(name: str, valid_names: List[str]) -> None:
    if name not in valid_names:
        raise ValueError(f"Invalid name: {name}. Must be one of {valid_names}")


class MutationOperatorConfig(BaseModel):
    name: str
    params: Optional[dict]

    @validator("name")
    @classmethod
    def validate_name(cls, name: str) -> str:
        _validate_config_name(name, list(MUTATION_OPERATOR_MAP.keys()))
        return name

    class Config:  # noqa: D106
        extra = Extra.forbid


def build_mutation_operators(
    configs: List[MutationOperatorConfig] | MutationOperatorConfig,
) -> List[base.BaseMutationOperator]:
    """Builds a list of mutation operators from a list of configs"""
    if isinstance(configs, MutationOperatorConfig):
        configs = [configs]

    mutation_operators = []
    for config in configs:
        mutation_operators.append(MUTATION_OPERATOR_MAP[config.name](**config.params))

    return mutation_operators


class BaseMutatorConfig(BaseModel):
    name: str
    mutation_operator_configs: List[MutationOperatorConfig] | MutationOperatorConfig

    class Config:  # noqa: D106
        extra = Extra.forbid


class CharacterMutatorConfig(BaseMutatorConfig):
    match_character_type: bool
    single_mutation_per_word: bool
    seed: int


class CaseMutatorConfig(BaseMutatorConfig):
    case_mode: Literal["upper", "lower", "both"]
    single_mutation_per_word: bool
    seed: int


class WhiteSpaceMutatorConfig(BaseMutatorConfig):
    seed: int


def _build_character_mutator(
    config: CharacterMutatorConfig | dict,
) -> mutators.CharacterLevelMutator:
    if isinstance(config, dict):
        config = CharacterMutatorConfig.parse_obj(config)

    mutation_operators = build_mutation_operators(config.mutation_operator_configs)

    word_validators = DEFAULT_WORD_VALIDATORS.copy()
    # Excludes punctuation-only words if matching type
    if config.match_character_type:
        word_validators.append(
            WordValidator(validation_func=lambda x: not is_punctuation(x), mode="any"),
        )

    return mutators.CharacterLevelMutator(
        mutation_operators=mutation_operators,
        word_validators=word_validators,
        seed=config.seed,
        single_mutation_per_word=config.single_mutation_per_word,
    )


def _build_case_mutator(
    config: CaseMutatorConfig | dict,
) -> mutators.CharacterLevelMutator:
    if isinstance(config, dict):
        config = CaseMutatorConfig.parse_obj(config)

    mutation_operators = build_mutation_operators(config.mutation_operator_configs)

    word_validators = DEFAULT_WORD_VALIDATORS.copy()
    if config.case_mode == "upper":
        word_validators.append(WordValidator(validation_func=str.isupper, mode="any"))
    elif config.case_mode == "lower":
        word_validators.append(WordValidator(validation_func=str.islower, mode="any"))
    elif config.case_mode == "both":
        word_validators.append(WordValidator(validation_func=str.isalpha, mode="any"))
    else:
        raise ValueError(
            f"Case Mode: {config.case_mode} is invalid. "
            "Valid case modes are `upper`, `lower` or `both`"
        )

    return mutators.CharacterLevelMutator(
        mutation_operators=mutation_operators,
        word_validators=word_validators,
        seed=config.seed,
        single_mutation_per_word=config.single_mutation_per_word,
    )


def _build_whitespace_mutator(
    config: WhiteSpaceMutatorConfig | dict,
) -> mutators.WhiteSpaceMutator:
    if isinstance(config, dict):
        config = WhiteSpaceMutatorConfig.parse_obj(config)

    mutation_operators = build_mutation_operators(config.mutation_operator_configs)

    return mutators.WhiteSpaceMutator(
        mutation_operators=mutation_operators, seed=config.seed
    )


def build_mutator(config: dict) -> base.BaseMutator:
    """Builds a mutator class from a config"""
    if "name" not in config.keys():
        raise ValueError("Config missing 'name' key")

    builder_map = {
        "character_level": _build_character_mutator,
        "case": _build_case_mutator,
        "whitespace": _build_whitespace_mutator,
    }

    if config["name"] not in builder_map:
        raise ValueError(
            f"Invalid mutator name '{config['name']}'. "
            f"Valid options are: {list(builder_map.keys())}"
        )

    return builder_map[config["name"]](config)
