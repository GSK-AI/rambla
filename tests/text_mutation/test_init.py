import string
from unittest import mock

import pytest

from rambla import text_mutation
from rambla.text_mutation.mutators import CharacterLevelMutator, WhiteSpaceMutator
from rambla.text_mutation.operators import (
    InsertCharacterBetweenWordsOperator,
    SwapCharacterOperator,
    SwitchCaseOperator,
)


def test_mutation_operator_config_invalid_name() -> None:
    mock_operator_map = {"mock_operator": None}
    mock_mutator_operator_config = {"name": "invalid_name", "params": {}}

    with mock.patch.object(
        text_mutation, "MUTATION_OPERATOR_MAP", new=mock_operator_map
    ):
        with pytest.raises(ValueError):
            _ = text_mutation.MutationOperatorConfig.parse_obj(
                mock_mutator_operator_config
            )


@pytest.fixture
def character_level_mutator_config() -> dict:
    return {
        "name": "character_level",
        "mutation_operator_configs": [
            {"name": "swap_character", "params": {"match_character_type": True}}
        ],
        "match_character_type": True,
        "single_mutation_per_word": True,
        "seed": 123,
    }


def test_build_mutator_character_level(character_level_mutator_config: dict) -> None:
    mutator = text_mutation.build_mutator(character_level_mutator_config)

    # Check operators instantiated correctly
    assert isinstance(mutator.mutation_operators[0], SwapCharacterOperator)
    assert mutator.mutation_operators[0].match_character_type

    # Check mutator instantiated with correct args
    assert isinstance(mutator, CharacterLevelMutator)
    assert mutator.single_mutation_per_word

    # Check expected number of word validators
    num_default_validators = len(text_mutation.DEFAULT_WORD_VALIDATORS)
    assert len(mutator.word_validators) == num_default_validators + 1


@pytest.fixture
def case_mutator_config() -> dict:
    return {
        "name": "case",
        "mutation_operator_configs": [
            {"name": "switch_case", "params": {"case_mode": "upper", "seed": 123}}
        ],
        "case_mode": "upper",
        "single_mutation_per_word": True,
        "seed": 123,
    }


def test_build_mutator_case(case_mutator_config: dict) -> None:
    mutator = text_mutation.build_mutator(case_mutator_config)

    # Check operators instantiated correctly
    assert isinstance(mutator.mutation_operators[0], SwitchCaseOperator)
    assert mutator.mutation_operators[0].case_mode == "upper"

    # Check mutator instantiated with correct args
    assert isinstance(mutator, CharacterLevelMutator)
    assert mutator.single_mutation_per_word

    # Check expected number of word validators
    num_default_validators = len(text_mutation.DEFAULT_WORD_VALIDATORS)
    assert len(mutator.word_validators) == num_default_validators + 1


@pytest.fixture
def whitespace_mutator_config() -> dict:
    return {
        "name": "whitespace",
        "mutation_operator_configs": [
            {
                "name": "insert_character_between_words",
                "params": {
                    "insert_character_opts": list(string.whitespace),
                    "seed": 123,
                },
            }
        ],
        "seed": 123,
    }


def test_build_mutator_whitespace(whitespace_mutator_config: dict) -> None:
    mutator = text_mutation.build_mutator(whitespace_mutator_config)

    # Check operators instantiated correctly
    assert isinstance(
        mutator.mutation_operators[0], InsertCharacterBetweenWordsOperator
    )

    # Check mutator instantiated with correct args
    assert isinstance(mutator, WhiteSpaceMutator)
    assert mutator.seed == 123
