import string
from typing import Sequence
from unittest import mock

import pytest

from rambla.text_mutation import base, mutators, operators, word_validation


@pytest.fixture
def word_validators() -> Sequence[word_validation.WordValidator]:
    return [
        word_validation.WordValidator(
            validation_func=lambda x: not x.isspace(), mode="all"
        )
    ]


@pytest.mark.parametrize(
    "input_text, n_mut",
    [
        ("What is the capital of France?", 2),
        ("What is the capital of France?", 4),
        ("What does this context\ntell us about the causes of heart disease?", 6),
        ("Is p63 more sensitive and specific than 34βE12 in adenocarcinoma?", 7),
    ],
)
def test_character_level_mutator(
    input_text: str,
    n_mut: int,
    word_validators: Sequence[word_validation.WordValidator],
) -> None:
    # Mock mutation operator simply replaces the word with "MUTATED"
    mock_mutation_operator = mock.MagicMock(spec=base.BaseMutationOperator)
    mock_mutation_operator.transform.return_value = "MUTATED"

    mutator = mutators.CharacterLevelMutator(
        mutation_operators=[mock_mutation_operator],
        word_validators=word_validators,
        seed=123,
        single_mutation_per_word=True,
    )
    output_text = mutator.mutate(input_text, n_mut)

    assert output_text.count("MUTATED") == n_mut

    # Checks the whitespace hasn't been mutated
    original_whitespace = len([c for c in input_text if c.isspace()])
    new_whitespace = len([c for c in output_text if c.isspace()])
    assert original_whitespace == new_whitespace


@pytest.mark.parametrize(
    "input_text, n_mut", [("What is the capital of France?", 7), ("\n", 1)]
)
def test_character_level_mutator_invalid_n_mutations(
    input_text: str,
    n_mut: int,
    word_validators: Sequence[word_validation.WordValidator],
) -> None:
    mock_mutation_operator = mock.MagicMock(spec=base.BaseMutationOperator)

    mutator = mutators.CharacterLevelMutator(
        mutation_operators=[mock_mutation_operator],
        word_validators=word_validators,
        seed=123,
        single_mutation_per_word=True,
    )
    with pytest.raises(ValueError):
        _ = mutator.mutate(input_text, n_mut)


def test_character_level_mutator_same_seed(
    word_validators: Sequence[word_validation.WordValidator],
) -> None:
    n_mut = 4
    input_text = "What is the capital of France?"

    mock_mutation_operator = mock.MagicMock(spec=base.BaseMutationOperator)
    mock_mutation_operator.transform.return_value = "MUTATED"

    base_mutator = mutators.CharacterLevelMutator(
        mutation_operators=[mock_mutation_operator],
        word_validators=word_validators,
        seed=123,
        single_mutation_per_word=True,
    )
    original_output = base_mutator.mutate(input_text, n_mut)

    mutator_same_seed = mutators.CharacterLevelMutator(
        mutation_operators=[mock_mutation_operator],
        word_validators=word_validators,
        seed=123,
        single_mutation_per_word=True,
    )
    same_seed_output = mutator_same_seed.mutate(input_text, n_mut)

    assert same_seed_output == original_output


def test_character_level_mutator_diff_seed(
    word_validators: Sequence[word_validation.WordValidator],
) -> None:
    n_mut = 4
    input_text = "What is the capital of France?"

    mock_mutation_operator = mock.MagicMock(spec=base.BaseMutationOperator)
    mock_mutation_operator.transform.return_value = "MUTATED"

    base_mutator = mutators.CharacterLevelMutator(
        mutation_operators=[mock_mutation_operator],
        word_validators=word_validators,
        seed=123,
        single_mutation_per_word=True,
    )
    original_output = base_mutator.mutate(input_text, n_mut)

    mutator_same_seed = mutators.CharacterLevelMutator(
        mutation_operators=[mock_mutation_operator],
        word_validators=word_validators,
        seed=42,
        single_mutation_per_word=True,
    )
    diff_seed_output = mutator_same_seed.mutate(input_text, n_mut)

    assert diff_seed_output != original_output


@pytest.mark.parametrize(
    "input_text, num_mutations",
    [
        ("Random text with only spaces", 2),
        ("Random text with only spaces", 4),
        ("Random text\nwith whitespace and spaces", 2),
        ("Random text\nwith whitespace and spaces", 4),
        ("Random text.\nWith whitespace and\ttabs. And spaces.", 3),
        ("Random text.\nWith whitespace and\ttabs. And spaces.", 5),
    ],
)
def test_whitespace_mutator(input_text: str, num_mutations: int) -> None:
    start_whitespace = sum([letter.isspace() for letter in input_text])

    mutator = mutators.WhiteSpaceMutator(
        [
            operators.InsertCharacterBetweenWordsOperator(
                insert_character_opts=list(string.whitespace)
            )
        ]
    )
    output_text = mutator.mutate(input_text, num_mutations)

    end_whitespace = sum([letter.isspace() for letter in output_text])

    assert end_whitespace == (start_whitespace + num_mutations)
