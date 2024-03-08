import random
from typing import List, Optional, Sequence

import numpy as np

from rambla.text_mutation.base import BaseMutationOperator, BaseMutator
from rambla.text_mutation.utils import split_into_words_and_whitespace
from rambla.text_mutation.word_validation import WordValidator
from rambla.utils.misc import seed_everything


class CharacterLevelMutator(BaseMutator):
    def __init__(
        self,
        mutation_operators: Sequence[BaseMutationOperator] | BaseMutationOperator,
        word_validators: Sequence[WordValidator],
        seed: Optional[int] = None,
        single_mutation_per_word: bool = True,
    ) -> None:
        """Mutates characters of individual words in a text sequence

        Parameters
        ----------
        mutation_operators : Sequence[BaseMutationOperator] | BaseMutationOperator
            Operators to randomly select from to apply mutations to individual words
        word_validators : Sequence[WordValidator]
            List of validators for checking validity of individual words
        seed : Optional[int]
            Random seed to initialize randomizer with
        single_mutation_per_word : bool, optional
            If true then a mutation won't be applied to the same word more than once,
            by default True
        """
        super().__init__(mutation_operators)
        self.word_validators = word_validators
        self.seed = seed
        self.single_mutation_per_word = single_mutation_per_word

        if self.seed:
            seed_everything(self.seed)

    @staticmethod
    def _get_valid_word_inds(
        words: List[str], word_validators: Sequence[WordValidator]
    ) -> List[int]:
        valid_inds = []
        for validator in word_validators:
            valid_inds.append(
                {idx for idx, word in enumerate(words) if validator.validate(word)}
            )

        intersection_inds = set.intersection(*valid_inds)
        return list(intersection_inds)

    def mutate(self, text: str, num_mutations: int) -> str:
        """Mutates the words in `text` `n_mutations` times

        Parameters
        ----------
        text : str
            Text to mutate
        num_mutations : int
            Number of times to mutate the text (If `self.single_mutation_per_word` then
            must be less than number of words in text)

        Returns
        -------
        str
            Mutated text string
        """
        # Splits string, preserving whitespace for joining purposes later
        words = split_into_words_and_whitespace(text)

        # Get indexes of valid words
        word_inds = self._get_valid_word_inds(words, self.word_validators)

        if self.single_mutation_per_word and (num_mutations > len(word_inds)):
            raise ValueError(
                f"Number of valid words in input {len(word_inds)} "
                f"is less than the number of mutations: {num_mutations}"
            )

        replace = not self.single_mutation_per_word

        mutate_inds = np.random.choice(word_inds, size=num_mutations, replace=replace)

        mutated_words = words.copy()
        for ind in mutate_inds:
            mutator = random.choice(self.mutation_operators)
            mutated_words[ind] = mutator.transform(words[ind])

        return "".join(mutated_words)


class WhiteSpaceMutator(BaseMutator):
    def __init__(
        self,
        mutation_operators: Sequence[BaseMutationOperator],
        seed: Optional[int] = None,
    ) -> None:
        """Class for applying whitespace mutations to text

        TODO: Convert this class into a more general-purpose class for applying
        mutations to whole sentences
        """
        super().__init__(mutation_operators)
        self.seed = seed

        if self.seed:
            seed_everything(self.seed)

    def mutate(self, text: str, num_mutations: int) -> str:
        for _ in range(num_mutations):
            mutator = random.choice(self.mutation_operators)
            text = mutator.transform(text)

        return text
