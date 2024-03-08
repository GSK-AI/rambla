import random
import string
from typing import List, Literal, Optional

import numpy as np

from rambla.text_mutation.base import BaseMutationOperator
from rambla.text_mutation.utils import (
    get_character_collection,
    is_punctuation,
    split_into_words_and_whitespace,
)
from rambla.utils.misc import seed_everything


class SwapCharacterOperator(BaseMutationOperator):
    def __init__(
        self,
        match_character_type: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """Class for performing a character-swap mutation on text

        Parameters
        ----------
        skip_punctuation : bool, optional
            If True then won't apply mutations to punctuation characters, by default
            True
        match_character_type : bool, optional
           If True then ensures character mutations replace character with characters
           of same type.
           Character types:
               upper: Uppercase characters
               lower: Lowercase characters
               digit: Numerical characters
               punctuation: Punctuation characters (if not `skip_punctuation`)
            By default True
        seed : Optional[int], optional
            Random seed to initialize randomizer with, by default None
        """
        self.match_character_type = match_character_type
        self.seed = seed

        if self.seed:
            seed_everything(self.seed)

    @staticmethod
    def _get_valid_char_indices(text: str, skip_punctuation: bool) -> List[int]:
        """Returns indices of characters allowed to mutate"""
        if skip_punctuation:
            char_inds = [i for i in range(len(text)) if not is_punctuation(text[i])]
        else:
            char_inds = list(range(len(text)))

        if not char_inds:
            raise ValueError(f"No valid mutate characters found in string {text}")

        return char_inds

    @staticmethod
    def _get_char_mutation_opts(
        curr_char: str, match_character_type: bool
    ) -> List[str]:
        """Returns a list of acceptable characters to sample a mutation from"""
        if match_character_type:
            char_collection = get_character_collection(curr_char)
        else:
            char_collection = string.ascii_letters

        return [c for c in char_collection if c != curr_char]

    def transform(self, text: str) -> str:
        char_inds = self._get_valid_char_indices(text, self.match_character_type)

        mutate_ind = np.random.choice(char_inds)
        curr_char = text[mutate_ind]
        char_opts = self._get_char_mutation_opts(curr_char, self.match_character_type)

        chars = list(text)
        chars[mutate_ind] = np.random.choice(char_opts)

        return "".join(chars)


class SwitchCaseOperator(BaseMutationOperator):
    def __init__(
        self, case_mode: Literal["upper", "lower", "both"], seed: Optional[int] = None
    ) -> None:
        self.case_mode = case_mode
        self.seed = seed

        if self.seed:
            seed_everything(self.seed)

    @staticmethod
    def _get_case_inds(
        text, letter_type: Literal["upper", "lower", "both"]
    ) -> List[int]:
        check_func = {"upper": str.isupper, "lower": str.islower, "both": str.isalpha}
        case_inds = [
            idx for idx, letter in enumerate(text) if check_func[letter_type](letter)
        ]
        if not case_inds:
            raise ValueError(
                f"Input '{text}' does not contain any {letter_type} characters"
            )

        return case_inds

    def transform(self, text: str) -> str:
        case_inds = self._get_case_inds(text, self.case_mode)  # type: ignore
        switch_case = random.choice(case_inds)
        letters = list(text)
        letters[switch_case] = str.swapcase(letters[switch_case])

        return "".join(letters)


class RemoveCharacterOperator(BaseMutationOperator):
    def transform(self, text: str) -> str:
        raise NotImplementedError


class AddCharacterOperator(BaseMutationOperator):
    def transform(self, text: str) -> str:
        raise NotImplementedError


class InsertCharacterBetweenWordsOperator(BaseMutationOperator):
    def __init__(
        self,
        insert_character_opts: List[str],
        seed: Optional[int] = None,
    ) -> None:
        """Inserts random characters from a selection between words of a piece of text.

        Example Usage:

            Adding Random Whitespace:
                ```
                import string

                operator = InsertCharacterBetweenWordsOperator(
                    insert_character_opts=list(string.whitespace)
                )
                ```

        Parameters
        ----------
        insert_character_opts : List[str]
            List of characters options for inserting between words.
        seed : Optional[int], optional
            Random seed to initialize randomizer with, by default None
        """
        self.insert_character_opts = insert_character_opts

        self.seed = seed
        if self.seed:
            seed_everything(self.seed)

    def transform(self, text: str) -> str:
        words = split_into_words_and_whitespace(text)
        insert_index = random.choice(range(len(words)))
        whitespace_choice = random.choice(self.insert_character_opts)
        words.insert(insert_index, whitespace_choice)

        return "".join(words)
