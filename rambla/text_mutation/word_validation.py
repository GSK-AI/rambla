from typing import Callable, Literal

ValidationFunctionType = Callable[[str], bool]


class WordValidator:
    def __init__(
        self, validation_func: ValidationFunctionType, mode: Literal["all", "any"]
    ) -> None:
        """Validates a word according to a user-defined validation function

        Examples
        --------
        # Returns True only if every character is an alphabet character
        >>> validator = WordValidator(str.isalpha, "all")

        # Returns True if any character is whitespace
        >>> validator = WordValidator(str.isspace, "any")

        Parameters
        ----------
        validation_func : Callable[[str], bool]
            Function to validate characters in word
        mode : Literal["all", "any"]
            Validation mode
                'all': Validator must return True on all characters in word
                'any': Validator must return True on at least one character in word

        """
        self.validation_func = validation_func
        self.mode = mode
        if mode not in ["all", "any"]:
            raise ValueError(
                f"Validation mode: {self.mode} is invalid. "
                "Valid modes are: 'all' or 'any'"
            )

    def _validation_func(self, word: str) -> bool:
        if self.mode == "all":
            return all([self.validation_func(letter) for letter in word])
        elif self.mode == "any":
            return any([self.validation_func(letter) for letter in word])
        else:
            raise ValueError(
                f"Validation mode: {self.mode} is invalid. "
                "Valid modes are: 'all' or 'any'"
            )

    @staticmethod
    def all_letter_validate(word: str, validation_func: ValidationFunctionType) -> bool:
        return all([validation_func(letter) for letter in word])

    @staticmethod
    def any_letter_validate(word: str, validation_func: ValidationFunctionType) -> bool:
        return any([validation_func(letter) for letter in word])

    def validate(self, word: str) -> bool:
        # Automatically return False on empty strings
        if len(word) == 0:
            return False

        return self._validation_func(word)
