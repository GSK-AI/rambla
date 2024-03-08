import abc
from typing import Sequence


class BaseMutationOperator(abc.ABC):
    @abc.abstractmethod
    def transform(self, text: str) -> str:
        ...


class BaseMutator(abc.ABC):
    def __init__(
        self, mutation_operators: Sequence[BaseMutationOperator] | BaseMutationOperator
    ) -> None:
        if isinstance(mutation_operators, BaseMutationOperator):
            mutation_operators = [mutation_operators]
        self.mutation_operators = mutation_operators

    @abc.abstractmethod
    def mutate(self, text: str, num_mutations: int) -> str:
        ...
