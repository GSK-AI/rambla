from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
from datasets import Dataset
from pydantic import BaseModel, root_validator

from rambla.datasets.io import MCQADatasetConfig, prepare_dataset
from rambla.utils.task import BaseComponent


def sample_examples_from_dataset(
    pos_examples: Dataset,
    neg_examples: Dataset,
    order: List[str],
    test_qns: Dataset,
    *,
    pos_label: str,
    neg_label: str,
    index_field: str,
    source_question_field: str,
    dest_question_field: str,
) -> List[List[int]]:
    """Samples random example questions of given order.

    Given a lists of positive and negative questions, this generates a set of questions
    that follow the distribution of positive and negative labels given in order.
    Repeat the process of generating example questions for each question in the test
    dataset, making sure that none of the examples are the same as the test question.

    Args
    ----
        pos_examples (Dataset): The subset of the dataset that has positive answer.
        neg_examples (Dataset): The subset of the dataset that has negative answer.
        order (List[str]): A list of positive and negative labels for the few shot
                            prompt examples to follow.
        test_qns (Dataset): The balanced dataset that will be used to test the LLM.
        pos_label (str): The answer label of positive questions (eg. Yes).
        neg_label (str): The answer label of negative questions (eg. No).
        index_field (str): The name of the column to use as indexing (eg. pmid).
        source_question_field (str): The name of the column of the test dataset
        containing the queries.
        dest_question_field (str): The name of the column of the dataset from which we
                            will sample the examples from, containing the queries.
    """
    examples_all_qns = []
    n_pos = order.count(pos_label)
    n_neg = order.count(neg_label)
    for testqn in test_qns:
        pos_counter, neg_counter = 0, 0
        examples_single_qn = []

        pre_pos_qns = pos_examples.filter(
            lambda x: x[source_question_field] != testqn[dest_question_field]
        )
        pos_qns = np.random.choice(pre_pos_qns[index_field], n_pos, replace=False)

        pre_neg_qns = neg_examples.filter(
            lambda x: x[source_question_field] != testqn[dest_question_field]
        )
        neg_qns = np.random.choice(pre_neg_qns[index_field], n_neg, replace=False)

        for clss in order:
            if clss == pos_label:
                examples_single_qn.append(pos_qns[pos_counter])
                pos_counter += 1
            elif clss == neg_label:
                examples_single_qn.append(neg_qns[neg_counter])
                neg_counter += 1
            else:
                raise ValueError(
                    f"""Must input few-shot question distribution as a list of
                    {pos_label} and/or {neg_label}. You entered invalid string
                    {clss}."""
                )
        examples_all_qns.append(examples_single_qn)

    return examples_all_qns


class ExamplesGeneratingModuleConfig(BaseModel):
    seed: int
    order: List[str]
    index_field: str
    source_dataset_config: Optional[MCQADatasetConfig]

    positive_label: Optional[str] = "yes"
    negative_label: Optional[str] = "no"
    source_question_field: Optional[str] = "question"
    dest_question_field: Optional[str] = "question"
    source_target_field: Optional[str] = "final_decision"
    examples_column_name: Optional[str] = "examples"

    @root_validator()
    @classmethod
    def validate_order(cls, values):
        if not set(values["order"]).issubset(
            set([values["positive_label"], values["negative_label"]])
        ):
            raise ValueError(
                f"""{values["order"]} has an entry not equal to
                        {values["positive_label"]} or {values["negative_label"]}."""
            )
        return values


class ExamplesGeneratingModule(BaseComponent):
    """Adds examples to be used in the few-shot-prompt to the dataset.

    Takes a source dataset which contains the questions you wish to query to the LLM,
    and adds a column containing a list of indeces which refer to the questions
    from the destination dataset which are to be used for the few-shot prompt. The
    questions follow the sequence of positive and negative answers as given by "order".
    """

    def __init__(
        self,
        seed: int,
        order: List[str],
        index_field: str,
        source_dataset: Optional[Dataset] = None,
        *,
        positive_label: Optional[str] = "yes",
        negative_label: Optional[str] = "no",
        source_question_field: Optional[str] = "question",
        dest_question_field: Optional[str] = "question",
        source_target_field: Optional[str] = "final_decision",
        examples_column_name: Optional[str] = "examples",
    ):
        self.seed = seed
        self.order = order
        self.index_field = index_field
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.source_dataset = source_dataset
        self.source_question_field = source_question_field
        self.dest_question_field = dest_question_field
        self.source_target_field = source_target_field
        self.examples_column_name = examples_column_name

    @classmethod
    def from_config(
        cls, config: Union[dict, ExamplesGeneratingModuleConfig]
    ) -> ExamplesGeneratingModule:
        if not isinstance(config, ExamplesGeneratingModuleConfig):
            config = ExamplesGeneratingModuleConfig.parse_obj(config)

        source_dataset = None
        if config.source_dataset_config:
            source_dataset = prepare_dataset(config.source_dataset_config.dict())

        return cls(
            seed=config.seed,
            order=config.order,
            index_field=config.index_field,
            source_dataset=source_dataset,
            positive_label=config.positive_label,
            negative_label=config.negative_label,
            source_question_field=config.source_question_field,
            dest_question_field=config.dest_question_field,
            source_target_field=config.source_target_field,
            examples_column_name=config.examples_column_name,
        )

    def run(self, dataset: Dataset) -> Dataset:
        if self.source_dataset is None:
            self.source_dataset = dataset

        np.random.seed(self.seed)

        pos_examples = self.source_dataset.filter(
            lambda x: x[self.source_target_field] == self.positive_label
        )
        neg_examples = self.source_dataset.filter(
            lambda x: x[self.source_target_field] == self.negative_label
        )

        examples = sample_examples_from_dataset(
            pos_examples=pos_examples,
            neg_examples=neg_examples,
            order=self.order,
            test_qns=dataset,
            pos_label=self.positive_label,
            neg_label=self.negative_label,
            index_field=self.index_field,
            source_question_field=self.source_question_field,
            dest_question_field=self.dest_question_field,
        )
        dataset = dataset.add_column(self.examples_column_name, examples)

        return dataset
