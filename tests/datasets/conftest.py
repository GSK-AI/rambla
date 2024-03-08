import pytest
from datasets import Dataset


@pytest.fixture
def pqa_dataset_config() -> dict:
    return {
        "name": "pubmed_qa",
        "params": {"path": "pubmed_qa", "subset": "pqa_labeled", "split": "train"},
        "index_field": "pubid",
        "question_field": "question",
        "target_field": "final_decision",
        "categories_to_keep": ["yes", "no", "maybe"],
    }


@pytest.fixture
def pqa_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "final_decision": ["yes", "yes", "yes", "maybe", "no"],
            "pmid": list("12345"),
            "question": list("abcde"),
            "contexts": [
                ["The aim", "of this experiment", "is to ", "analyse "],
                [
                    "the performance",
                    "of both models",
                    "on the PubmedQA dataset",
                    "with a bias",
                ],
                ["without context or", "examples is near random"],
                [" few-shot", "prompts with", "only"],
                ["ending"],
            ],
        }
    )


@pytest.fixture
def flat_pqa_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "final_decision": ["yes", "yes", "yes", "maybe", "no"],
            "pmid": list("12345"),
            "question": list("abcde"),
            "context": [
                "The aim of this experiment is to analyse ",
                "the performance of both models on the PubmedQA dataset with a bias",
                "without context or examples is near random",
                " few-shot prompts with only",
                "ending",
            ],
        }
    )


@pytest.fixture
def filtered_pqa_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "final_decision": ["yes", "yes", "yes", "no"],
            "pmid": list("1235"),
            "question": list("abce"),
            "context": [
                "The aim of this experiment is to analyse ",
                "the performance of both models on the PubmedQA dataset with a bias",
                "without context or examples is near random",
                "ending",
            ],
        }
    )


@pytest.fixture
def balanced_pqa_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "final_decision": ["yes", "no"],
            "pmid": list("15"),
            "question": list("ae"),
            "context": [
                "The aim of this experiment is to analyse ",
                "ending",
            ],
        }
    )
