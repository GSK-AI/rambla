from typing import List

import pytest
from datasets import Dataset


@pytest.fixture
def question_dataset() -> Dataset:
    question_column = list("ABCD")
    other_column = list("abcd")

    input_dataset = Dataset.from_dict(
        {
            "question": question_column,
            "other": other_column,
        }
    )
    return input_dataset


@pytest.fixture
def response_cache_fname() -> str:
    return "response.json"


@pytest.fixture
def cache_dir() -> str:
    return "dummy cache dir"


@pytest.fixture
def index_field() -> str:
    return "index"


@pytest.fixture
def target_field() -> str:
    return "final_decision"


@pytest.fixture
def question_field() -> str:
    return "question"


@pytest.fixture
def response_field_name() -> str:
    return "response"


@pytest.fixture
def null_category() -> str:
    return "null"


@pytest.fixture
def categories() -> List[str]:
    return ["yes", "no", "maybe"]


@pytest.fixture
def dataset_config(
    categories: List[str], index_field: str, question_field: str, target_field: str
) -> dict:
    return {
        "name": "pubmed_qa",
        "params": {"path": "pubmed_qa", "subset": "pqa_labeled", "split": "train"},
        "index_field": index_field,
        "question_field": question_field,
        "target_field": target_field,
        "categories_to_keep": categories,
    }


@pytest.fixture
def prompt_formatter_config(question_field: str, index_field: str) -> dict:
    return {
        "template": "mock template with {question}",
        "var_map": {question_field: question_field},
        "index_field": index_field,
    }


@pytest.fixture
def response_formatter_config(
    response_field_name: str, categories: List[str], null_category: str
) -> dict:
    return {
        "response_field_name": response_field_name,
        "categories": categories,
        "string_formatter_name": "basic",
        "null_category": null_category,
    }


@pytest.fixture
def evaluator_config(
    categories: List[str],
    response_field_name: str,
    target_field: str,
    null_category: str,
) -> dict:
    return {
        "categories": categories + [null_category],
        "response_field": response_field_name,
        "target_field": target_field,
    }


@pytest.fixture
def response_quality_evaluator_config(response_field_name: str) -> dict:
    return {"field_names": response_field_name, "encoding": "text-davinci-003"}
