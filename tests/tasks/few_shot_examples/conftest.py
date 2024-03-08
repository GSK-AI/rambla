from typing import List

import numpy as np
import pytest

from tests.conftest import generate_random_string


@pytest.fixture
def seed() -> int:
    return 24


@pytest.fixture
def order() -> List[str]:
    return ["yes", "yes", "no"]


@pytest.fixture
def response_cache_fname() -> str:
    return "response.json"


@pytest.fixture
def cache_dir() -> str:
    return "dummy cache dir"


@pytest.fixture
def index_field() -> str:
    return "pmid"


@pytest.fixture
def target_field() -> str:
    return "final_decision"


@pytest.fixture
def target_field_in_template() -> str:
    return "answer"


@pytest.fixture
def question_field() -> str:
    return "question"


@pytest.fixture
def context_field() -> str:
    return "context"


@pytest.fixture
def response_field_name() -> str:
    return "response"


@pytest.fixture
def null_category() -> str:
    return "null"


@pytest.fixture
def categories() -> List[str]:
    return ["yes", "no"]


@pytest.fixture
def dataset_config(
    categories: List[str], index_field: str, question_field: str, target_field: str
) -> dict:
    return {
        "name": "balanced_pubmed_qa",
        "params": {"path": "pubmed_qa", "subset": "pqa_labeled", "split": "train"},
        "index_field": index_field,
        "question_field": question_field,
        "target_field": target_field,
        "categories_to_keep": categories,
    }


@pytest.fixture
def source_dataset_config(
    categories: List[str], index_field: str, question_field: str, target_field: str
) -> dict:
    return {
        "name": "flat_pubmed_qa",
        "params": {"path": "pubmed_qa", "subset": "pqa_labeled", "split": "train"},
        "index_field": index_field,
        "question_field": question_field,
        "target_field": target_field,
        "categories_to_keep": categories,
    }


@pytest.fixture
def examples_module_config_no_source(
    seed: int, order: List[str], index_field: str, question_field: str
) -> dict:
    return {
        "seed": seed,
        "order": order,
        "index_field": index_field,
        "question_field": question_field,
    }


@pytest.fixture
def examples_module_config_with_source(
    seed: int,
    order: List[str],
    index_field: str,
    question_field: str,
    source_dataset_config: dict,
) -> dict:
    return {
        "seed": seed,
        "order": order,
        "index_field": index_field,
        "question_field": question_field,
        "source_dataset_config": source_dataset_config,
    }


@pytest.fixture
def examples_prompt_formatter_config(
    index_field: str,
    question_field: str,
    context_field: str,
    target_field: str,
    target_field_in_template: str,
) -> dict:
    var_map = {
        question_field: question_field,
        context_field: context_field,
        target_field: target_field_in_template,
    }
    return {
        "var_map": var_map,
        "index_field": index_field,
        "target_field": target_field,
        "allow_duplicates": False,
        "intro_template": "you are smart",
        "examples_template": "dummy {question} and {context} and finally {answer}.",
        "final_question_template": "{question}how many tests can you do {context}?",
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
def fewshot_task_config_no_source(
    dataset_config: dict,
    examples_module_config_no_source: dict,
    examples_prompt_formatter_config: dict,
    response_formatter_config: dict,
    evaluator_config: dict,
    response_component_config: dict,
) -> dict:
    return {
        "dataset_config": dataset_config,
        "examples_module_config": examples_module_config_no_source,
        "prompt_formatter_config": examples_prompt_formatter_config,
        "response_formatter_config": response_formatter_config,
        "evaluator_config": evaluator_config,
        "response_component_config": response_component_config,
    }


@pytest.fixture
def fewshot_task_config_with_source(
    dataset_config: dict,
    examples_module_config_with_source: dict,
    examples_prompt_formatter_config: dict,
    response_formatter_config: dict,
    evaluator_config: dict,
    response_component_config: dict,
) -> dict:
    return {
        "dataset_config": dataset_config,
        "examples_module_config": examples_module_config_with_source,
        "prompt_formatter_config": examples_prompt_formatter_config,
        "response_formatter_config": response_formatter_config,
        "evaluator_config": evaluator_config,
        "response_component_config": response_component_config,
    }


# dataset fixtures::


@pytest.fixture
def mock_flat_pubmedqa_dataset():
    def _mock_flat_pubmedqa_dataset(
        seed,
        n_samples=20,
        long_answer_length=100,
        context_length=100,
        labels_length=30,
        meshes_length=50,
    ):
        from datasets import Dataset

        np.random.seed(seed)

        data_dict = {
            "question": [
                generate_random_string(long_answer_length) for _ in range(n_samples)
            ],
            "long_answer": [
                generate_random_string(long_answer_length) for _ in range(n_samples)
            ],
            "final_decision": np.random.choice(["yes", "no"], n_samples).tolist(),
            "contexts": [
                generate_random_string(context_length) for _ in range(n_samples)
            ],
            "labels": [generate_random_string(labels_length) for _ in range(n_samples)],
            "meshes": [generate_random_string(meshes_length) for _ in range(n_samples)],
            "pmid": np.random.randint(0, 100_000, n_samples).tolist(),
        }
        dataset = Dataset.from_dict(data_dict)
        return dataset

    return _mock_flat_pubmedqa_dataset


@pytest.fixture
def mock_balanced_dataset():
    def _mock_balanced_dataset(seed, length):
        from datasets import Dataset

        if length % 2 != 0:
            raise ValueError("The balanced dataset must be of even length.")

        np.random.seed(seed)
        question_length = 100
        context_length = 100

        data_dict = {
            "question": [
                generate_random_string(question_length) for _ in range(length)
            ],
            "final_decision": ["yes"] * int(length / 2) + ["no"] * int(length / 2),
            "context": [generate_random_string(context_length) for _ in range(length)],
            "pmid": np.random.randint(0, 100_000, length).tolist(),
        }

        dataset = Dataset.from_dict(data_dict)
        return dataset

    return _mock_balanced_dataset


@pytest.fixture
def mock_questions():
    def _mock_questions(seed, length):
        np.random.seed(seed)
        return [generate_random_string(50) for _ in range(length)]

    return _mock_questions


@pytest.fixture
def mock_examples_list():
    def _mock_examples_list(seed, length, order_length):
        np.random.seed(seed)
        exs = []
        for i in range(length):
            exs.append(np.random.randint(0, 100_000, order_length).tolist())

        return exs

    return _mock_examples_list


@pytest.fixture
def mock_results() -> dict:
    confusion_matrix = np.array([[5, 2, 3], [3, 2, 3], [5, 6, 2]])
    results = {
        "confusion_matrix": confusion_matrix,
        "f1": 0.9,
        "recall": 0.8,
        "precision": 0.7,
    }
    return results


@pytest.fixture
def mock_results_with_bias() -> dict:
    confusion_matrix = np.array([[5, 2, 3], [3, 2, 3], [5, 6, 2]])
    results = {
        "confusion_matrix": confusion_matrix,
        "f1": 0.9,
        "recall": 0.8,
        "precision": 0.7,
        "bias_for_yes": 2 / 3,
        "bias_for_no": 1 / 3,
    }
    return results
