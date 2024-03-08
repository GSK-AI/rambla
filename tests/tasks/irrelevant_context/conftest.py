from typing import List

import pytest


@pytest.fixture
def response_cache_fname() -> str:
    return "response.json"


@pytest.fixture
def cache_dir(tmpdir) -> str:
    return tmpdir


@pytest.fixture
def index_field() -> str:
    return "pmid"


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
def context_field_name() -> str:
    return "context"


@pytest.fixture
def categories() -> List[str]:
    return ["yes", "no", "maybe"]


@pytest.fixture
def dataset_config(
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
def context_augmenting_module_config(context_field_name: str) -> dict:
    return {
        "n_contexts": 3,
        "position_of_original_context": 1,
        "field_name": context_field_name,
        "seed": 1234,
        "separator": "-----",
    }


@pytest.fixture
def prompt_formatter_config(
    context_field_name: str, question_field: str, index_field: str
) -> dict:
    return {
        "template": "mock template with {question} and {context}",
        "var_map": {
            question_field: question_field,
            context_field_name: context_field_name,
        },
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
def distracting_context_task_config(
    dataset_config: dict,
    context_augmenting_module_config: dict,
    prompt_formatter_config: dict,
    response_formatter_config: dict,
    evaluator_config: dict,
    response_component_config: dict,
    cache_dir: str,
    index_field: str,
    response_cache_fname: str,
) -> dict:
    return {
        "dataset_config": dataset_config,
        "context_augmenting_module_config": context_augmenting_module_config,
        "prompt_formatter_config": prompt_formatter_config,
        "response_formatter_config": response_formatter_config,
        "evaluator_config": evaluator_config,
        "cache_dir": cache_dir,
        "index_field": index_field,
        "response_cache_fname": response_cache_fname,
        "response_component_config": response_component_config,
    }


@pytest.fixture
def shuffling_module_config() -> dict:
    return {
        "field_name": "context",
        "seed": 1234,
    }


@pytest.fixture
def text_trimmer_config() -> dict:
    return {
        "field_name": "context",
        "n_sentences": 3,
    }


@pytest.fixture
def irrelevant_context_task_config(
    dataset_config: dict,
    shuffling_module_config: dict,
    prompt_formatter_config: dict,
    response_formatter_config: dict,
    evaluator_config: dict,
    response_component_config: dict,
    text_trimmer_config: dict,
) -> dict:
    return {
        "dataset_config": dataset_config,
        "shuffling_module_config": shuffling_module_config,
        "prompt_formatter_config": prompt_formatter_config,
        "response_formatter_config": response_formatter_config,
        "evaluator_config": evaluator_config,
        "text_trimmer_config": text_trimmer_config,
        "response_component_config": response_component_config,
    }


@pytest.fixture
def dataset_mixer_config(
    dataset_config: dict,
) -> dict:
    return {
        "source_dataset_config": dataset_config,
        "source_field_name": "context",
        "dest_field_name": "context",
        "seed": 1234,
        "with_replacement": False,
    }


@pytest.fixture
def irrelevant_context_different_dataset_task_config(
    dataset_config: dict,
    dataset_mixer_config: dict,
    prompt_formatter_config: dict,
    response_formatter_config: dict,
    evaluator_config: dict,
    response_component_config: dict,
    text_trimmer_config: dict,
) -> dict:
    return {
        "dataset_config": dataset_config,
        "dataset_mixer_config": dataset_mixer_config,
        "prompt_formatter_config": prompt_formatter_config,
        "response_formatter_config": response_formatter_config,
        "evaluator_config": evaluator_config,
        "text_trimmer_config": text_trimmer_config,
        "response_component_config": response_component_config,
    }
