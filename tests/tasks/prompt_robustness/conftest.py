import pytest


@pytest.fixture
def task_config(
    dataset_config: dict,
    prompt_formatter_config: dict,
    response_formatter_config: dict,
    evaluator_config: dict,
    response_quality_evaluator_config: dict,
    response_component_config: dict,
) -> dict:
    return {
        "dataset_config": dataset_config,
        "prompt_formatter_config": prompt_formatter_config,
        "response_formatter_config": response_formatter_config,
        "evaluator_config": evaluator_config,
        "response_quality_evaluator_config": response_quality_evaluator_config,
        "response_component_config": response_component_config,
    }


@pytest.fixture
def mutator_config() -> dict:
    return {
        "name": "character_level",
        "mutation_operator_configs": [
            {"name": "swap_character", "params": {"match_character_type": True}}
        ],
        "match_character_type": True,
        "single_mutation_per_word": True,
        "seed": 123,
    }


@pytest.fixture
def prompt_robustness_config(task_config: dict, mutator_config: dict) -> dict:
    return {
        "subtask": {
            "name": "MCQABaselineTask",
            "config": task_config,
        },
        "mutator_config": mutator_config,
        "mutation_schedule": [10, 20, 30, 40, 50],
        "field_to_mutate": "context",
    }
