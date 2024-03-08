import pytest


@pytest.fixture
def baseline_task_config(
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
