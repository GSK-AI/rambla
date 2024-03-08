import pytest


@pytest.fixture
def llm_config() -> dict:
    return {
        "name": "openai_chat",
        "params": {"temperature": 1.0, "is_async": False},
    }


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
def context_field() -> str:
    return "context"


@pytest.fixture
def response_field() -> str:
    return "response"


@pytest.fixture
def dataset_config(
    index_field: str,
    question_field: str,
    target_field: str,
) -> dict:
    return {
        "name": "mock_dataset",
        "params": {"path": "mock_path", "split": "train"},
        "index_field": index_field,
        "question_field": question_field,
        "target_field": target_field,
        "categories_to_keep": ["yes", "no"],
    }


@pytest.fixture
def longform_task_config(
    llm_config: dict,
    dataset_config: dict,
    response_component_config: dict,
    target_field: str,
    question_field: str,
    context_field: str,
    index_field: str,
    response_field: str,
) -> dict:
    return {
        "dataset_config": dataset_config,
        "longform_prompt_formatter_config": {
            "template": "another mock template with {context}",
            "var_map": {context_field: context_field},
            "index_field": index_field,
        },
        "scoring_model_config": llm_config,
        "question_response_formatter_config": {
            "template": "another mock template with {question} and {context}",
            "var_map": {question_field: question_field, context_field: context_field},
            "index_field": index_field,
        },
        "response_formatter_config": {
            "response_field_name": f"scored_{response_field}",
            "categories": ["yes", "no"],
            "string_formatter_name": "basic",
        },
        "evaluator_config": {
            "categories": ["yes", "no", "null"],
            "target_field": target_field,
            "response_field": f"scored_{response_field}",
        },
        "question_field": question_field,
        "target_field": target_field,
        "subsample_size": None,
        "response_component_config": response_component_config,
    }
