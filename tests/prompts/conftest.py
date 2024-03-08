import pytest
from datasets import Dataset


@pytest.fixture
def placeholders() -> str:
    return ["question", "context", "answer"]


@pytest.fixture
def var_map() -> str:
    return {"question": "question", "context": "context", "final_decision": "answer"}


@pytest.fixture
def index_field() -> str:
    return "index"


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
def examples_field() -> str:
    return "examples"


@pytest.fixture
def mock_dataset(
    index_field: str,
    question_field: str,
    context_field: str,
    target_field: str,
    examples_field: str,
) -> Dataset:
    return Dataset.from_dict(
        {
            index_field: list("1234"),
            question_field: ["Qn:A", "Qn:B", "Qn:C", "Qn:D"],
            context_field: ["Ctx:A", "Ctx:B", "Ctx:C", "Ctx:D"],
            target_field: ["Ans:A", "Ans:B", "Ans:C", "Ans:D"],
            examples_field: [list("123"), list("432"), list("134"), list("423")],
        }
    )


@pytest.fixture
def column_prompt_formatter_config(
    index_field: str, question_field: str, context_field: str, tmpdir
) -> dict:
    var_map = {question_field: question_field, context_field: context_field}
    return {
        "var_map": var_map,
        "index_field": index_field,
        "output_filepath": tmpdir,
        "allow_duplicates": False,
        "template": "dummy template with {question} and {context}",
    }


@pytest.fixture
def column_prompt_formatter_placeholders() -> str:
    pass


@pytest.fixture
def examples_prompt_formatter_config(
    index_field: str,
    question_field: str,
    context_field: str,
    target_field: str,
    examples_field: str,
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
        "examples_column_name": examples_field,
        "allow_duplicates": False,
        "intro_template": "you are smart",
        "examples_template": "dummy {question} and {context} and finally {answer}.",
        "final_question_template": "{question}how many tests can you do {context}?",
    }
