from typing import Callable
from unittest import mock

import numpy as np
import pytest
from datasets import Dataset
from pydantic.error_wrappers import ValidationError

from rambla.evaluation import build_eval_component
from rambla.evaluation.longform import ResponseQualityEvalComponent
from rambla.prompts.formatters import ColumnPromptFormatter
from rambla.response_formatting.formatting import (
    MappingResponseFormatter,
    MCQAResponseFormatter,
)
from rambla.response_generation.response import ResponseComponent
from rambla.tasks.longform import longform
from rambla.text_to_text_components import build_text_to_text_module
from rambla.text_to_text_components.llm_similarity_component import (
    LLMTextToTextSimilarity,
)
from rambla.utils.misc import EnvCtxManager
from tests.conftest import make_mock_llm

# flake8: noqa: N802


@mock.patch("rambla.tasks.longform.longform.build_llm")
@mock.patch("rambla.tasks.longform.longform.prepare_dataset")
def test_mcqa_longform_task_run_task(
    mock_prepare_dataset,
    mock_build_llm,
    longform_task_config: dict,
    mock_flat_pubmedqa_dataset: Dataset,
    make_mock_llm: Callable,
):
    mock_prepare_dataset.return_value = mock_flat_pubmedqa_dataset

    expected_scoring_responses = [
        "yes",
        "no",
        "yes",
        "yes",
        "yes",
        "no",
        "yes",
        "no",
        "no",
        "no",
    ]
    scoring_llm = make_mock_llm(expected_scoring_responses)
    mock_build_llm.return_value = scoring_llm

    # acting
    task = longform.MCQALongFormTask.from_config(longform_task_config)
    responses = [
        "This text comments on AI.",
        "This text comments on Biology.",
        "I do not understand what this text is about.",
        "A dummy conclusion generated.",
        "This text cannot be summarised in three sentences.",
        "No.",
        "Yes, yes yes yes yes",
        "One word summary: Yes.",
        "yes and no",
        "no",
        "Yes...",
    ]

    mock_llm = make_mock_llm(responses)

    task_output = task.run_task(mock_llm)

    output_dataset = task_output.datasets["final_dataset"]  # noqa: F841
    output_confmat = task_output.artifacts["results"]["confusion_matrix"]  # noqa: F841

    # Asserts
    assert output_dataset["scored_response"] == expected_scoring_responses

    # Longform response prompts
    prompt_template = longform_task_config["longform_prompt_formatter_config"][
        "template"
    ]
    expected_prompts = [
        prompt_template.format(context=context)
        for context in mock_flat_pubmedqa_dataset["context"]
    ]
    assert output_dataset["prompt"] == expected_prompts

    # Checking whether the model was called with the right prompts
    for ii, prompt in enumerate(expected_prompts):
        assert mock_llm.generate.call_args_list[ii].kwargs == {"prompt": prompt}

    # Scoring response prompts
    scoring_prompt_template = longform_task_config[
        "question_response_formatter_config"
    ]["template"]
    expected_scoring_prompts = [
        scoring_prompt_template.format(context=context, question=question)
        for question, context in zip(
            mock_flat_pubmedqa_dataset["question"],
            mock_flat_pubmedqa_dataset["context"],
        )
    ]
    assert output_dataset["score_prompt"] == expected_scoring_prompts
    # Checking whether the scoring model was called with the right prompts
    for ii, prompt in enumerate(expected_scoring_prompts):
        assert scoring_llm.generate.call_args_list[ii].kwargs == {"prompt": prompt}

    # Label encoder
    output_label_encoder = task_output.artifacts["label_encoder"]
    assert output_label_encoder.keys() == set(["yes", "no", "null"])
    assert set(output_label_encoder.values()) == set(range(3))

    # Confmat
    expected_confmat = np.array(
        [
            [4, 0, 1],
            [0, 0, 0],
            [1, 0, 4],
        ]
    )

    confmat_order = ["no", "null", "yes"]
    confmat_order_indices = [output_label_encoder[clss] for clss in confmat_order]
    reordered_output_confmat = output_confmat[confmat_order_indices, :][
        :, confmat_order_indices
    ]
    assert expected_confmat.shape == reordered_output_confmat.shape
    assert np.allclose(expected_confmat, reordered_output_confmat)


def test_mcqa_longform_task_config(longform_task_config: dict) -> None:
    longform_task_config["scoring_model_config"]["name"] = "openai_chat"
    config = longform.MCQALongFormTaskConfig.parse_obj(longform_task_config)
    for field in longform_task_config:
        assert field in config.__fields__.keys()


def test_mcqa_longform_task_config_invalid_target_field(
    longform_task_config: dict,
) -> None:
    longform_task_config["evaluator_config"]["target_field"] = "invalid_target"
    with pytest.raises(ValidationError) as exc_info:
        _ = longform.MCQALongFormTaskConfig.parse_obj(longform_task_config)

    assert "target_field" in str(exc_info.value)
    assert "not the same as" in str(exc_info.value)


def test_mcqa_longform_task_config_invalid_response_field(
    longform_task_config: dict,
) -> None:
    longform_task_config["response_formatter_config"][
        "response_field_name"
    ] = "invalid_response"
    with pytest.raises(ValidationError) as exc_info:
        _ = longform.MCQALongFormTaskConfig.parse_obj(longform_task_config)

    assert "response_field" in str(exc_info.value)
    assert "not the same as" in str(exc_info.value)


def test_mcqa_longform_task_config_invalid_categories(
    longform_task_config: dict,
) -> None:
    invalid_response_formatter_config = longform_task_config.copy()
    invalid_response_formatter_config["response_formatter_config"]["categories"] = [
        "yes",
        "no",
        "maybe",
    ]
    with pytest.raises(ValidationError):
        _ = longform.MCQALongFormTaskConfig.parse_obj(invalid_response_formatter_config)

    # Checks an error raised if null category not included
    invalid_evaluator_config = longform_task_config.copy()
    invalid_evaluator_config["evaluator_config"]["categories"] = ["yes", "no"]
    with pytest.raises(ValidationError):
        _ = longform.MCQALongFormTaskConfig.parse_obj(invalid_evaluator_config)


@mock.patch.object(longform, "build_llm")
@mock.patch.object(longform, "slice_dataset")
@mock.patch.object(longform, "prepare_dataset")
@mock.patch.object(longform.MCQALongFormTask, "__init__", return_value=None)
def test_mcqa_longform_task_from_config(
    mock_init: mock.MagicMock,
    mock_prepare_dataset: mock.MagicMock,
    mock_slice_dataset: mock.MagicMock,
    mock_build_llm: mock.MagicMock,
    longform_task_config: dict,
) -> None:
    longform_task_config["subsample_size"] = 7
    longform_task_config["scoring_model_config"]["name"] = "openai_chat"
    mock_response_formatter = mock.MagicMock(spec=MCQAResponseFormatter)
    with mock.patch.object(
        longform.MCQAResponseFormatter,
        "from_config",
        return_value=mock_response_formatter,
    ):
        _ = longform.MCQALongFormTask.from_config(longform_task_config)

    mock_prepare_dataset.assert_called
    assert (
        mock_prepare_dataset.call_args[0][0]["name"]
        == longform_task_config["dataset_config"]["name"]
    )
    assert (
        mock_slice_dataset.call_args[1]["stop_slice"]
        == longform_task_config["subsample_size"]
    )
    mock_build_llm.assert_called_once_with(longform_task_config["scoring_model_config"])

    assert (
        mock_init.call_args[1]["question_field"]
        == longform_task_config["question_field"]
    )
    assert (
        mock_init.call_args[1]["target_field"] == longform_task_config["target_field"]
    )


@pytest.fixture
def text_to_text_config(tmpdir, response_component_config: dict) -> dict:
    template = (
        "Dummy template to assess the similarity of {statement_1} and {statement_2}"
    )
    index_field = "index"
    llm_config = {
        "name": "openai_chat",
        "params": {"temperature": 0.1, "engine": "gpt-4"},
    }

    var_map = {
        "response": "statement_1",
        "answer": "statement_2",
    }
    prompt_formatter_config = {
        "template": template,
        "var_map": var_map,
        "index_field": index_field,
    }
    config = {
        "name": "llm_component",
        "params": {
            "llm_config": llm_config,
            "prompt_formatter_config": prompt_formatter_config,
            "response_field_name": "evaluation_response",
            "response_component_config": response_component_config,
        },
    }

    return config


@pytest.fixture
def long_form_similarity_task_config(
    tmpdir, response_component_config: dict, text_to_text_config: dict
) -> dict:
    response = "response"
    response_categories = ["yes", "no", "null"]
    evaluator_categories = ["1", "0", "null"]

    dataset_config = {
        "name": "dummy",
        "params": {"path": "dummy", "subset": "dummy", "split": "dummy"},
    }

    renaming_map = {"yes": "1", "no": "0", "null": "null"}
    response_formatter_config = {
        "response_field_name": "response",
        "renaming_map": renaming_map,
        "string_formatter_name": "basic",
        "null_category": "null",
        "categories": response_categories,
    }

    evaluator_config = {
        "name": "shortform",
        "params": {
            "categories": evaluator_categories,
            "response_field": "response",
            "target_field": "target",
            "metric_names": ["recall", "f1", "precision"],
        },
    }

    template = "Question: {question}, Context: {context}"

    var_map = {
        "question": "question",
        "context": "context",
    }

    index_field = "index"

    prompt_formatter_config = {
        "template": template,
        "var_map": var_map,
        "index_field": index_field,
        "allow_duplicates": False,
    }

    response_quality_evaluator_config = {
        "field_names": ["unformatted_response", "answer"],
        "encoding": "text-davinci-003",
    }

    config = {
        "dataset_config": dataset_config,
        "prompt_formatter_config": prompt_formatter_config,
        "response_component_config": response_component_config,
        "text_to_text_component_config": text_to_text_config,
        "response_formatter_config": response_formatter_config,
        "evaluator_config": evaluator_config,
        "response_field_name": response,
        "response_quality_evaluator_config": response_quality_evaluator_config,
    }

    return config


@EnvCtxManager(
    OPENAI_API_KEY="__dummy_openai_api_key__",
    OPENAI_API_BASE="__dummy_openai_api_base__",
    OPENAI_API_VERSION="__dummy_openai_api_version__",
)
@mock.patch("rambla.tasks.longform.longform.prepare_dataset")
def test_SimilarityBasedLongFormTask_from_config(
    mock_prepare_dataset,
    long_form_similarity_task_config: dict,
) -> None:
    mock_prepare_dataset.return_value = "dummy"

    # From config
    long_form_similarity_task_from_config = (
        longform.SimilarityBasedLongFormTask.from_config(
            long_form_similarity_task_config
        )
    )

    # Assert
    assert long_form_similarity_task_from_config.dataset == "dummy"
    assert (
        long_form_similarity_task_config["prompt_formatter_config"]
        == long_form_similarity_task_from_config.prompt_formatter.__dict__
    )
    assert (
        long_form_similarity_task_from_config.response_field_name
        == long_form_similarity_task_config["response_field_name"]
    )
    assert (
        long_form_similarity_task_from_config.response_quality_evaluator.__dict__
        == long_form_similarity_task_config["response_quality_evaluator_config"]
    )
    for key in set(
        long_form_similarity_task_config["response_formatter_config"].keys()
    ) & set(long_form_similarity_task_from_config.response_formatter.__dict__.keys()):
        assert (
            long_form_similarity_task_from_config.response_formatter.__dict__[key]
            == long_form_similarity_task_config["response_formatter_config"][key]
        )
    for key in set(
        long_form_similarity_task_config["evaluator_config"]["params"].keys()
    ) & set(long_form_similarity_task_from_config.evaluator.__dict__.keys()):
        assert (
            long_form_similarity_task_from_config.evaluator.__dict__[key]
            == long_form_similarity_task_config["evaluator_config"]["params"][key]
        )

    for key in set(
        long_form_similarity_task_config["response_component_config"].keys()
    ) & set(long_form_similarity_task_from_config.response_component.__dict__.keys()):
        assert (
            long_form_similarity_task_from_config.response_component.__dict__[key]
            == long_form_similarity_task_config["response_component_config"][key]
        )

    for key in set(
        long_form_similarity_task_config["text_to_text_component_config"][
            "params"
        ].keys()
    ) & set(
        long_form_similarity_task_from_config.text_to_text_component.__dict__.keys()
    ):
        assert (
            long_form_similarity_task_from_config.text_to_text_component.__dict__[key]
            == long_form_similarity_task_config["text_to_text_component_config"][
                "params"
            ][key]
        )


@EnvCtxManager(
    OPENAI_API_KEY="__dummy_openai_api_key__",
    OPENAI_API_BASE="__dummy_openai_api_base__",
    OPENAI_API_VERSION="__dummy_openai_api_version__",
)
@mock.patch("rambla.tasks.longform.longform.prepare_dataset")
def test_SimilarityBasedLongFormTask_init_vs_from_config(
    mock_prepare_dataset,
    long_form_similarity_task_config: dict,
) -> None:
    mock_prepare_dataset.return_value = "dummy"

    # From config
    long_form_similarity_task_from_config = (
        longform.SimilarityBasedLongFormTask.from_config(
            long_form_similarity_task_config
        )
    )

    # From init
    long_form_similarity_task_from_init = longform.SimilarityBasedLongFormTask(
        dataset=mock_prepare_dataset.return_value,
        prompt_formatter=ColumnPromptFormatter.from_config(
            long_form_similarity_task_config["prompt_formatter_config"]
        ),
        response_component=ResponseComponent.from_config(
            long_form_similarity_task_config["response_component_config"]
        ),
        text_to_text_component=build_text_to_text_module(
            long_form_similarity_task_config["text_to_text_component_config"]
        ),
        response_formatter=MappingResponseFormatter.from_config(
            long_form_similarity_task_config["response_formatter_config"]
        ),
        evaluator=build_eval_component(
            long_form_similarity_task_config["evaluator_config"]
        ),
        response_field_name=long_form_similarity_task_config["response_field_name"],
        response_quality_evaluator=ResponseQualityEvalComponent.from_config(
            long_form_similarity_task_config["response_quality_evaluator_config"]
        ),
        dataset_filterer=None,
    )

    # Assert
    assert long_form_similarity_task_from_config.dataset == "dummy"
    assert (
        long_form_similarity_task_from_config.dataset
        == long_form_similarity_task_from_init.dataset
    )
    assert (
        long_form_similarity_task_from_init.prompt_formatter.__dict__
        == long_form_similarity_task_from_config.prompt_formatter.__dict__
    )
    assert (
        long_form_similarity_task_from_config.response_formatter.__dict__
        == long_form_similarity_task_from_init.response_formatter.__dict__
    )
    assert (
        long_form_similarity_task_from_config.evaluator.__dict__
        == long_form_similarity_task_from_init.evaluator.__dict__
    )
    assert (
        long_form_similarity_task_from_config.response_field_name
        == long_form_similarity_task_from_init.response_field_name
    )
    assert (
        long_form_similarity_task_from_config.response_quality_evaluator.__dict__
        == long_form_similarity_task_from_init.response_quality_evaluator.__dict__
    )

    response_component_config_dict = (
        long_form_similarity_task_from_config.evaluator.__dict__
    )
    response_component_init_dict = (
        long_form_similarity_task_from_init.evaluator.__dict__
    )

    for key in response_component_config_dict:
        if key == "_limiter":
            assert (
                response_component_config_dict[key].keys()
                == response_component_init_dict[key].keys()
            )
        else:
            assert (
                response_component_config_dict[key] == response_component_init_dict[key]
            )

    text_to_text_component_config_dict = (
        long_form_similarity_task_from_config.text_to_text_component.__dict__
    )
    text_to_text_component_init_dict = (
        long_form_similarity_task_from_init.text_to_text_component.__dict__
    )

    for key in text_to_text_component_config_dict:
        if key == "response_field_name":
            assert (
                text_to_text_component_config_dict[key]
                == text_to_text_component_init_dict[key]
            )
        else:
            assert key in text_to_text_component_init_dict


def test_SimilarityBasedLongFormTask_from_config_invalid_categories(
    long_form_similarity_task_config: dict,
) -> None:
    invalid_response_formatter_config = long_form_similarity_task_config.copy()
    invalid_response_formatter_config["response_formatter_config"]["renaming_map"] = {
        "yes": "1",
        "no": "0",
        "maybe": "maybe",
    }
    with pytest.raises(ValidationError):
        _ = longform.SimilarityBasedLongFormTaskConfig.parse_obj(
            invalid_response_formatter_config
        )

    # Checks an error raised if null category not included
    invalid_evaluator_config = long_form_similarity_task_config.copy()
    invalid_evaluator_config["evaluator_config"]["params"]["categories"] = ["yes", "no"]
    with pytest.raises(ValidationError):
        _ = longform.SimilarityBasedLongFormTaskConfig.parse_obj(
            invalid_evaluator_config
        )


def test_SimilarityBasedLongFormTask_from_config_invalid_response_field(
    long_form_similarity_task_config: dict,
) -> None:
    long_form_similarity_task_config["response_formatter_config"][
        "response_field_name"
    ] = "invalid_response"
    with pytest.raises(ValidationError) as exc_info:
        _ = longform.SimilarityBasedLongFormTaskConfig.parse_obj(
            long_form_similarity_task_config
        )

    assert "response_field" in str(exc_info.value)
    assert "not the same as" in str(exc_info.value)


@EnvCtxManager(
    OPENAI_API_KEY="__dummy_openai_api_key__",
    OPENAI_API_BASE="__dummy_openai_api_base__",
    OPENAI_API_VERSION="__dummy_openai_api_version__",
)
def test_SimilarityBasedLongFormTask_run_task(
    make_mock_llm: Callable,
    long_form_similarity_task_config: dict,
) -> None:
    # Mock dataset
    dataset = Dataset.from_dict(
        {
            "index": [0, 1],
            "question": [
                "Is James the son of Kate?",
                "Is Kate the daughter of James?",
            ],
            "context": [
                "James is Kates son",
                "James is Kates son",
            ],
            "answer": ["James is Kates son", "James is Kates son"],
            "target": [
                "1",
                "0",
            ],
        }
    )

    # Mock LLM in text_to_text_component
    expected_scoring_responses = [
        "yes",
        "no",
    ]
    scoring_llm = make_mock_llm(expected_scoring_responses)
    mock_text_to_text_component = build_text_to_text_module(
        long_form_similarity_task_config["text_to_text_component_config"]
    )
    mock_text_to_text_component.llm = scoring_llm

    # Define task
    long_form_similarity_task = longform.SimilarityBasedLongFormTask(
        dataset=dataset,
        prompt_formatter=ColumnPromptFormatter.from_config(
            long_form_similarity_task_config["prompt_formatter_config"]
        ),
        response_component=ResponseComponent.from_config(
            long_form_similarity_task_config["response_component_config"]
        ),
        text_to_text_component=mock_text_to_text_component,
        response_formatter=MappingResponseFormatter.from_config(
            long_form_similarity_task_config["response_formatter_config"]
        ),
        evaluator=build_eval_component(
            long_form_similarity_task_config["evaluator_config"]
        ),
        response_field_name=long_form_similarity_task_config["response_field_name"],
        response_quality_evaluator=ResponseQualityEvalComponent.from_config(
            long_form_similarity_task_config["response_quality_evaluator_config"]
        ),
        dataset_filterer=None,
    )

    # Define mock LLM
    responses = [
        "Yes James is Kates son",
        "No James is Kates son",
        "This is some random text",
        "This is also some random text",
    ]
    mock_llm = make_mock_llm(responses)

    # Run
    task_output = long_form_similarity_task.run_task(mock_llm)

    expected_dataset = Dataset.from_dict(
        {
            "index": [0, 1],
            "question": [
                "Is James the son of Kate?",
                "Is Kate the daughter of James?",
            ],
            "context": [
                "James is Kates son",
                "James is Kates son",
            ],
            "answer": ["James is Kates son", "James is Kates son"],
            "target": [
                "1",
                "0",
            ],
            "prompt": [
                "Question: Is James the son of Kate?, Context: James is Kates son",
                "Question: Is Kate the daughter of James?, Context: James is Kates son",
            ],
            "unformatted_response": [
                "Yes James is Kates son",
                "No James is Kates son",
            ],
            "evaluation_prompt": [
                "Dummy template to assess the similarity of Yes James is Kates son and James is Kates son",
                "Dummy template to assess the similarity of No James is Kates son and James is Kates son",
            ],
            "evaluation_response": ["yes", "no"],
            "response": ["1", "0"],
        }
    )
    # Assert
    for ii, prompt in enumerate(expected_dataset["prompt"]):
        assert mock_llm.generate.call_args_list[ii].kwargs == {"prompt": prompt}
    for ii, prompt in enumerate(expected_dataset["evaluation_prompt"]):
        assert scoring_llm.generate.call_args_list[ii].kwargs == {"prompt": prompt}

    expected_task_output_metrics = {
        "recall": 1.0,
        "f1": 1.0,
        "precision": 1.0,
        "quality_eval/unformatted_response": {"mean": 6.0, "median": 6.0, "std": 0.0},
        "quality_eval/answer": {"mean": 5.0, "median": 5.0, "std": 0.0},
    }
    assert task_output.metrics == expected_task_output_metrics
    assert task_output.artifacts == {
        "response_quality_eval": {
            "unformatted_response": {"mean": 6.0, "median": 6.0, "std": 0.0},
            "answer": {"mean": 5.0, "median": 5.0, "std": 0.0},
        },
        "results": {"recall": 1.0, "f1": 1.0, "precision": 1.0},
        "label_encoder": {"1": 1, "0": 0, "null": 2},
    }
    for column_name in expected_dataset.column_names:
        assert (
            task_output.datasets["final_dataset"][column_name]
            == expected_dataset[column_name]
        )
