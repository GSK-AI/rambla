import asyncio
import os
import random
import shutil
import string
import tempfile
from typing import Dict, Generator, Union
from unittest import mock

import numpy as np
import pytest
from datasets import Dataset

from rambla.models.openai_models import OpenaiBaseModel

TEMP_DIR = tempfile.mkdtemp()
LABEL_CATEGORIES = list("ABCDEFG")
MESH_OPTIONS = list("abcdefg")


def pytest_configure(config):
    # Set the environment variable to the temporary local path
    # to avoid writing permission error in CI
    os.environ["HF_HOME"] = TEMP_DIR

    os.environ["NLTK_DATA"] = TEMP_DIR

    import nltk

    nltk.download("punkt")

    from datasets import disable_caching

    disable_caching()


def pytest_unconfigure(config):
    os.environ.pop("HF_HOME", None)
    shutil.rmtree(TEMP_DIR)


def hf_datasets_are_same(dt0, dt1) -> bool:
    dt0_features = dt0.features.keys()
    dt1_features = dt1.features.keys()

    if dt0_features != dt1_features:
        return False

    for feat in list(dt0_features):
        if dt0[feat] != dt1[feat]:
            return False

    return True


def generate_random_string(length: int) -> str:
    str_characters = "!@#$%^&*" + string.ascii_lowercase + string.digits
    return "".join(random.choice(str_characters) for _ in range(length))


def generate_random_context_entry(
    n_contexts: int = 3, n_labels: int = 2, n_meshes: int = 2
) -> dict:
    contexts = [generate_random_string(10) for _ in range(n_contexts)]
    labels = np.random.choice(LABEL_CATEGORIES, size=n_labels).tolist()
    meshes = np.random.choice(MESH_OPTIONS, size=n_meshes).tolist()

    return {"contexts": contexts, "labels": labels, "meshes": meshes}


@pytest.fixture
def mock_response_dict():
    return {"index": 1234, "response": "mock_response"}


@pytest.fixture
def mock_model_dict() -> Dict[str, Union[str, int, float]]:
    return {
        "engine": "text-davinci-003",
        "temperature": 1,
        "top_p": 0.95,
    }


@pytest.fixture
def mock_llm(mock_model_dict: dict):
    from rambla.models.base_model import BaseLLM

    mock_llm = mock.MagicMock(spec=BaseLLM)
    mock_llm.async_generate.return_value = "mock_response"
    mock_llm.generate.return_value = "mock_response"
    mock_llm._model_dict = mock_model_dict

    return mock_llm


@pytest.fixture
def mock_flat_pubmedqa_dataset():
    # NOTE: Must not global import `datasets`/`evaluate` outside of functions within `conftest.py`!  # noqa
    # When `Dataset` got imported, it imports datasets.config, where it assigns constant variables like `HF_DATASETS_CACHE`  # noqa: E501
    # by using environment variables at the import time.
    # Will have to patch all different constant variables if not done in this way  # noqa: E501
    from datasets import Dataset

    n_samples = 10
    long_answer_length = 100

    data_dict = {
        "pmid": np.random.randint(0, 100_000, n_samples).tolist(),
        "question": [
            generate_random_string(long_answer_length) for _ in range(n_samples)
        ],
        "final_decision": [
            "yes",
            "no",
            "yes",
            "no",
            "yes",
            "no",
            "yes",
            "no",
            "yes",
            "no",
        ],
        "context": [
            "first context.",
            "another piece of context.",
            "This context. Spans two sentences.",
            "Fourth context!",
            "Fifth.",
            "6th context.",
            "yes another context.",
            "context context context.",
            "7392320.",
            "$@$£$^$^$.",
        ],
    }

    dataset = Dataset.from_dict(data_dict)
    return dataset


@pytest.fixture
def mock_pubmedqa_dataset():
    from datasets import Dataset

    n_samples = 10
    long_answer_length = 100

    data_dict = {
        "pubid": np.random.randint(0, 100_000, n_samples).tolist(),
        "question": [
            generate_random_string(long_answer_length) for _ in range(n_samples)
        ],
        "context": [generate_random_context_entry() for _ in range(n_samples)],
        "long_answer": [
            generate_random_string(long_answer_length) for _ in range(n_samples)
        ],
        "final_decision": np.random.choice(["yes", "no", "maybe"], n_samples).tolist(),
    }

    dataset = Dataset.from_dict(data_dict)
    return dataset


@pytest.fixture
def mock_dict_of_lists():
    n_samples = 10
    long_answer_length = 100

    data_dict = {
        "pmid": np.random.randint(0, 100_000, n_samples).tolist(),
        "question": [
            generate_random_string(long_answer_length) for _ in range(n_samples)
        ],
    }
    return data_dict


@pytest.fixture
def make_mock_llm():
    def inner(responses: list):
        mock_llm = mock.create_autospec(spec=OpenaiBaseModel, instance=True)
        mock_llm.generate = mock.MagicMock(side_effect=responses)
        mock_llm.async_generate = mock.AsyncMock(side_effect=responses)
        mock_llm._model_dict = {"a": 1}
        mock_llm.is_async = False
        return mock_llm

    return inner


@pytest.fixture
def response_component_config() -> dict:
    return {
        "cache_base_dir": None,
        "response_cache_fname": "response.pkl",
        "max_rate": 4,
        "run_async": False,
        "time_period": 60,
        "backoff_decorator_config": "DEFAULT",
    }


@pytest.fixture
def mock_prompt_dataset() -> Dataset:
    dataset_dict = {
        "prompt": ["first prompt", "second prompt", "third prompt"],
        "question": ["q1", "q2", "q3"],
        "index": ["001", "002", "003"],
        "final_decision": ["yes", "no", "yes"],
    }
    return Dataset.from_dict(dataset_dict)


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def label_field() -> str:
    return "label"


@pytest.fixture
def index_field() -> str:
    return "index"


@pytest.fixture
def text_field_1() -> str:
    return "text_1"


@pytest.fixture
def text_field_2() -> str:
    return "text_2"


@pytest.fixture
def mock_text_to_text_dataset(
    index_field: str,
    text_field_1: str,
    text_field_2: str,
    label_field: str,
) -> Dataset:
    n = 10
    context_length = 20

    return Dataset.from_dict(
        {
            index_field: list(range(n)),
            text_field_1: [generate_random_string(context_length) for _ in range(n)],
            text_field_2: [generate_random_string(context_length) for _ in range(n)],
            label_field: list("0101010101"),
        }
    )
