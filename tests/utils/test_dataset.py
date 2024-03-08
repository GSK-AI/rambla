import numpy as np
import pytest
from datasets import Dataset

from rambla.utils import dataset
from rambla.utils.dataset import DatasetFilterer, DatasetFilteringCondition
from tests.conftest import generate_random_string

# flake8: noqa: N802


def test_from_dict_to_dataset():
    n_samples = 10
    long_answer_length = 100

    data_dict = {
        "pmid": np.random.randint(0, 100_000, n_samples).tolist(),
        "question": [
            generate_random_string(long_answer_length) for _ in range(n_samples)
        ],
    }
    output_dataset = dataset.from_dict_to_dataset(data_dict)

    assert output_dataset["pmid"] == data_dict["pmid"]
    assert output_dataset["question"] == data_dict["question"]

    assert output_dataset.features["pmid"].dtype == "int32"
    assert output_dataset.features["question"].dtype == "string"


@pytest.fixture
def mock_dest_dataset() -> Dataset:
    n_samples = 10

    data_dict = {
        "index": list(range(n_samples)),
        "prompt": [generate_random_string(10) for _ in range(n_samples)],
    }

    return Dataset.from_dict(data_dict)


def test_add_fields_to_dataset_single_field(
    mock_pubmedqa_dataset: Dataset,
    mock_dest_dataset: Dataset,
) -> None:
    field_name = "question"

    new_dataset = dataset.add_fields_to_dataset(
        mock_pubmedqa_dataset, mock_dest_dataset, field_name
    )

    assert "question" in new_dataset.features.keys()
    assert new_dataset["question"] == mock_pubmedqa_dataset["question"]

    # Checks original keys still in dataset
    assert "index" in new_dataset.features.keys()
    assert "prompt" in new_dataset.features.keys()


def test_add_fields_to_dataset_multi_field(
    mock_pubmedqa_dataset: Dataset,
    mock_dest_dataset: Dataset,
) -> None:
    new_field = [generate_random_string(20) for _ in range(10)]

    mock_pubmedqa_dataset = mock_pubmedqa_dataset.add_column(
        "new_field", new_field  # type: ignore
    )

    field_names = ["question", "new_field"]

    new_dataset = dataset.add_fields_to_dataset(
        mock_pubmedqa_dataset, mock_dest_dataset, field_names
    )

    assert "question" in new_dataset.features.keys()
    assert new_dataset["question"] == mock_pubmedqa_dataset["question"]
    assert "new_field" in new_dataset.features.keys()
    assert new_dataset["new_field"] == mock_pubmedqa_dataset["new_field"]

    # Checks original keys still in dataset
    assert "index" in new_dataset.features.keys()
    assert "prompt" in new_dataset.features.keys()


def test_add_fields_to_dataset_invalid_field(
    mock_pubmedqa_dataset: Dataset,
    mock_dest_dataset: Dataset,
) -> None:
    with pytest.raises(KeyError):
        _ = dataset.add_fields_to_dataset(
            mock_pubmedqa_dataset, mock_dest_dataset, "invalid_field"
        )


def test_slice_dataset_start_and_stop(mock_pubmedqa_dataset: Dataset) -> None:
    start_slice = len(mock_pubmedqa_dataset) // 4
    stop_slice = len(mock_pubmedqa_dataset) // 2

    sliced_dataset = dataset.slice_dataset(
        mock_pubmedqa_dataset, start_slice, stop_slice
    )

    assert len(sliced_dataset) == (stop_slice - start_slice)
    assert sliced_dataset["pubid"][0] == mock_pubmedqa_dataset["pubid"][start_slice]
    assert sliced_dataset["pubid"][-1] == mock_pubmedqa_dataset["pubid"][stop_slice - 1]


def test_slice_dataset_start_only(mock_pubmedqa_dataset: Dataset) -> None:
    start_slice = len(mock_pubmedqa_dataset) // 2

    sliced_dataset = dataset.slice_dataset(
        mock_pubmedqa_dataset, start_slice=start_slice
    )

    assert len(sliced_dataset) == len(mock_pubmedqa_dataset) - start_slice
    assert sliced_dataset["pubid"][0] == mock_pubmedqa_dataset["pubid"][start_slice]
    assert sliced_dataset["pubid"][-1] == mock_pubmedqa_dataset["pubid"][-1]


def test_slice_dataset_stop_only(mock_pubmedqa_dataset: Dataset) -> None:
    stop_slice = len(mock_pubmedqa_dataset) // 2

    sliced_dataset = dataset.slice_dataset(mock_pubmedqa_dataset, stop_slice=stop_slice)

    assert len(sliced_dataset) == stop_slice
    assert sliced_dataset["pubid"][0] == mock_pubmedqa_dataset["pubid"][0]
    assert sliced_dataset["pubid"][-1] == mock_pubmedqa_dataset["pubid"][stop_slice - 1]


def test_slice_dataset_invalid_slice(mock_pubmedqa_dataset: Dataset) -> None:
    with pytest.raises(ValueError):
        _ = dataset.slice_dataset(mock_pubmedqa_dataset)

    stop_slice = len(mock_pubmedqa_dataset) + 1

    with pytest.raises(ValueError):
        _ = dataset.slice_dataset(mock_pubmedqa_dataset, stop_slice=stop_slice)


def test_DatasetFilteringCondition_single_exclude():
    dataset = Dataset.from_dict(
        {"id": list("ABCD"), "response": ["hi", "hey", "-1", "hello"]}
    )

    config = {
        "filtering_conditions": {
            "field_name": "response",
            "field_values": ["-1"],
            "filter_out": True,
        }
    }

    dataset_filterer = DatasetFilterer.from_config(config)

    output_dataset = dataset_filterer.run(dataset)

    assert output_dataset["id"] == list("ABD")


def test_DatasetFilteringCondition_single_include():
    dataset = Dataset.from_dict(
        {"id": list("ABCD"), "response": ["hi", "hey", "-1", "hello"]}
    )

    config = {
        "filtering_conditions": {
            "field_name": "response",
            "field_values": ["-1"],
            "filter_out": False,
        }
    }

    dataset_filterer = DatasetFilterer.from_config(config)

    output_dataset = dataset_filterer.run(dataset)

    assert output_dataset["id"] == list("C")


def test_DatasetFilteringCondition_multiple_exclude():
    dataset = Dataset.from_dict(
        {
            "id": list("ABCD"),
            "field_0": ["hi", "hey", "-1", "hello"],
            "field_1": ["dummy", "again", "also", "dummy"],
        }
    )

    config = {
        "filtering_conditions": [
            {
                "field_name": "field_0",
                "field_values": ["-1"],
                "filter_out": True,
            },
            {
                "field_name": "field_1",
                "field_values": ["dummy"],
                "filter_out": True,
            },
        ]
    }

    dataset_filterer = DatasetFilterer.from_config(config)

    output_dataset = dataset_filterer.run(dataset)

    assert output_dataset["id"] == list("B")


def test_DatasetFilteringCondition_mix_exclude_and_include():
    dataset = Dataset.from_dict(
        {
            "id": list("ABCD"),
            "field_0": ["hi", "hey", "-1", "hello"],
            "field_1": ["dummy", "again", "also", "dummy"],
        }
    )

    config = {
        "filtering_conditions": [
            {
                "field_name": "field_0",
                "field_values": ["hi"],
                "filter_out": True,
            },
            {
                "field_name": "field_1",
                "field_values": ["dummy"],
                "filter_out": False,
            },
        ]
    }

    dataset_filterer = DatasetFilterer.from_config(config)

    output_dataset = dataset_filterer.run(dataset)

    assert output_dataset["id"] == list("D")


def test_DatasetFilteringCondition_true():
    config = {
        "field_name": "field_0",
        "field_values": "-1",
        "filter_out": True,
    }

    condition = DatasetFilteringCondition.parse_obj(config)
    entry = {"field_0": 10}

    output = condition(entry)

    assert output


def test_DatasetFilteringCondition_false():
    config = {
        "field_name": "field_0",
        "field_values": "-1",
        "filter_out": False,
    }

    condition = DatasetFilteringCondition.parse_obj(config)
    entry = {"field_0": 10}

    output = condition(entry)

    assert not output


def test_DatasetFilteringCondition_empty_output_dataset():
    dataset = Dataset.from_dict(
        {"id": list("ABCD"), "response": ["hi", "hey", "-1", "hello"]}
    )

    config = {
        "filtering_conditions": {
            "field_name": "response",
            "field_values": ["not present"],
            "filter_out": False,
        }
    }

    dataset_filterer = DatasetFilterer.from_config(config)

    output_dataset = dataset_filterer.run(dataset)

    assert not output_dataset["id"]
