import pytest
from datasets import Dataset

from rambla.datasets.utils import (
    add_label_column_for_similarity_task,
    create_twoway_dataset_split,
)


@pytest.fixture
def dataset_fixture() -> Dataset:
    dataset_dict = {
        "final_decision": ["yes", "yes", "no", "no", "maybe", "maybe"],
        "pmid": [0, 1, 2, 3, 4, 5],
    }
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def test_create_twoway_dataset_split(dataset_fixture):
    index_field = "pmid"
    target_field = "final_decision"
    seed = 1234

    output = create_twoway_dataset_split(
        dataset=dataset_fixture,
        index_field=index_field,
        target_field=target_field,
        seed=seed,
    )

    keys = ["validation", "test"]
    for key in keys:
        assert key in output

        assert len(output[key]) == 3

        assert len(output[key].filter(lambda x: x["final_decision"] == "yes")) == 1
        assert len(output[key].filter(lambda x: x["final_decision"] == "no")) == 1
        assert len(output[key].filter(lambda x: x["final_decision"] == "maybe")) == 1


def test_create_twoway_dataset_split_same_seed(dataset_fixture):
    index_field = "pmid"
    target_field = "final_decision"
    seed = 1234

    output_0 = create_twoway_dataset_split(
        dataset=dataset_fixture,
        index_field=index_field,
        target_field=target_field,
        seed=seed,
    )

    output_1 = create_twoway_dataset_split(
        dataset=dataset_fixture,
        index_field=index_field,
        target_field=target_field,
        seed=seed,
    )

    for key in ["validation", "test"]:
        for field in ["pmid", "final_decision"]:
            assert output_0[key][field] == output_1[key][field]


def test_add_label_column_for_similarity_task():
    dataset = Dataset.from_dict({"text_1": ["Dummy statement 1"]})

    dataset = add_label_column_for_similarity_task(dataset)

    expected_dataset_dict = {
        "text_1": ["Dummy statement 1"],
        "label": ["1"],
    }

    expected_dataset = Dataset.from_dict(expected_dataset_dict)

    assert dataset["text_1"] == expected_dataset["text_1"]
    assert dataset["label"] == expected_dataset["label"]
