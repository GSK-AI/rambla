import os
from pathlib import Path
from unittest import mock

import pytest
from datasets import Dataset, load_dataset

from rambla.datasets.pubmedqa import (
    balance_pubmedqa,
    create_flat_pubmedqa,
    create_split_flat_pubmedqa,
    flatten_pubmedqa,
)
from tests.conftest import hf_datasets_are_same


@pytest.fixture
def filtered_pqa_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "final_decision": ["yes", "yes", "yes", "no"],
            "pmid": list("1235"),
            "question": list("abce"),
            "context": [
                "The aim of this experiment is to analyse ",
                "the performance of both models on the PubmedQA dataset with a bias",
                "without context or examples is near random",
                "ending",
            ],
        }
    )


@pytest.fixture
def balanced_pqa_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "final_decision": ["yes", "no"],
            "pmid": list("15"),
            "question": list("ae"),
            "context": [
                "The aim of this experiment is to analyse ",
                "ending",
            ],
        }
    )


@pytest.mark.fileio
@mock.patch("rambla.datasets.pubmedqa.load_dataset")
def test_create_flat_pubmedqa_with_json_output(
    mock_load_dataset, tmpdir, mock_pubmedqa_dataset
):
    mock_load_dataset.return_value = mock_pubmedqa_dataset
    output_dataset = create_flat_pubmedqa(
        "pqa_labeled", output_filepath=Path(tmpdir) / "test.json"
    )

    entry = output_dataset[0]
    assert all(isinstance(value, (str, int)) for _, value in entry.items())
    assert os.path.exists(Path(tmpdir) / "test.json")

    saved_output_dataset = load_dataset(
        "json",
        data_files=str(Path(tmpdir) / "test.json"),
        cache_dir=str(Path(tmpdir) / "alt_hf_cache"),
    )
    assert saved_output_dataset["train"].shape == output_dataset.shape
    assert saved_output_dataset["train"][-1] == output_dataset[-1]


@mock.patch("rambla.datasets.pubmedqa.load_dataset")
def test_create_flat_pubmedqa__without_json_output(
    mock_load_dataset, mock_pubmedqa_dataset
):
    mock_load_dataset.return_value = mock_pubmedqa_dataset
    output_dataset = create_flat_pubmedqa("pqa_labeled")

    entry = output_dataset[0]
    assert all(isinstance(value, (str, int)) for _, value in entry.items())


def test_flatten_pubmedqa(mock_pubmedqa_dataset):
    output_dataset = flatten_pubmedqa(mock_pubmedqa_dataset)

    entry = output_dataset[0]
    assert all(isinstance(value, (str, int)) for _, value in entry.items())


@pytest.mark.fileio
def test_create_split_flat_pubmedqa():
    seed = 1234

    output_dataset = create_split_flat_pubmedqa(
        seed=seed,
        name="pqa_labeled",
        path="pubmed_qa",
    )

    keys = ["validation", "test"]
    for key in keys:
        assert key in output_dataset

        assert len(output_dataset[key]) == 500

    for clss in ["yes", "no", "maybe"]:
        n_clss_in_val = len(
            output_dataset["validation"].filter(lambda x: x["final_decision"] == clss)
        )
        n_clss_in_test = len(
            output_dataset["test"].filter(lambda x: x["final_decision"] == clss)
        )

        assert n_clss_in_val == n_clss_in_test


@pytest.mark.fileio
def test_create_split_flat_pubmedqa_same_seed():
    seed = 1234

    output_dataset_0 = create_split_flat_pubmedqa(
        seed=seed,
        name="pqa_labeled",
        path="pubmed_qa",
    )

    output_dataset_1 = create_split_flat_pubmedqa(
        seed=seed,
        name="pqa_labeled",
        path="pubmed_qa",
    )

    for key in ["validation", "test"]:
        for field in ["pmid", "final_decision"]:
            assert output_dataset_0[key][field] == output_dataset_1[key][field]


@pytest.mark.fileio
def test_create_split_flat_pubmedqa_storing(tmpdir):
    seed = 1234

    path = Path(tmpdir) / "test_dataset_00"
    path.mkdir()

    output_dataset = create_split_flat_pubmedqa(
        seed=seed, name="pqa_labeled", path="pubmed_qa", output_filepath=path
    )

    assert (path / "validation.parquet").is_file()
    assert (path / "test.parquet").is_file()

    loaded_dataset = load_dataset(
        "parquet",
        data_files={
            "validation": str(path / "validation.parquet"),
            "test": str(path / "test.parquet"),
        },
    )

    for key in ["validation", "test"]:
        for field in ["pmid", "final_decision"]:
            assert output_dataset[key][field] == loaded_dataset[key][field]

        split_loaded_dataset = load_dataset(str(path), split=key)

        for field in ["pmid", "final_decision"]:
            split_loaded_dataset[field] == output_dataset[key]


@pytest.mark.fileio
def test_create_split_flat_pubmedqa_valueerror(tmpdir):
    seed = 1234

    path = Path(tmpdir) / "test_dataset"
    path.mkdir()
    _ = create_split_flat_pubmedqa(
        seed=seed, name="pqa_labeled", path="pubmed_qa", output_filepath=path
    )

    assert path.is_dir()

    with pytest.raises(ValueError):
        _ = create_split_flat_pubmedqa(
            seed=seed, name="pqa_labeled", path="pubmed_qa", output_filepath=path
        )


def test_balance_pubmedqa(filtered_pqa_dataset, balanced_pqa_dataset):
    output = balance_pubmedqa(filtered_pqa_dataset, ["yes", "no"], "final_decision")

    assert hf_datasets_are_same(output, balanced_pqa_dataset)
