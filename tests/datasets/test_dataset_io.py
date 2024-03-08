from unittest import mock

import pytest

from rambla.datasets.io import (
    DatasetParams,
    _get_str_from_list_of_dict,
    _load_pubmedqa,
    prepare_dataset,
    prepare_generic_hf_dataset,
    prepare_local_dataset,
    prepare_mcqa_dataset,
    process_pubmed_qa_long_form,
)
from tests.conftest import hf_datasets_are_same


@pytest.mark.parametrize("subset", ["pqa_labeled", "pqa_unlabeled", "pqa_artificial"])
@pytest.mark.parametrize("split", ["train", "train[:100]", "train[:35%]"])
@mock.patch("rambla.datasets.io.load_dataset")
def test_load_pubmedqa_train(mock_load_dataset, subset, split):
    mock_load_dataset.return_value = "dummy return value"
    params = DatasetParams(path="pubmed_qa", subset=subset, split=split)
    output = _load_pubmedqa(params)

    #
    assert output == "dummy return value"
    mock_load_dataset.assert_called_with(
        path="pubmed_qa", name=params.subset, split=params.split
    )


@pytest.mark.parametrize("subset", ["pqa_labeled"])
@pytest.mark.parametrize(
    "split",
    ["validation", "validation[:100]", "validation[:35%]", "test", "test[:100]"],
)
@mock.patch("rambla.datasets.io.load_dataset")
def test_load_pubmedqa_validation_and_test(mock_load_dataset, subset, split):
    mock_load_dataset.return_value = "dummy return value"
    params = DatasetParams(path="dummy path", subset=subset, split=split)
    output = _load_pubmedqa(params)

    #
    assert output == "dummy return value"
    mock_load_dataset.assert_called_with(path="dummy path", split=params.split)


@pytest.mark.parametrize("subset", ["pqa_unlabeled", "pqa_artificial"])
@pytest.mark.parametrize(
    "split", ["validation", "validation[:100]", "test", "test[:100]"]
)
def test_load_pubmedqa_valueerror(subset, split):
    params = DatasetParams(path="pubmed_qa", subset=subset, split=split)

    with pytest.raises(ValueError) as exc_info:
        _load_pubmedqa(params)

    assert (
        str(exc_info.value)
        == f"Only `pqa_labeled` supported. Found {params.subset} instead"
    )


@mock.patch("rambla.datasets.io._load_pubmedqa")
def test_prepare_mcqa_dataset(mock_load_pubmedqa, pqa_dataset, pqa_dataset_config):
    pqa_dataset_config["categories_to_keep"] = None
    mock_load_pubmedqa.return_value = pqa_dataset

    output = prepare_mcqa_dataset(pqa_dataset_config)

    mock_load_pubmedqa.assert_called_with(DatasetParams(**pqa_dataset_config["params"]))
    assert hf_datasets_are_same(output, pqa_dataset)


@mock.patch("rambla.datasets.io._load_pubmedqa")
def test_prepare_mcqa_dataset_with_filtering(
    mock_load_pubmedqa, pqa_dataset, pqa_dataset_config
):
    pqa_dataset_config["categories_to_keep"] = ["yes", "no"]
    mock_load_pubmedqa.return_value = pqa_dataset

    output = prepare_mcqa_dataset(pqa_dataset_config)

    mock_load_pubmedqa.assert_called_with(DatasetParams(**pqa_dataset_config["params"]))
    filtered_dataset = pqa_dataset.filter(
        lambda x: x["final_decision"] in ["yes", "no"]
    )
    assert hf_datasets_are_same(output, filtered_dataset)


@mock.patch("rambla.datasets.io._load_pubmedqa")
@mock.patch("rambla.datasets.io.flatten_pubmedqa")
def test_prepare_mcqa_dataset_flat_pubmed(
    mock_flatten_pubmedqa,
    mock_load_pubmedqa,
    pqa_dataset,
    pqa_dataset_config,
    flat_pqa_dataset,
):
    pqa_dataset_config["name"] = "flat_pubmed_qa"
    mock_load_pubmedqa.return_value = pqa_dataset
    mock_flatten_pubmedqa.return_value = flat_pqa_dataset

    output = prepare_mcqa_dataset(pqa_dataset_config)

    mock_load_pubmedqa.assert_called_with(DatasetParams(**pqa_dataset_config["params"]))
    mock_flatten_pubmedqa.assert_called_with(pqa_dataset)

    assert hf_datasets_are_same(output, flat_pqa_dataset)


@mock.patch("rambla.datasets.io._load_pubmedqa")
@mock.patch("rambla.datasets.io.flatten_pubmedqa")
@mock.patch("rambla.datasets.io.balance_pubmedqa")
def test_prepare_mcqa_dataset_balanced_pubmed_no_cats(
    mock_balance_pubmedqa,
    mock_flatten_pubmedqa,
    mock_load_pubmedqa,
    pqa_dataset,
    pqa_dataset_config,
    flat_pqa_dataset,
    filtered_pqa_dataset,
    balanced_pqa_dataset,
):
    pqa_dataset_config["name"] = "balanced_pubmed_qa"
    pqa_dataset_config["categories_to_keep"] = None
    mock_load_pubmedqa.return_value = pqa_dataset
    mock_flatten_pubmedqa.return_value = flat_pqa_dataset
    mock_balance_pubmedqa.return_value = balanced_pqa_dataset

    output = prepare_mcqa_dataset(pqa_dataset_config)

    mock_load_pubmedqa.assert_called_with(DatasetParams(**pqa_dataset_config["params"]))
    mock_flatten_pubmedqa.assert_called_with(pqa_dataset)

    called_with = mock_balance_pubmedqa.call_args_list[0]
    called_with_dataset = called_with[0][0]
    called_with_categories = called_with[0][1]
    called_with_field = called_with[0][2]

    assert called_with_categories == ["yes", "no"]
    assert called_with_field == "final_decision"
    assert hf_datasets_are_same(called_with_dataset, filtered_pqa_dataset)

    assert hf_datasets_are_same(output, balanced_pqa_dataset)


@mock.patch("rambla.datasets.io._load_pubmedqa")
@mock.patch("rambla.datasets.io.flatten_pubmedqa")
@mock.patch("rambla.datasets.io.balance_pubmedqa")
def test_prepare_mcqa_dataset_balanced_pubmed_wrong_cats(
    mock_balance_pubmedqa,
    mock_flatten_pubmedqa,
    mock_load_pubmedqa,
    pqa_dataset,
    pqa_dataset_config,
    flat_pqa_dataset,
    filtered_pqa_dataset,
    balanced_pqa_dataset,
):
    pqa_dataset_config["name"] = "balanced_pubmed_qa"
    pqa_dataset_config["categories_to_keep"] = ["positive", "negative"]
    mock_load_pubmedqa.return_value = pqa_dataset
    mock_flatten_pubmedqa.return_value = flat_pqa_dataset
    mock_balance_pubmedqa.return_value = balanced_pqa_dataset

    output = prepare_mcqa_dataset(pqa_dataset_config)

    mock_load_pubmedqa.assert_called_with(DatasetParams(**pqa_dataset_config["params"]))
    mock_flatten_pubmedqa.assert_called_with(pqa_dataset)

    called_with = mock_balance_pubmedqa.call_args_list[0]
    called_with_dataset = called_with[0][0]
    called_with_categories = called_with[0][1]
    called_with_field = called_with[0][2]

    assert called_with_categories == ["yes", "no"]
    assert called_with_field == "final_decision"
    assert hf_datasets_are_same(called_with_dataset, filtered_pqa_dataset)

    assert hf_datasets_are_same(output, balanced_pqa_dataset)


def test_prepare_mcqa_dataset_error(pqa_dataset_config):
    pqa_dataset_config["name"] = "anything"

    with pytest.raises(ValueError) as exc_info:
        _ = prepare_mcqa_dataset(pqa_dataset_config)
    assert "anything" in str(exc_info.value)


@mock.patch("rambla.datasets.io.prepare_mcqa_dataset")
@mock.patch("rambla.datasets.io.prepare_generic_hf_dataset")
def test_prepare_dataset_mcqa(mock_prepare_generic, mock_prepare_mcqa):
    mock_prepare_mcqa.return_value = "dummy output"
    config = {"name": "pubmed_qa", "params": {"other": "t"}}
    output = prepare_dataset(config)

    #
    assert output == "dummy output"
    mock_prepare_generic.assert_not_called()
    mock_prepare_mcqa.assert_called_once_with(config)


@pytest.mark.parametrize(
    "config",
    [
        {
<<<<<<< HEAD
            "name": "cnn_dailymail",
            "params": {"other": "t"},
        },
        {"name": "bigbio_biosses", "params": {"other": "t"}},
=======
            "name": "pubmed_qa_long_form",
            "params": {"other": "t"},
        },
>>>>>>> master
    ],
)
@mock.patch("rambla.datasets.io.prepare_mcqa_dataset")
@mock.patch("rambla.datasets.io.prepare_generic_hf_dataset")
def test_prepare_dataset_generic(mock_prepare_generic, mock_prepare_mcqa, config):
    mock_prepare_generic.return_value = "dummy output"

    #
    output = prepare_dataset(config)

    #
    assert output == "dummy output"
    mock_prepare_mcqa.assert_not_called()
    mock_prepare_generic.assert_called_once_with(config)


@mock.patch("rambla.datasets.io.prepare_mcqa_dataset")
@mock.patch("rambla.datasets.io.prepare_generic_hf_dataset")
def test_prepare_dataset_not_found(mock_prepare_generic, mock_prepare_mcqa):
    config = {"name": "dummy", "params": {"other": "t"}}

    with pytest.raises(ValueError) as exc_info:
        prepare_dataset(config)

    #
    assert "dummy" in str(exc_info.value)
    mock_prepare_mcqa.assert_not_called()
    mock_prepare_generic.assert_not_called()


@pytest.mark.parametrize(
    "config",
    [
        {
            "name": "pubmed_qa_long_form",
            "params": {
                "path": "pubmed_qa",
                "subset": "pqa_labeled",
                "split": "train",
            },
        },
    ],
)
@mock.patch("rambla.datasets.io.process_pubmed_qa_long_form")
@mock.patch("rambla.datasets.io.load_dataset")
def test_prepare_generic_hf_dataset(
    mock_load_dataset, mock_process_pubmed_qa_long_form, config
):
    mock_load_dataset.return_value = "mock output"
    mock_process_pubmed_qa_long_form.return_value = "mock output 2"

    #
    output = prepare_generic_hf_dataset(config)

    #
    mock_process_pubmed_qa_long_form.assert_called_with("mock output")

    assert output == "mock output 2"
    mock_load_dataset.assert_called_once_with(
        path=config["params"]["path"],
        name=config["params"]["subset"],
        split=config["params"]["split"],
    )


@mock.patch("rambla.datasets.io.load_dataset")
def test_prepare_generic_hf_dataset_not_found(mock_load_dataset):
    config = {
        "name": "dummy name",
        "index_field": "id",
        "params": {
            "path": "dummy path",
            "subset": "1.2.3",
            "split": "validation",
        },
    }

    #
    with pytest.raises(ValueError) as exc_info:
        prepare_generic_hf_dataset(config)

    #
    assert "dummy name" in str(exc_info.value)
    mock_load_dataset.assert_not_called()


@mock.patch("rambla.datasets.io.load_dataset")
def test_prepare_local_dataset_dataset_not_found(mock_load_dataset):
    config = {
        "name": "dummy name",
        "params": {
            "split": "train",
        },
    }

    #
    with pytest.raises(ValueError) as exc_info:
        prepare_local_dataset(config)

    #
    assert "dummy name" in str(exc_info.value)
    mock_load_dataset.assert_not_called()


@pytest.mark.fileio
def test_process_pubmed_qa_long_form(mock_pubmedqa_dataset):
    dataset = process_pubmed_qa_long_form(mock_pubmedqa_dataset)

    entry = dataset[0]

    assert isinstance(entry["answer"], str)
    assert isinstance(entry["context"], str)
    assert isinstance(entry["question"], str)
    assert entry["label"] == "1"


def test_get_str_from_list_of_dict():
    dummy_list_of_dicts = [
        {"XXX": "Ignore this", "text": "Test"},
        {"XXX": "Ignore this", "text": " passed"},
        {"XXX": "Ignore this", "text": "!"},
    ]

    output = _get_str_from_list_of_dict(dummy_list_of_dicts)

    assert output == "Test passed!"
