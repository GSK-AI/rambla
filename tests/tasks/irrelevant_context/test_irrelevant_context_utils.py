from unittest import mock

import numpy as np
import pytest
from datasets import Dataset

from rambla.tasks.irrelevant_context.utils import (
    ContextAugmentingModule,
    DatasetMixerConfig,
    DatasetMixerModule,
    ShufflingModule,
    create_shuffled_copies_of_column,
    merge_lists,
    shuffle_column,
)
from tests.conftest import generate_random_string, hf_datasets_are_same

# flake8: noqa: N802


@pytest.fixture
def dest_field_name() -> str:
    return "dest_field_name"


@pytest.fixture
def source_field_name() -> str:
    return "source_field_name"


@pytest.fixture
def sourse_dataset_config() -> dict:
    return {
        "name": "dummy dataset name",
        "index_field": "id",
        "params": {
            "path": "dummy dataset path",
            "subset": "3.0.0",
            "split": "validation",
        },
    }


@pytest.fixture
def dataset_mixer_config_dict(
    sourse_dataset_config: dict, dest_field_name: str, source_field_name: str
) -> dict:
    return {
        "source_dataset_config": sourse_dataset_config,
        "source_field_name": source_field_name,
        "dest_field_name": dest_field_name,
        "seed": 1234,
    }


@pytest.fixture
def dataset_mixer_config_basemodel(
    dataset_mixer_config_dict: dict,
) -> DatasetMixerConfig:
    return DatasetMixerConfig.parse_obj(dataset_mixer_config_dict)


def test_shuffle_column():
    seed = 1234

    pmids = list(range(100))
    output_column = shuffle_column(column=pmids, seed=seed)

    assert output_column != pmids


def test_shuffle_column_same_seed():
    seed = 1234

    pmids = list(range(100))
    output_column_0 = shuffle_column(column=pmids, seed=seed)
    output_column_1 = shuffle_column(column=pmids, seed=seed)

    assert output_column_0 == output_column_1
    assert output_column_0 != pmids


def test_shuffle_column_different_seed():
    seed_0 = 1234
    seed_1 = 5678

    pmids = list(range(100))
    output_column_0 = shuffle_column(column=pmids, seed=seed_0)
    output_column_1 = shuffle_column(column=pmids, seed=seed_1)

    assert output_column_0 != output_column_1
    assert output_column_0 != pmids
    assert output_column_1 != pmids


@pytest.fixture
def dataset_fixture() -> Dataset:
    n_rows = 100
    return Dataset.from_dict(
        {
            "index": list(range(n_rows)),
            "question": [generate_random_string(56) for _ in range(n_rows)],
        }
    )


def test_ShufflingModule_run(dataset_fixture: Dataset):
    field_name = "question"
    seed = 1234

    shuffling_module = ShufflingModule(field_name=field_name, seed=seed)
    output_dataset = shuffling_module.run(dataset_fixture)

    #
    assert output_dataset[f"unshuffled_{field_name}"] == dataset_fixture[field_name]
    assert output_dataset[field_name] != dataset_fixture[field_name]
    assert output_dataset["index"] == dataset_fixture["index"]

    assert output_dataset.features.keys() == set(
        ["question", "index", "unshuffled_question"]
    )


def test_ShufflingModule_run_same_seed(dataset_fixture: Dataset):
    field_name = "question"
    seed = 1234

    shuffling_module_0 = ShufflingModule(field_name=field_name, seed=seed)
    output_dataset_0 = shuffling_module_0.run(dataset_fixture)

    shuffling_module_1 = ShufflingModule(field_name=field_name, seed=seed)
    output_dataset_1 = shuffling_module_1.run(dataset_fixture)

    #
    assert output_dataset_0[field_name] == output_dataset_1[field_name]
    assert output_dataset_0[field_name] != dataset_fixture[field_name]
    assert output_dataset_1[field_name] != dataset_fixture[field_name]


def test_ShufflingModule_run_different_seed(dataset_fixture: Dataset):
    field_name = "question"

    seed_0 = 1234
    shuffling_module_0 = ShufflingModule(field_name=field_name, seed=seed_0)
    output_dataset_0 = shuffling_module_0.run(dataset_fixture)

    seed_1 = 5678
    shuffling_module_1 = ShufflingModule(field_name=field_name, seed=seed_1)
    output_dataset_1 = shuffling_module_1.run(dataset_fixture)

    #
    assert output_dataset_0[field_name] != output_dataset_1[field_name]
    assert output_dataset_0[field_name] != dataset_fixture[field_name]
    assert output_dataset_1[field_name] != dataset_fixture[field_name]


def test_DatasetMixerModule_smaller_dest_dataset():
    source_field_name = "other_context"
    dest_field_name = "context"
    seed = 1234

    source_dataset = Dataset.from_dict({source_field_name: list("ABCD")})
    dest_dataset = Dataset.from_dict({dest_field_name: list("123")})

    mixer_module = DatasetMixerModule(
        source_dataset=source_dataset,
        source_field_name=source_field_name,
        dest_field_name=dest_field_name,
        seed=seed,
        with_replacement=False,
    )

    #
    output_dataset = mixer_module.run(dest_dataset)

    #
    assert output_dataset.features.keys() == set(["context", "original_context"])
    assert len(set(output_dataset["context"])) == len(output_dataset["context"])
    assert set(output_dataset["context"]).issubset(set(source_dataset["other_context"]))
    assert output_dataset["original_context"] == dest_dataset["context"]


def test_DatasetMixerModule():
    source_field_name = "other_context"
    dest_field_name = "new_context"
    seed = 1234

    source_dataset = Dataset.from_dict({source_field_name: list("ABCD")})
    dest_dataset = Dataset.from_dict({"context": list("12356")})

    mixer_module = DatasetMixerModule(
        source_dataset=source_dataset,
        source_field_name=source_field_name,
        dest_field_name=dest_field_name,
        seed=seed,
        with_replacement=True,
    )

    #
    output_dataset = mixer_module.run(dest_dataset)

    #
    assert output_dataset.features.keys() == set(["context", "new_context"])
    assert set(output_dataset["new_context"]).issubset(
        set(source_dataset["other_context"])
    )
    assert output_dataset["context"] == dest_dataset["context"]


def test_DatasetMixerModule_column_exists():
    source_field_name = "context"
    dest_field_name = "context"
    seed = 1234

    source_dataset = Dataset.from_dict({source_field_name: list("ABCD")})
    dest_dataset = Dataset.from_dict({"context": list("12356")})

    mixer_module = DatasetMixerModule(
        source_dataset=source_dataset,
        source_field_name=source_field_name,
        dest_field_name=dest_field_name,
        seed=seed,
        with_replacement=True,
    )

    #
    output_dataset = mixer_module.run(dest_dataset)

    #
    assert output_dataset.features.keys() == set(["context", "original_context"])
    assert set(output_dataset["context"]).issubset(set(source_dataset["context"]))
    assert output_dataset["original_context"] == dest_dataset["context"]


def test_DatasetMixerModule_same_seed():
    source_field_name = "other_context"
    dest_field_name = "context"
    seed = 1234

    source_dataset = Dataset.from_dict({source_field_name: list("ABCD")})
    dest_dataset = Dataset.from_dict({dest_field_name: list("12356")})

    #
    mixer_module_0 = DatasetMixerModule(
        source_dataset=source_dataset,
        source_field_name=source_field_name,
        dest_field_name=dest_field_name,
        seed=seed,
        with_replacement=True,
    )

    output_dataset_0 = mixer_module_0.run(dest_dataset)

    mixer_module_1 = DatasetMixerModule(
        source_dataset=source_dataset,
        source_field_name=source_field_name,
        dest_field_name=dest_field_name,
        seed=seed,
        with_replacement=True,
    )

    output_dataset_1 = mixer_module_1.run(dest_dataset)

    #
    assert hf_datasets_are_same(output_dataset_0, output_dataset_1)


def test_DatasetMixerModule_different_seed():
    source_field_name = "other_context"
    dest_field_name = "context"

    source_dataset = Dataset.from_dict({source_field_name: list("ABCD")})
    dest_dataset = Dataset.from_dict({dest_field_name: list("12356")})

    #
    seed_0 = 1234
    mixer_module_0 = DatasetMixerModule(
        source_dataset=source_dataset,
        source_field_name=source_field_name,
        dest_field_name=dest_field_name,
        seed=seed_0,
    )

    output_dataset_0 = mixer_module_0.run(dest_dataset)

    seed_1 = 5678
    mixer_module_1 = DatasetMixerModule(
        source_dataset=source_dataset,
        source_field_name=source_field_name,
        dest_field_name=dest_field_name,
        seed=seed_1,
    )

    output_dataset_1 = mixer_module_1.run(dest_dataset)

    #
    assert not hf_datasets_are_same(output_dataset_0, output_dataset_1)


@mock.patch("rambla.tasks.irrelevant_context.utils.prepare_dataset")
def test_DatasetMixerModule_from_config_dict(
    mock_prepare_dataset,
    dataset_mixer_config_dict,
):
    mock_dataset = Dataset.from_dict(
        {dataset_mixer_config_dict["source_field_name"]: ["dummy"]}
    )
    mock_prepare_dataset.return_value = mock_dataset

    module_from_config = DatasetMixerModule.from_config(dataset_mixer_config_dict)
    module_from_init = DatasetMixerModule(
        source_dataset=mock_dataset,
        source_field_name=dataset_mixer_config_dict["source_field_name"],
        dest_field_name=dataset_mixer_config_dict["dest_field_name"],
        seed=dataset_mixer_config_dict["seed"],
    )

    for key in dataset_mixer_config_dict.keys():
        if key in ["source_dataset_config"]:
            continue
        assert getattr(module_from_config, key) == getattr(module_from_init, key)


@mock.patch("rambla.tasks.irrelevant_context.utils.prepare_dataset")
def test_DatasetMixerModule_from_config_basemodel(
    mock_prepare_dataset, dataset_mixer_config_basemodel
):
    mock_dataset = Dataset.from_dict(
        {dataset_mixer_config_basemodel.source_field_name: ["dummy"]}
    )
    mock_prepare_dataset.return_value = mock_dataset

    module_from_config = DatasetMixerModule.from_config(dataset_mixer_config_basemodel)
    module_from_init = DatasetMixerModule(
        source_dataset=mock_dataset,
        source_field_name=dataset_mixer_config_basemodel.source_field_name,
        dest_field_name=dataset_mixer_config_basemodel.dest_field_name,
        seed=dataset_mixer_config_basemodel.seed,
    )

    for key in dataset_mixer_config_basemodel.dict().keys():
        if key in ["source_dataset_config"]:
            continue
        assert getattr(module_from_config, key) == getattr(module_from_init, key)


@mock.patch("rambla.tasks.irrelevant_context.utils.prepare_dataset")
def test_DatasetMixerModule_valueerror(mock_prepare_dataset, dataset_mixer_config_dict):
    mock_dataset = Dataset.from_dict(
        {dataset_mixer_config_dict["source_field_name"]: ["dummy"]}
    )
    mock_prepare_dataset.return_value = mock_dataset

    dataset_mixer_config_dict["source_field_name"] = "__dummy_field_name__"
    with pytest.raises(ValueError) as exc_info:
        DatasetMixerModule.from_config(dataset_mixer_config_dict)

    assert "__dummy_field_name__" in str(exc_info.value)


def test_merge_lists_two_lists():
    lst_a = list("ABCD")
    lst_b = list("abcd")
    data = [lst_a, lst_b]
    separator = ""

    output = merge_lists(data, separator)
    assert output == ["Aa", "Bb", "Cc", "Dd"]


def test_merge_lists_three_lists():
    lst_a = list("ABCD")
    lst_b = list("abcd")
    lst_c = list("1234")
    data = [lst_a, lst_b, lst_c]
    separator = ""

    output = merge_lists(data, separator)
    assert output == ["Aa1", "Bb2", "Cc3", "Dd4"]


def test_create_shuffled_data():
    data = list("ABCD")
    seed = 123

    output = create_shuffled_copies_of_column(
        data=data,
        n_contexts=2,
        position_of_original_context=0,
        seed=seed,
    )

    assert output[0] == data


def test_create_shuffled_data_shuffling_works():
    data = list(np.random.choice(list("ABCD"), 1_000))

    seed = 123

    output = create_shuffled_copies_of_column(
        data=data,
        n_contexts=2,
        position_of_original_context=0,
        seed=seed,
    )

    assert output[0] == data
    assert output[1] != data


def test_create_shuffled_data_shuffling_works_four():
    data = list(np.random.choice(list("ABCD"), 1_000))
    seed = 123

    output = create_shuffled_copies_of_column(
        data=data,
        n_contexts=4,
        position_of_original_context=0,
        seed=seed,
    )

    assert output[0] == data
    assert output[1] != data
    assert output[2] != data
    assert output[2] != output[1]
    assert output[3] != output[1]
    assert output[3] != output[2]


def test_create_shuffled_data_four():
    data = list("ABCDEF")
    seed = 123

    output = create_shuffled_copies_of_column(
        data=data,
        n_contexts=4,
        position_of_original_context=2,
        seed=seed,
    )

    assert output[2] == data


def test_create_shuffled_data_same_seed():
    data = list("ABCD")
    seed = 123

    output_0 = create_shuffled_copies_of_column(
        data=data,
        n_contexts=2,
        position_of_original_context=0,
        seed=seed,
    )

    output_1 = create_shuffled_copies_of_column(
        data=data,
        n_contexts=2,
        position_of_original_context=0,
        seed=seed,
    )

    assert output_0[0] == output_1[0]


@pytest.fixture
def context_augmenting_module_config() -> dict:
    def inner(seed):
        return {
            "n_contexts": 3,
            "position_of_original_context": 0,
            "field_name": "context",
            "seed": seed,
            "separator": "\n\n",
        }

    return inner


@pytest.fixture
def dummy_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "index": list(range(10)),
            "context": list(map(str, range(10))),
        }
    )


@mock.patch("rambla.tasks.irrelevant_context.utils.merge_lists")
@mock.patch("rambla.tasks.irrelevant_context.utils.create_shuffled_copies_of_column")
def test_ContextAugmentingModule_with_mocks(
    mock_create_shuffled_copies_of_column,
    mock_merge_lists,
    dummy_dataset: Dataset,
    context_augmenting_module_config: dict,
):
    config = context_augmenting_module_config(1234)
    augmenting_module = ContextAugmentingModule.from_config(config)

    mock_create_shuffled_copies_of_column.return_value = "dummy return value"
    mock_merge_lists.return_value = list("ABCDEFGHIJ")
    #
    output_dataset = augmenting_module.run(dummy_dataset)

    #
    assert output_dataset["original_context"] == dummy_dataset["context"]
    assert output_dataset["context"] == list("ABCDEFGHIJ")

    mock_create_shuffled_copies_of_column.assert_called_with(
        data=dummy_dataset["context"],
        n_contexts=config["n_contexts"],
        position_of_original_context=config["position_of_original_context"],
        seed=config["seed"],
    )

    mock_merge_lists.assert_called_with(
        data="dummy return value", separator=config["separator"]
    )


def test_ContextAugmentingModule(
    dummy_dataset: Dataset, context_augmenting_module_config: dict
):
    config = context_augmenting_module_config(1234)
    augmenting_module = ContextAugmentingModule.from_config(config)

    #
    output_dataset = augmenting_module.run(dummy_dataset)

    #
    assert output_dataset["original_context"] == dummy_dataset["context"]
    original_data = list(
        map(
            lambda x: x.split(config["separator"])[
                config["position_of_original_context"]
            ],
            output_dataset["context"],
        )
    )
    assert original_data == dummy_dataset["context"]


def test_ContextAugmentingModule_same_seed(
    dummy_dataset: Dataset, context_augmenting_module_config: dict
):
    seed_0 = 1234
    config_0 = context_augmenting_module_config(seed_0)
    augmenting_module_0 = ContextAugmentingModule.from_config(config_0)

    seed_1 = 1234
    config_1 = context_augmenting_module_config(seed_1)
    augmenting_module_1 = ContextAugmentingModule.from_config(config_1)

    #
    output_dataset_0 = augmenting_module_0.run(dummy_dataset)
    output_dataset_1 = augmenting_module_1.run(dummy_dataset)

    #
    assert hf_datasets_are_same(output_dataset_0, output_dataset_1)


def test_ContextAugmentingModule_different_seed(
    dummy_dataset: Dataset, context_augmenting_module_config: dict
):
    seed_0 = 1234
    config_0 = context_augmenting_module_config(seed_0)
    augmenting_module_0 = ContextAugmentingModule.from_config(config_0)

    seed_1 = 1234
    config_1 = context_augmenting_module_config(seed_1)
    augmenting_module_1 = ContextAugmentingModule.from_config(config_1)

    #
    output_dataset_0 = augmenting_module_0.run(dummy_dataset)
    output_dataset_1 = augmenting_module_1.run(dummy_dataset)

    #
    assert hf_datasets_are_same(output_dataset_0, output_dataset_1)
