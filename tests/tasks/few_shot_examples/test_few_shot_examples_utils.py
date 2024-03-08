from copy import deepcopy
from typing import List
from unittest import mock

import pytest
from datasets import Dataset, concatenate_datasets

from rambla.tasks.few_shot_examples.utils import (
    ExamplesGeneratingModule,
    ExamplesGeneratingModuleConfig,
    sample_examples_from_dataset,
)

# flake8: noqa: N802

different_order_inputs = pytest.mark.parametrize(
    "test_order, pos_label, neg_label, pos_counter, neg_counter",
    [
        (["Yes", "Yes", "Yes", "No"], "Yes", "No", 3, 1),
        (["Yes", "No"], "Yes", "No", 1, 1),
        (["yes", "yes", "no", "no"], "yes", "no", 2, 2),
        (["Positive", "Negative", "Positive"], "Positive", "Negative", 2, 1),
        (["Y"], "Y", "N", 1, 0),
    ],
)


@different_order_inputs
def test_sample_examples_from_same_source_dataset(
    mock_flat_pubmedqa_dataset: Dataset,
    test_order,
    pos_label,
    neg_label,
    pos_counter,
    neg_counter,
):
    pos_examples = mock_flat_pubmedqa_dataset(
        seed=1,
        n_samples=20,
        long_answer_length=100,
        context_length=100,
        labels_length=30,
        meshes_length=50,
    )
    neg_examples = mock_flat_pubmedqa_dataset(
        seed=2,
        n_samples=20,
        long_answer_length=100,
        context_length=100,
        labels_length=30,
        meshes_length=50,
    )
    test_qns = concatenate_datasets([pos_examples, neg_examples])

    output = sample_examples_from_dataset(
        pos_examples,
        neg_examples,
        test_order,
        test_qns,
        pos_label=pos_label,
        neg_label=neg_label,
        index_field="pmid",
        source_question_field="question",
        dest_question_field="question",
    )

    assert all(
        [
            (len(set(out).intersection(set(pos_examples["pmid"]))) == pos_counter)
            for out in output
        ]
    )
    assert all(
        [
            (len(set(out).intersection(set(neg_examples["pmid"]))) == neg_counter)
            for out in output
        ]
    )
    assert len(output) == len(test_qns)
    assert all(len(out) == len(test_order) for out in output)
    assert all([(len(set(out)) == len(out)) for out in output])
    assert all(
        [
            (len(set(output[i]).intersection(test_qns[i])) == 0)
            for i in range(len(output))
        ]
    )


@different_order_inputs
def test_sample_examples_from_different_source_dataset(
    mock_flat_pubmedqa_dataset: Dataset,
    mock_balanced_dataset: Dataset,
    test_order,
    pos_label,
    neg_label,
    pos_counter,
    neg_counter,
):
    pos_examples = mock_flat_pubmedqa_dataset(
        seed=1,
        n_samples=20,
        long_answer_length=100,
        context_length=100,
        labels_length=30,
        meshes_length=50,
    )
    neg_examples = mock_flat_pubmedqa_dataset(
        seed=2,
        n_samples=20,
        long_answer_length=100,
        context_length=100,
        labels_length=30,
        meshes_length=50,
    )
    test_qns = mock_balanced_dataset(1, 46)

    output = sample_examples_from_dataset(
        pos_examples,
        neg_examples,
        test_order,
        test_qns,
        pos_label=pos_label,
        neg_label=neg_label,
        index_field="pmid",
        source_question_field="question",
        dest_question_field="question",
    )
    assert all(
        [
            (len(set(out).intersection(set(pos_examples["pmid"]))) == pos_counter)
            for out in output
        ]
    )
    assert all(
        [
            (len(set(out).intersection(set(neg_examples["pmid"]))) == neg_counter)
            for out in output
        ]
    )
    assert len(output) == len(test_qns)
    assert all(len(out) == len(test_order) for out in output)
    assert all([(len(set(out)) == len(out)) for out in output])
    assert all(
        [
            (len(set(output[i]).intersection(test_qns[i])) == 0)
            for i in range(len(output))
        ]
    )


@pytest.mark.parametrize(
    "test_order, pos_label, neg_label",
    [
        (["Yes", "Yes", "Yes", "No"], "yes", "no"),
        (["Yes", "No"], "Yes ", "No"),
        (["no"], "yes", "No"),
    ],
)
def test_sample_examples_from_dataset_error(
    mock_flat_pubmedqa_dataset: Dataset,
    mock_balanced_dataset: Dataset,
    test_order,
    pos_label,
    neg_label,
):
    pos_examples = mock_flat_pubmedqa_dataset(
        seed=1,
        n_samples=10,
        long_answer_length=100,
        context_length=100,
        labels_length=30,
        meshes_length=50,
    )
    neg_examples = mock_flat_pubmedqa_dataset(
        seed=2,
        n_samples=10,
        long_answer_length=100,
        context_length=100,
        labels_length=30,
        meshes_length=50,
    )
    test_qns = mock_balanced_dataset(1, 30)
    with pytest.raises(ValueError) as exc_info:
        _ = sample_examples_from_dataset(
            pos_examples,
            neg_examples,
            test_order,
            test_qns,
            pos_label=pos_label,
            neg_label=neg_label,
            index_field="pmid",
            source_question_field="question",
            dest_question_field="question",
        )

    assert "invalid string" in str(exc_info.value)
    assert str(pos_label) in str(exc_info.value)
    assert str(neg_label) in str(exc_info.value)


@mock.patch("rambla.tasks.few_shot_examples.utils.sample_examples_from_dataset")
def test_examplesmodule_run_same_source(
    mock_sample_examples_from_dataset: Dataset,
    mock_balanced_dataset: Dataset,
    mock_examples_list: List[str],
):
    dest_dataset = mock_balanced_dataset(1, 20)
    input_order = ["yes", "no", "yes"]

    examplesgeneratingmodule = ExamplesGeneratingModule(
        seed=3, order=input_order, index_field="pmid"
    )
    assert examplesgeneratingmodule.source_dataset is None

    mock_sample_examples_from_dataset.return_value = mock_examples_list(
        seed=12, length=len(dest_dataset), order_length=len(input_order)
    )

    output = examplesgeneratingmodule.run(dest_dataset)
    assert examplesgeneratingmodule.source_dataset == dest_dataset

    called_with = mock_sample_examples_from_dataset.call_args_list
    assert len(called_with) == 1
    variables = called_with[0].kwargs
    out_pos_examples = variables["pos_examples"]
    out_neg_examples = variables["neg_examples"]

    dt_pos = examplesgeneratingmodule.source_dataset.filter(
        lambda x: x["question"] in out_pos_examples
    )["final_decision"]
    assert all([k == "yes" for k in dt_pos])

    dt_neg = examplesgeneratingmodule.source_dataset.filter(
        lambda x: x["question"] in out_neg_examples
    )["final_decision"]
    assert all([k == "no" for k in dt_neg])

    assert variables["order"] == input_order
    assert variables["test_qns"] == dest_dataset

    assert len(output.column_names) == len(dest_dataset.column_names) + 1
    new_column = set(output.column_names).difference(set(dest_dataset.column_names))
    assert list(new_column)[0] == "examples"
    assert all([isinstance(out, str) for out in group] for group in output["examples"])


@mock.patch("rambla.tasks.few_shot_examples.utils.sample_examples_from_dataset")
def test_examplesmodule_run_different_source(
    mock_sample_examples_from_dataset: Dataset,
    mock_balanced_dataset: Dataset,
    mock_examples_list: List[str],
):
    dest_dataset = mock_balanced_dataset(1, 20)
    source_dataset = mock_balanced_dataset(2, 30)
    input_order = ["yes", "no", "yes"]

    examplesgeneratingmodule = ExamplesGeneratingModule(
        seed=3,
        order=input_order,
        source_dataset=source_dataset,
        examples_column_name="ex",
        index_field="pmid",
    )
    assert examplesgeneratingmodule.source_dataset == source_dataset

    mock_sample_examples_from_dataset.return_value = mock_examples_list(
        seed=12, length=len(dest_dataset), order_length=len(input_order)
    )
    output = examplesgeneratingmodule.run(dest_dataset)
    assert examplesgeneratingmodule.source_dataset == source_dataset

    called_with = mock_sample_examples_from_dataset.call_args_list
    assert len(called_with) == 1
    variables = called_with[0].kwargs
    out_pos_examples = variables["pos_examples"]
    out_neg_examples = variables["neg_examples"]

    dt_pos = examplesgeneratingmodule.source_dataset.filter(
        lambda x: x["pmid"] in out_pos_examples["pmid"]
    )["final_decision"]
    assert all([k == "yes" for k in dt_pos])

    dt_neg = examplesgeneratingmodule.source_dataset.filter(
        lambda x: x["pmid"] in out_neg_examples["pmid"]
    )["final_decision"]
    assert all([k == "no" for k in dt_neg])

    assert len(dt_pos) + len(dt_neg) == len(examplesgeneratingmodule.source_dataset)

    assert variables["order"] == input_order
    assert variables["test_qns"] == dest_dataset

    assert len(output.column_names) == len(dest_dataset.column_names) + 1
    new_column = set(output.column_names).difference(set(dest_dataset.column_names))
    assert list(new_column)[0] == "ex"
    assert all([isinstance(out, str) for out in group] for group in output["ex"])


@pytest.mark.parametrize("seed", [(1), (24), (133)])
def test_examplesmodule_run_same_seed(mock_balanced_dataset: Dataset, seed):
    dest_dataset = mock_balanced_dataset(1, 20)
    input_order = ["yes", "no", "yes"]

    examplesmodule1 = ExamplesGeneratingModule(
        seed=seed, order=input_order, index_field="pmid"
    )
    examplesmodule2 = ExamplesGeneratingModule(
        seed=seed, order=input_order, index_field="pmid"
    )

    output1 = examplesmodule1.run(dest_dataset)
    output2 = examplesmodule2.run(dest_dataset)

    for i in range(len(output1)):
        example_questions1 = output1["examples"][i]
        example_questions2 = output2["examples"][i]
        assert all(
            example_questions1[j] == example_questions2[j]
            for j in range(len(input_order))
        )


@pytest.mark.parametrize("seed1, seed2", [(1, 2), (24, 30), (133, 3)])
def test_examplesmodule_run_different_seed(
    mock_balanced_dataset: Dataset, seed1, seed2
):
    dest_dataset = mock_balanced_dataset(1, 20)
    input_order = ["yes", "no", "yes"]

    examplesmodule1 = ExamplesGeneratingModule(
        seed=seed1, order=input_order, index_field="pmid"
    )
    examplesmodule2 = ExamplesGeneratingModule(
        seed=seed2, order=input_order, index_field="pmid"
    )

    output1 = examplesmodule1.run(dest_dataset)
    output2 = examplesmodule2.run(dest_dataset)

    for i in range(len(output1)):
        example_questions1 = output1["examples"][i]
        example_questions2 = output2["examples"][i]
        assert any(
            example_questions1[j] != example_questions2[j]
            for j in range(len(input_order))
        )


def test_ExamplesGeneratingModuleConfig_order_validation(
    examples_module_config_no_source: dict,
):
    config = deepcopy(examples_module_config_no_source)

    config["order"] = ["yes", "yes", "q"]
    with pytest.raises(ValueError) as exc_info:
        _ = ExamplesGeneratingModuleConfig.parse_obj(config)

    assert str(config["order"]) in str(exc_info.value)
    assert str("yes") in str(exc_info.value)
