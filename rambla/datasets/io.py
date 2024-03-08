import csv
import json
import logging
from typing import List, Optional, Union

import pandas as pd
from datasets import Dataset, Value, load_dataset
from pydantic import BaseModel, Extra, Field, validator

from rambla.datasets.pubmedqa import balance_pubmedqa, flatten_pubmedqa
from rambla.datasets.utils import add_label_column_for_similarity_task
from rambla.utils.misc import get_dataset_path

# flake8: noqa: E501


MCQA_DATASETS_LIST = [
    "pubmed_qa",
    "flat_pubmed_qa",
    "balanced_pubmed_qa",
]

GENERIC_HF_DATASETS_LIST = [
    "sick",
    "glue_mrpc",
    "pubmed_qa_long_form",
]

LOCAL_STORAGE_DATASETS_LIST = [
    "bioasq",
]


# NOTE: Relative paths
LOCAL_STORAGE_DATASET_DICT = {
    "bioasq": {
        "train": "bioasq_training11b.json",
    },
}

AVAILABLE_DATASETS = (
    MCQA_DATASETS_LIST + GENERIC_HF_DATASETS_LIST + LOCAL_STORAGE_DATASETS_LIST
)


logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logger = logging.getLogger(__file__)


class DatasetParams(BaseModel):
    """Example usage:

    Example 1.
    ```python
    DatasetParams(
        path="pubmed_qa",
        subset="pqa_labeled",
        split="train",
    )
    ```
    """

    path: Optional[str] = None
    subset: Optional[str] = None
    split: Optional[str] = None
    extra: Optional[dict] = None


class GenericDatasetConfig(BaseModel):
    name: str
    params: DatasetParams
    index_field: Optional[str]
    question_field: Optional[str]
    context_field: Optional[str]
    answer_field: Optional[str]  # Long form text answers
    target_field: Optional[str]  # Final label for evalaution
    # categories_to_keep required for
    # irrelevant_context_different_dataset_task_config
    categories_to_keep: Optional[list]

    class Config:  # noqa: D106
        extra = Extra.forbid


class MCQADatasetConfig(BaseModel):
    name: str
    params: DatasetParams
    index_field: str

    question_field: str = Field(
        "question", description="dataset column name for question"
    )
    target_field: str = Field(..., description="dataset column name for label")
    categories_to_keep: List[str] = None

    class Config:  # noqa: D106
        extra = Extra.forbid

    @validator("categories_to_keep")
    @classmethod
    def validate_categories_to_keep(cls, v, values, **kwargs) -> List[str]:
        if v:
            error_msg = (
                "If `categories_to_keep` are provided, "
                "a `target_field` needs to be provided."
            )
            assert values["target_field"] is not None, error_msg
        return v


def _load_pubmedqa(params: DatasetParams) -> Dataset:
    if params.split.startswith("train"):
        return load_dataset(path=params.path, name=params.subset, split=params.split)
    elif params.split.startswith(("validation", "test")):
        if params.subset not in ["pqa_labeled"]:
            raise ValueError(
                f"Only `pqa_labeled` supported. Found {params.subset} instead"
            )
        return load_dataset(path=params.path, split=params.split)
    else:
        raise ValueError(f"params split {params.split} not supported.")


def _get_str_from_list_of_dict(row: List):
    str = ""
    for dict in row:
        str += dict["text"]
    return str


def _load_bioasq_dataframe(params: DatasetParams) -> pd.DataFrame:
    """Loads the bioasq dataset from a predefined location.

    BIOASQ Task b - biomedical questions in English,
    along with their gold concepts. When loaded from json the "snippets"
    column contains a lits of dicts with contexts from many different
    papers. We combine all of these into one str in a "context" column.
    """
    # LOCAL_STORAGE_DATASET_DICT is used to hard code the paths to
    # these downloaded datasets based on the split requested
    relative_path = LOCAL_STORAGE_DATASET_DICT["bioasq"][params.split]
    path = get_dataset_path() / relative_path

    with open(path, "r") as f:
        data = json.load(f)

    df = pd.json_normalize(data["questions"], meta=["body", "ideal_answer", "snippets"])
    df = df[["body", "ideal_answer", "snippets"]]
    df["ideal_answer"] = df["ideal_answer"].str[0]
    df.rename(columns={"ideal_answer": "answer", "body": "question"}, inplace=True)
    # Combine all contexts into one string
    df["context"] = df.apply(
        lambda x: _get_str_from_list_of_dict(x["snippets"]), axis=1
    )
    # Allow for subset of data to be tested
    if params.subset:
        df = df.iloc[: int(params.subset)]
    dataset = Dataset.from_pandas(df)
    # Add label for final eval
    dataset = add_label_column_for_similarity_task(dataset)
    return dataset


def prepare_mcqa_dataset(dataset_config: dict) -> Dataset:
    """Prepares HF dataset according to config.

    TODO: Consider adding a post-loading step that can do filtering/slicing etc

    1. Loads the request dataset
    2. Filters dataset based on `target_field` and `categories_to_keep`
    3. Balances dataset (so equal data points of each class) if required.
    """
    config = MCQADatasetConfig.parse_obj(dataset_config)

    if config.name == "pubmed_qa":
        dataset = _load_pubmedqa(config.params)

        if config.categories_to_keep is not None:
            dataset = dataset.filter(
                lambda x: x[config.target_field] in config.categories_to_keep
            )
    elif config.name == "flat_pubmed_qa":
        dataset = _load_pubmedqa(config.params)
        dataset = flatten_pubmedqa(dataset)

        if config.categories_to_keep is not None:
            dataset = dataset.filter(
                lambda x: x[config.target_field] in config.categories_to_keep
            )
    elif config.name == "balanced_pubmed_qa":
        dataset = _load_pubmedqa(config.params)
        dataset = flatten_pubmedqa(dataset)

        if (not config.categories_to_keep) or (
            set(config.categories_to_keep) != set(["yes", "no"])
        ):
            logger.info(
                f"""Balancing dataset categories is currently only supported when
                    categories_to_keep is ["yes", "no"]. As you supplied
                    {config.categories_to_keep}, this value will be overwritten."""
            )
            config.categories_to_keep = ["yes", "no"]

        dataset = dataset.filter(
            lambda x: x[config.target_field] in config.categories_to_keep
        )
        dataset = balance_pubmedqa(
            dataset, config.categories_to_keep, config.target_field
        )
    else:
        raise ValueError(
            f"{config.name} not recognised. "
            f"Available datasets are: {MCQA_DATASETS_LIST}."
        )

    return dataset


def process_pubmed_qa_long_form(dataset: Dataset) -> Dataset:
    """Processes the pubmed_qa dataset for the similarity_long_from task

    1. Flatten pubmedqa
    2. Add label of 1's for final eval (needed as no ground truth
    in dataset but we want all LLM outputs to be similar)
    """
    dataset = flatten_pubmedqa(dataset)
    dataset = dataset.rename_column("long_answer", "answer")
    dataset = add_label_column_for_similarity_task(dataset)
    return dataset


def prepare_generic_hf_dataset(dataset_config: dict) -> Dataset:  # noqa: D103
    config = GenericDatasetConfig.parse_obj(dataset_config)
    if config.name in (GENERIC_HF_DATASETS_LIST):
        dataset = load_dataset(
            path=config.params.path,
            name=config.params.subset,
            split=config.params.split,
        )

        if config.name == "glue_mrpc":
            dataset = dataset.rename_column("sentence1", "text_1")
            dataset = dataset.rename_column("sentence2", "text_2")
            dataset = dataset.cast_column("label", Value("string"))
        if config.name == "sick":
            dataset = dataset.rename_column("sentence_A", "text_1")
            dataset = dataset.rename_column("sentence_B", "text_2")
            dataset = dataset.cast_column("label", Value("string"))
            """NOTE: We map "contradiction" and "neutral" to the same class.
            Original:
                0 - Entailment
                1 - Neutral
                2 - Contradiction
            Afterwards:
                0 - Neutral / Contradiction
                1 - Entailment
            """
            label_rename_map = {"0": "1", "1": "0", "2": "0"}
            dataset = dataset.map(
                lambda x: {"label": label_rename_map[x]}, input_columns="label"
            )
        if config.name == "pubmed_qa_long_form":
            dataset = process_pubmed_qa_long_form(dataset)
    else:
        raise ValueError(
            f"{config.name} not recognised. "
            f"Available datasets are: {GENERIC_HF_DATASETS_LIST}."
        )

    return dataset


def prepare_local_dataset(dataset_config: dict) -> Dataset:  # noqa: D103
    config = GenericDatasetConfig.parse_obj(dataset_config)
    if config.name in (LOCAL_STORAGE_DATASETS_LIST):
        splits = ["train", "dev", "test"]
        if config.params.split not in splits:
            raise ValueError(
                f"""{config.params.split} split not in supported.
                Try one of {splits=}"""
            )
        if config.name == "bioasq":
            dataset = _load_bioasq_dataframe(config.params)
    else:
        raise ValueError(
            f"{config.name} not recognised. "
            f"Available datasets are: {GENERIC_HF_DATASETS_LIST}."
        )
    return dataset


def prepare_dataset(config: dict) -> Dataset:  # noqa: D103
    # NOTE: consider allowing basemodel input.
    if config["name"] in MCQA_DATASETS_LIST:
        return prepare_mcqa_dataset(config)
    elif config["name"] in GENERIC_HF_DATASETS_LIST:
        return prepare_generic_hf_dataset(config)
    elif config["name"] in LOCAL_STORAGE_DATASETS_LIST:
        return prepare_local_dataset(config)
    else:
        raise ValueError(
            f"{config['name']=} not supported. Try one of {AVAILABLE_DATASETS=}."
        )
