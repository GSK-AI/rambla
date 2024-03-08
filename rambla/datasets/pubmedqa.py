import logging
from pathlib import Path
from typing import List, Literal, Optional, Union

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from datasets.arrow_dataset import Dataset as ArrowDataset

from rambla.datasets.utils import create_twoway_dataset_split

logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logger = logging.getLogger(__file__)


def flatten_pubmedqa(dataset: Dataset) -> Dataset:
    """Flattens pubmedqa dataset.

    NOTE: also renamed `pubid` to `pmid`
    NOTE: entries `contexts.reasoning_free_pred` and `contexts.reasoning_requiers_pred`
    left out

    Parameters
    ----------
    dataset : Dataset
        Original pubmedqa dataset to be flattened out

    Returns
    -------
    Dataset
        Flat version of the input dataset
    """
    df = dataset.to_pandas()
    df.rename(columns={"pubid": "pmid"}, inplace=True)
    df = df.astype({"pmid": str})
    df.set_index("pmid", inplace=True)

    context = df["context"].apply(pd.Series)
    df.drop(columns="context", inplace=True)

    df["context"] = context["contexts"].apply(lambda x: "\n\n".join(x))
    df["labels"] = context["labels"].apply(lambda x: ", ".join(x))
    df["meshes"] = context["meshes"].apply(lambda x: ", ".join(x))
    # NOTE: under "context" pubmedqa also has:
    # `reasoning_free_pred` and `reasoning_required_pred`
    dataset = ArrowDataset.from_pandas(df)
    return dataset


def create_flat_pubmedqa(
    name: Literal["pqa_labeled", "pqa_unlabeled", "pqa_artificial"],
    path: str = "pubmed_qa",
    split: str = "train",
    output_filepath: Optional[Union[str, Path]] = None,
) -> ArrowDataset:
    r"""Creates a flat version of pubmedqa dataset.

    TODO: try to replace pandas functionality

    Flattens out the following fields:
        - `contexts`: concatenates all items using "\n\n"
        - `labels`: concatenates all items using ", "
        - `meshes`: concatenates all items using ", "

    Also renames "pubid" field to "pmid" to be consistent with "nlp_shared_pipeline"

    Parameters
    ----------
    name : str
        Argument for `datasets.load_dataset` function.
        Options: "pqa_labeled", "pqa_unlabeled", "pqa_artificial"
    path : str
        Argument for `datasets.load_dataset` function.
        NOTE: should either be "pubmed_qa" or a path pointing to the dataset.
    split : str
        Argument for `datasets.load_dataset` function.
        Default to be `train`.
    output_filepath : Optional[Union[str, Path]] = None
        Location to store the resulting dataset.
        Default to be `None`

    Returns
    -------
    Dataset
        A huggingface dataset with a flat structure.
    """
    dataset = load_dataset(path=path, name=name, split=split)

    flat_dataset = flatten_pubmedqa(dataset)

    if output_filepath is not None:
        flat_dataset.to_json(output_filepath)
        logger.info(f"Dumped dataset at {output_filepath}")

    return flat_dataset


def create_split_flat_pubmedqa(
    seed: int,
    name: Literal["pqa_labeled", "pqa_unlabeled", "pqa_artificial"],
    path: str = "pubmed_qa",
    output_filepath: Optional[Union[str, Path]] = None,
) -> DatasetDict:
    """Creates val/test stratified splits for PQA.

    Example usage:
    ```python
        dataset_dict = create_split_flat_pubmedqa(
            seed=1234,
            output_filepath="tmp/test",
            name="pqa_labeled",
            path="pubmed_qa"
        )

        dt_dict = load_dataset("tmp/test")
        val_dt = load_dataset("tmp/test", split="validation")
        part_val_dt = load_dataset("tmp/test", split="validation[:100]")

    ```
    Parameters
    ----------
    seed : int
        Seed to use to set random state.
    output_filepath : Union[str, Path]
        Where to store the generated `DatasetDict`.
    name : Literal['pqa_labeled', 'pqa_unlabeled', 'pqa_artificial']
        _description_
    path : str, optional
        _description_, by default "pubmed_qa"

    Returns
    -------
    DatasetDict
        Has two keys: `val` and `test` that each hold
        a stratified split of the dataset.

    Raises
    ------
    ValueError
        If file already exists.
    """
    flat_pqa = create_flat_pubmedqa(
        name=name,
        output_filepath=None,
        path=path,
    )

    pqa_dataset_dict = create_twoway_dataset_split(
        dataset=flat_pqa,
        index_field="pmid",
        target_field="final_decision",
        seed=seed,
    )

    if output_filepath:
        for split_name, dataset in pqa_dataset_dict.items():
            filename = Path(output_filepath) / f"{split_name}.parquet"
            if filename.is_file():
                raise ValueError
            dataset.to_parquet(filename)
    return pqa_dataset_dict


def balance_pubmedqa(
    dataset: Dataset, categories_to_keep: List[str], target_field: str
) -> Dataset:
    """Balances dataset according to the target classes.

    Parameters
    ----------
    dataset : Dataset
        The (possibly unbalanced) dataset.
    categories_to_keep : List[str]
        The list of classes that must be in equal amounts in the final dataset.
    target_field : str
        The column name containing the classes/categories.

    Returns
    -------
    Dataset
        The balanced dataset.
    """
    category_lengths = []
    category_datasets = []
    for category in categories_to_keep:
        category_dataset = dataset.filter(lambda x: x[target_field] == category)
        category_lengths.append(len(category_dataset))
        category_datasets.append(category_dataset)
    min_category_length = min(category_lengths)
    dataset = concatenate_datasets(
        [data.select(range(min_category_length)) for data in category_datasets]
    )
    return dataset
