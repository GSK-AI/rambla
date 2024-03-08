from datasets import Dataset, DatasetDict
from sklearn.model_selection import StratifiedKFold


def create_twoway_dataset_split(
    dataset: Dataset,
    index_field: str,
    target_field: str,
    seed: int,
) -> DatasetDict:
    """Creates a DatasetDict with balanced val and test splits.

    Parameters
    ----------
    dataset : Dataset
        HF dataset to be split.
    index_field : str
        _description_
    target_field : str
        The field used for stratifing the split.
    seed : int
        Seed to use to set random state of splitting.

    Returns
    -------
    DatasetDict
        Has two keys: `val` and `test` that each hold
        a stratified split of the dataset.

    """
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    splits = skf.split(dataset[index_field], dataset[target_field])

    # NOTE: Two-way split; wlog we take the first one.
    val_indices, test_indices = list(splits)[0]

    val_dataset = dataset.select(val_indices)
    test_dataset = dataset.select(test_indices)

    dataset_dict = DatasetDict({"validation": val_dataset, "test": test_dataset})

    return dataset_dict


def add_label_column_for_similarity_task(
    dataset: Dataset, column_name: str = "label"
) -> Dataset:
    """Adds a column of 1s

    Function to add a label column consisting of all 1's
    to datasets for the similarity_long_form task where
    we would aim for all LLM outputs to be similar
    """
    new_column = "1" * len(dataset)
    dataset = dataset.add_column(column_name, new_column)
    return dataset
