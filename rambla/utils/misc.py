import functools
import logging
import os
import random
import subprocess
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from rambla.utils.types import LabelDictType

# flake8: noqa: N806


def flatten_dictionary(dictionary: dict):  # noqa: D103
    out = {}

    def flatten(x, key=""):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], key + a + "/")
        elif isinstance(x, list):
            for i, a in enumerate(x):
                flatten(a, key + str(i) + "/")
        else:
            out[key[:-1]] = x

    flatten(dictionary)
    return out


def get_dataset_path() -> Path:
    """Loads the env variable to `pathlib.Path`."""
    try:
        path = Path(os.environ["DATASET_STORAGE_PATH"])
    except KeyError:
        raise KeyError(
            "Please add `DATASET_STORAGE_PATH` in your environment variables."
        )
    return path


def initialize_logger(
    logger_name: str = "", level: int = logging.INFO
) -> logging.Logger:
    """
    Gets logger and assigns its level.

    NOTE: This is a copy from `aiml_kga_utils`

    To set up a root logger, use the following:
        > logging.basicConfig(level=logging.INFO)
        > logger = initialize_logger(logger_name=__name__, level=logging.INFO)

    Parameters
    ----------
    logger_name : str, optional
        Name used in the logger, by default ""
    level : [type], optional
        [description], by default logging.INFO

    Returns
    -------
    logger: logging.Logger
        Formatted logger
    """
    logger = logging.getLogger(logger_name)
    logger.level = level

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter("%(asctime)s  %(name)s  [%(levelname)s]: %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def squeeze_dict_dim(dictionary: dict[str, dict]) -> dict:
    """Merges the first two dimensions of the dict."""
    new_dict = {}
    for k0, v0 in dictionary.items():
        for k1, v1 in v0.items():
            new_key = f"{k0}/{k1}"
            new_dict[new_key] = v1
    return new_dict


def run_cmd(cmd: Union[str, List[str]]) -> subprocess.CompletedProcess:
    """Runs a bash command as a subprocess

    Parameters
    ----------
    cmd : str | List[str]
        Command or list of commands to pass as run arguments

    Raises
    ------
    subprocess.CalledProcessError
        Error captured by stderr of subprocess
    """
    if isinstance(cmd, str):
        cmd = [item for item in cmd.split(" ") if item]

    process = subprocess.run(cmd, capture_output=True, shell=False)
    if process.returncode != 0:
        print(process.stdout.decode())
        print(process.stderr.decode())
        raise subprocess.CalledProcessError(
            returncode=process.returncode,
            cmd=cmd,
            output=process.stdout,
            stderr=process.stderr,
        )
    return process


def validate_tasks(tasks: list[str]):
    """Validates the provided tasks against all availablle tasks."""
    available_tasks = get_available_task_yaml_files()
    for task in tasks:
        if task not in available_tasks:
            raise RuntimeError(f"{task=} not recognised. Try one of {available_tasks=}")


def validate_models(models: list[str]):
    """Validates the provided models against all availablle models."""
    available_models = get_available_model_yaml_files()
    for model in models:
        if model not in available_models:
            raise RuntimeError(
                f"{model=} not recognised. Try one of {available_models=}"
            )


def get_hydra_conf_dir() -> Path:
    """Returns path for hydra conf directory."""
    return Path(__file__).parent.parent.parent / "rambla/conf"


def get_available_model_yaml_files() -> list[str]:
    """Finds all available yaml files defining a model."""
    models_dir = get_hydra_conf_dir() / "model"
    all_models = list(map(lambda x: x.split(".yaml")[0], os.listdir(models_dir)))
    return all_models


def get_available_task_yaml_files() -> list[str]:
    """Finds all available yaml files defining a task."""
    tasks_dir = get_hydra_conf_dir() / "task"
    all_tasks = list(map(lambda x: x.split(".yaml")[0], os.listdir(tasks_dir)))
    return all_tasks


def merge_dicts(dicts: list[dict]) -> dict:
    """Merges a list of dicts."""
    final_dict = {}
    for dictionary in dicts:
        final_dict.update(dictionary)
    return final_dict


class EnvCtxManager:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._prev_values = {}

    def __enter__(self):
        """Capturing previous values and setting the new ones."""
        for key, value in self.kwargs.items():
            self._prev_values[key] = os.environ.get(key)
            os.environ[key] = value

    def __exit__(self, type, value, traceback):
        """Reverting env vars back to their prev state."""
        for key in self.kwargs.keys():
            if self._prev_values[key] is None:
                del os.environ[key]
            else:
                os.environ[key] = self._prev_values[key]

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


def add_prefix_to_dict_keys(dictionary: dict, prefix: str, sep: str = "/") -> dict:
    """Returns a new dict where all keys are prefixed with the prefix."""
    new_dict = {}
    for k, v in dictionary.items():
        new_dict[f"{prefix}{sep}{k}"] = v
    return new_dict


def dict_argmax(d: LabelDictType) -> str:
    """Argmax for dictionary keys based on values."""
    return max(d.keys(), key=lambda x: d[x])


def dict_softmax(d: LabelDictType) -> LabelDictType:
    """Softmax for dictionary."""
    den = np.exp(np.fromiter(d.values(), dtype=float)).sum()
    return {k: np.exp(v) / den for k, v in d.items()}


def make_json_serialisable(obj: Any) -> Any:
    """Makes object json serialisable."""
    if isinstance(obj, np.ndarray):
        dtype = obj.dtype
        if np.issubdtype(dtype, np.integer):
            obj = obj.astype(np.int32)
        obj = obj.tolist()
    elif isinstance(obj, Path):
        obj = obj.as_posix()
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = make_json_serialisable(v)
    elif isinstance(obj, list):
        for ii, item in enumerate(obj):
            obj[ii] = make_json_serialisable(item)
    elif isinstance(obj, (int, float, str)):
        pass
    elif isinstance(obj, np.integer):
        obj = int(obj)
    elif isinstance(obj, type(None)):
        pass
    elif isinstance(obj, type) and issubclass(obj, Exception):
        obj = obj.__name__
    else:
        raise TypeError(f"{type(obj)} can't be handled.")
    return obj


def seed_everything(random_seed: int) -> None:
    """Seeds everything."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def generate_report_from_cmat(cmat: np.ndarray) -> pd.Series:
    """Compute several metrics from confusion matrix."""
    if list(cmat.shape) != [2, 2]:
        raise ValueError(
            "`cmat` needs to have shape `[2, 2]` but found " f"cmat with {cmat.shape=}"
        )

    TN, FP, FN, TP = cmat.ravel()

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    report = pd.Series(
        {
            "TPR": TPR,
            "TNR": TNR,
            "PPV": PPV,
            "NPV": NPV,
            "FPR": FPR,
            "FNR": FNR,
            "FDR": FDR,
            "ACC": ACC,
        }
    )

    return report


def split_text_on_sep(
    text: str, sequence_sep: str, expected_sequences: Optional[int] = None
) -> list[str]:
    """Splits a text string into sequences using a sequence_sep string"""
    if sequence_sep not in text:
        raise ValueError(f"No separator '{sequence_sep}' found in input: {text}")
    sequences = text.split(sequence_sep)
    if expected_sequences and len(sequences) > expected_sequences:
        raise ValueError(
            f"Invalid number of sequences found: {len(sequences)}."
            f"Input must contain {expected_sequences} sequences"
        )
    return sequences


def list_of_dicts_to_dict_of_lists(
    dict_list: List[Dict[str, Any]]
) -> Dict[str, List[Any]]:
    """Converts list of dicts into dict of lists"""
    all_keys = set(chain.from_iterable(dict_list))
    if not all([set(d.keys()) == all_keys for d in dict_list]):
        raise KeyError("All dictionaries in list must contain the same keys")

    artifacts_dict = {k: [] for k in all_keys}
    for i_dict in dict_list:
        for k, v in i_dict.items():
            artifacts_dict[k].append(v)

    return artifacts_dict


def convert_dict_key_to_str(d: dict) -> dict:
    """Converts all dict keys to str for mlflog logging"""
    keys_to_remove = []
    new_kwargs = {}

    d_copy = deepcopy(d)

    for k, v in d_copy.items():
        new_v = v
        if isinstance(v, dict):
            new_v = convert_dict_key_to_str(v)
            d_copy[k] = new_v

        if not isinstance(k, str):
            keys_to_remove.append(k)
            new_kwargs[str(k)] = new_v

    for k in keys_to_remove:
        del d_copy[k]

    d_copy.update(new_kwargs)

    return d_copy


def prepare_dicts_for_logging(
    *,
    eval_results: dict,
    quality_eval: Optional[dict] = None,
) -> tuple[dict, dict]:
    """Util function for packing together metrics."""
    to_log_as_metrics = {}
    to_log_as_dicts = {}

    to_log_as_metrics.update(
        {
            k: v
            for k, v in eval_results["results"].items()
            if isinstance(v, (int, float, dict, list, np.ndarray))
        }
    )
    if quality_eval:
        to_log_as_metrics.update(
            add_prefix_to_dict_keys(quality_eval, "quality_eval", "/")
        )

    if "label_encoder" in eval_results:
        to_log_as_dicts["label_encoder"] = make_json_serialisable(
            eval_results["label_encoder"]
        )
    return to_log_as_metrics, to_log_as_dicts


def get_available_text_to_text_task_yaml_files() -> list[str]:
    """Finds all available yaml files defining a task."""
    tasks_dir = get_hydra_conf_dir() / "text_to_text_task"
    all_text_to_text_tasks = list(
        map(lambda x: x.split(".yaml")[0], os.listdir(tasks_dir))
    )
    return all_text_to_text_tasks


def validate_text_to_text_tasks(tasks: list[str]):
    """Validates the provided text_to_text_tasks against all availablle text_to_text_tasks."""
    available_text_to_text_tasks = get_available_text_to_text_task_yaml_files()
    for task in tasks:
        if task not in available_text_to_text_tasks:
            raise RuntimeError(
                f"{task=} not recognised. Try one of {available_text_to_text_tasks=}"
            )


def get_available_text_to_text_components_yaml_files() -> list[str]:
    """Finds all available yaml files defining a text_to_text_components."""
    components_dir = get_hydra_conf_dir() / "text_to_text_component"
    all_components = list(
        map(lambda x: x.split(".yaml")[0], os.listdir(components_dir))
    )
    return all_components


def validate_text_to_text_components(components: list[str]):
    """Validates the provided text_to_text_components against all availablle text_to_text_components."""
    available_components = get_available_text_to_text_components_yaml_files()
    for component in components:
        if component not in available_components:
            raise RuntimeError(
                f"{component=} not recognised. Try one of {available_components=}"
            )


def compute_class_counts_from_confmat(
    confmat: np.ndarray,
    label_encoder: dict[str, int],
) -> dict[str, int | float]:
    """Computes totals and proportions of each class for predictions and targets."""
    preds = confmat.sum(axis=1)
    targets = confmat.sum(axis=0)
    n_total = confmat.sum()

    out = {}
    for key, value in label_encoder.items():
        out[f"n_pred_{key}"] = preds[value]
        out[f"prop_pred_{key}"] = preds[value] / n_total

        out[f"n_target_{key}"] = targets[value]
        out[f"prop_target_{key}"] = targets[value] / n_total

    return out
