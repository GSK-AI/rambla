import hashlib
import re
from pathlib import Path
from typing import Any, Dict

from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
from omegaconf import DictKeyType
from sklearn import metrics as sklearn_metrics

from rambla.tasks.base import RunTaskReturnType
from rambla.utils.caching import json_serialise
from rambla.utils.io import dump
from rambla.utils.misc import make_json_serialisable


class RunExistsError(Exception):
    def __init__(self, run_id: int):
        self.run_id = run_id
        message = f"A run with ID {run_id} already exists"
        super().__init__(message)


def get_task_name(key: str) -> str:
    """Gets the name of the task passed to hydra"""
    hydra_cfg = HydraConfig.get()
    choices = dict(hydra_cfg.runtime.choices)

    if key in choices:
        return hydra_cfg.runtime.choices[key]
    else:
        raise KeyError(f"{key=} parameter not defined")


def compute_config_hash(config: Dict[DictKeyType, Any]) -> str:
    """Computes hash for the provided config."""
    serialized_config = json_serialise(config)
    hash_key = hashlib.md5()
    hash_key.update(serialized_config.encode())
    return hash_key.hexdigest()


def get_run_id(config_dir: Path) -> str:
    """Iterates run_id if existing runs found"""
    subdir_names = [d.name for d in config_dir.rglob("*") if d.is_dir()]
    if not subdir_names:
        return "000"
    numeric_subdirs = sorted(
        [int(sdir) for sdir in subdir_names if re.match(r"^[0-9]{3}$", sdir)]
    )
    next_value = max(numeric_subdirs) + 1
    return f"{next_value:03d}"


def get_save_path(task_name: str, config: Dict[DictKeyType, Any]) -> Path:
    """Gets a directory path based on task and config hash to save metrics to"""
    if "metrics_save_dir" not in config.keys():
        raise KeyError(
            f"Config contains keys: {list(config.keys())}, "
            "must contain `metrics_save_dir`"
        )

    # Check whether a `run_id` is set in the config
    run_id = config.get("run_id")
    if run_id:
        if run_id.isnumeric():
            raise ValueError(
                f"`run_id`: {run_id} is invalid. "
                "Must provide a non-numeric ID when manually setting `run_id`"
            )
        # Remove run_id so it doesn't affect the config hash
        config = {i: config[i] for i in config if i != "run_id"}

    config_hash = compute_config_hash(config)

    config_dir = Path(config["metrics_save_dir"]) / task_name / config_hash
    if not run_id:
        run_id = get_run_id(config_dir)

    save_path = config_dir / run_id

    if save_path.is_dir():
        raise RunExistsError(run_id)

    save_path.mkdir(parents=True, exist_ok=True)

    return save_path


# TODO: needs to be tested
def store_task_output(
    task_output: RunTaskReturnType,
    save_path: Path,
):  # noqa: D103
    # metrics
    if task_output.metrics:
        metrics = make_json_serialisable(task_output.metrics)
        dump(metrics, save_path / "metrics.json")

    # artifacts
    artifacts_dict = task_output.artifacts
    if artifacts_dict:
        artifacts_extension = task_output.artifact_storing_format
        if artifacts_extension == "json":
            artifacts_dict = make_json_serialisable(artifacts_dict)

        for key, value in artifacts_dict.items():
            dump(value, save_path / f"artifact_{key}.{artifacts_extension}")

    # datasets
    datasets_dict = task_output.datasets
    if datasets_dict:
        for key, value in datasets_dict.items():
            value.to_json(save_path / f"{key}.json")

    # plots
    plots_dict = task_output.plots
    if plots_dict:
        for key, value in plots_dict.items():
            fig, ax = plt.subplots()
            value.plot(ax=ax)
            plt.savefig(save_path / f"{key}.png")

    # other
    if task_output.other:
        for key, value in task_output.other.items():
            dump(value, save_path / f"{key}.pkl")


def create_plots(task_output: RunTaskReturnType) -> RunTaskReturnType:
    """Creates plots and adds them to task_output"""
    results = task_output.artifacts["results"]
    if "roc_curve" in results and "auc" in results:
        display = sklearn_metrics.RocCurveDisplay(
            fpr=results["roc_curve"]["fpr"],
            tpr=results["roc_curve"]["tpr"],
            roc_auc=results["auc"],
        )
        task_output.plots = {"roc_curve": display}

    return task_output
