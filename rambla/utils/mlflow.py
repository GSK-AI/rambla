import logging
import os
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import mlflow
from mlflow.entities import ViewType
from mlflow.entities.experiment import Experiment as MlflowExperiment
from mlflow.entities.run import Run as MlflowRun

from rambla.utils.io import dump
from rambla.utils.misc import (
    add_prefix_to_dict_keys,
    convert_dict_key_to_str,
    flatten_dictionary,
    make_json_serialisable,
    squeeze_dict_dim,
)
from rambla.utils.mlflow_types import MlflowRunDict

logger = logging.getLogger(__file__)


def load_dicts_for_run(
    run: Union[dict, MlflowRun], dict_names: list[str]
) -> dict[str, dict]:
    """Loads dicts that were logged for the run.

    That is, dicts that were logged with `logger.log_dict`

    Parameters
    ----------
    run : Union[dict, MlflowRun]
        An mlflow.entities.run for which we want to load dicts.
    dict_names : list[str]
        The names of dictionaries to be loaded.

    Returns
    -------
    list[dict]
    """
    out = {}
    if isinstance(run, MlflowRun):
        artifact_uri = run.info._artifact_uri
    else:
        artifact_uri = run["info"]["artifact_uri"]

    for name in dict_names:
        try:
            run_stored_dict = mlflow.artifacts.load_dict(f"{artifact_uri}/{name}.json")
        except mlflow.exceptions.MlflowException as err:
            logger.info(f"Caught {err=}")
        else:
            out[name] = run_stored_dict
    return out


def log_artifacts(*, extension: str, **kwargs):
    """Logs artifacts using mlflow logger."""
    if not kwargs:
        return
    home_dir = os.path.expanduser("~")
    base_path = Path(home_dir) / "rambla_tmp" / uuid.uuid4().hex
    base_path.mkdir(parents=True)
    for key, value in kwargs.items():
        parsed = make_json_serialisable(deepcopy(value))
        fpath = base_path / f"{key}.{extension}"
        dump(parsed, fpath)
        mlflow.log_artifact(fpath)


def mlflow_log(
    project_name: str,
    experiment_name: str,
    run_name: Optional[str] = None,
    *,
    tags: Optional[dict[str, str]] = None,
    config: Optional[dict] = None,
    artifacts: Optional[dict] = None,
    metrics: Optional[dict] = None,
    dictionaries: Optional[dict[str, dict]] = None,  # Needs to be json-able!
    extension: str = "json",
):
    """Wrapper around mlflow logging config, metrics and artifacts."""
    if not any([config, artifacts, metrics, dictionaries]):
        return

    if not dictionaries:
        dictionaries = {}

    full_experiment_name = f"{project_name}/{experiment_name}"
    _ = mlflow.set_experiment(experiment_name=full_experiment_name)
    with mlflow.start_run(
        run_name=run_name,
        tags=tags,
    ):
        if artifacts:
            log_artifacts(extension=extension, **artifacts)

        if metrics:
            # NOTE: Any entry that's a dict will _not_ be logged as a metric
            # but will instead be logged as a dict.
            metrics_copy = deepcopy(metrics)
            to_remove = []
            for k, v in metrics.items():
                if not isinstance(v, (int, float)):
                    to_remove.append(k)

            if to_remove:
                if not dictionaries:
                    dictionaries = {}
                if "metrics" not in dictionaries:
                    dictionaries["metrics"] = {}

                for k in to_remove:
                    del metrics_copy[k]
                    dictionaries["metrics"][k] = metrics[k]

            mlflow.log_metrics(metrics_copy)

        if dictionaries:
            for key, value in dictionaries.items():
                mlflow.log_dict(make_json_serialisable(value), f"{key}.json")

            # NOTE: This will make loading the dicts easier.
            if not config:
                config = {}

            if "logged_dicts" not in config:
                config["logged_dicts"] = list(dictionaries.keys())

        if config:
            # Ensure all dict keys are of type str
            parsed_config = convert_dict_key_to_str(config)
            flat_config = flatten_dictionary(parsed_config)
            mlflow.log_params(flat_config)


def find_experiments(project_name: str) -> list[MlflowExperiment]:
    """Finds all experiments under a project."""
    # full_experiment_name = "/{project_name}/{experiment_name}"
    filter_str = f"name ILIKE '/{project_name}%'"
    experiments = mlflow.search_experiments(
        view_type=ViewType.ALL, filter_string=filter_str
    )
    return experiments


def prepare_run_dict(
    run: Union[MlflowRunDict, MlflowRun],
    experiment_name: str,
    keys_to_expand: list[str],
) -> dict:
    """Prepares an Mlflow run with other useful information.

    - Loads any dicts logged with `.log_dict`
    - Expands nested dicts.


    NOTE: `run` is a nested dict (see `MlflowRunDict`)
        - info: dict
            artifact_uri: str
            end_time: int
            experiment_id: str
            lifecycle_stage: str
            run_id: str
            run_name: str
            run_uuid: str
            start_time: int
            status: str
            user_id: str
            experiment_name: str
        - data: dict
            metrics: dict
            params: dict
            tags: dict
        - inputs: dict

    This function will flatten this dict of dicts
    my combining their keys.

    The output `squeezed_run_dict` will contain:
    `info/artifact_uri` -> run["info"]["artifact_uri"]
    ...
    `info/data/metrics/{key}` -> {value} for all key,value
    pairs inside `info["data"]["metrics"]`.

    """
    if isinstance(run, MlflowRun):
        run: MlflowRunDict = run.to_dictionary()  # type: ignore
    run["info"]["experiment_name"] = experiment_name

    # If any dicts were logged, then we load them
    dict_names = [
        value
        for key, value in run["data"]["params"].items()
        if key.startswith("logged_dicts")
    ]
    if dict_names:
        loaded_dicts = load_dicts_for_run(run, dict_names)
        for k, v in loaded_dicts.items():
            if k in run["data"]:
                run["data"][k].update(loaded_dicts[k])
            else:
                run["data"][k] = loaded_dicts[k]

    # We flatten the first two dims of the dictionary to make the final
    # dataframe easier to manipulate
    squeezed_run_dict = squeeze_dict_dim(run)
    for key in keys_to_expand:
        squeezed_run_dict.update(add_prefix_to_dict_keys(squeezed_run_dict[key], key))

    for key in keys_to_expand:
        squeezed_run_dict.pop(key)

    return squeezed_run_dict
