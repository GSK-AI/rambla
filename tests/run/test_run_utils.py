from __future__ import annotations

import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Union
from unittest import mock

import numpy as np
import pytest
from datasets import Dataset
from omegaconf import DictKeyType, OmegaConf

from rambla.models.base_model import BaseLLM
from rambla.run import run_task, utils
from rambla.tasks.base import RunTaskReturnType


def testget_task_name() -> None:
    key = "task"
    mock_choices = {key: "mock_task"}
    mock_hydra_cfg = mock.MagicMock()
    mock_hydra_cfg.runtime.choices = mock_choices
    with mock.patch.object(utils.HydraConfig, "get", return_value=mock_hydra_cfg):
        task_name = utils.get_task_name(key)

    assert task_name == mock_choices[key]


def testget_task_name_no_task() -> None:
    key = "invalid_arg"
    mock_choices = {"task": "invalid_param"}
    mock_hydra_cfg = mock.MagicMock()
    mock_hydra_cfg.runtime.choices = mock_choices
    with mock.patch.object(utils.HydraConfig, "get", return_value=mock_hydra_cfg):
        with pytest.raises(KeyError):
            _ = utils.get_task_name(key)


@pytest.fixture
def mock_config() -> dict:
    return {
        "task": {
            "dataset_config": {"name": "mock_dataset"},
            "evaluator": "mock_evaluator",
            "class_key": "mock_task",
        },
        "model": {
            "name": "openai_chat",
            "params": {"engine": "gpt-4", "temperature": 0.4},
        },
        "metrics_save_dir": "mock/save/dir",
    }


def test_getcompute_config_hash(mock_config: dict) -> None:
    def omega_config(config_dict: dict) -> Dict[DictKeyType, Any]:
        cfg = OmegaConf.create(mock_config)
        return OmegaConf.to_container(cfg)  # type: ignore

    # Convert to omegaconf and back to check serializer can handle
    config = omega_config(mock_config)
    config_hash = utils.compute_config_hash(config)  # type: ignore

    assert isinstance(config_hash, str)

    # Checks changing params changes hash
    new_config = mock_config.copy()
    new_config["model"]["params"]["temperature"] += 1

    new_hash = utils.compute_config_hash(new_config)

    assert config_hash != new_hash


@pytest.fixture
def n_existing_runs() -> int:
    return 4


@pytest.fixture
def mock_metric_subdirs(n_existing_runs: int) -> List[mock.MagicMock]:
    mock_dirs = []
    for i in range(n_existing_runs):
        mock_dir = mock.MagicMock(spec=Path)
        mock_dir.is_dir.return_value = True
        mock_dir.name = f"{i:03d}"
        mock_dirs.append(mock_dir)

    # Shuffle to ensure function is robust to order
    random.shuffle(mock_dirs)

    return mock_dirs


def testget_run_id_existing_runs(
    n_existing_runs: int, mock_metric_subdirs: List[mock.MagicMock]
) -> None:
    mock_config_dir = mock.MagicMock(spec=Path)
    mock_config_dir.rglob.return_value = mock_metric_subdirs

    run_id = utils.get_run_id(mock_config_dir)
    assert run_id == f"{n_existing_runs:03d}"


def testget_run_id_no_existing_runs() -> None:
    mock_config_dir = mock.MagicMock(spec=Path)
    mock_config_dir.rglob.return_value = []

    assert utils.get_run_id(mock_config_dir) == "000"


def test_get_run_id_existing_runs_other_dirs(
    n_existing_runs: int, mock_metric_subdirs: List[mock.MagicMock]
) -> None:
    other_dir = mock.MagicMock(spec=Path)
    other_dir.name = "mock_id_004"
    other_dir.is_dir.return_value = True
    other_file = mock.MagicMock(spec=Path)
    other_file.name = "some_file.json"
    other_file.is_dir.return_value = False

    mock_config_dir = mock.MagicMock(spec=Path)
    mock_config_dir.rglob.return_value = mock_metric_subdirs

    run_id = utils.get_run_id(mock_config_dir)
    assert run_id == f"{n_existing_runs:03d}"


def test_get_save_path(mock_config: dict) -> None:
    mock_save_path = mock.MagicMock(spec=Path)
    mock_save_path.is_dir.return_value = False

    mock_config_dir = mock.MagicMock(spec=Path)
    mock_config_dir.__truediv__.return_value = mock_save_path

    mock_task_dir = mock.MagicMock(spec=Path)
    mock_task_dir.__truediv__.return_value = mock_config_dir

    mock_metrics_save_dir = mock.MagicMock(spec=Path)
    mock_metrics_save_dir.__truediv__.return_value = mock_task_dir

    mock_config_hash = "3423sdhsdfh32435"
    mock_run_id = "004"

    with mock.patch.object(
        utils, "compute_config_hash", return_value=mock_config_hash
    ) as mock_getconfig_hash, mock.patch.object(
        utils, "Path", return_value=mock_metrics_save_dir
    ), mock.patch.object(
        utils, "get_run_id", return_value=mock_run_id
    ) as mock_get_run_id:
        save_path = utils.get_save_path("mock_task", mock_config)

    assert save_path == mock_save_path
    mock_getconfig_hash.assert_called_once_with(mock_config)
    mock_get_run_id.assert_called_once_with(mock_config_dir)
    # Checks output from `get_run_id` was appended onto `config_dir`
    mock_config_dir.__truediv__.assert_called_once_with(mock_run_id)


def test_get_save_path_run_id(mock_config: dict) -> None:
    mock_run_id = "mock_run_id"
    mock_config["run_id"] = mock_run_id

    mock_save_path = mock.MagicMock(spec=Path)
    mock_save_path.is_dir.return_value = False

    mock_config_dir = mock.MagicMock(spec=Path)
    mock_config_dir.__truediv__.return_value = mock_save_path

    mock_task_dir = mock.MagicMock(spec=Path)
    mock_task_dir.__truediv__.return_value = mock_config_dir

    mock_metrics_save_dir = mock.MagicMock(spec=Path)
    mock_metrics_save_dir.__truediv__.return_value = mock_task_dir

    mock_config_hash = "3423sdhsdfh32435"

    with mock.patch.object(
        utils, "compute_config_hash", return_value=mock_config_hash
    ) as mock_getconfig_hash, mock.patch.object(
        utils, "Path", return_value=mock_metrics_save_dir
    ), mock.patch.object(
        utils, "get_run_id", return_value=mock_run_id
    ) as mock_get_run_id:
        save_path = utils.get_save_path("mock_task", mock_config)

    assert save_path == mock_save_path

    mock_config_no_run_id = mock_config.copy()
    del mock_config_no_run_id["run_id"]

    mock_getconfig_hash.assert_called_once_with(mock_config_no_run_id)

    assert not mock_get_run_id.called

    # Checks output from `get_run_id` was appended onto `config_dir`
    mock_config_dir.__truediv__.assert_called_once_with(mock_run_id)


def test_get_save_path_run_id_id_exists(mock_config: dict) -> None:
    mock_run_id = "mock_run_id"
    mock_config["run_id"] = mock_run_id

    mock_save_path = mock.MagicMock(spec=Path)
    mock_save_path.is_dir.return_value = True

    mock_config_dir = mock.MagicMock(spec=Path)
    mock_config_dir.__truediv__.return_value = mock_save_path

    mock_task_dir = mock.MagicMock(spec=Path)
    mock_task_dir.__truediv__.return_value = mock_config_dir

    mock_metrics_save_dir = mock.MagicMock(spec=Path)
    mock_metrics_save_dir.__truediv__.return_value = mock_task_dir

    mock_config_hash = "3423sdhsdfh32435"

    with mock.patch.object(
        utils, "compute_config_hash", return_value=mock_config_hash
    ), mock.patch.object(
        utils, "Path", return_value=mock_metrics_save_dir
    ), mock.patch.object(
        utils, "get_run_id", return_value=mock_run_id
    ):
        with pytest.raises(utils.RunExistsError):
            _ = utils.get_save_path("mock_task", mock_config)


def test_get_save_path_run_id_invalid_id(mock_config: dict) -> None:
    mock_run_id = "009"

    mock_config["run_id"] = mock_run_id

    mock_save_path = mock.MagicMock(spec=Path)
    mock_save_path.is_dir.return_value = False

    mock_config_dir = mock.MagicMock(spec=Path)
    mock_config_dir.__truediv__.return_value = mock_save_path

    mock_task_dir = mock.MagicMock(spec=Path)
    mock_task_dir.__truediv__.return_value = mock_config_dir

    mock_metrics_save_dir = mock.MagicMock(spec=Path)
    mock_metrics_save_dir.__truediv__.return_value = mock_task_dir

    mock_config_hash = "3423sdhsdfh32435"

    with mock.patch.object(
        utils, "compute_config_hash", return_value=mock_config_hash
    ), mock.patch.object(
        utils, "Path", return_value=mock_metrics_save_dir
    ), mock.patch.object(
        utils, "get_run_id", return_value=mock_run_id
    ):
        with pytest.raises(ValueError):
            _ = utils.get_save_path("mock_task", mock_config)


def test_get_save_path_no_metrics_save_dir(mock_config: dict) -> None:
    mock_config_no_metrics_save_dir = mock_config.copy()
    del mock_config_no_metrics_save_dir["metrics_save_dir"]

    with pytest.raises(KeyError):
        _ = utils.get_save_path("mock_task", mock_config_no_metrics_save_dir)


@pytest.fixture
def mock_metrics() -> Dict[str, Union[float, np.ndarray]]:
    return {
        "precision": 0.5,
        "recall": 0.4,
        "f1": 0.4,
    }


@pytest.fixture
def mock_artifacts() -> Dict[str, Union[float, np.ndarray]]:
    return {
        "precision": 0.5,
        "recall": 0.4,
        "f1": 0.4,
        "confmat": np.array([[10, 20], [5, 15]]),
    }


@pytest.fixture
def mock_datasets() -> Dict[str, Union[float, np.ndarray]]:
    return {
        "final_dataset": Dataset.from_dict(
            {"response": list("abcd"), "final_decision": list("0101")}
        )
    }


@pytest.fixture
def mock_task_output(
    mock_metrics: dict, mock_artifacts: dict, mock_datasets: dict
) -> RunTaskReturnType:
    return RunTaskReturnType(
        metrics=mock_metrics,
        artifacts=mock_artifacts,
        datasets=mock_datasets,
        other=mock_artifacts,
    )


@pytest.fixture
def mock_task_class(mock_task_output: RunTaskReturnType) -> type:
    class MockTask:
        def __init__(self, dataset: str, evaluator: str) -> None:
            self.dataset = dataset
            self.evaluator = evaluator

        @classmethod
        def from_config(cls, config: dict) -> MockTask:
            return cls(dataset=config["dataset_config"], evaluator=config["evaluator"])

        def run_task(self, model: BaseLLM) -> RunTaskReturnType:
            return mock_task_output

    return MockTask


@mock.patch("rambla.run.run_task.build_llm")
@mock.patch("rambla.run.run_task.get_task_name")
@mock.patch("rambla.run.run_task.get_save_path")
@mock.patch("rambla.run.run_task.dump")
@mock.patch("rambla.run.run_task.mlflow_log")
@mock.patch("rambla.run.run_task.store_task_output")
def test_main(
    mock_store_task_output,
    mock_mlflow_log,
    mock_dump,
    mock_get_save_path,
    mock_get_task_name,
    mock_build_llm,
    mock_config: dict,
    mock_task_class: type,
    mock_task_output: RunTaskReturnType,
) -> None:
    # Gets hydra-decorated function
    main_func = run_task.main.__wrapped__

    mock_project_name = "mock_project_name"
    mock_task_name = "mock_task"
    mock_task_map = {mock_task_name: mock_task_class}
    mock_get_task_name.return_value = mock_task_name
    mock_build_llm.return_value = mock.MagicMock(spec=BaseLLM)
    mock_save_path = Path("dummy path")
    mock_get_save_path.return_value = mock_save_path

    cfg = OmegaConf.create(mock_config)

    with mock.patch.object(run_task, "TASK_MAP", new=mock_task_map), mock.patch.dict(
        os.environ, {"MLFLOW_PROJECT_NAME": mock_project_name}, clear=True
    ):
        main_func(cfg)

    #
    mock_get_task_name.assert_called_once()
    mock_build_llm.assert_called_once_with(mock_config["model"])

    cfg_without_class_key = deepcopy(dict(mock_config))
    cfg_without_class_key["task"].pop("class_key")

    mock_get_save_path.assert_called_once_with(mock_task_name, cfg_without_class_key)
    mock_dump.assert_called_once_with(
        cfg_without_class_key, mock_save_path / "config.json"
    )
    mock_store_task_output.assert_called_once_with(mock_task_output, mock_save_path)

    mock_mlflow_log.assert_called_once_with(
        project_name=mock_project_name,
        experiment_name=mock_task_name,
        run_name=None,
        tags={
            "model_name": "openai_chat",
            "class_key": "mock_task",
            "yaml_file_name": "mock_task",
            "model_name_identifier": "gpt-4",
            "save_path": str(mock_save_path),
            "dataset": "mock_dataset",
        },
        config=cfg_without_class_key,
        metrics=mock_task_output.metrics,
        artifacts=mock_task_output.artifacts,
        extension=mock_task_output.artifact_storing_format,
        dictionaries=mock_task_output.dictionaries,
    )
