import os

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from rambla.run.utils import (
    create_plots,
    get_save_path,
    get_task_name,
    store_task_output,
)
from rambla.text_to_text_components import build_text_to_text_module
from rambla.text_to_text_tasks import TEXT_TO_TEXT_TASK_MAP
from rambla.text_to_text_tasks.base import BaseTextToTextTask
from rambla.utils.io import dump
from rambla.utils.misc import make_json_serialisable
from rambla.utils.mlflow import mlflow_log


@hydra.main(
    version_base=None,
    config_path="../conf/",
    config_name="config_text_to_text",
)
def main(cfg: DictConfig) -> None:
    """Runs a full evaluation for a text_to_text task"""
    task_name = get_task_name("text_to_text_task")
    yaml_file_component_name = get_task_name("text_to_text_component")

    config = OmegaConf.to_container(cfg, resolve=True)

    dataset_name = config["text_to_text_task"]["dataset_config"]["name"]
    component_name = config["text_to_text_component"]["name"]

    if not isinstance(config, dict):
        raise TypeError(f"Config is of type: {type(config)}. Must be of type dict.")

    task_class_name = config["text_to_text_task"].pop("class_key")

    # Load the relevant task
    if task_class_name not in TEXT_TO_TEXT_TASK_MAP.keys():
        raise KeyError(
            f"Invalid task: {task_class_name}. "
            f"Currently supported tasks are: {list(TEXT_TO_TEXT_TASK_MAP.keys())}"
        )

    task: BaseTextToTextTask = TEXT_TO_TEXT_TASK_MAP[task_class_name].from_config(
        config["text_to_text_task"]
    )

    # Load the relevant component
    component = build_text_to_text_module(config["text_to_text_component"])

    # execution
    task_output = task.run_task(component)  # type: ignore

    # create plots
    task_output = create_plots(task_output)

    # storing and logging
    config = make_json_serialisable(config)
    save_path = get_save_path(task_name, config)

    dump(config, save_path / "config.json")

    store_task_output(task_output, save_path)

    tags = {
        "class_key": task_class_name,
        "yaml_file_name": task_name,
        "dataset_name": dataset_name,
        "component_name": component_name,
        "component_yaml_file_name": yaml_file_component_name,
    }

    mlflow_log(
        project_name=os.environ["MLFLOW_PROJECT_NAME"],
        experiment_name=task_name,
        run_name=config.get("run_id"),
        tags=tags,
        config=config,
        metrics=task_output.metrics,
        artifacts=task_output.artifacts,
        extension=task_output.artifact_storing_format,
        dictionaries=task_output.dictionaries,
    )


if __name__ == "__main__":
    load_dotenv()
    main()
