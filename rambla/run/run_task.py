import os

# flake8: noqa: E402
# fmt: off
# isort: off
from dotenv import load_dotenv; load_dotenv()
# fmt: on
# isort: on

import hydra
from omegaconf import DictConfig, OmegaConf

from rambla.models import build_llm
from rambla.run.utils import get_save_path, get_task_name, store_task_output
from rambla.tasks import TASK_MAP
from rambla.tasks.base import BaseTask
from rambla.utils.io import dump
from rambla.utils.misc import make_json_serialisable
from rambla.utils.mlflow import mlflow_log


@hydra.main(
    version_base=None,
    config_path="../conf/",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Runs a full evaluation for a task"""
    task_name = get_task_name("task")
    config = OmegaConf.to_container(cfg, resolve=True)

    if not isinstance(config, dict):
        raise TypeError(f"Config is of type: {type(config)}. Must be of type dict.")

    task_class_name = config["task"].pop("class_key")

    # Load the relevant task
    if task_class_name not in TASK_MAP.keys():
        raise KeyError(
            f"Invalid task: {task_class_name}. "
            f"Currently supported tasks are: {list(TASK_MAP.keys())}"
        )

    task: BaseTask = TASK_MAP[task_class_name].from_config(config["task"])

    # Load the relevant model
    model = build_llm(config["model"])

    # execution
    task_output = task.run_task(model)  # type: ignore

    # storing and logging
    config = make_json_serialisable(config)
    save_path = get_save_path(task_name, config)

    dump(config, save_path / "config.json")

    store_task_output(task_output, save_path)

    # Adding tags
    tags = {
        "model_name": config["model"]["name"],
        "class_key": task_class_name,
        "yaml_file_name": task_name,
        "save_path": str(save_path),
    }

    ACCEPTED_MODELS = ["huggingface_llm", "openai_chat"]  # noqa: N806

    if tags["model_name"] == "huggingface_llm":
        tags["model_name_identifier"] = config["model"]["params"]["model_name"]
    elif tags["model_name"] in ["openai_chat", "openai_35_chat"]:
        tags["model_name_identifier"] = config["model"]["params"]["engine"]
    else:
        raise ValueError(
            f"{config['model']['name']=} not recognised. "
            f"Accepted models are: {ACCEPTED_MODELS}."
        )

    if tags["class_key"] in ["PromptRobustness"]:
        tags["dataset"] = config["task"]["subtask"]["config"]["dataset_config"]["name"]
    else:
        tags["dataset"] = config["task"]["dataset_config"]["name"]

    mlflow_log(
        project_name=os.environ.get("MLFLOW_PROJECT_NAME", "LLM_EVAL"),
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
    main()
