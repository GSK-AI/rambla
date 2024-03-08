import argparse
import logging
from typing import Optional

from rambla.utils.misc import (
    get_available_model_yaml_files,
    get_available_task_yaml_files,
    run_cmd,
    validate_models,
    validate_tasks,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s  %(name)s  [%(levelname)s]: %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def parse_args():  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        "-t",
        type=str,
        nargs="+",
        help=(
            "Tasks to run. Each one of these will be ran "
            "against each of the models. If no input is provided "
            "then this will default to the full list of yaml files "
            "under /conf/tasks."
        ),
    )
    parser.add_argument(
        "--models",
        "-m",
        type=str,
        nargs="+",
        help=(
            "Models to run the tasks against. "
            "Each one of these will be ran against each of the tasks. "
            "If no input is provided then this will default "
            "to the full list of yaml files under /conf/models."
        ),
    )
    parser.add_argument(
        "--subset",
        "-s",
        type=int,
        help=(
            "Subset of dataset to use. "
            "If no input is provided then will deafult to the whole dataset."
        ),
    )
    args = parser.parse_args()
    return args


def main(
    tasks: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    subset: Optional[int] = None,
):
    """Will run models against tasks."""
    if not tasks:
        tasks = get_available_task_yaml_files()
    else:
        validate_tasks(tasks)

    if not models:
        models = get_available_model_yaml_files()
    else:
        validate_models(models)

    tasks_string = ",".join(tasks)
    models_string = ",".join(models)

    if subset:
        # Note this command will not work for those datasets stored locally
        # If you wish to run tasks on a alternative subset for those datasets
        # please adjust the relevant config
        command = (
            "python rambla/run/run_task.py "
            "--multirun "
            f"task={tasks_string} "
            f"model={models_string} "
            f"task.dataset_config.params.split='train[:{subset}]'"
        )
    else:
        command = (
            "python rambla/run/run_task.py "
            "--multirun "
            f"task={tasks_string} "
            f"model={models_string} "
        )

    logger.info(
        f"About to run {len(models) * len(tasks)} jobs. "
        f"Derived from {len(models)=} and {len(tasks)=}. "
    )
    logger.info(f"Running command:\n{command}")

    run_cmd(command)


if __name__ == "__main__":
    args = parse_args()
    main(tasks=args.tasks, models=args.models, subset=args.subset)
