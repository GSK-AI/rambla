import argparse
import itertools
import os
import subprocess
import uuid
from pathlib import Path

from dotenv import load_dotenv

from rambla.utils.misc import (
    EnvCtxManager,
    get_available_task_yaml_files,
    initialize_logger,
    run_cmd,
    validate_models,
    validate_tasks,
)

logger = initialize_logger(__name__)


def parse_args():  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        "-t",
        type=str,
        nargs="+",
        help=(
            "Tasks to run integrations tests for. "
            "Tests will be ran for all combinations of tasks and models!"
        ),
    )
    parser.add_argument(
        "--models",
        "-m",
        type=str,
        nargs="+",
        default=["openai_chat"],
        help=(
            "LLMs to run integrations tests for. "
            "Tests will be ran for all combinations of tasks and models!"
        ),
    )
    parser.add_argument("--break-on-error", action="store_true", default=False)
    args = parser.parse_args()
    return args


def build_command(task_name: str, model_name: str, save_dir: str) -> str:
    """Builds the bash command to be executed."""
    script_path = Path(__file__).parent.parent / "rambla/run/run_task.py"
    command = (
        f"python {script_path} "
        f"save_dir={save_dir} "
        f"task={task_name} "
        f"model={model_name} "
    )

    if "fewshot" in task_name:
        # NOTE: changing the order to a minimal version
        order = ["yes"]

        if "parent" in task_name:
            command += f" task.orders=[{order}]"
        else:
            command += f" task.examples_module_config.order={order}"

    if "distracting_context" in task_name:
        # NOTE: This task requires a dataset of len>1 because of sampling.
        command += " task.dataset_config.params.split='train[:5]'"
    elif "spelling_robustness" in task_name:
        # NOTE: This task internally runs a subtask
        command += " task.subtask.config.dataset_config.params.split='train[:1]'"
    elif "fewshot" in task_name:
        if "parent" in task_name:
            command += " task.child_task_config.dataset_config.params.split='train[:6]'"
        else:
            command += " task.dataset_config.params.split='train[:6]'"
    elif "question_formation" in task_name or "long_form" in task_name:
        command += " task.dataset_config.params.subset=2"
    else:
        command += " task.dataset_config.params.split='train[:1]'"

    return command


def main(tasks: list[str], models: list[str], save_dir: Path, break_on_error: bool):
    """Runs commands produced by combining all models with all tasks."""
    if save_dir.is_dir():
        raise RuntimeError(f"{save_dir=} exists")

    validate_tasks(tasks)
    validate_models(models)

    run_bash_command = EnvCtxManager(MLFLOW_PROJECT_NAME="aiml_rai_llm_eval_testing")(
        run_cmd
    )
    for task_name, model_name in itertools.product(tasks, models):
        command = build_command(task_name, model_name, save_dir.as_posix())

        logger.info(f"Running: {command}")
        try:
            run_bash_command(command)
        except subprocess.CalledProcessError as err:
            if break_on_error:
                raise err


if __name__ == "__main__":
    load_dotenv()
    os.environ["MLFLOW_PROJECT_NAME"] = "aiml_rai_llm_eval_testing"
    args = parse_args()
    save_dir = (
        Path(os.path.expanduser("~"))
        / "rambla_integration_tests_dir"
        / str(uuid.uuid4())
    )

    if not args.tasks:
        args.tasks = get_available_task_yaml_files()

    main(args.tasks, args.models, save_dir, args.break_on_error)
