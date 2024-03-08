import argparse
import os
import subprocess
import uuid
from pathlib import Path

from dotenv import load_dotenv

from rambla.utils.misc import EnvCtxManager, initialize_logger, run_cmd

logger = initialize_logger(__name__)


COMPONENT_TO_TASK_MAP = {
    "llm_component": ["text_to_text_original"],
    "embeddings_component": ["text_to_text_cat_to_cont_mrpc"],
    "nli_bidirectional_component_strict": ["text_to_text_nli_bi_mrpc_train"],
    "nli_bidirectional_component_relaxed": ["text_to_text_nli_bi_mrpc_train"],
    "nli_unidirectional_component": ["text_to_text_original"],
    "nlp_component": ["text_to_text_cat_to_cont_mrpc"],
}


def parse_args():  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--components",
        "-c",
        type=str,
        nargs="+",
        default=list(COMPONENT_TO_TASK_MAP.keys()),
        help="Textual similarity components to run integrations tests for.",
    )
    parser.add_argument("--break-on-error", action="store_true", default=False)
    args = parser.parse_args()
    return args


def build_command(task_name: str, model_name: str, save_dir: str) -> str:
    """Builds the bash command to be executed."""
    script_path = (
        Path(__file__).parent.parent / "rambla/run/run_text_to_text.py"
    )
    command = (
        f"python {script_path} "
        f"save_dir={save_dir} "
        f"text_to_text_task={task_name} "
        f"text_to_text_component={model_name} "
        "text_to_text_task.dataset_config.params.split='train[:2]'"
    )
    return command


def main(components: list[str], save_dir: Path, break_on_error: bool):
    """Runs commands produced by combining all models with all tasks."""
    if save_dir.is_dir():
        raise RuntimeError(f"{save_dir=} exists")

    run_bash_command = EnvCtxManager(MLFLOW_PROJECT_NAME="aiml_rai_llm_eval_testing")(
        run_cmd
    )
    for component_name in components:
        if component_name not in COMPONENT_TO_TASK_MAP:
            raise RuntimeError(
                f"{component_name=} not supported. "
                f"Try one of {COMPONENT_TO_TASK_MAP.keys()=}"
            )

        for task_name in COMPONENT_TO_TASK_MAP[component_name]:
            command = build_command(task_name, component_name, save_dir.as_posix())
            logger.info(f"Running: {command}")
            try:
                run_bash_command(command)
            except subprocess.CalledProcessError as err:
                if break_on_error:
                    raise err


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    save_dir = (
        Path(os.path.expanduser("~"))
        / "rambla_integration_tests_dir"
        / str(uuid.uuid4())
    )

    main(args.components, save_dir, args.break_on_error)
