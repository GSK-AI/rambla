import argparse
from typing import Optional

from rambla.utils.misc import (
    get_available_text_to_text_components_yaml_files,
    get_available_text_to_text_task_yaml_files,
    initialize_logger,
    run_cmd,
    validate_text_to_text_components,
    validate_text_to_text_tasks,
)

# Note this is needed as not every text_to_text evaluation component can be used
# with every text_to_text_task. This dictionary defines a mapping for which
# components are suitiable for which tasks.
TEXT_TO_TEXT_CONFIG_COMBINATIONS = {
    "llm_component": [
        "text_to_text_mrpc",
        "text_to_text_continuous_sick",
    ],
    "llm_component_context": [
        "text_to_text_mrpc",
        "text_to_text_continuous_sick",
    ],
    "nli_bidirectional_component": [
        "text_to_text_nli_bi_mrpc",
        "text_to_text_nli_bi_sick",
    ],
    "nli_unidirectional_component": [
        "text_to_text_nli_uni_sick",
    ],
    "nlp_bleu_component": [
        "text_to_text_cat_to_cont_mrpc",
        "text_to_text_cat_to_cont_sick",
    ],
    "nlp_rouge1_component": [
        "text_to_text_cat_to_cont_mrpc",
        "text_to_text_cat_to_cont_sick",
    ],
    "nlp_rouge2_component": [
        "text_to_text_cat_to_cont_mrpc",
        "text_to_text_cat_to_cont_sick",
    ],
    "nlp_rougeL_component": [
        "text_to_text_cat_to_cont_mrpc",
        "text_to_text_cat_to_cont_sick",
    ],
    "nlp_component": [
        "text_to_text_cat_to_cont_mrpc",
        "text_to_text_cat_to_cont_sick",
    ],
}

logger = initialize_logger(__name__)


def parse_args():  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_to_text_tasks",
        "-t",
        type=str,
        nargs="+",
        help=(
            "Text_to_text_tasks to run. Each one of these will be ran "
            "against each of the apropriate models. If no input is provided "
            "then this will default to all available combinations."
        ),
    )
    parser.add_argument(
        "--text_to_text_components",
        "-c",
        type=str,
        nargs="+",
        help=(
            "Text_to_text_component to be used in the text_to_text_tasks. "
            "Each one of these will be ran against each of their respective tasks. "
            "If no input is provided then this will default "
            "to all available combinations."
        ),
    )

    args = parser.parse_args()
    return args


def main(
    text_to_text_tasks: Optional[list[str]] = None,
    text_to_text_components: Optional[list[str]] = None,
):
    """Will run text_to_text_components against text_to_text_tasks."""
    if not text_to_text_tasks:
        text_to_text_tasks = get_available_text_to_text_task_yaml_files()
    else:
        validate_text_to_text_tasks(text_to_text_tasks)

    if not text_to_text_components:
        text_to_text_components = get_available_text_to_text_components_yaml_files()
    else:
        validate_text_to_text_components(text_to_text_components)

    # This ensures that only the appropriate component and task combinations
    # (as outlined in the TEXT_TO_TEXT_CONFIG_COMBINATIONS dict above) are run together
    for key in text_to_text_components:
        available_tasks = TEXT_TO_TEXT_CONFIG_COMBINATIONS[key]
        tasks_to_use = set(available_tasks) & set(text_to_text_tasks)

        tasks_string = ",".join(tasks_to_use)

        command = (
            "python rambla/run/run_text_to_text.py "
            "--multirun "
            f"text_to_text_task={tasks_string} "
            f"text_to_text_component={key} "
        )

        logger.info(f"About to run {len(tasks_to_use)} jobs. ")
        logger.info(f"Running command:\n{command}")
        run_cmd(command)


if __name__ == "__main__":
    args = parse_args()
    main(
        text_to_text_tasks=args.text_to_text_tasks,
        text_to_text_components=args.text_to_text_components,
    )
