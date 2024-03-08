from typing import Dict, Type

from rambla.tasks.base import BaseTask
from rambla.tasks.few_shot_examples import FewShotExamplesTask
from rambla.tasks.few_shot_examples.few_shot_examples import ParentFewShotExamplesTask
from rambla.tasks.irrelevant_context import (
    DistractingContextTask,
    IrrelevantContextDiffDatasetTask,
    IrrelevantContextTask,
)
from rambla.tasks.longform import MCQALongFormTask
from rambla.tasks.longform.longform import SimilarityBasedLongFormTask
from rambla.tasks.mcqabaseline import MCQABaselineTask
from rambla.tasks.negation import NegationTask
from rambla.tasks.paraphrase import ParaphraseTask
from rambla.tasks.prompt_robustness import PromptRobustness

TASK_MAP: Dict[str, Type[BaseTask]] = {
    "ParaphraseTask": ParaphraseTask,
    "IrrelevantContextTask": IrrelevantContextTask,
    "MCQABaselineTask": MCQABaselineTask,
    "NegationTask": NegationTask,
    "IrrelevantContextDiffDatasetTask": IrrelevantContextDiffDatasetTask,
    "MCQALongFormTask": MCQALongFormTask,
    "DistractingContextTask": DistractingContextTask,
    "FewShotExamplesTask": FewShotExamplesTask,
    "ParentFewShotExamplesTask": ParentFewShotExamplesTask,
    "PromptRobustness": PromptRobustness,
    "SimilarityLongFormTask": SimilarityBasedLongFormTask,
}
