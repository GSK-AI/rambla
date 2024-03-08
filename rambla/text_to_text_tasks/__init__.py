from typing import Dict, Type

from rambla.text_to_text_tasks.base import BaseTextToTextTask
from rambla.text_to_text_tasks.text_to_text_similarity_eval import (
    TextToTextSimilarityEvaluation,
)

TEXT_TO_TEXT_TASK_MAP: Dict[str, Type[BaseTextToTextTask]] = {
    "TextToText": TextToTextSimilarityEvaluation,
    "TextToTextContinuous": TextToTextSimilarityEvaluation,
    "TextToTextCatToCont": TextToTextSimilarityEvaluation,
}
