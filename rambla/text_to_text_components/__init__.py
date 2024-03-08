from typing import Union

from pydantic import BaseModel, Extra, validator

from rambla.text_to_text_components.base import BaseTextToTextSimilarityComponent
from rambla.text_to_text_components.embeddings_component import (
    EmbeddingBasedTextToTextComponent,
)
from rambla.text_to_text_components.llm_similarity_component import (
    LLMTextToTextSimilarity,
)
from rambla.text_to_text_components.nli_wrappers import (
    NLIBidirectional,
    NLIUnidirectional,
)
from rambla.text_to_text_components.nlp_component import NgramTextToTextSimilarity

COMPONENT_MAP = {
    "llm_component": LLMTextToTextSimilarity,
    "llm_component_context": LLMTextToTextSimilarity,
    "embeddings_component": EmbeddingBasedTextToTextComponent,
    "nli_bidirectional_component": NLIBidirectional,
    "nli_unidirectional_component": NLIUnidirectional,
    "nlp_component": NgramTextToTextSimilarity,
    "nlp_rougeL_component": NgramTextToTextSimilarity,
    "nlp_rouge2_component": NgramTextToTextSimilarity,
    "nlp_rouge1_component": NgramTextToTextSimilarity,
    "nlp_bleu_component": NgramTextToTextSimilarity,
}


def _validate_module_name(module_name: str):
    """Checks provided name is in `COMPONENT MAP`"""
    if module_name not in COMPONENT_MAP.keys():
        raise ValueError(
            f"Invalid name: {module_name}. Name must "
            f"be one of {list(COMPONENT_MAP.keys())}"
        )


class ParentTextToTextModuleConfig(BaseModel):
    name: str
    params: dict

    class Config:  # noqa: D106
        extra = Extra.forbid

    @validator("name")
    @classmethod
    def validate_name(cls, v):
        _validate_module_name(v)
        return v


def build_text_to_text_module(
    config: Union[dict, ParentTextToTextModuleConfig]
) -> BaseTextToTextSimilarityComponent:
    """Prepares LLM based on config"""
    config = ParentTextToTextModuleConfig.parse_obj(config)
    module_class = COMPONENT_MAP[config.name]
    return module_class.from_config(config.params)
