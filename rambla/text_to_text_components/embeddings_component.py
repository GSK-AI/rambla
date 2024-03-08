from __future__ import annotations

from typing import Union

from datasets import Dataset
from pydantic import BaseModel, Extra

from rambla.models import LLMConfig, build_llm
from rambla.models.base_model import BaseAPIEmbeddingsModel
from rambla.response_generation.response import (
    ResponseComponent,
    ResponseComponentConfig,
)
from rambla.text_to_text_components.base import BaseTextToTextSimilarityComponent
from rambla.utils.similarity import (
    BaseSimilarityModule,
    SimilarityModuleConfig,
    build_similarity_module,
)

"""Example usage:

```python
    from dotenv import load_dotenv
    load_dotenv()

    model_config = {"name": "openai", "params": {"engine": "text-embedding-ada-002"}}
    similarity_module_config = {"name": "numpy_inner_product"}

    config = {
        "embeddings_model_config": model_config,
        "similarity_module_config": similarity_module_config,
        "cache_dir": "tmp",
        "index_field": "index",
        "text_field_1": "text_1",
        "text_field_2": "text_2",
        "response_cache_fname": "response.pkl",
        "response_field_name": "response",
    }

    dataset = Dataset.from_dict(
        {
            "index": [0, 1],
            "text_1": ["I hate ice-cream", "I love ice-cream"],
            "text_2": ["I love ice-cream", "I love vanilla ice-cream"],
        }
    )
    module = EmbeddingBasedTextToTextComponent.from_config(config=config)
    response = module.run(dataset)

```
"""


class EmbeddingBasedTextToTextComponentConfig(BaseModel):
    embeddings_model_config: LLMConfig
    similarity_module_config: SimilarityModuleConfig
    response_component_config: ResponseComponentConfig
    text_field_1: str
    text_field_2: str
    response_field_name: str

    class Config:  # noqa: D106
        extra = Extra.forbid


class EmbeddingBasedTextToTextComponent(BaseTextToTextSimilarityComponent):
    embedding_columns = ["response_text_field_1", "response_text_field_2"]

    def __init__(
        self,
        model: BaseAPIEmbeddingsModel,
        similarity_module: BaseSimilarityModule,
        response_component: ResponseComponent,
        *,
        text_field_1: str,
        text_field_2: str,
        response_field_name: str = "response",
    ):
        self.model = model
        self.similarity_module = similarity_module
        self.response_component = response_component

        self.text_field_1 = text_field_1
        self.text_field_2 = text_field_2
        self.response_field_name = response_field_name

    @classmethod
    def from_config(
        cls, config: Union[dict, EmbeddingBasedTextToTextComponentConfig]
    ) -> EmbeddingBasedTextToTextComponent:
        if not isinstance(config, EmbeddingBasedTextToTextComponentConfig):
            config = EmbeddingBasedTextToTextComponentConfig.parse_obj(config)

        model = build_llm(config.embeddings_model_config)
        similarity_module = build_similarity_module(config.similarity_module_config)

        response_component = ResponseComponent.from_config(
            config.response_component_config
        )

        return cls(
            model=model,
            similarity_module=similarity_module,
            response_component=response_component,
            text_field_1=config.text_field_1,
            text_field_2=config.text_field_2,
            response_field_name=config.response_field_name,
        )

    def run(self, dataset: Dataset) -> Dataset:
        # Generate responses
        response_dataset = self.response_component.batch_generate(
            model=self.model,
            prompt_dataset=dataset,
            prompt_field=self.text_field_1,
            response_field=self.embedding_columns[0],
        )
        response_dataset = self.response_component.batch_generate(
            model=self.model,
            prompt_dataset=response_dataset,
            prompt_field=self.text_field_2,
            response_field=self.embedding_columns[1],
        )

        # Inner product
        similarity_scores = self.similarity_module.run(
            arr0=response_dataset[self.embedding_columns[0]],
            arr1=response_dataset[self.embedding_columns[1]],
        )
        response_dataset = response_dataset.add_column(
            self.response_field_name, similarity_scores
        )  # type: ignore
        return response_dataset
