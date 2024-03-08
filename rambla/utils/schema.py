from typing import List, Optional, TypedDict, Union

from pydantic import BaseModel

from rambla.models import LLMConfig


class PromptConfig(BaseModel):
    # NOTE: Unformatted f-string
    template: str
    include_title: bool = False
    order: str = "desc"  # options: "desc", "asc"
    n_chunks: int = 5
    required_fields: List[str]
    tiktoken_encoding_name: Optional[str] = None
    token_budget: Optional[int] = None


class ResponseConfig(BaseModel):
    llm: LLMConfig
    prompt: PromptConfig


class ResponseInstance(TypedDict):
    index: Union[str, int]
    response: str
