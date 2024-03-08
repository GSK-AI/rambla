import pydantic
import pytest

from rambla.text_to_text_components import (
    COMPONENT_MAP,
    ParentTextToTextModuleConfig,
    _validate_module_name,
    build_text_to_text_module,
)
from rambla.text_to_text_components.llm_similarity_component import (
    LLMTextToTextSimilarity,
)
from rambla.utils.misc import EnvCtxManager

# flake8: noqa: N802


def test_ParentTextToTextModuleConfig(parent_llm_component_config):
    ParentTextToTextModuleConfig.parse_obj(parent_llm_component_config)
    assert True


def test_ParentTextToTextModuleConfig_wrong_name(parent_llm_component_config):
    parent_llm_component_config["name"] = "__dummy_module_name__"
    with pytest.raises(pydantic.ValidationError):
        ParentTextToTextModuleConfig.parse_obj(parent_llm_component_config)


def test_validate_module_name_valueerror():
    dummy_name = "__dummy_module_name__"
    with pytest.raises(ValueError):
        _validate_module_name(dummy_name)


@pytest.mark.parametrize("module_name", list(COMPONENT_MAP.keys()))
def test_validate_module_name(module_name: str):
    _validate_module_name(module_name)


@EnvCtxManager(
    OPENAI_API_KEY="__dummy_openai_api_key__",
    OPENAI_API_BASE="__dummy_openai_api_base__",
    OPENAI_API_VERSION="__dummy_openai_api_version__",
)
def test_build_text_to_text_module(parent_llm_component_config):
    output_module = build_text_to_text_module(parent_llm_component_config)
    assert isinstance(output_module, LLMTextToTextSimilarity)
