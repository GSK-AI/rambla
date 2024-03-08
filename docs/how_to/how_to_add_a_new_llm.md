## Goal
Define classes for LLMs that will be evaluated or be used as components of the evaluation pipeline.

# Overview
We define the base classes in `rambla/models/base_model.py`. 

We currently support two types of models:
 - API based models (i.e., models access through an API)
 - Huggingface based models (i.e., models accessed through the huggingface ecosystem)

All models need to support a `from_config(cls, config)` constructor and ideally have method for generating responses `.generate(self, prompt: str) -> str`.

The `.model_dict(self) -> Dict[str, Hashable]` property is used by the `ResponseComponent` class for creating a cache directory.

*NOTE:* These models are not accessed directly. They are accessed through the `ResponseComponent` (in `rambla/response_generation/response.py`) that handles:
    - sync/async generation (using `.generate` or `.async_generate`)
    - caching (using `._model_dict` to create a sub-directory)
      - See `rambla/utils/caching.py` for more information.
    - backing off (e.g., when rate limit is hit)
      - See `rambla/utils/requests.py` for more information.
    - skipping (e.g., when prompt is too big)
      - See `rambla/utils/requests.py` for more information.
These are handled in: `ResponseComponent._make_request_func`

## `BaseLLM`
This is the base class for API based LLMs. These models can optionally have an `async` method `.async_generate(self, prompt: str) -> str` for asynchronous generation. We don't put any restrictions over how the API is accessed. As an example please see the `OpenaiChatCompletionModel` and `OpenaiCompletionModel` classes in `rambla/models/openai_models.py`.


## `BaseHuggingFaceModel`
This is the base class for models loaded through the HF ecosystem. As an example see the `TextGenerationModel` class in `rambla/models/huggingface.py`.

# Accessing the models
If you add a new model class to make it accessible to the rest of the codebase please add it in the `MODEL_MAP` inside this file: `rambla/models/__init__.py`.

# Running experiments
We run experiments using [hydra](https://hydra.cc/). To make a new model available in your experiments you need to add a new yaml file in this directory: `rambla/conf/model/<yaml file name>.yaml`. This can then be referenced in other yaml files. This needs to follow the pattern of this class: `LLMConfig` in `rambla/models/__init__.py`.