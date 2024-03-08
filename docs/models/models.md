## Goal

The package currently supports two types of models:
1) API-based models. See `BaseAPIModel` in `rambla/models/base_model.py` for the abstract base class and `OpenaiChatCompletionModel` in `rambla/models/openai_models.py` for the class defining functionality for accessing the OpenAI API.

2) Models accessed through Huggingface functionality. See `BaseHuggingFaceModel` in `rambla/models/base_model.py` for the abstract base class and `TextGenerationModel` in `rambla/models/huggingface.py` for the class defining functionality for accessing the model. 

## Adding new API-based models
1) The OpenAI classes in `rambla/models/openai_models.py` should be used as a reference on how to build up the necessary functionality. This includes the class responsible for text generation and a `pydantic.BaseModel` responsible for config validation.
2) Once the functionality is ready, a key-value pair needs to be added in the `MODEL_MAP` in this script: `rambla/models/__init__.py`
3) The model can then be referenced with a new `.yaml` config file under `rambla/conf/model` with the following format:
```yaml
name: <this should mirror the `key` in the key-value pair added in `MODEL_MAP` in step 1 above.>
params:
    <This set of params should match the params expected by the new `pydantic.BaseModel` class.>
```

This is an example of `.yaml` file that can be used to access an OpenAI model:
```yaml
name: openai_chat
params:
  temperature: 0.0
  engine: gpt-4-32k
  async_calls: false
```

## Adding new local model
1) Create a `.yaml` config file under `rambla/conf/model` with the following format:
```yaml
name: huggingface_llm
params:
  model_name: <path/to/model/>
  loading_params:
    torch_dtype: bfloat16
  generation_config:
    max_new_tokens: 200
    do_sample: false
    top_p: 1.0
  is_finetuned: true
  device_map: auto
```
Further information on how these parameters are used can be found by inspecting the `TextGenerationModelConfig` and the `TextGenerationModel` classes in `rambla/models/huggingface.py`. Params under `params.generation_config` will be passed to `transformers.GenerationConfig` and can be model specific.

2) This `.yaml` file can then be referenced in the `rambla/conf/config.yaml` config file and get benchmarked against existing or new tasks.