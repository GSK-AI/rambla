from __future__ import annotations

import copy
from typing import Dict, Hashable, Literal, Optional, Union

import torch
import transformers
from pydantic import BaseModel, Extra, root_validator, validator

from rambla.models.base_model import BaseHuggingFaceModel
from rambla.models.utils import MessageFormatter
from rambla.utils.misc import initialize_logger, split_text_on_sep
from rambla.utils.pytorch import SUPPORTED_DEVICES

logger = initialize_logger(__name__)


class MaximumContextLengthExceededError(Exception):
    def __init__(self, max_supported_length: int, n_tokens: int):
        self.max_supported_length = max_supported_length
        self.n_tokens = n_tokens
        error_message = f"{max_supported_length=}, but {n_tokens=}."
        super().__init__(error_message)


class NLIModelParams(BaseModel):
    label_map: Dict[int, str]
    device: str
    return_mode: Literal["logits", "label", "dict"]
    sequence_sep: str

    @validator("device", pre=True)
    @classmethod
    def validate_device(cls, v, values, **kwargs):
        if v not in SUPPORTED_DEVICES:
            raise ValueError(
                f"Device: {v} not in supported devices: {SUPPORTED_DEVICES}"
            )
        return v


class NLIModelConfig(BaseModel):
    name: str
    params: NLIModelParams


class NLIModel(BaseHuggingFaceModel):
    def __init__(
        self,
        tokenizer: Union[
            transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
        ],
        model: transformers.PreTrainedModel,
        device: str,
        label_map: Dict[int, str],
        return_mode: Literal["logits", "label", "dict"] = "dict",
        sequence_sep: str = "[SEP]",
    ) -> None:
        """Interface for huggingface natural-language inference models

        Class provides an interface for any huggingface models for
        performing joint-classification of input sequences for natural-
        language inference. e.g. microsoft/deberta-large-mnli

        Parameters
        ----------
        tokenizer : Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
            Tokenizer for converting string into input for `model`
        model : PreTrainedModel
            Loaded huggingface model
        device : str
            Device to run inference on (e.g. "cpu")
        label_map : Dict[int, str]
            Map of logit indexes to classification labels.

            For example:
                {
                    0: "entailment",
                    1: "neutral",
                    2: "contradiction",
                }

        return_mode : Literal["logits", "label"], optional
            Mode for output type of model. Current options:
                "logits": Return a list of floats with the logits for each label
                "label": Returns the class label with the max logit
                "dict": Returns a dict with keys from the label_map and
                        logits for each label as floats for the values
            By default "dict"
        sequence_sep : str, optional
            String to use to separate sequences in inout, by default "[SEP]"
        """
        super().__init__(tokenizer, model, device)
        self.label_map = label_map
        self.return_mode = return_mode
        self.sequence_sep = sequence_sep

    @property
    def _model_dict(self) -> Dict[str, Hashable]:
        return {
            "model_name": self.model.name_or_path,
            "tokenizer_name": self.tokenizer.name_or_path,
            "label_map": self.label_map,  # type: ignore
            "return_mode": self.return_mode,
        }

    @classmethod
    def from_config(cls, config: Union[dict, NLIModelConfig]) -> "NLIModel":
        if isinstance(config, dict):
            config = NLIModelConfig.parse_obj(config)

        if "deberta" in config.name.lower():
            tokenizer = transformers.DebertaTokenizer.from_pretrained(config.name)
            model = transformers.DebertaForSequenceClassification.from_pretrained(
                config.name
            )
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(config.name)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                config.name
            )

        return cls(
            tokenizer=tokenizer,
            model=model,
            device=config.params.device,
            label_map=config.params.label_map,
            return_mode=config.params.return_mode,
            sequence_sep=config.params.sequence_sep,
        )

    def generate(self, prompt: str) -> dict:
        """Jointly classifies two sequences of text returning the max label

        Parameters
        ----------
        text : Input sequences separated by separator string.
            For example for the following 2 sequences:

            text1 = "I like ice cream"
            text2 = "I love ice cream"

            text = "I like ice cream[SEP]I love ice cream"
        """
        sequences = split_text_on_sep(prompt, self.sequence_sep, 2)
        sequence1, sequence2 = sequences[0], sequences[1]
        tokens = self.tokenizer(
            [sequence1], [sequence2], padding=True, truncation=True, return_tensors="pt"
        )
        tokens = tokens.to(self.device)
        with torch.no_grad():
            # Returns prediction of type SequenceClassifierOutput
            prediction = self.model(**tokens)

        logits = prediction.logits[0]

        if self.return_mode == "logits":
            return {"response": logits.tolist()}
        elif self.return_mode == "label":
            return {"response": self.label_map[int(torch.argmax(logits))]}
        elif self.return_mode == "dict":
            return_dict = {}
            logits = logits.tolist()
            for key, value in self.label_map.items():
                return_dict.update({value: logits[key]})
            return {"response": return_dict}
        else:
            raise ValueError(f"Invalid return mode: {self.return_mode}")


TORCH_DTYPE_MAP = {
    "torch.float16": torch.float16,
    "float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "float32": torch.float32,
}


class HuggingfaceModelLoadingKwargs(BaseModel):
    torch_dtype: Optional[Union[Literal["auto"], torch.dtype]]

    @validator("torch_dtype", pre=True)
    @classmethod
    def validate_torch_dtype(cls, torch_dtype):
        #  `torch_dtype` can be either `torch.dtype` or `"auto"`
        if isinstance(torch_dtype, str) and torch_dtype != "auto":
            try:
                torch_dtype = TORCH_DTYPE_MAP[torch_dtype]
            except KeyError:
                raise KeyError(
                    f"Found {torch_dtype=}, try one of {TORCH_DTYPE_MAP.keys()=}."
                )
        return torch_dtype

    class Config:  # noqa: D106
        # NOTE: We want to allow other parameters as well given that
        # there's a ton of HF model kwargs.
        extra = Extra.allow
        arbitrary_types_allowed = True


class TextGenerationModelConfig(BaseModel):
    model_name: str
    # NOTE: If `tokenizer_name` is not provided, then `model_name`
    # will be used to load the tokenizer.
    tokenizer_name: Optional[str]
    loading_params: Optional[HuggingfaceModelLoadingKwargs]

    # NOTE: will be parsed to `transformers.GenerationConfig`.
    generation_config: Optional[dict] = {}

    device: Optional[str]
    # NOTE: Requires `accelerate` to be installed.
    device_map: Optional[str] = "auto"

    is_finetuned: bool = False
    # NOTE: If this is provided then we use it.
    message_formatting_template: Optional[str]
    local_files_only: bool = False

    class Config:  # noqa: D106
        extra = Extra.forbid

    @root_validator()
    @classmethod
    def validate_device_and_device_map(cls, values):
        device = values.get("device", None)
        device_map = values.get("device_map", None)

        if device is not None and device_map is not None:
            raise ValueError(
                f"Can't provide both {values['device']=} and {values['device_map']=}."
            )
        return values

    @validator("message_formatting_template")
    @classmethod
    def validate_message_formatting_template(cls, message_formatting_template, values):
        is_finetuned = values["is_finetuned"]
        if message_formatting_template and not is_finetuned:
            logger.info(
                f"{message_formatting_template=} was provided but `{is_finetuned=}`. "
                f"`message_formatting_template` will be ignored."
            )
        return message_formatting_template


class TextGenerationModel(BaseHuggingFaceModel):
    def __init__(
        self,
        tokenizer: Union[
            transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast
        ],
        model: transformers.PreTrainedModel,
        pipeline: transformers.Pipeline,
        message_formatter: Optional[MessageFormatter] = None,
        generation_config: Optional[transformers.GenerationConfig] = None,
        loading_params: Optional[HuggingfaceModelLoadingKwargs] = None,
        device: Optional[str] = None,
        # NOTE: Requires `accelerate` to be installed.
        device_map: Optional[str] = "auto",
        is_finetuned: bool = False,
    ) -> None:
        """Interface class for huggingface text-generation models

        Class provides an interface for huggingface models which take
        text as input and output generated text.

        Parameters
        ----------
        tokenizer : Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
            Tokenizer for converting string into input for `model`
        model : PreTrainedModel
            Loaded huggingface model
        loading_params: Optional[HuggingfaceModelLoadingKwargs]
            After being passed to the class, it will not be used directly.
            It will only be used inside `._model_dict`.
        pipeline : transformers.Pipeline
            Text generation pipeline
        message_formatter : MessageFormatter
            Formats the query to match what the model expects.
        generation_config : Optional[transformers.GenerationConfig], optional
            Config to be passed on to the `pipeline`, by default None
        NOTE: Only one of `device` and `device_map` should be provided.
        device : Optional[str]
            Device to run inference on (e.g. "cpu")
        device_map : Optional[str]
            Used by `transformers` for model parallelisation.
            NOTE: Requires `accelerate` to be installed.
        """
        if device is not None and device_map is not None:
            raise ValueError(f"Can't provide both {device=} and {device_map=}.")

        super().__init__(tokenizer, model, device)
        self.loading_params = loading_params
        self.pipeline = pipeline
        self.message_formatter = message_formatter
        self.device_map = device_map

        self._update_generation_config_token_params(generation_config)
        self.generation_config = generation_config
        self.is_finetuned = is_finetuned

    @property
    def _model_dict(self) -> Dict[str, Hashable]:
        loading_params_str = ""
        if self.loading_params:
            loading_params_str = str(self.loading_params)

        generation_config_str = ""
        if self.generation_config:
            generation_config_str = self.generation_config.to_json_string(
                ignore_metadata=True
            )
        return {
            "model_name": self.model.name_or_path,
            "tokenizer_name": self.tokenizer.name_or_path,
            "loading_params": loading_params_str,
            "generation_config": generation_config_str,
        }

    def _update_generation_config_token_params(
        self, generation_config: transformers.GenerationConfig
    ):
        """Add eos and pad token to the generation config."""
        generation_config.update(eos_token_id=self.model.config.eos_token_id)
        generation_config.update(pad_token_id=self.model.config.pad_token_id)

    def _count_tokens(self, message: str) -> int:
        return len(self.tokenizer(message)["input_ids"])

    def _get_max_supported_new_tokens(self, message: str) -> int:
        """Get max supported new tokens."""
        max_supported_length = self.model.config.max_position_embeddings
        n_tokens = self._count_tokens(message)
        if n_tokens > max_supported_length:
            raise MaximumContextLengthExceededError(max_supported_length, n_tokens)
        return max_supported_length - n_tokens

    def _update_generation_config_max_new_tokens(
        self, generation_config: transformers.GenerationConfig, message: str
    ):
        """Update max_new_tokens in the config."""
        max_supported_new_tokens = self._get_max_supported_new_tokens(message)
        if (
            not generation_config.max_new_tokens
            or max_supported_new_tokens < generation_config.max_new_tokens
        ):
            generation_config.update(max_new_tokens=max_supported_new_tokens)

    @classmethod
    def from_config(
        cls, config: Union[dict, TextGenerationModelConfig]
    ) -> TextGenerationModel:
        if not isinstance(config, TextGenerationModelConfig):
            config = TextGenerationModelConfig.parse_obj(config)

        if config.loading_params:
            model_kwargs = config.loading_params.dict()
        else:
            model_kwargs = {}

        device_or_device_map = {}
        pipeline_kwargs = {}
        if config.device:
            device_or_device_map["device"] = config.device
            pipeline_kwargs["device"] = config.device
        elif config.device_map:
            device_or_device_map["device_map"] = config.device_map

        # NOTE: If `tokenizer_name` is not provided, then `model_name`
        # will be used to load the tokenizer.
        tokenizer_name = config.tokenizer_name
        if not tokenizer_name:
            tokenizer_name = config.model_name

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name,
            local_files_only=config.local_files_only,
            return_special_tokens_mask=True,
            **device_or_device_map,
        )

        if config.device_map:
            model_kwargs.update(device_or_device_map)

        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            local_files_only=config.local_files_only,
            **model_kwargs,
        )

        if config.device is not None:
            model = model.to(config.device)

        if "llama-2" in config.model_name and "7b" in config.model_name:
            model.config.max_position_embeddings = 4096

        pipeline = transformers.pipeline(
            task="text-generation", model=model, tokenizer=tokenizer, **pipeline_kwargs
        )

        message_formatter = None
        if config.is_finetuned:
            message_formatter = MessageFormatter(
                tokenizer=tokenizer,
                template=config.message_formatting_template,
            )

        generation_config = transformers.GenerationConfig(**config.generation_config)

        return cls(
            tokenizer=tokenizer,
            model=model,
            loading_params=config.loading_params,
            message_formatter=message_formatter,
            generation_config=generation_config,
            pipeline=pipeline,
            is_finetuned=config.is_finetuned,
        )

    def generate(self, prompt: str) -> str:
        generation_config = copy.deepcopy(self.generation_config)

        if self.message_formatter:
            prompt = self.message_formatter.format(prompt)

        self._update_generation_config_max_new_tokens(generation_config, prompt)

        inference_result = self.pipeline(
            prompt, return_full_text=False, generation_config=generation_config
        )
        return inference_result[0]["generated_text"]
