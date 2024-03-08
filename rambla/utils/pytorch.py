from typing import List

from torch import nn

SUPPORTED_DEVICES: List[str] = ["cpu", "cuda", "mps"]


def switch_model_device(model: nn.Module, device: str) -> nn.Module:
    """Switches the device of a model"""
    if device not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Device: {device} is invalid. Must be one of: {SUPPORTED_DEVICES}"
        )

    if str(model.device) != device:
        model = model.to(device)

    return model
