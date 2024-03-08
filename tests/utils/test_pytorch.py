from unittest import mock

import pytest
from torch import nn

from rambla.utils import pytorch


@mock.patch.object(pytorch, "SUPPORTED_DEVICES", new=["mock_device"])
def test_switch_model_device_same_device() -> None:
    mock_device = "mock_device"

    mock_model = mock.MagicMock(spec=nn.Module)
    mock_model.device = mock_device

    mock_model = pytorch.switch_model_device(mock_model, mock_device)

    assert mock_model.device == mock_device
    assert not mock_model.to.called


@mock.patch.object(pytorch, "SUPPORTED_DEVICES", new=["mock_device1", "mock_device2"])
def test_switch_model_device_different_device() -> None:
    current_device = "mock_device1"
    new_device = "mock_device2"

    mock_to = mock.MagicMock()
    mock_model = mock.MagicMock(spec=nn.Module)
    mock_model.device = current_device
    mock_model.to = mock_to

    mock_model = pytorch.switch_model_device(mock_model, new_device)
    mock_to.assert_called_once_with(new_device)


@mock.patch.object(pytorch, "SUPPORTED_DEVICES", new=["mock_device"])
def test_switch_model_device_invalid_device() -> None:
    mock_model = mock.MagicMock(spec=nn.Module)
    with pytest.raises(ValueError):
        _ = pytorch.switch_model_device(mock_model, "invalid_device")
