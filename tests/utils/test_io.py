import asyncio
from pathlib import Path
from typing import Generator
from unittest import mock

import aiofiles
import pytest
from aiofiles.threadpool.binary import AsyncBufferedIOBase

from rambla.utils import io

aiofiles.threadpool.wrap.register(mock.MagicMock)(  # type: ignore
    lambda *args, **kwargs: AsyncBufferedIOBase(*args, **kwargs)
)


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def test_async_dump_json(
    event_loop: asyncio.AbstractEventLoop,
) -> None:
    mock_file = mock.MagicMock()
    mock_save_path = mock.MagicMock(spec=Path)

    dump_obj = {"mock_param": "mock_value"}

    mock_save_path.suffix = ".json"

    with mock.patch.object(
        aiofiles.threadpool, "sync_open", return_value=mock_file
    ), mock.patch.object(io, "json") as mock_json:
        _ = event_loop.run_until_complete(io.async_dump(dump_obj, mock_save_path))

    mock_file.write.assert_called()
    mock_json.dumps.assert_called_once_with(dump_obj)


def test_async_dump_pickle(
    event_loop: asyncio.AbstractEventLoop,
) -> None:
    mock_file = mock.MagicMock()
    mock_save_path = mock.MagicMock(spec=Path)

    dump_obj = {"mock_param": "mock_value"}

    mock_save_path.suffix = ".pkl"

    with mock.patch.object(
        aiofiles.threadpool, "sync_open", return_value=mock_file
    ), mock.patch.object(io, "pickle") as mock_pickle:
        _ = event_loop.run_until_complete(io.async_dump(dump_obj, mock_save_path))

    mock_file.write.assert_called()
    mock_pickle.dumps.assert_called_once_with(dump_obj)


def test_async_dump_invalid_file(
    event_loop: asyncio.AbstractEventLoop,
) -> None:
    mock_file = mock.MagicMock()
    mock_save_path = mock.MagicMock(spec=Path)

    dump_obj = {"mock_param": "mock_value"}

    mock_save_path.suffix = ".invalid_file_suffix"

    with mock.patch.object(aiofiles.threadpool, "sync_open", return_value=mock_file):
        with pytest.raises(ValueError):
            _ = event_loop.run_until_complete(io.async_dump(dump_obj, mock_save_path))


def test_async_load_json(
    event_loop: asyncio.AbstractEventLoop,
) -> None:
    mock_file = mock.MagicMock()
    mock_save_path = mock.MagicMock(spec=Path)

    mock_save_path.suffix = ".json"

    with mock.patch.object(
        aiofiles.threadpool, "sync_open", return_value=mock_file
    ), mock.patch.object(io, "json") as mock_json:
        _ = event_loop.run_until_complete(io.async_load(mock_save_path))

    mock_file.read.assert_called()
    mock_json.loads.assert_called()


def test_async_load_pickle(
    event_loop: asyncio.AbstractEventLoop,
) -> None:
    mock_file = mock.MagicMock()
    mock_save_path = mock.MagicMock(spec=Path)

    mock_save_path.suffix = ".pkl"

    with mock.patch.object(
        aiofiles.threadpool, "sync_open", return_value=mock_file
    ), mock.patch.object(io, "pickle") as mock_pickle:
        _ = event_loop.run_until_complete(io.async_load(mock_save_path))

    mock_file.read.assert_called()
    mock_pickle.loads.assert_called()


def test_async_load_invalid_file(
    event_loop: asyncio.AbstractEventLoop,
) -> None:
    mock_file = mock.MagicMock()
    mock_save_path = mock.MagicMock(spec=Path)

    mock_save_path.suffix = ".invalid_file_suffix"

    with mock.patch.object(aiofiles.threadpool, "sync_open", return_value=mock_file):
        with pytest.raises(ValueError):
            _ = event_loop.run_until_complete(io.async_load(mock_save_path))


def test_dump_json() -> None:
    mock_save_path = mock.MagicMock(spec=Path)

    dump_obj = {"mock_param": "mock_value"}

    mock_save_path.suffix = ".json"

    with mock.patch("builtins.open") as mock_file, mock.patch.object(
        io, "json"
    ) as mock_json:
        io.dump(dump_obj, mock_save_path)

    assert mock_json.dump.called_once_with(dump_obj, mock_file)


def test_dump_pickle() -> None:
    mock_save_path = mock.MagicMock(spec=Path)

    dump_obj = {"mock_param": "mock_value"}

    mock_save_path.suffix = ".pkl"

    with mock.patch("builtins.open") as mock_file, mock.patch.object(
        io, "pickle"
    ) as mock_pickle:
        io.dump(dump_obj, mock_save_path)

    assert mock_pickle.dump.called_once_with(dump_obj, mock_file)


def test_dump_yaml() -> None:
    mock_save_path = mock.MagicMock(spec=Path)

    dump_obj = {"mock_param": "mock_value"}

    mock_save_path.suffix = ".yaml"

    with mock.patch("builtins.open") as mock_file, mock.patch.object(
        io, "yaml"
    ) as mock_yaml:
        io.dump(dump_obj, mock_save_path)

    assert mock_yaml.dump.called_once_with(dump_obj, mock_file)


def test_dump_invalid_file() -> None:
    mock_save_path = mock.MagicMock(spec=Path)

    dump_obj = {"mock_param": "mock_value"}

    mock_save_path.suffix = ".invalid_file_suffix"

    with mock.patch("builtins.open"):
        with pytest.raises(ValueError):
            io.dump(dump_obj, mock_save_path)


def test_load_json() -> None:
    mock_save_path = mock.MagicMock(spec=Path)

    mock_save_path.suffix = ".json"

    with mock.patch("builtins.open") as mock_file, mock.patch.object(
        io, "json"
    ) as mock_json:
        _ = io.load(mock_save_path)

    assert mock_json.load.called_once_with(mock_file)


def test_load_pickle() -> None:
    mock_save_path = mock.MagicMock(spec=Path)

    mock_save_path.suffix = ".pkl"

    with mock.patch("builtins.open") as mock_file, mock.patch.object(
        io, "pickle"
    ) as mock_pickle:
        _ = io.load(mock_save_path)

    assert mock_pickle.load.called_once_with(mock_file)


def test_load_yaml() -> None:
    mock_save_path = mock.MagicMock(spec=Path)

    mock_save_path.suffix = ".yaml"

    with mock.patch("builtins.open") as mock_file, mock.patch.object(
        io, "yaml"
    ) as mock_yaml:
        _ = io.load(mock_save_path)

    assert mock_yaml.load.called_once_with(mock_file)


def test_load_invalid_file() -> None:
    mock_save_path = mock.MagicMock(spec=Path)

    mock_save_path.suffix = ".invalid_file_suffix"

    with mock.patch("builtins.open"):
        with pytest.raises(ValueError):
            io.load(mock_save_path)
