import asyncio
import types
from pathlib import Path
from typing import Generator
from unittest import mock

import pytest

from rambla.utils.caching import (
    AsyncFileCache,
    BaseFileCache,
    FileCache,
    file_cache,
    json_serialise,
)
from rambla.utils.io import dump


def test_BaseFileCache__init__(tmpdir):  # noqa: N802
    func = mock.MagicMock()
    cache_dir = Path(tmpdir) / "dummy_path"
    fname = "item.json"
    file_cache = BaseFileCache(function=func, cache_dir=cache_dir, fname=fname)

    assert cache_dir.is_dir()
    assert file_cache.fname == fname
    assert file_cache.cache_dir == cache_dir


def test_BaseFileCache_hash_dict_snapshot():  # noqa: N802
    info = {"a": 1, "b": 3}
    output_hash = BaseFileCache.hash_dict(info)
    assert output_hash == "f0dafc6d23ce6bd6fbaa5c94a2c9a2c0"


def test_json_serialise_fail():
    obj = {"a": 1, "b": Path("dummy/path")}
    with pytest.raises(TypeError):
        json_serialise(obj, False)


def test_json_serialise_with_path():
    obj = {"a": 1, "b": Path("dummy/path")}
    out = json_serialise(obj, True)
    assert out == '{"a": 1, "b": "dummy/path"}'


def test_json_serialise_with_path_nested():
    obj = {"outer": {"a": 1, "b": Path("dummy/path")}}
    out = json_serialise(obj, True)
    assert out == '{"outer": {"a": 1, "b": "dummy/path"}}'


def test_BaseFileCache_getfilepath_create(tmpdir):  # noqa: N802
    func = mock.MagicMock()
    cache_dir = Path(tmpdir) / "dummy_path"
    fname = "item.json"
    file_cache = BaseFileCache(function=func, cache_dir=cache_dir, fname=fname)

    info = {"a": 1, "b": 3}
    output_filepath = file_cache.get_filepath(info, create=True)

    assert output_filepath.parent.is_dir()


def test_BaseFileCache_getfilepath_fname(tmpdir):  # noqa: N802
    func = mock.MagicMock()
    cache_dir = Path(tmpdir) / "dummy_path"
    fname = "item.json"
    file_cache = BaseFileCache(function=func, cache_dir=cache_dir, fname=fname)

    info = {"a": 1, "b": 3}
    output_filepath = file_cache.get_filepath(info)
    assert output_filepath.name == fname


def test_BaseFileCache_getfilepath_not_create(tmpdir):  # noqa: N802
    func = mock.MagicMock()
    cache_dir = Path(tmpdir) / "dummy_path"
    fname = "item.json"
    file_cache = BaseFileCache(function=func, cache_dir=cache_dir, fname=fname)

    info = {"a": 1, "b": 3}
    output_filepath = file_cache.get_filepath(info)

    assert not output_filepath.parent.is_dir()


def test_BaseFileCache_exists_true(tmpdir):  # noqa: N802
    func = mock.MagicMock()
    cache_dir = Path(tmpdir) / "dummy_path"
    fname = "item.json"
    file_cache = BaseFileCache(function=func, cache_dir=cache_dir, fname=fname)

    info = {"a": 1, "b": 3}
    path = cache_dir / BaseFileCache.hash_dict(info)
    path.mkdir()
    dump("dummy_object", path / "item.json")
    output_exists = file_cache.exists(info)
    assert output_exists


def test_BaseFileCache_exists_false(tmpdir):  # noqa: N802
    func = mock.MagicMock()
    cache_dir = Path(tmpdir) / "dummy_path"
    fname = "item.json"
    file_cache = BaseFileCache(function=func, cache_dir=cache_dir, fname=fname)

    info = {"a": 1, "b": 3}
    output_exists = file_cache.exists(info)
    assert not output_exists


def test_FileCache__init__(tmpdir):  # noqa: N802
    func = mock.MagicMock()
    cache_dir = Path(tmpdir) / "dummy_path"
    fname = "item.json"
    file_cache = FileCache(function=func, cache_dir=cache_dir, fname=fname)

    assert cache_dir.is_dir()
    assert file_cache.fname == fname
    assert file_cache.cache_dir == cache_dir


def test_AsyncFileCache__init__(tmpdir):  # noqa: N802
    func = mock.MagicMock()
    cache_dir = Path(tmpdir) / "dummy_path"
    fname = "item.json"
    file_cache = AsyncFileCache(function=func, cache_dir=cache_dir, fname=fname)

    assert cache_dir.is_dir()
    assert file_cache.fname == fname
    assert file_cache.cache_dir == cache_dir


@mock.patch("rambla.utils.caching.load")
def test_FileCache_get(mock_load, tmpdir):  # noqa: N802
    func = mock.MagicMock()
    cache_dir = Path(tmpdir) / "dummy_cache_dir"
    fname = "item.json"
    file_cache = FileCache(function=func, cache_dir=cache_dir, fname=fname)

    # mocking
    mock_load_return_value = "dummy load output"
    mock_load.return_value = mock_load_return_value

    mock_get_filepath = mock.MagicMock()
    mock_get_filepath_return_value = "dummy get_filepath output"
    mock_get_filepath.return_value = mock_get_filepath_return_value
    file_cache.get_filepath = mock_get_filepath

    info = {"a": 1, "b": 2}

    # call
    output = file_cache.get(info)

    # checks
    assert output == mock_load_return_value
    mock_load.assert_called_once()
    mock_load.assert_called_with(mock_get_filepath_return_value)

    mock_get_filepath.assert_called_with(info)
    mock_get_filepath.assert_called_once()


@mock.patch("rambla.utils.caching.dump")
def test_FileCache_put(mock_dump, tmpdir):  # noqa: N802
    func = mock.MagicMock()
    cache_dir = Path(tmpdir) / "dummy_cache_dir"
    fname = "item.json"
    file_cache = FileCache(function=func, cache_dir=cache_dir, fname=fname)
    dummy_object = "dummy object"

    # mocking
    mock_get_filepath = mock.MagicMock()
    mock_get_filepath_return_value = "dummy get_filepath output"
    mock_get_filepath.return_value = mock_get_filepath_return_value
    file_cache.get_filepath = mock_get_filepath

    info = {"a": 1, "b": 2}

    # call
    file_cache.put(dummy_object, info)

    # checks
    mock_dump.assert_called_once()
    mock_dump.assert_called_with(dummy_object, mock_get_filepath_return_value)

    mock_get_filepath.assert_called_with(info, create=True)
    mock_get_filepath.assert_called_once()


def test_FileCache__call__file_does_not_exist(tmpdir):  # noqa: N802
    mock_function = mock.MagicMock()
    mock_function_return_value = "dummy mock function return value"
    mock_function.return_value = mock_function_return_value

    cache_dir = Path(tmpdir) / "dummy_cache_dir"
    fname = "item.json"
    file_cache = FileCache(function=mock_function, cache_dir=cache_dir, fname=fname)

    # mocking
    mock_exists = mock.MagicMock()
    mock_exists.return_value = False
    file_cache.exists = mock_exists

    mock_put = mock.MagicMock()
    file_cache.put = mock_put

    mock_get = mock.MagicMock()
    file_cache.get = mock_get

    # call
    dummy_input = "dummy input value"
    output = file_cache(dummy_input=dummy_input)

    # checks
    assert output == mock_function_return_value

    mock_put.assert_called_once()
    mock_put.assert_called_with(
        mock_function_return_value, {"dummy_input": dummy_input}
    )

    mock_get.assert_not_called()

    mock_exists.assert_called_once()
    mock_exists.assert_called_with({"dummy_input": dummy_input})


def test_FileCache__call__file_exists(tmpdir):  # noqa: N802
    mock_function = mock.MagicMock()

    cache_dir = Path(tmpdir) / "dummy_cache_dir"
    fname = "item.json"
    file_cache = FileCache(function=mock_function, cache_dir=cache_dir, fname=fname)

    # mocking
    mock_exists = mock.MagicMock()
    mock_exists.return_value = True
    file_cache.exists = mock_exists

    mock_put = mock.MagicMock()
    file_cache.put = mock_put

    mock_get = mock.MagicMock()
    mock_get_return_value = "mock_get return_value"
    mock_get.return_value = mock_get_return_value
    file_cache.get = mock_get

    # call
    dummy_input = "dummy input value"
    output = file_cache(dummy_input=dummy_input)

    # checks
    assert output == mock_get_return_value

    mock_put.assert_not_called()

    mock_get.assert_called_once()
    mock_get.assert_called_with({"dummy_input": dummy_input})

    mock_exists.assert_called_once()
    mock_exists.assert_called_with({"dummy_input": dummy_input})


@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@mock.patch("rambla.utils.caching.async_load")
def test_AsyncFileCache_get(  # noqa: N802
    mock_load,
    tmpdir,
    event_loop: asyncio.AbstractEventLoop,
):
    func = mock.MagicMock()
    cache_dir = Path(tmpdir) / "dummy_cache_dir"
    fname = "item.json"
    file_cache = AsyncFileCache(function=func, cache_dir=cache_dir, fname=fname)

    # mocking
    mock_load_return_value = "dummy load output"
    mock_load.return_value = mock_load_return_value

    mock_get_filepath = mock.MagicMock()
    mock_get_filepath_return_value = "dummy get_filepath output"
    mock_get_filepath.return_value = mock_get_filepath_return_value
    file_cache.get_filepath = mock_get_filepath

    info = {"a": 1, "b": 2}

    # call
    output = event_loop.run_until_complete(file_cache.get(info))

    # checks
    assert output == mock_load_return_value
    mock_load.assert_called_once()
    mock_load.assert_called_with(mock_get_filepath_return_value)

    mock_get_filepath.assert_called_with(info)
    mock_get_filepath.assert_called_once()


@mock.patch("rambla.utils.caching.async_dump")
def test_AsyncFileCache_put(  # noqa: N802
    mock_dump,
    tmpdir,
    event_loop: asyncio.AbstractEventLoop,
):  # noqa: N802
    func = mock.MagicMock()
    cache_dir = Path(tmpdir) / "dummy_cache_dir"
    fname = "item.json"
    file_cache = AsyncFileCache(function=func, cache_dir=cache_dir, fname=fname)
    dummy_object = "dummy object"

    # mocking
    mock_get_filepath = mock.MagicMock()
    mock_get_filepath_return_value = "dummy get_filepath output"
    mock_get_filepath.return_value = mock_get_filepath_return_value
    file_cache.get_filepath = mock_get_filepath

    info = {"a": 1, "b": 2}

    # call
    event_loop.run_until_complete(file_cache.put(dummy_object, info))

    # checks
    mock_dump.assert_called_once()
    mock_dump.assert_called_with(dummy_object, mock_get_filepath_return_value)

    mock_get_filepath.assert_called_with(info, create=True)
    mock_get_filepath.assert_called_once()


async def mock_async_function(*args, **kwargs):
    return "dummy mock function return value"


def test_AsyncFileCache__call__file_does_not_exist(tmpdir, event_loop):  # noqa: N802
    mock_function = mock_async_function

    cache_dir = Path(tmpdir) / "dummy_cache_dir_00"
    fname = "item.json"
    file_cache = AsyncFileCache(
        function=mock_function, cache_dir=cache_dir, fname=fname
    )

    # mocking
    mock_exists = mock.MagicMock()
    mock_exists.return_value = False
    file_cache.exists = mock_exists

    async def async_mock_put(*args, **kwargs):
        return None

    file_cache.put = types.MethodType(async_mock_put, file_cache)

    async def async_mock_get(*args, **kwargs):
        return "mock_get return_value"

    file_cache.get = types.MethodType(async_mock_get, file_cache)

    # call
    dummy_input = "dummy input value"

    output = event_loop.run_until_complete(file_cache(dummy_input=dummy_input))

    # checks
    assert output == "dummy mock function return value"


def test_AsyncFileCache__call__file_exists(tmpdir, event_loop):  # noqa: N802
    mock_function = mock.MagicMock()

    cache_dir = Path(tmpdir) / "dummy_cache_dir"
    fname = "item.json"
    file_cache = AsyncFileCache(
        function=mock_function, cache_dir=cache_dir, fname=fname
    )

    # mocking
    mock_exists = mock.MagicMock()
    mock_exists.return_value = True
    file_cache.exists = mock_exists

    async def async_mock_put(*args, **kwargs):
        return None

    file_cache.put = types.MethodType(async_mock_put, file_cache)

    async def async_mock_get(*args, **kwargs):
        return "mock_get return_value"

    file_cache.get = types.MethodType(async_mock_get, file_cache)

    # call
    dummy_input = "dummy input value"

    output = event_loop.run_until_complete(file_cache(dummy_input=dummy_input))

    # checks
    assert output == "mock_get return_value"


def test_file_cache(tmpdir):
    mock_func = mock.MagicMock()
    dummy_return_value = "dummy return value"
    mock_func.return_value = dummy_return_value

    def wrapper_func(first_arg, second_arg):
        return mock_func(first_arg, second_arg)

    cache_dir = Path(tmpdir) / "dummy_path"
    fname = "item.json"

    cached_function = file_cache(cache_dir=cache_dir, fname=fname)(wrapper_func)

    out_1 = cached_function(first_arg=1, second_arg=2)
    out_2 = cached_function(first_arg=1, second_arg=2)

    assert out_1 == dummy_return_value
    assert out_2 == dummy_return_value
    mock_func.assert_called_once()
