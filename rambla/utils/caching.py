import functools
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from rambla.utils.io import async_dump, async_load, dump, load


def _dict_parser(d: dict) -> dict:
    """Recursively parses a dictionary to serialisable format."""
    for k, v in d.items():
        if isinstance(v, Path):
            d[k] = str(v)
        elif isinstance(v, dict):
            d[k] = _dict_parser(v)
    return d


def json_serialise(obj: dict, safe_parse: bool = True) -> str:
    """Serialises a dictionary.

    If `safe_parse=True` we attempt to parse certain types
    to a serialisable format.
    """
    try:
        return json.dumps(obj, sort_keys=True)
    except TypeError as err:
        if safe_parse:
            new_dict = _dict_parser(deepcopy(obj))
            return json_serialise(new_dict, safe_parse=False)
        raise err


class BaseFileCache:
    def __init__(
        self, cache_dir: Path, function: Callable = None, fname: str = "item.json"
    ):
        """Creates cache_dir if not already a directory.

        NOTE: if a function is provided it should accept named parameters only.
        """
        self.function = function
        self.fname = fname
        self.cache_dir = Path(cache_dir)

        if not self.cache_dir.is_dir():
            self.cache_dir.mkdir()

    @staticmethod
    def hash_dict(info: dict, safe_parse: bool = True) -> str:
        """Hashing function to be used for caching.

        NOTE: `info` should be serialisable.
        NOTE: if `safe_parse=True` we will attempt to parse
        certain types into a serialisable format.
        """
        hash_key = hashlib.md5()
        serialised_obj = json_serialise(info, safe_parse=safe_parse)
        hash_key.update(serialised_obj.encode())
        return hash_key.hexdigest()

    def get_filepath(self, info: dict, create: bool = False) -> Path:
        """Returns filepath based on the hash of the `info` dict.

        Option for creating it if it does not exist.
        """
        path = self.cache_dir / BaseFileCache.hash_dict(info)
        if create and not path.is_dir():
            path.mkdir()
        return path / self.fname

    def exists(self, info: dict) -> bool:
        """Checks whether filepath exists."""
        return self.get_filepath(info).is_file()


class AsyncFileCache(BaseFileCache):
    async def __call__(self, **kwargs) -> Any:
        if self.exists(kwargs):
            return await self.get(kwargs)

        out = await self.function(**kwargs)
        await self.put(out, kwargs)

        return out

    async def put(self, obj: Any, info: dict):
        filepath = self.get_filepath(info, create=True)
        await async_dump(obj, filepath)

    async def get(self, info: dict) -> Any:
        return await async_load(self.get_filepath(info))


class FileCache(BaseFileCache):
    def __call__(self, **kwargs) -> Any:
        if self.exists(kwargs):
            return self.get(kwargs)

        out = self.function(**kwargs)
        self.put(out, kwargs)
        return out

    def get(self, info: dict) -> Any:
        """Loads file from filepath derived from the hash of the `info` dict."""
        return load(self.get_filepath(info))

    def put(self, obj: Any, info: dict):
        """Stores file at filepath derived from the hash of the `info` dict."""
        dump(obj, self.get_filepath(info, create=True))


def file_cache(cache_dir: Path, fname: str) -> Callable:
    """Function decorator for caching for _synchronous_ base function."""

    def _cache(function: Callable) -> Callable:
        file_cache = FileCache(function=function, cache_dir=cache_dir, fname=fname)
        decorator = functools.wraps(function)(file_cache)
        return decorator

    return _cache


def async_file_cache(cache_dir: Path, fname: str) -> Callable:
    """Function decorator for caching for _asynchronous_ base function."""

    def _cache(function: Callable) -> Callable:
        file_cache = AsyncFileCache(function=function, cache_dir=cache_dir, fname=fname)
        decorator = functools.wraps(function)(file_cache)
        return decorator

    return _cache
