import json
import pickle
from pathlib import Path
from typing import Any, Union

import aiofiles
import yaml


async def async_dump(obj: Any, filepath: Union[Path, str]) -> None:
    """Asynchronous storing function."""
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    extension = filepath.suffix

    if extension == ".pkl":
        async with aiofiles.open(filepath, "wb") as f:
            byte_obj = pickle.dumps(obj)
            await f.write(byte_obj)
    elif extension == ".json":
        async with aiofiles.open(filepath, "w") as f:
            byte_obj = json.dumps(obj)
            await f.write(byte_obj)
    else:
        raise ValueError(f"{extension=} for {filepath=} not supported.")


async def async_load(filepath: Union[Path, str]) -> Any:
    """Asynchronous loading function."""
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    extension = filepath.suffix

    if extension == ".pkl":
        async with aiofiles.open(filepath, "rb") as f:
            byte_obj = await f.read()
        obj = pickle.loads(byte_obj)
    elif extension == ".json":
        async with aiofiles.open(filepath, "rb") as f:
            byte_obj = await f.read()
        obj = json.loads(byte_obj)
    else:
        raise ValueError(f"{extension=} for {filepath=} not supported.")

    return obj


def dump(obj: Any, filepath: Union[Path, str], **kwargs) -> None:
    """Function to dump an object to a filepath.

    Args
    ----
        obj (Any): object to be stored.
        filename (Path): path to dump the object.

    Raises
    ------
        ValueError: if storing method for extension is not implemented.
    """
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    extension = filepath.suffix

    if extension == ".json":
        with open(filepath, "w") as f:
            json.dump(obj, f, **kwargs)
    elif extension == ".pkl":
        with open(filepath, "wb") as f:
            pickle.dump(obj, f)
    elif extension == ".yaml":
        with open(filepath, "w") as f:
            yaml.dump(obj, f, **kwargs)
    else:
        raise ValueError(f"{extension=} for {filepath=} not supported.")


def load(filepath: Union[Path, str], **kwargs) -> Any:
    """Function to load an object from a file according to the extension.

    Args
    ----
        filepath (Path): path for object.

    Raises
    ------
        ValueError: if loading method for extension is not implemented.

    Returns
    -------
        Any: the object at the location
    """
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    extension = filepath.suffix

    if extension == ".json":
        with open(filepath, "r") as f:
            obj = json.load(f)
    elif extension == ".pkl":
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
    elif extension == ".yaml":
        with open(filepath, "r") as f:
            obj = yaml.load(f, Loader=yaml.Loader)
    else:
        raise ValueError(f"{extension=} for {filepath=} not supported.")

    return obj
