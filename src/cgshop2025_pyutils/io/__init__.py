import functools
from pathlib import Path

from ..data_schemas.instance import Cgshop2025Instance
from ..data_schemas.solution import Cgshop2025Solution


def open_file(func):
    """
    Decorator to open a file before calling the function and close it afterwards,
    if passed as string or pathlib.Path.
    """

    @functools.wraps(func)
    def wrapper(file, *args, **kwargs):
        if isinstance(file, str):
            file = Path(file)
        if isinstance(file, Path):
            with file.open() as f:
                return func(f, *args, **kwargs)
        return func(file, *args, **kwargs)

    return wrapper


@open_file
def read_instance(file) -> Cgshop2025Instance:
    """
    Read an instance from a file.
    :param file: File object or path to the file.
    :return: Instance object
    """
    content = file.read()
    return Cgshop2025Instance.model_validate_json(content)


@open_file
def read_solution(file) -> Cgshop2025Solution:
    """
    Read a solution from a file.
    :param file: File object or path to the file.
    :return: Solution object
    """
    content = file.read()
    return Cgshop2025Solution.model_validate_json(content)
