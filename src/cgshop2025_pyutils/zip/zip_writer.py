import zipfile
from pathlib import Path

from cgshop2025_pyutils.data_schemas.instance import Cgshop2025Instance
from cgshop2025_pyutils.data_schemas.solution import Cgshop2025Solution


class ZipWriter:
    def __init__(self, path: str | Path):
        self._path = str(path)
        if Path(self._path).exists():
            msg = f"File {self._path} already exists."
            raise FileExistsError(msg)
        self._zip = zipfile.ZipFile(path, "w")

    def add_instance(self, instance: Cgshop2025Instance):
        self._zip.writestr(
            f"{instance.instance_uid}.instance.json", instance.model_dump_json()
        )

    def add_solution(self, solution: Cgshop2025Solution):
        self._zip.writestr(
            f"{solution.instance_uid}.solution.json", solution.model_dump_json()
        )

    def close(self):
        self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
