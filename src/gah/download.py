"""Fetch the pinned upstream HAFusion arrays when they are absent locally."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from .data import ARRAY_FILES, LABEL_FILES


SOURCE_COMMIT = "73fbe911901ada71bb40129e19d86dedb9b9602c"
SOURCE_URL = f"https://github.com/MiRuacle24/HAFusion/archive/{SOURCE_COMMIT}.zip"
SOURCE_ROOT = f"HAFusion-{SOURCE_COMMIT}/"
DATA_DIRS = ("data_NY", "data_Chi", "data_SF")
FILES = tuple(dict.fromkeys((*ARRAY_FILES.values(), *LABEL_FILES.values())))


def ensure_data(root: Path) -> None:
    missing = [
        (directory, filename)
        for directory in DATA_DIRS
        for filename in FILES
        if not (root / directory / filename).is_file()
    ]
    if not missing:
        return
    print(f"Downloading HAFusion data from {SOURCE_URL}", flush=True)
    with urlopen(SOURCE_URL, timeout=120) as response:
        archive_bytes = response.read()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        for directory, filename in missing:
            data = archive.read(f"{SOURCE_ROOT}{directory}/{filename}")
            destination = root / directory / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)

