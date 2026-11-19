"""Resolve references to the frozen run tree inside this portable release."""
from pathlib import Path, PurePosixPath


def lokasi_aset(value):
    parts = PurePosixPath(value).parts
    if "runs" not in parts or ".." in parts:
        raise ValueError(f"Referensi aset model tidak dikenali: {value}")
    suffix = parts[parts.index("runs") + 1:]
    return Path(__file__).resolve().parent.parent.joinpath("runs", *suffix)
