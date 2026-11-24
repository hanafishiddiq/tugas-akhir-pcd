"""Unduh sekali dan periksa berkas model yang dipakai demo v45."""
import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
URL = "https://github.com/hanafishiddiq/tugas-akhir-pcd/releases/download/v45.1/model-v45.zip"


def periksa_model(root, expected):
    for name, reference in expected.items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"Berkas model belum lengkap: {name}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != reference:
                raise ValueError(f"SHA-256 berbeda untuk {name}")


def siapkan(archive):
    manifest = json.loads((HERE / "MANIFEST.json").read_bytes())
    expected = {k: v for k, v in manifest["sha256"].items() if k.startswith("runs/")}
    if not expected:
        raise ValueError("Manifest tidak memuat berkas model.")
    destination = HERE / "runs"
    if destination.exists():
        periksa_model(HERE, expected)
        return
    with tempfile.TemporaryDirectory(dir=HERE) as temp:
        staging = Path(temp)
        if archive is None:
            archive = staging / "model.zip"
            print("Mengunduh model v45 …", flush=True)
            with urllib.request.urlopen(URL, timeout=120) as source, archive.open("wb") as target:
                shutil.copyfileobj(source, target)
        with zipfile.ZipFile(archive) as bundle:
            if sorted(bundle.namelist()) != sorted(expected):
                raise ValueError("Daftar berkas ZIP tidak cocok dengan manifest v45.")
            for name in expected:
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
        periksa_model(staging, expected)
        try:
            (staging / "runs").rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            periksa_model(HERE, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arsip", type=Path, help="ZIP model v45 yang sudah diunduh")
    try:
        siapkan(parser.parse_args().arsip)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Gagal menyiapkan model: {exc}\n")
    print("Model v45 siap.")
