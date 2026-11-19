"""Demo lokal kepadatan visual pada satu gambar CCTV Bantul."""
import argparse
import hashlib
import io
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PROJECT = HERE
STAGE4 = HERE
sys.path.insert(0, str(STAGE4 / "kode"))
from refined_inference import RefinedModel
from vision import save_json, sha
from tampilan import angka, persen, simpan

PIPELINE = STAGE4 / "runs/localization_20261117_v45/PIPELINE.json"
PIPELINE_HASH = "34931e28b9c5707eb0b002363522c21b3a087bc1db2a2ead3b86330bd16d0dcc"
MODEL_ID = "9bb33935eec4e5c174b3"
CAMERAS = {"01_paseban_selatan": "Paseban Selatan", "02_paseban_barat": "Paseban Barat",
           "04_bejen_barat": "Bejen Barat"}
EXAMPLES = json.loads((HERE / "contoh.json").read_text())


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--contoh", choices=EXAMPLES, help="Contoh lokal (bawaan: S033)")
    inputs.add_argument("--foto", type=Path, help="Foto asli JPG/PNG dari kamera terpilih")
    parser.add_argument("--kamera", choices=CAMERAS, help="Identitas kamera untuk --foto")
    parser.add_argument("--hasil", type=Path, default=HERE/"hasil", help="Folder induk untuk hasil baru")
    args = parser.parse_args()
    if bool(args.foto) != bool(args.kamera):
        parser.error("Untuk foto sendiri, berikan --foto dan --kamera bersama.")
    return args


def load_frame(path, camera, spec, example):
    if not path.is_file():
        raise ValueError(f"Foto tidak ditemukan: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if example and digest != EXAMPLES[example]["source_sha256"]:
        raise ValueError("Berkas contoh berbeda dari sumber yang diverifikasi.")
    for item in EXAMPLES.values():
        if item["source_sha256"] == digest and item["camera_id"] != camera:
            raise ValueError("Kamera yang dipilih berbeda dari kamera sumber foto contoh.")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.format not in ("PNG", "JPEG") or getattr(im, "n_frames", 1) != 1:
                raise ValueError("Gunakan satu gambar JPG atau PNG statis.")
            expected = (spec["width"], spec["height"])
            if im.size != expected:
                raise ValueError(f"Ukuran {im.width}×{im.height}. {CAMERAS[camera]} memerlukan "
                                 f"{expected[0]}×{expected[1]} dengan sudut kamera asli.")
            pixels = np.asarray(im.convert("RGB"))
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Foto tidak dapat dibaca sebagai JPG/PNG yang utuh.") from exc
    return pixels, digest


def run(args):
    key = None if args.foto else args.contoh or "S033"
    path = PROJECT/EXAMPLES[key]["source_image"] if key else args.foto.expanduser()
    camera = EXAMPLES[key]["camera_id"] if key else args.kamera
    rois = json.loads((PROJECT/"ROIS.json").read_text())
    rgb, digest = load_frame(path, camera, rois[camera], key)
    if sha(PIPELINE) != PIPELINE_HASH:
        raise ValueError("Checksum konfigurasi tidak cocok dengan v45.")
    target = args.hasil.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    model = RefinedModel(STAGE4/"runs", PIPELINE)
    if model.model_id != MODEL_ID or model.rois[camera] != rois[camera]:
        raise ValueError("Identitas model atau ROI berbeda dari versi demo.")
    print(f"Menganalisis {path.name} · {CAMERAS[camera]} · v45 …", file=sys.stderr, flush=True)
    with threadpool_limits(limits=1):
        result = model.predict(rgb, camera)
    if result["status"] != "ok":
        raise ValueError("Inferensi tidak selesai.")
    result.update(image_sha256=digest, source_name=path.name, example=key, roi=model.rois[camera]["polygon_xy"],
                  image_size=[rgb.shape[1],rgb.shape[0]], roi_version=model.rois[camera]["version"],
                  coverage_definition="100 × luas gabungan kotak inside di dalam ROI / luas ROI (piksel)")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = target / f"{key or camera}_{stamp}_{uuid4().hex[:10]}"
    with tempfile.TemporaryDirectory(dir=target, prefix=".proses-") as tmp:
        staging = Path(tmp)
        simpan(rgb, result, model.rois[camera], CAMERAS[camera], staging)
        save_json(staging/"hasil.json",result)
        staging.rename(out)
    s = result["summary"]
    print(f"{CAMERAS[camera]}  |  {s['vehicle_count']} kendaraan  |  Kepadatan visual {persen(s['coverage_percent'])}")
    print(" · ".join(f"{k.capitalize()} {v}" for k,v in s["count_by_class"].items()))
    print(f"Cakupan: {angka(s['covered_area_px'])} dari {angka(s['roi_area_px'])} piksel ROI.")
    print(f"Gambar: {out/'hasil.jpg'}\nRincian: {out/'hasil.json'}\nMasker: {out/'cakupan.png'}")


if __name__ == "__main__":
    cv2.setNumThreads(1)
    args = options()
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        print(f"Gagal memproses gambar: {exc or type(exc).__name__}", file=sys.stderr)
        sys.exit(1)
