"""Check extracted package integrity and compare four fresh inference results."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def check():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--laporan", type=Path)
    parser.add_argument("--integritas", action="store_true", help="Periksa berkas tanpa menjalankan model")
    args = parser.parse_args()
    manifest = json.loads((HERE/"MANIFEST.json").read_text())
    for name, expected in manifest["sha256"].items():
        if digest(HERE/name) != expected:
            raise ValueError(f"Checksum tidak cocok: {name}")
    print(f"{len(manifest['sha256'])} berkas cocok dengan manifest.", flush=True)
    if args.integritas:
        return
    import numpy as np
    from PIL import Image
    examples = json.loads((HERE/"contoh.json").read_text())
    reference = json.loads((HERE/"acuan_demo.json").read_text())
    checks = []
    max_score_difference = 0.0
    changed_scores = []
    with tempfile.TemporaryDirectory(prefix="cek-hasil-v45-") as tmp:
        for ident, example in examples.items():
            out = Path(tmp)/ident
            command = [sys.executable, "-B", str(HERE/"demo.py"), "--hasil", str(out)]
            # Exercise the normal photo path too, not just the built-in examples.
            if ident == "NB006":
                command += ["--foto", str(HERE/example["source_image"]), "--kamera", example["camera_id"]]
            else:
                command += ["--contoh", ident]
            subprocess.run(command, check=True)
            (record,) = out.glob("*/hasil.json")
            value = json.loads(record.read_bytes())
            expected = reference[ident]
            for key in ("model_id", "model_provenance", "status", "camera_id", "image_sha256", "roi", "summary"):
                if value[key] != expected[key]:
                    raise ValueError(f"{ident}: {key} berbeda")
            for actual, target in zip(value["predictions"], expected["predictions"], strict=True):
                if actual.keys() != target.keys():
                    raise ValueError(f"{ident}: struktur prediksi berbeda")
                for key in actual:
                    if key == "confidence":
                        # HOG and floating-point kernels can differ slightly between CPU builds.
                        difference = abs(actual[key] - target[key])
                        max_score_difference = max(max_score_difference, difference)
                        if not np.isfinite(actual[key]):
                            raise ValueError(f"{ident}: skor deteksi tidak valid")
                        if difference > 1e-6:
                            changed_scores.append((ident, actual["class_name"], actual["bbox_xyxy"], target[key], actual[key]))
                    elif key == "contact_xy":
                        np.testing.assert_allclose(actual[key], target[key], rtol=0, atol=1e-9)
                    elif actual[key] != target[key]:
                        raise ValueError(f"{ident}: {key} berbeda")
            with Image.open(record.with_name("cakupan.png")) as mask:
                if hashlib.sha256(mask.convert("L").tobytes()).hexdigest() != expected["mask_pixels_sha256"]:
                    raise ValueError(f"{ident}: masker berbeda")
            with Image.open(record.with_name("hasil.jpg")) as overlay:
                overlay.load()
                if list(overlay.size) != expected["overlay_size"]:
                    raise ValueError(f"{ident}: ukuran gambar hasil berbeda")
            checks.append(dict(contoh=ident, kamera=value["camera_id"], **value["summary"]))
    report = dict(lulus=True, jumlah_berkas=len(manifest["sha256"]), model_id=value["model_id"], hasil=checks)
    report["selisih_skor_maksimum"] = max_score_difference
    print(f"Selisih skor maksimum: {max_score_difference:.3g}")
    if changed_scores:
        raise ValueError(f"Skor di luar acuan: {changed_scores}")
    if args.laporan:
        args.laporan.write_text(json.dumps(report, indent=2)+"\n")
    print("Keempat contoh cocok dengan keluaran v45 yang dibekukan.")
if __name__ == "__main__":
    check()
