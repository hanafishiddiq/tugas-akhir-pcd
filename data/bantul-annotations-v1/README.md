# Bantul annotations v1

222 images with AI visual cross-audit, explicit physical ROI membership and localized uncertainty. Training and split are pending.

[Original images](https://drive.google.com/drive/folders/1MaLqNqmCQNgn2FsZjGHeiZsNZtVYlwRE) · [Annotated previews and complete labels/audit](https://drive.google.com/drive/folders/1d3aNsyLqk_OYutcidcFhzXS0GG-C4Rq2)

# Anotasi CCTV Bantul — 222 gambar

Status 17 November 2026: Anotasi seluruh 222 gambar selesai: 2087 bbox, 87 gambar acuan hitung lengkap dan 135 gambar label parsial dengan ignore. Semua 222 menjalani self-review dan audit visual reviewer Sol berbeda; verifikasi struktur/integritas serta 444 perhitungan luas independen lulus. Ini audit AI, belum sertifikasi ground truth manusia. Belum ada split terkunci atau training.

Masukan adalah 222 gambar kurasi v3 dengan tiga ROI jalan yang telah dikunci. Bbox memakai koordinat piksel asli [xmin, ymin, xmax, ymax); kanan/bawah eksklusif. Bbox mencakup bagian kendaraan yang terlihat, termasuk pengendara/penumpang motor, tanpa menebak bagian tersembunyi. Kelas sasaran adalah mobil, motor, bus dan truk; pickup/cargo van dipisahkan sebagai other, kelas meragukan sebagai unknown.

Keanggotaan ROI ditentukan melalui pemeriksaan posisi fisik/kontak jalan. Atap atau kepala pengendara yang berada di luar poligon tidak otomatis membuat kendaraan di luar ROI. Objek pada batas atau tertutup yang belum dapat dipastikan diberi boundary_review/unknown dan alasan ignore. Jalan di luar cakupan pendekat sasaran tidak dianotasi secara menyeluruh dan tidak boleh dijadikan negatif training tanpa masker cakupan tambahan.

ANNOTATIONS.json adalah sumber utama. COCO.json menyertakan seluruh objek dengan atribut ignore dan area ignore pada images[].ignore_regions. Pembaca COCO umum belum tentu menerapkan kedua ekstensi ini; pembaca khusus wajib menggunakannya atau memakai ANNOTATIONS.json sebagai sumber utama. FULL_COUNT_REFERENCE.json hanya memuat frame dengan seluruh target yang terlihat dapat diputuskan. PARTIAL_REFERENCE.json menyimpan label yang jelas beserta ketidakpastian terlokalisasi; nilai hitungan penuh null, bukan nol.

Persentase cakupan = luas gabungan bbox target yang diterima dan dipotong masker ROI / luas masker ROI × 100%. Ini cakupan kotak pada bidang gambar, bukan segmentasi badan kendaraan, kepadatan fisik per meter, atau luas permukaan jalan sesungguhnya. Hitungan dan cakupan memakai daftar accepted_object_ids yang sama. Batas atas per kelas bersifat marginal dan tidak dijumlahkan untuk menebak total ketika kelas unknown.

Audit mencakup pemeriksaan gambar asli, overlay oleh anotator, pemeriksaan visual oleh reviewer Sol berbeda, dan validasi struktur/aritmetika independen. Lulus audit ini bukan sertifikasi ground truth oleh manusia. Jumlah objek lintas frame bukan jumlah kendaraan unik; episode_group masih provisional. Split terkunci, pelatihan, evaluasi model dan paper belum dikerjakan dalam tahap ini.

Batas atas cakupan bernilai null bila ada area ignore relevan yang tidak membatasi extent kendaraan. observed_envelope_area_px hanya gabungan persegi panjang yang tercatat untuk diagnosis; jangan menafsirkannya sebagai batas atas kendaraan tersembunyi.

Gunakan gambar asli yang ditautkan melalui IMAGE_INDEX.json untuk eksperimen. JPEG dalam overlays hanya untuk meninjau label; ada kotak, teks dan footer tambahan, sehingga tidak menjadi masukan training atau evaluasi model. Koordinat anotasi selalu mengacu pada dimensi gambar asli.


## Hasil per kamera

| Kamera | Gambar | Acuan lengkap | Parsial/ignore |
|---|---:|---:|---:|
| 01_paseban_selatan | 81 | 13 | 68 |
| 02_paseban_barat | 78 | 51 | 27 |
| 04_bejen_barat | 63 | 23 | 40 |

## Dukungan target yang diterima di ROI

| Kelas | Instance diterima | Gambar | Kelompok sementara |
|---|---:|---:|---:|
| mobil | 424 | 145 | 131 |
| motor | 1154 | 190 | 172 |
| bus | 14 | 12 | 12 |
| truk | 6 | 6 | 4 |

Instance lintas frame dapat menampilkan kendaraan yang sama. Kelompok sementara belum membuktikan episode independen. Dukungan ini meliputi bagian jelas dari frame parsial; evaluator hitungan penuh hanya boleh memakai FULL_COUNT_REFERENCE.json.

Berikutnya tinjau independensi kelompok lokasi/sesi/episode dan kebutuhan tambahan bus/truk, lalu kunci split dan jalankan pilot HOG-SVM pada gambar utuh. Aturan keanggotaan ROI otomatis masih perlu diuji.


Image binaries are on Drive. SOURCE_MANIFEST and original acquisition archives preserve the project-relative source paths. core.py/test_core.py implement the metric contract; run tests with `uv run --with numpy python code/test_core.py` from this directory. Source images and ROI masks are required for reproducing full integrity/area checks; scripts and raw audit inputs are included in LABEL_DAN_AUDIT.zip. Local visual review grids remain in the project workspace.
