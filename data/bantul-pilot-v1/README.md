# Bantul Pilot v1

Paket pilot empat kelas untuk CCTV tetap, dikunci pada 17 November 2026 sebelum training. Paket mencakup 248 gambar utama, termasuk 10 embargo dan 2 karantina, serta 12 crop Palbapang khusus pelatihan classifier. Pembagian aktif berisi 99 gambar train, 61 validation, dan 76 test. Evaluasi dirancang sebagai holdout temporal pada kamera tetap. Tahap ini memverifikasi kesiapan data dan kontrak evaluasi. Hasil training belum tersedia.

- [Kartu dataset](DATASET_CARD.md) dan [kontrak penggunaan](DATA_CONTRACT.md)
- [Pembagian terkunci](SPLIT_LOCK.json) dan [laporan kesiapan](READINESS_REPORT.json)
- [Indeks sumber dan atribusi](SOURCE_INDEX.json) dan [jejak revisi anotasi](REVISION_LINEAGE.json)
- [Log 25 tes paket](code/TEST_RESULTS.txt) dan [panduan workspace asli](WORKSPACE_README.md)
- [Indeks 260 gambar dan crop](DAFTAR_GAMBAR.csv) dan [panduan pemulihan](restore/PULIHKAN_DATASET.md)
- [Gambar, overlay, dan berkas pemulihan di Drive](https://drive.google.com/drive/folders/1dL_hXMOnOQvsBkwFW4IsZBJYo-ulJcR5)

## Pemulihan dan verifikasi

GitHub menyimpan metadata serta kode. Unduh media dan paket pendukung dari folder Drive. Ikuti panduan dan manifest pemulihan yang menyertai paket untuk menempatkan aset ke jalur proyek yang tercatat. Gunakan `Data/Eksperimen/Bantul_Pilot_v1/` untuk paket pilot dan pertahankan jalur relatif pada `INPUT_PATHS.json`. Aset pendukung mencakup paket anotasi historis, ROI, dan bukti audit untuk fixture dan validator.

Sebagian bukti historis memakai jalur absolut sebagai entri terikat hash. Panduan pemulihan memetakan jalur tersebut ke workspace portabel tanpa mengubah byte metadata terkunci. `CHECKSUMS.json` mencatat susunan paket workspace, termasuk `README.md` asli yang disalin ke GitHub sebagai `WORKSPACE_README.md`, dan overlay yang disimpan di Drive. Galeri `index.html` bekerja setelah struktur workspace dan media dipulihkan.

Dari root workspace yang telah dipulihkan, gunakan Python 3.12 dan dependensi dalam [requirements.txt](requirements.txt). Ikuti contoh pemanggilan pada panduan Drive untuk workspace portabel. Perintah berikut berlaku dalam workspace asli.

```sh
P=Data/Eksperimen/Bantul_Pilot_v1
python -B "$P/code/run_tests.py"
python -B "$P/code/readiness.py" --root "$PWD" --paths "$P/INPUT_PATHS.json"
```

Gunakan `assignments` pada `SPLIT_LOCK.json` untuk pembagian model. File berakhiran `_PROPOSED` merekam snapshot adaptor sebelum penguncian.

CSV pada GitHub memakai akhir baris LF; salinan Drive memakai CRLF. Nilai seluruh baris tetap sama. Hash pada manifest publikasi mengacu pada byte berkas Drive.
