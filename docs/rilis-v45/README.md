# Menjalankan demo v45

## Empat contoh bawaan

| Foto | Kamera | Jumlah kendaraan | Cakupan visual |
|---|---|---:|---:|
| S033 | Paseban Selatan | 16 | 59,42% |
| W046 | Paseban Barat | 3 | 33,09% |
| NB006 | Bejen Barat | 13 | 46,77% |
| W041 | Paseban Barat | 0 | 0,00% |

## Pemasangan

Unduh [kode proyek (ZIP)](https://github.com/hanafishiddiq/tugas-akhir-pcd/archive/refs/heads/main.zip) dan ekstrak, atau clone repositori. Buka terminal di folder yang memuat `requirements.txt`. Buat lingkungan virtual dengan Python 3.12 64-bit.

| Sistem | Buat lingkungan | Aktifkan |
|---|---|---|
| Windows (Command Prompt) | `py -3.12 -m venv .venv` | `.venv\Scripts\activate.bat` |
| Linux atau macOS (bash/zsh) | `python3.12 -m venv .venv` | `source .venv/bin/activate` |

Jalankan perintah berikut di terminal yang sama:

```sh
python -m pip install -r requirements.txt
python src/v45/siapkan_model.py
python src/v45/demo.py --contoh S033
```

Pengunduh memeriksa SHA-256 setiap berkas sebelum menempatkannya di `src/v45/runs`. Untuk membawa bobot dari komputer lain, salin [model-v45.zip](https://github.com/hanafishiddiq/tugas-akhir-pcd/releases/download/v45.1/model-v45.zip) ke folder proyek dan jalankan `python src/v45/siapkan_model.py --arsip model-v45.zip`. Setelah pustaka dan model terpasang, demo dapat dijalankan tanpa internet. [SHA-256 arsip model](SHA256SUMS).

## Foto sendiri dan pemeriksaan hasil

```sh
python src/v45/demo.py --foto "foto saya.jpg" --kamera 04_bejen_barat
python src/v45/uji_contoh.py --laporan verifikasi.json
```

Pilih kamera asal foto: `01_paseban_selatan`, `02_paseban_barat`, atau `04_bejen_barat`. Pakai foto JPG atau PNG pada resolusi dan sudut kamera asli. Setiap proses menghasilkan `hasil.jpg` (kotak kendaraan dan ROI), `cakupan.png` (piksel jalan yang dihitung), dan `hasil.json` (rincian prediksi). Tambahkan `--hasil nama_folder` untuk memilih folder keluaran.

Perintah pemeriksaan menjalankan keempat foto validasi pada tabel di atas. Kotak, kelas, hitungan dan piksel masker dibandingkan dengan acuan yang sama pada setiap sistem. Untuk satu motor pada NB006, pengujian menghasilkan dua skor keyakinan, 0,882102 dan 0,877629. Kedua skor memberi kelas dan kotak yang sama. Pemeriksa menerima kedua nilai itu hanya pada deteksi tersebut. Untuk skor lain, toleransi absolut terhadap acuan adalah 0,000001. [Hasil uji deteksi dan kepadatan pada data uji](../../evaluasi/README.md).

## Sumber gambar

Foto bersumber dari CCTV Pemerintah Kabupaten Bantul / Diskominfo Bantul. Empat contoh berasal dari kamera Paseban Selatan, Paseban Barat dan Bejen Barat. Foto contoh mempertahankan piksel sumber. Visualisasi demo menambahkan ROI, kotak kendaraan dan teks. Panel evaluasi dan paper juga memotong gambar dan memasangkan anotasi dengan prediksi. Metadata pemutar yang disimpan pada saat akuisisi mencantumkan Creative Commons Attribution 4.0. [Portal CCTV Bantul](https://bantulkab.go.id/cctv.html) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Kartu dataset](../../data/README.md)
