# Menjalankan demo v45

## Empat contoh bawaan

| Foto | Kamera | Jumlah kendaraan | Cakupan visual |
|---|---|---:|---:|
| S033 | Paseban Selatan | 16 | 59,42% |
| W046 | Paseban Barat | 3 | 33,09% |
| NB006 | Bejen Barat | 13 | 46,77% |
| W041 | Paseban Barat | 0 | 0,00% |

Dari folder hasil ekstraksi ZIP Mac, jalankan `./Jalankan.command --contoh W041` untuk mencoba jalan kosong. Perintah `./Jalankan.command --uji --laporan verifikasi.json` menjalankan ulang empat gambar. Pemeriksa membandingkan kelas, kotak, hitungan, ROI, piksel masker, dan identitas model. Empat foto berasal dari validasi. Hasil uji resmi tersedia di [laporan evaluasi](../../evaluasi/README.md).

## Menjalankan sumber dari clone

Langkah berikut untuk Mac Apple Silicon. Mulai dari root clone, dengan berkas ZIP rilis di folder Downloads. Periksa sidik jarinya sebelum memuat model. Lanjutkan ke ekstraksi setelah `shasum` menampilkan `OK`.

```sh
zip_file="$HOME/Downloads/PoC_Bantul_v45_macOS_arm64.zip"
(cd "$(dirname "$zip_file")" && shasum -a 256 -c -) < docs/rilis-v45/SHA256SUMS
```

```sh
mkdir -p .paket
ditto -x -k "$zip_file" .paket
```

Hubungkan bobot dengan kode sumber satu kali pada clone baru. Sesudahnya, gunakan interpreter paket untuk menjalankan sumber yang ada di repositori. Kode, definisi ROI, dan foto acuan yang dipakai perintah berikut berasal dari checkout.

```sh
ln -s ../../.paket/PoC_Bantul_v45_macOS_arm64/runs src/v45/runs
python_paket=".paket/PoC_Bantul_v45_macOS_arm64/runtime/bin/python3.12"
"$python_paket" -EsB src/v45/demo.py --contoh S033
"$python_paket" -EsB src/v45/uji_contoh.py
```

Pada Mac yang sudah memiliki paket hasil ekstraksi, pengguna demo cukup menjalankan `Jalankan.command` dari paket itu. Berkas [requirements.txt](../../requirements.txt) mencatat versi pustaka inferensi untuk pengembang. Pengujian paket dilakukan pada macOS arm64 dengan Python 3.12.13.

## Sumber gambar

Foto bersumber dari CCTV Pemerintah Kabupaten Bantul / Diskominfo Bantul. Empat contoh berasal dari kamera Paseban Selatan, Paseban Barat dan Bejen Barat. Foto contoh mempertahankan piksel sumber. Visualisasi demo menambahkan ROI, kotak kendaraan dan teks. Panel evaluasi dan paper juga memotong gambar dan memasangkan anotasi dengan prediksi. Metadata pemutar yang disimpan pada saat akuisisi mencantumkan Creative Commons Attribution 4.0. [Portal CCTV Bantul](https://bantulkab.go.id/cctv.html) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Kartu dataset](../../data/bantul-pilot-v1/DATASET_CARD.md)
