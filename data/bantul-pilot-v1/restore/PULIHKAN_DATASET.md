# Memulihkan Bantul Pilot v1

Paket ini memulihkan dataset yang dikunci beserta kode, label, bukti audit, dan galeri tanpa mengambil rekaman video mentah. Gunakan daftar unduhan dan checksum dari rilis yang sama.

## Berkas yang dibutuhkan

- `restore_portable.py`, `build_restore_plan.py` dan `RESTORE_MANIFEST.json`.
- Enam ZIP bukti dan `SUPPORT_ARCHIVE_CHECKSUMS.json`. Setiap ZIP berisi subset objek berdasarkan SHA-256. Simpan nama archive seperti yang tercantum di sidecar.
- Berkas gambar dalam `DRIVE_DOWNLOADS.csv`. Daftar mencakup 222 gambar core lama, 215 overlay lama yang tetap dipakai galeri, enam overlay lama yang masih menjadi bukti audit, 26 gambar core tambahan, 33 overlay tambahan atau revisi, dan 12 crop auxiliary. Dua belas sumber asli auxiliary sudah ada di ZIP bukti.

Kolom `drive_url` memuat ID dari receipt metadata berkas lama dan 71 berkas baru atau diperbarui. Simpan berkas ke subfolder sesuai kolom `upload_group` bila ada atau `kind` bila kosong. Ini menjaga gambar asli dan overlay dengan nama sama tetap terpisah. Skrip juga mengenali berkas yang dinamai memakai SHA-256. Setiap unduhan diverifikasi dari isinya sebelum validator berjalan.

## Jalankan pemulihan

Gunakan Python 3.12 dengan NumPy dan Pillow seperti pada environment Tahap 03 proyek. Jalankan dari direktori yang memuat skrip dan manifest.

```sh
python -B restore_portable.py restore \
  --downloads /lokasi/unduhan-gambar \
  --archives /lokasi/enam-zip \
  --integrity /lokasi/enam-zip/SUPPORT_ARCHIVE_CHECKSUMS.json \
  --destination /lokasi/root-proyek-baru \
  --run-tests
```

Pilih direktori tujuan baru. Pemulihan menguji SHA-256 arsip, daftar anggota, ukuran byte setiap aset, SHA-256 semua jalur keluaran, dan kecocokan input, kode, aset, statistik serta split dengan lock asli. Bila pemeriksaan gagal, tujuan akhir tidak terbit. Opsi `--run-tests` menambahkan suite 25 uji. Buka `Data/Eksperimen/Bantul_Pilot_v1/index.html` di dalam folder yang dipulihkan untuk menjelajah galeri.

Enam puluh jalur absolut lama pada bukti audit dipetakan berdasarkan daftar izin yang terikat hash ke berkas di root hasil pemulihan. Adapter routing dijalankan di memori saat memanggil validator. Isi file anotasi, helper, protocol, dan split lock tetap persis. Gunakan kembali pemetaan yang sama saat memeriksa folder yang sudah dipulihkan.

```sh
python -B restore_portable.py verify --root /lokasi/root-proyek-baru --run-tests
```

Uji lokal menggunakan enam ZIP final dan salinan berhash sama sebagai pengganti unduhan Drive. Bukti hasilnya ada di `RESTORE_TEST.json`.
