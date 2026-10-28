# Analisis Kepadatan dan Klasifikasi Kendaraan pada Lalu Lintas

Tugas akhir mata kuliah **Pengolahan Citra Digital (PCD)**.

## Ruang lingkup

- **Input:** citra statis lalu lintas dari dataset.
- **Proses:** deteksi kendaraan dalam citra, klasifikasi jenis kendaraan, dan estimasi kepadatan visual di area jalan (*region of interest* / ROI).
- **Output yang ditargetkan:** citra beranotasi, kategori dan jumlah kendaraan per kelas, indikator kepadatan, serta hasil evaluasi.
- **Pendekatan:** pengolahan citra klasik dan *machine learning*.
- **Di luar scope:** video dan pelacakan antar-frame, pengendalian lampu lalu lintas, serta deteksi ambulans khusus.

## Struktur Repository

- `docs/` — ruang lingkup, antarmuka modul, dan aturan kolaborasi.
- `src/` — modul implementasi (belum dibuat selain inisialisasi paket).
- `scripts/` — rencana skrip training/evaluasi.
- `tests/` — pengujian otomatis.
- `configs/` — konfigurasi ROI, kelas, dan parameter eksperimen.
- `data/` — petunjuk data lokal; **dataset besar tidak disimpan di Git**.
