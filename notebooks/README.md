# Notebook Tugas Akhir PCD

Untuk pengumpulan, gunakan [analisis_kepadatan_bantul.ipynb](analisis_kepadatan_bantul.ipynb). Notebook menyimpan kode, foto, grafik, tabel, dan keluaran inferensi v45. [Versi HTML](analisis_kepadatan_bantul.html) membuka hasil yang sama langsung di browser saat luring.

## Menghitung ulang

Buka Jupyter dari folder yang memuat berkas `analisis_kepadatan_bantul.ipynb` dan jalankan *Restart Kernel and Run All*. Kernel referensi memakai Python 3.12, NumPy, Pillow, Matplotlib, dan OpenCV dengan HOG (`opencv-contrib-python-headless==5.0.0.93`). Mode bawaan `bukti` membaca cuplikan data dan prediksi beku yang telah ditanam di dalam notebook. Bila nama berkas diganti, arahkan variabel lingkungan `BANTUL_NOTEBOOK` ke berkas tersebut.

Untuk menghasilkan prediksi baru, ikuti [pemasangan demo](../docs/rilis-v45/README.md) lalu pasang Jupyter dengan interpreter yang sama: `python -m pip install jupyterlab matplotlib`. Dari folder proyek, jalankan `python -m jupyterlab` dan buka notebook. Pilih `MODE = "inferensi"` pada sel konfigurasi. Kernel akan menjalankan `demo.py` memakai Python yang sedang aktif. Jika notebook disalin ke folder lain, isi `PAKET` dengan lokasi folder `src/v45`. Gunakan `FOTO_BARU` dan `KAMERA` untuk memeriksa foto sendiri.

Pada S033, periksa 1 mobil, 13 motor, 2 bus, 0 truk dan cakupan 59,42%. Pada W041, periksa jumlah nol dan cakupan 0,00%. Sel evaluasi membaca kembali rekap deteksi per kelas untuk 76 foto, 76 catatan waktu dan 31 acuan lengkap untuk jumlah serta cakupan. Metrik deteksi dihitung dari jumlah TP, FP dan FN per kelas. Waktu yang ditampilkan berasal dari pengukuran uji resmi.

Sel riwayat memuat 45 iterasi. Ubah `SEMUA_ITERASI` untuk melihat rinciannya. Opsi `ULANG_BASELINE` memerlukan workspace pengembangan dan hanya menjalankan baseline awal pada folder keluaran baru.
