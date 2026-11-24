# Analisis Kepadatan dan Klasifikasi Kendaraan pada Lalu Lintas

[Paper IEEE (PDF)](paper/paper.pdf) · [Notebook dengan hasil](notebooks/analisis_kepadatan_bantul.ipynb) · [Jalankan demo Python](docs/rilis-v45/README.md)

Untuk tugas akhir Pengolahan Citra Digital, kami mencoba menghitung kendaraan yang mengantre di Simpang Paseban dan Simpang Bejen, Bantul. Masukannya satu foto CCTV. Program menandai mobil, motor, bus, dan truk, lalu menghitung berapa persen area jalan di foto yang tertutup kotak-kotak kendaraan.

![Antrean di Paseban Selatan dengan batas area jalan dan kotak hasil deteksi](docs/rilis-v45/S033.jpg)

Pada foto Paseban Selatan ini (S033), program menemukan 13 motor, 1 mobil, dan 2 bus. Totalnya 16 kendaraan, sama dengan hitungan pada anotasi. Kotak hasil deteksi menutupi 59,42% area jalan yang kami tandai, sedangkan kotak pada anotasi menutupi 60,78%. Bagian kotak yang bertumpuk dihitung sekali. Foto berasal dari CCTV Diskominfo Bantul. Garis batas dan kotak kendaraan kami tambahkan untuk analisis. [Sumber gambar dan lisensi](docs/rilis-v45/README.md#sumber-gambar)

## Menjalankan demo

Buka terminal di folder proyek dan aktifkan lingkungan virtual Python 3.12. Langkahnya tersedia di [panduan Windows, Linux, dan macOS](docs/rilis-v45/README.md#pemasangan). Kemudian jalankan:

```sh
python -m pip install -r requirements.txt
python src/v45/siapkan_model.py
python src/v45/demo.py --contoh S033
```

Perintah kedua mengunduh bobot model satu kali. Pada S033, hasilnya 1 mobil, 13 motor, 2 bus, dan cakupan 59,42%. Program menampilkan folder yang berisi gambar beranotasi, masker cakupan, dan angka dalam JSON. [Contoh foto sendiri, pemasangan luring, dan uji empat foto](docs/rilis-v45/README.md).

## Hasil pengujian

Model ini memakai pengolahan citra klasik, termasuk HOG untuk membaca pola tepi dan SVM untuk mengenali kendaraan. Kami memakai 99 gambar utama untuk pelatihan dan 61 gambar untuk validasi. Model yang kami pilih mendapat macro-F1 85,50% pada validasi. Saat diuji pada 76 gambar lain dari tiga kamera yang sama, skornya turun menjadi 59,11%. Bus dan truk masih sering terlewat. Dari 8 bus dan 6 truk pada anotasi uji, model menemukan 3 bus dan 2 truk dengan benar.

Untuk menilai jumlah dan kepadatan, kami memakai 31 gambar uji yang anotasinya lengkap. Rata-rata selisih absolut terhadap anotasi adalah 0,45 kendaraan per gambar untuk jumlah dan 3,32 poin persentase untuk cakupan jalan. [Tabel hasil per kelas dan contoh kesalahan deteksi](evaluasi/README.md)

[Data dan anotasi](data/README.md) · [Kode model](src/v45/) · [Sumber LaTeX](paper/paper.tex) · [Unduh notebook sebagai HTML](https://github.com/hanafishiddiq/tugas-akhir-pcd/raw/refs/heads/main/notebooks/analisis_kepadatan_bantul.html)
