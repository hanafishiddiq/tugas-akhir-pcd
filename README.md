# Analisis Kepadatan dan Klasifikasi Kendaraan pada Lalu Lintas

[Paper IEEE (PDF)](paper/paper.pdf) · [Notebook dengan hasil](notebooks/analisis_kepadatan_bantul.ipynb) · [Demo untuk Mac Apple Silicon](https://github.com/hanafishiddiq/tugas-akhir-pcd/releases/download/v45/PoC_Bantul_v45_macOS_arm64.zip)

Untuk tugas akhir Pengolahan Citra Digital, kami mencoba menghitung kendaraan yang mengantre di Simpang Paseban dan Simpang Bejen, Bantul. Masukannya satu foto CCTV. Program menandai mobil, motor, bus, dan truk, lalu menghitung berapa persen area jalan di foto yang tertutup kotak-kotak kendaraan.

![Antrean di Paseban Selatan dengan batas area jalan dan kotak hasil deteksi](docs/rilis-v45/S033.jpg)

Pada foto Paseban Selatan ini (S033), program menemukan 13 motor, 1 mobil, dan 2 bus. Totalnya 16 kendaraan, sama dengan hitungan pada anotasi. Kotak hasil deteksi menutupi 59,42% area jalan yang kami tandai, sedangkan kotak pada anotasi menutupi 60,78%. Bagian kotak yang bertumpuk dihitung sekali. Foto berasal dari CCTV Diskominfo Bantul. Garis batas dan kotak kendaraan kami tambahkan untuk analisis. [Sumber gambar dan lisensi](docs/rilis-v45/README.md#sumber-gambar)

## Menjalankan demo

Di Mac Apple Silicon, unduh [paket demo v45](https://github.com/hanafishiddiq/tugas-akhir-pcd/releases/download/v45/PoC_Bantul_v45_macOS_arm64.zip), ekstrak, lalu klik dua kali `Jalankan.command`. Paket ini sudah berisi Python, model, dan empat foto contoh. Demo akan membuka Terminal dan memproses foto Paseban Selatan di atas.

Untuk mencoba foto lain, buka Terminal di folder hasil ekstraksi. Contoh berikut memakai foto bawaan dari Bejen Barat.

```sh
./Jalankan.command --foto contoh/NB006.jpg --kamera 04_bejen_barat
```

Ganti `contoh/NB006.jpg` dengan foto JPG atau PNG yang ingin diperiksa. Pilih kode kamera sesuai asal fotonya (`01_paseban_selatan`, `02_paseban_barat`, atau `04_bejen_barat`). Gunakan foto dengan sudut dan resolusi asli karena tiap kamera punya batas area jalan sendiri.

Terminal akan menampilkan lokasi folder hasil. Buka `hasil.jpg` untuk melihat kotak kendaraan, `cakupan.png` untuk melihat piksel yang dihitung, atau `hasil.json` untuk mengambil angkanya. Untuk memeriksa keempat contoh sekaligus, jalankan `./Jalankan.command --uji`. [Panduan menjalankan kode dari clone](docs/rilis-v45/README.md)

## Hasil pengujian

Model ini memakai pengolahan citra klasik, termasuk HOG untuk membaca pola tepi dan SVM untuk mengenali kendaraan. Kami memakai 99 gambar utama untuk pelatihan dan 61 gambar untuk validasi. Model yang kami pilih mendapat macro-F1 85,50% pada validasi. Saat diuji pada 76 gambar lain dari tiga kamera yang sama, skornya turun menjadi 59,11%. Bus dan truk masih sering terlewat. Dari 8 bus dan 6 truk pada anotasi uji, model menemukan 3 bus dan 2 truk dengan benar.

Untuk menilai jumlah dan kepadatan, kami memakai 31 gambar uji yang anotasinya lengkap. Rata-rata selisih absolut terhadap anotasi adalah 0,45 kendaraan per gambar untuk jumlah dan 3,32 poin persentase untuk cakupan jalan. [Tabel hasil per kelas dan contoh kesalahan deteksi](evaluasi/README.md)

[Data dan anotasi](data/README.md) · [Kode model](src/v45/) · [Sumber LaTeX](paper/paper.tex) · [Unduh notebook sebagai HTML](https://github.com/hanafishiddiq/tugas-akhir-pcd/raw/refs/heads/main/notebooks/analisis_kepadatan_bantul.html)
