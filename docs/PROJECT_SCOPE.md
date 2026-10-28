# Scope dan Kriteria Penerimaan (Draf v0.1)

## Judul
Analisis Kepadatan dan Klasifikasi Kendaraan pada Lalu Lintas

## Masukan dan keluaran
- Input: **satu gambar statis** dari dataset (JPG/PNG), bukan video.
- Output: lokasi kendaraan, kelas kendaraan, jumlah per kelas, perkiraan kepadatan visual di ROI jalan, visualisasi, dan metrik evaluasi.
- Kelas awal yang diusulkan: motor, mobil, bus, truk; disesuaikan dengan anotasi dataset final.
- Indikator kepadatan: jumlah terdeteksi dan/atau rasio cakupan objek terhadap ROI. Hindari menghitung luas bounding box yang saling tumpang tindih dua kali.
- Kategori rendah/sedang/tinggi dan ambang numerik **belum final**; perlu dikalibrasi terhadap data dan pengamatan berlabel.

## Batasan
- Gunakan pengolahan citra klasik dan machine learning sederhana, **tanpa deep learning**.
- Tidak mengolah video, tracking temporal, atau kontrol lampu merah.
- Tidak ada target khusus deteksi ambulans.
- Klaim *kemacetan* atau kecepatan kendaraan tidak bisa dibuktikan hanya dari satu citra.

## Keputusan teknis yang belum dikunci
- Dataset dan ketentuan lisensi; anotasi dan metode split.
- Detektor citra statis (MOG2 tidak cocok untuk single-image tanpa referensi latar).
- Kelas kendaraan, ukuran minimum objek, ROI, metode kepadatan, dan ambang kategori.
- Format antarmuka demo (CLI/batch/notebook/UI), kriteria akurasi numerik, serta instruksi final dari dosen.

## Acceptance gates awal
1. Dataset dapat dibaca, anotasi cocok dengan gambar, dan pembagian train/val/test tidak bocor antarsumber/adegan.
2. Detektor menemukan kendaraan pada gambar baru, bukan menggunakan label sebagai prediksi.
3. Klasifikasi jenis dievaluasi pada data yang tidak pernah digunakan untuk pelatihan.
4. Kepadatan visual diuji terhadap ground truth independen yang didefinisikan eksplisit.
5. Program dapat direproduksi dengan instruksi menjalankan, versi dependensi, dan parameter tersimpan.

**Status:** draf untuk disepakati kelompok; bukan spesifikasi tugas dosen yang sudah diverifikasi.
