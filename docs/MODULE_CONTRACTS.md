# Kontrak Antarmodul — Draf v0.1

**Tujuan:** anggota dapat mengembangkan dan menguji modul secara independen. Semua kontrak bersifat usulan sampai dataset diputuskan.

## Representasi data
- Gambar: array NumPy RGB/BGR dengan dokumentasi urutan kanal.
- Bounding box: koordinat piksel [x_min, y_min, x_max, y_max], titik asal di kiri atas, batas kanan/bawah **eksklusif**.
- Deteksi: daftar {bbox, score?}; koordinat dalam ukuran gambar asli.
- Klasifikasi: daftar {bbox, class_name, score?} dengan panjang sama seperti deteksi yang valid.
- ROI: mask biner ukuran (height,width); dapat berupa poligon jalan.
- Kepadatan: {vehicle_count, count_by_class, roi_area_px, covered_area_px, coverage_ratio, density_level?}.

## Alur yang diusulkan
1. `preprocessing`: image -> (prepared_image, roi_mask).
2. `detection`: prepared_image + roi_mask -> daftar bounding box kandidat.
3. `features`: image crops dari bounding box -> matriks fitur (HOG/LBP/geometri, setelah diputuskan).
4. `classification`: matriks fitur -> kelas dan skor tiap kandidat.
5. `density`: roi_mask + kotak/segmen yang diterima -> indikator kepadatan visual.
6. `main`: rangkai, tampilkan/simpan hasil dan tangani kegagalan.

## Prinsip evaluasi
- Pisahkan **deteksi** (precision/recall, IoU), **klasifikasi** (confusion matrix, macro-F1), **perhitungan jumlah** (MAE), dan **estimasi kepadatan** (kesalahan terhadap ground truth).
- Pengujian end-to-end menggunakan hasil deteksi, tidak boleh mengganti hasilnya dengan anotasi asli.
- Bila cakupan memakai bounding box, hitung gabungan area yang beririsan dengan ROI agar tidak melebihi 100%; bedakan dari okupansi piksel kendaraan sesungguhnya.
- Standar label, confidence, ambang, dan jenis score belum ditetapkan.

## Aturan integrasi
- Semua fungsi memiliki unit test dengan gambar/array sintetis kecil.
- Format input-output tidak boleh diganti sepihak: ajukan issue/PR berisi migrasi dan update pengujian.
- Belum ada implementasi; file ini **rancangan kontrak**, bukan API Python yang sudah tersedia.
