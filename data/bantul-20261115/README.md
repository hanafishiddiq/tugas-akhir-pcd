# Arsip sumber CCTV Bantul 15 November 2026

**Status penyimpanan 17 November 2026:** 82 ZIP/part (7.500.124.695 byte) dan 38 JPEG duplikat v2 (20.998.592 byte) telah dipindahkan ke Sampah Drive, total 7,52 GB. Ini belum penghapusan permanen; antarmuka Drive menyatakan item Sampah dihapus selamanya setelah 30 hari. Tidak ada klaim kuota sudah bebas.

Dataset aktif tetap 222 gambar v3 dan 222 overlay anotasi. Seluruh 791 video bukti lokal lolos hash dan arsip sumbernya tetap dipertahankan. Sebanyak 86 paket lama (7.538.401.919 byte) masih disimpan: 83 memuat bukti video aktif, dua menyimpan snapshot metadata yang berubah, dan satu memuat laporan. ZIP kecil label/audit terbaru juga tetap tersedia.

Rilis awal tidak lagi tersedia sebagai 168 paket di folder aktif. CHECKSUMS.json, PACKAGE_MANIFEST.json, restore_dataset.py dan peta pemulihan lama merupakan bukti historis; pemulihan penuh memerlukan item dari Sampah dan tidak dijamin setelah retensi berakhir. Gunakan paket v3 serta anotasi terbaru untuk pekerjaan aktif.

[Daftar status arsip](STORAGE_STATUS_20261117.json). Bagian berikut merekam rilis historis 15 November.

Arsip akuisisi untuk PoC analisis jumlah mobil, motor, bus dan truk serta persentase cakupan ROI pada satu gambar CCTV persimpangan. Paket gambar yang kini dipilih untuk anotasi dan pratinjau tersedia melalui [Kurasi CCTV Bantul v3](../bantul-curated-v3/README.md). Label percobaan dalam arsip ini masih perlu revisi.

## Isi paket

- Rekaman lima preset selama jendela pengamatan 13.19.32–15.19.32 WIB: 15.168 segmen MPEG-TS asli, 1.550 JPEG pratinjau dan bukti audit. Durasi sumber tersedia: Paseban Selatan 73,10 menit; Paseban Barat 118,74; Klodran Selatan 119,34; Bejen Barat 75,08; Palbapang Selatan 119,36. Klodran dan Palbapang tetap cadangan. Bejen awalnya 404 lalu pulih; Paseban Selatan sering terputus.
- Pengambilan awal 11.43 WIB: 513 frame sumber beserta video, manifest dan bukti audit awal.
- Kurasi awal Bantul v1 dan 26 anotasi percobaan (194 bbox). Aturan pusat bbox belum valid untuk semua objek; jangan memperlakukan label diagnostik sebagai ground truth final.
- Laporan audit, anotasi percobaan dan pengambilan terbaru dalam Word serta Markdown.

Total 168 berkas paket, sekitar 14,01 GiB. Ini satu rilis sumber/diagnosis, bukan 168 dataset atau episode independen. MIO, NGSIM, referensi sintetis dan arsip percakapan tidak termasuk rilis Bantul ini.

## Penyimpanan

[Folder dataset Google Drive](https://drive.google.com/drive/folders/15nr118aERl3e0-US7FjSQzIRzWRanTky) menggunakan akses kelompok yang sudah ada.

1. [Rekaman 120 Menit](https://drive.google.com/drive/folders/1nITUhdL7cMfbFdaVuUyNziFDWNqI3Ejp): indeks/pratinjau dan segmen asli per kamera.
2. [Pengambilan Awal](https://drive.google.com/drive/folders/1S-wGvt-49ySaREnCH66ymiU1B5NTguXC): sumber audit awal.
3. [Kurasi Awal](https://drive.google.com/drive/folders/1N-hahgiMvSY5_IERjKcXYyz48Wlq0ctr).
4. [Anotasi Percobaan](https://drive.google.com/drive/folders/1U1LEx3njVbfTOx1IkKPM2twjv6bYOmvq).
5. [Laporan](https://drive.google.com/drive/folders/1aXkkjgKPmT6bJptWgA9iDZqL0dT-FxoV).

## Membuka paket

Unduh seluruh ZIP dan berkas .partNN dari subfolder ke satu folder lokal, bersama CHECKSUMS.json dan restore_dataset.py. Jangan mengubah nama berkas. Jalankan `python3 restore_dataset.py /lokasi/folder-unduhan`. Skrip memverifikasi SHA-256 seluruh paket dan menyatukan empat ZIP video lama yang dibagi karena batas unggahan 100 MiB. ZIP lainnya dapat diekstrak langsung.

Setelah verifikasi berhasil, ekstrak semua ZIP ke satu folder kosong yang sama. Jalur relatif dalam ZIP membentuk struktur Data/ dan Jawaban/ sehingga tautan galeri tetap berfungsi. Jangan membuat satu folder ekstraksi terpisah untuk setiap ZIP. File .partNN bukan ZIP mandiri.

Buka `Data/Sumber/Bantul_CCTV/20261115_131932_WIB_120min/index.html` untuk galeri baru, `Data/Kurasi/Bantul_CCTV_v1/index.html` untuk kurasi awal, dan `Data/Anotasi/Bantul_26_v1/index.html` untuk anotasi percobaan. Pemutar HLS lokal dapat membuka recording.m3u8; waktu pemutaran dapat melompati celah, sehingga gunakan COVERAGE.json untuk durasi sumber.

## Verifikasi dan batasan

Seluruh 644 klip dari batch 120 menit lulus hash/decode; 1.550 JPEG diperiksa dan 10 ekstraksi ulang identik. Tinjauan visual hanya sampel selama pengambilan, bukan persetujuan seluruh frame. Noise, watermark dan waktu kamera dipertahankan. Peringatan timestamp concat dicatat. Setiap paket diuji CRC ZIP sebelum diunggah; CHECKSUMS.json menyimpan hash untuk pemeriksaan unduhan.

Sumber: portal publik Pemerintah Kabupaten Bantul https://bantulkab.go.id/cctv.html dan Diskominfo Bantul. Halaman/konfigurasi sumber disimpan pada metadata; konfigurasi pemutar mencantumkan CC BY 4.0. Video sumber dipakai untuk memilih gambar dan memeriksa konteks berhenti; input PoC tetap satu gambar statis.

Lanjutkan anotasi gambar pilihan v3 dengan ROI jalan terkunci, kemudian validasi keanggotaan ROI dan dukungan kelas sebelum menentukan split berdasarkan lokasi, sesi, dan episode.
