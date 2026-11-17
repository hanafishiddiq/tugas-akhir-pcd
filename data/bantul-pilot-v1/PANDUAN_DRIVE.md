# Bantul Pilot v1 — mulai di sini

Paket dataset siap untuk pilot empat kelas, 17 November 2026. Split dikunci sebelum training. Tujuan: PoC satu gambar CCTV tetap dengan jumlah mobil, motor, bus, truk dan cakupan bbox pada ROI; hasil eksperimen untuk laporan IEEE.

- [Daftar 260 gambar dan crop beserta tautan preview](https://drive.google.com/file/d/17wVGks9exsjdxIlMu2zSiO0ymdxZeUDg/view?usp=drivesdk)
- [26 gambar utama tambahan](https://drive.google.com/drive/folders/1M0JLrKgnqwZC1Imnyj-qA4V2gAkWjE78)
- [33 overlay baru atau hasil koreksi](https://drive.google.com/drive/folders/1vGOo5qgmip9MRuSdbxq5eAbyYll32JNF)
- [12 crop Palbapang khusus classifier train](https://drive.google.com/drive/folders/1lZeoYOa5HU-1GYfPyX7MVe4bXNphkc1z)
- [Label, pembagian data dan protokol](https://drive.google.com/drive/folders/1_I1axBgl9f2idZrWelt9J-ovrWXl5ppd)
- [Panduan pemulihan](https://drive.google.com/file/d/1kHzi0FEgm5EYU3g45yVAkMm5fGZLvqHd/view?usp=drivesdk) dan [enam arsip bukti terpilih](https://drive.google.com/drive/folders/138G64xxylYz5ZH01D9CYOl3dHmi-j7_G)
- [Metadata dan kode GitHub](https://github.com/hanafishiddiq/tugas-akhir-pcd/tree/main/data/bantul-pilot-v1)

248 gambar utama mencakup 99 train, 61 validation, 76 test, 10 embargo karena dekat batas waktu, dan dua karantina. Acuan hitungan lengkap tersedia pada 42/23/31 gambar train/validation/test. Label parsial dipakai dengan aturan ignore. Tambahan 12 crop hanya untuk classifier train. Dukungan classifier: 153 mobil, 422 motor, 10 bus, dan 10 truk. Cakupan pilot terbatas pada kamera yang sama; training dan skor model belum tersedia.

Gambar lama 222 tetap memakai file yang sudah diterbitkan pada folder 07; overlay lama yang belum berubah memakai folder 08. Tautan tiap gambar ada di daftar di atas. Enam ZIP baru hanya berisi bukti audit dan dependensi pemulihan terpilih, tanpa arsip rekaman mentah. Gambar utama dan crop tetap dapat dipratinjau satu per satu.

Pemeriksaan mandiri: validator nol galat, 25 tes lulus, split dan sumber terikat SHA-256. Pemulihan lokal ke folder kosong memulihkan 1.403 berkas dan 782 tautan galeri; input unduhan gambar disimulasikan dengan salinan lokal berhash sama. Readback Drive mencocokkan nama, ukuran, MIME, dan folder; beberapa unduhan sampel juga dibandingkan hash. Audit label dilakukan AI.

Sumber: Pemerintah Kabupaten Bantul / Diskominfo Bantul melalui https://bantulkab.go.id/cctv.html. Atribusi dan metadata lisensi pemutar disimpan dalam paket. Crop, bbox, ROI dan overlay adalah modifikasi untuk penelitian.
