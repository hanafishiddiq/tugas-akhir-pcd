# Bantul CCTV, pilot empat kelas

**Siap untuk pilot.** Paket turunan telah diverifikasi independen dan split dikunci pada 17 November 2026 WIB melalui SPLIT_LOCK.json. Tidak ada pelatihan atau skor model.

Masukan satu gambar statis CCTV dari kamera tetap pada pendekat bersinyal. Kelas sasaran mobil, motor, bus, truk. Cakupan jalan adalah persentase union bbox objek diterima setelah dipotong mask ROI.

Ada 248 gambar core versi ini dan 12 gambar auxiliary Palbapang untuk crop classifier train-only. Gambar asli dirujuk melalui manifest dan tidak disalin. Sebanyak 7 koreksi turunan menautkan hash anotasi lama dan baru. Paket awal Bantul 222 tidak berubah.

## Pembagian terkunci

| Subset core | Gambar | Acuan lengkap | Kontrol kosong lengkap | Adegan bus positif lengkap | Adegan truk positif lengkap |
|---|---:|---:|---:|---:|---:|
| train | 99 | 42 | 13 | 0 | 0 |
| validation | 61 | 23 | 4 | 2 | 2 |
| test | 76 | 31 | 5 | 2 | 2 |

Sebanyak 10 gambar core terkena embargo dan 2 berada di karantina. Batas 15 November Paseban 13:54 dan 14:42 WIB, Bejen 14:42 WIB. Batas Bejen 17 November 09:03:10 WIB menjaga blok awal test dan blok berikutnya validation, masing-masing dipisahkan embargo 120 detik per sisi. Semua core Paseban tanggal 17 November untuk test. Tidak ada pengujian prakiraan lalu lintas atau generalisasi kamera/lokasi baru. Bukti tiap batas ada pada GROUPS.json.

Kecocokan armada/identitas dicatat konservatif, termasuk pasangan lintas simpang B049/S068 yang terhubung melalui sumber Palbapang cadangan. S026 dan W030 dikarantina agar potensi kemunculan lagi armada Trans Jogja tidak melintasi split. VS001 dan S043 adalah dua varian satu adegan dan satu bus Tiga Putra; hanya VS001 menjadi acuan lengkap. NB002/NB005, NB003/NB006, dan NS004/NS006 masing-masing berbagi unit pertemuan. Kekosongan 00029 dan pergantian kendaraan di Bejen mendukung adegan truk NB005 dan NB006 terpisah walau keduanya diperlukan dalam blok validation yang sama. Hitungan pertemuan visual kelas umum dikelompokkan konservatif per episode dan kelas; ini batas bawah operasional, bukan identitas seluruh kendaraan unik.

## Dukungan dan kegunaan

Classifier train menyediakan 153 mobil, 422 motor, 10 bus, dan 10 truk. Crop tambahan hanya mendukung classifier. Deteksi, hitungan/coverage, dan negatif tetap memakai core dengan aturan ROI dan ignore. Frame parsial boleh menyumbang deteksi dengan ignore, tetapi jumlah penuh dan cakupan acuan tetap null.


Anotasi dibuat dan ditinjau secara visual oleh reviewer AI berbeda. Audit kode dan data tidak mengukur akurasi model. Frame rapat/tertutup, sampel bus sedikit, dua lokasi core, dan akuisisi terbatas membatasi penggunaan untuk pilot eksploratif.

## Sumber dan penggunaan ulang

CCTV Pemerintah Kabupaten Bantul / Diskominfo Bantul. Atribusi pemutar akuisisi menyatakan Creative Commons Attribution 4.0. Beri kredit Pemerintah Kabupaten Bantul / Diskominfo Bantul, cantumkan sumber kamera dan lisensi, serta tandai crop, anotasi dan overlay sebagai modifikasi. Tautan sumber dan bukti pengambilan ada pada manifest sesi. Atribusi mengikuti metadata pemutar yang disimpan.

[Atribusi sesi 17 November](../../Sumber/Bantul_CCTV/20261117_085757_WIB_targeted/SOURCE_ATTRIBUTION.json) · [Pemetaan waktu](GROUPS.json) · [Audit versi](REVISION_LINEAGE.json) · [Galeri](index.html) · [Laporan gate](READINESS_REPORT.json)

Paket ini menggunakan jalur sumber relatif terhadap root proyek. Dari root proyek, jalankan code/run_tests.py dan code/readiness.py menggunakan environment Tahap 03. Perintah siap pakai, dependensi dan cara merakit ulang ada di README.md. Metadata daftar positif, negatif, dan acuan evaluasi berakhiran _PROPOSED adalah snapshot adaptor; assignments pada SPLIT_LOCK.json merupakan pembagian final.
