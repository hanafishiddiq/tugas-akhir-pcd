# Pedoman Kerja Kelompok

## Satu sumber kebenaran
- GitHub: source code, konfigurasi, test, dokumentasi teknis, issue/PR, dan manifest data.
- [Google Drive Tugas Akhir PCD](https://drive.google.com/drive/folders/1iTYGMPbZpLsHn7tZXGonYunil6_m7RpG): proposal, dataset besar, screenshot/hasil evaluasi besar, laporan, dan presentasi.
- Hindari menaruh salinan source code yang diedit aktif di Drive. Drive adalah pusat aset, bukan pengganti version control.

## Aturan Git
1. Buat issue dengan deskripsi, batasan, input-output, contoh pengujian, dan pemilik.
2. Mulai dari `main` terbaru, lalu gunakan branch `feat/<topik>`, `fix/<topik>`, atau `docs/<topik>`.
3. Gunakan commit kecil, jelaskan perubahan yang dibuat, dan jangan commit dataset/model besar.
4. Buka pull request; jelaskan apa yang diuji dan cara mereproduksinya.
5. Minimal satu rekan review sebelum merge ke main. Jangan force-push main.
6. Perubahan kontrak modul wajib disepakati pemilik modul terdampak.

## Pembagian tanggung jawab (usulan, belum ditugaskan)
- Dataset & preprocessing.
- Deteksi kendaraan single-image.
- Ekstraksi fitur & classifier.
- Estimasi kepadatan.
- Integrasi, eksperimen, dan evaluasi.

## Standar data
- Simpan data asli terpisah dari data kurasi dan hasil label.
- Catat asal dataset, lisensi, URL, tanggal pengambilan, kelas, format anotasi, dan metode split.
- Jangan mengubah test split setelah evaluasi final. Hindari kebocoran frame/scene serupa antar-split.
- Di repo hanya manifest, konfigurasi dan sampel minimal yang berlisensi untuk dibagikan.
- Setiap eksperimen melaporkan dataset version, commit SHA, parameter, dan metrik.

## Keamanan & privasi
- Jangan commit token, credential, informasi pribadi, atau dataset yang melanggar lisensi.
- Folder Drive awal saat penyiapan dapat diakses editor melalui link; pemilik sebaiknya membatasi writer ke lima anggota sebelum mengunggah aset penting.
- Repository saat bootstrap berstatus **public**; jangan unggah data yang tidak boleh dipublikasikan.
