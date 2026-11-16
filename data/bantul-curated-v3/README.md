# Kurasi CCTV Bantul v3

Paket anotasi 16 November 2026 berisi 222 gambar CCTV asli dari dua jendela akuisisi pada 15 November 2026. Ada 187 gambar inti (171 antrean dan 16 kontrol kosong) dan 35 gambar tantangan. Seluruh 38 kandidat awal v2 dipertahankan dan 184 gambar ditambahkan.

| Kamera | Gambar |
|---|---:|
| Paseban Selatan | 81 |
| Paseban Barat | 78 |
| Bejen Barat | 63 |

Gambar dapat dibuka satu per satu dalam subfolder kamera dan kategori di Drive. Seluruh 222 gambar telah terunggah; jumlah, nama, ukuran, MIME, ID dan folder telah diperiksa ulang. [Folder Drive gambar v3](https://drive.google.com/drive/folders/1MaLqNqmCQNgn2FsZjGHeiZsNZtVYlwRE).

## Metadata

- [Manifest 222 pilihan](MANIFEST.csv) mencatat nama gambar di Drive, sumber relatif terhadap folder sesi, dimensi, ukuran byte, SHA-256 lokal, kategori kurasi, kelompok sementara, dan status anotasi. Daftar kelas visual dipisah dengan `|`.
- [Ringkasan](SUMMARY.json) dan [verifikasi lokal](verification.json) mencatat jumlah pilihan, kecocokan dengan sumber, keunikan byte/piksel, dan konsistensi framing.
- [ROI jalan terkunci](ROIS.json) dan [aturan anotasi](ANNOTATION_POLICY.json) menjadi acuan pemeriksaan posisi kendaraan dari gambar utuh. Keanggotaan ROI otomatis masih perlu divalidasi.

Bus terlihat pada 12 gambar dan truk pada 6 gambar. Angka tersebut menghitung gambar dengan kelas terlihat. Keanggotaan ROI dan jumlah kendaraan unik belum ditetapkan. Paket menandai 25 gambar dengan anotasi percobaan yang perlu direvisi. Belum ada ground truth baru, split terkunci, atau training. Gambar tantangan memerlukan peninjauan keanggotaan dan label ignore sebelum digunakan untuk evaluasi hitung total atau cakupan. Gunakan kelompok lokasi, sesi, dan episode sementara saat merancang split.

Arsip akuisisi lengkap dan petunjuk pemulihan tetap tersedia di Drive. Pembersihan lokal yang disetujui mencakup sekitar 33,5 GB video, media, dan ZIP salinan yang tidak terpakai. Pemeriksaan 16 November 2026 mencocokkan ID dan ukuran 168 paket di Drive, memverifikasi SHA-256 lokal untuk 168 paket, dan menguji CRC isi ZIP. Pencocokan Drive memakai metadata, sementara pemeriksaan hash dilakukan pada berkas lokal. Gambar pilihan dan sumber temporal yang dipakai sebagai bukti dipertahankan. Pembersihan lokal selesai: 33.503.244.573 byte dihapus; ruang kosong terukur bertambah 33.334.562.816 byte. Sebanyak 791 video sumber/bukti dan seluruh 222 gambar tetap lokal. [Arsip sumber dan petunjuk pemulihan](../bantul-20261115/README.md).
