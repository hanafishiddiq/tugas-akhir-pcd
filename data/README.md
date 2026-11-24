# Data CCTV Bantul

Foto berasal dari CCTV Pemerintah Kabupaten Bantul. Kami meninjau lima sudut kamera dan memakai tiga sudut untuk menguji model pada gambar utuh, yaitu Paseban Selatan, Paseban Barat, dan Bejen Barat. [Foto dan anotasi di Google Drive](https://drive.google.com/drive/folders/1dL_hXMOnOQvsBkwFW4IsZBJYo-ulJcR5) · [Indeks gambar per kamera](bantul-pilot-v1/DAFTAR_GAMBAR.csv).

| Gambar utama | Latih | Validasi | Uji |
|---|---:|---:|---:|
| Semua gambar dalam pembagian aktif | 99 | 61 | 76 |
| Gambar dengan acuan hitungan lengkap | 42 | 23 | 31 |

Dari 248 gambar utama, 10 dikeluarkan karena terlalu dekat dengan batas waktu antarpembagian dan dua dikarantina karena ada kemungkinan kendaraan yang sama muncul kembali. Dua belas potongan tambahan dari Palbapang digunakan khusus untuk melatih pengenal jenis kendaraan. [Pembagian final per gambar](bantul-pilot-v1/SPLIT_LOCK.json) · [Kelompok adegan dan kendaraan](bantul-pilot-v1/GROUPS.json).

Anotator dan peninjau kedua menggunakan AI untuk memeriksa gambar dan kotak kendaraan. Untuk hitungan dan cakupan, objek yang masih meragukan dapat membuat acuan lengkap suatu foto bernilai `null`. Jumlah total nol pada foto beracuan lengkap berarti tidak ada kendaraan sasaran di area jalan yang diperiksa. [Anotasi kendaraan](bantul-pilot-v1/ANNOTATIONS.json) · [Poligon area jalan per kamera](../src/v45/ROIS.json) · [Kode evaluasi](bantul-pilot-v1/code/evaluate.py) · [Hasil uji](../evaluasi/README.md).

Tiga ringkasan berikut menjadi sumber angka di notebook. [Pengambilan rekaman](bantul-20261115/acquisition_summary.json) · [Pemilihan 222 gambar](bantul-curated-v3/SUMMARY.json) · [Pemeriksaan anotasi awal](bantul-annotations-v1/SUMMARY.json).

Untuk mengulang pemeriksaan seluruh dataset, gunakan [paket pilot lengkap](https://github.com/hanafishiddiq/tugas-akhir-pcd/tree/cf8ae4305b281a209861c754f5d259f4bc38a13c/data/bantul-pilot-v1) dan [panduan pemulihan](https://github.com/hanafishiddiq/tugas-akhir-pcd/blob/cf8ae4305b281a209861c754f5d259f4bc38a13c/data/bantul-pilot-v1/restore/PULIHKAN_DATASET.md) dari revisi yang sama. [Aturan anotasi 222 gambar](https://github.com/hanafishiddiq/tugas-akhir-pcd/blob/cf8ae4305b281a209861c754f5d259f4bc38a13c/data/bantul-annotations-v1/ANNOTATION_PROTOCOL.md) dan [kontrak evaluasi](https://github.com/hanafishiddiq/tugas-akhir-pcd/blob/cf8ae4305b281a209861c754f5d259f4bc38a13c/data/bantul-pilot-v1/DATA_CONTRACT.md) mencatat cara menangani objek di tepi area jalan, kelas ambigu, dan prediksi yang bertumpuk.

Sumber gambar adalah Pemerintah Kabupaten Bantul / Diskominfo Bantul. Metadata pemutar saat pengambilan mencantumkan CC BY 4.0. Tim menambahkan kotak, poligon area jalan, dan panel perbandingan. [Portal CCTV](https://bantulkab.go.id/cctv.html) · [Lisensi](https://creativecommons.org/licenses/by/4.0/).
