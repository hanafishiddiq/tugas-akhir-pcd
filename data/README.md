# Dataset lokal

Dataset **belum dipilih**. Unduh hanya dari sumber yang telah diperiksa izin/licensinya dan catat asalnya.

Contoh struktur lokal ketika dataset telah disetujui:
```text
data/
  raw/            # unduhan asli; gitignored
  processed/      # hasil transformasi; gitignored
  annotations/    # label besar; gitignored
  splits/         # train/validation/test manifests kecil yang dapat direview
```

Jangan memasukkan gambar dataset ke Git. Bagi tautan versi data melalui Google Drive proyek dan catat SHA/checksum bila diperlukan.
