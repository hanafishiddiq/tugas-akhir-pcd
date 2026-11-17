# Bantul Pilot v1

Paket pilot empat kelas telah lolos pemeriksaan independen dan split dikunci pada 17 November 2026. SPLIT_LOCK.json mengikat input, aset bukti, kode, dan pembagian data. Belum ada training atau skor model.

- [Split terkunci](SPLIT_LOCK.json)
- [Dataset card](DATASET_CARD.md)
- [Galeri source dan overlay](index.html)
- [Gate dan hash input](READINESS_REPORT.json)
- [Pemetaan sumber serta lisensi](SOURCE_INDEX.json)
- [Skenario uji adaptor](code/TEST_RESULTS.txt)

Gunakan dari root proyek, dengan environment Tahap 03.

```sh
PY=Jawaban/Perencanaan_Baseline/Tahap_03/kode/.venv/bin/python
P=Data/Eksperimen/Bantul_Pilot_v1
"$PY" -B "$P/code/run_tests.py"
"$PY" -B "$P/code/readiness.py" --root "$PWD" --paths "$P/INPUT_PATHS.json"
```

Media sumber, ROI, dan bukti lama tetap pada jalur proyek yang tercatat. Paket ini dapat dijalankan langsung dalam workspace proyek dan tidak menggandakan media asli. Beberapa rujukan bukti audit asli mengandung path absolut lokal yang terikat hash, sehingga pemindahan ke workspace lain memerlukan pemetaan aset atau rebinding path yang terdokumentasi. Publikasi terpilih dan petunjuk pemulihan berada di ../../../Publikasi/Bantul_Pilot_v1_20261117.

Perakitan ulang mengambil audit serta sumber yang didaftarkan di SOURCE_INDEX.json. Jalankan skrip assemble.py dari folder kerja .scratch/bantul-readiness-20261117/assembled_candidate dengan environment yang sama. package.py memproyeksikan hasil lolos ke folder kandidat baru dan menolak menimpa folder yang sudah ada. Pada akuisisi yang masih menambah log, validasi kembali hash snapshot sebelum mengganti kandidat. Tidak mengubah Bantul_222_v1.

File berakhiran _PROPOSED merekam keluaran adaptor sebelum penguncian. Pembagian yang berlaku adalah assignments di SPLIT_LOCK.json; nama file snapshot dipertahankan agar riwayat dapat ditelusuri. Jangan menyesuaikan pembagian setelah melihat hasil model test.
