# Kontrak kesiapan pilot Bantul, 17 November 2026

Tujuan: paket untuk eksperimen PoC eksploratif empat kelas pada kamera tetap. Ini bukan sertifikasi generalisasi populasi, dataset besar, atau jaminan performa. Tidak melatih model atau menghitung skor model sekarang.

## Pembagian dan dukungan

Paket lama222 immutable. Tambahan dan koreksi berlabel menghasilkan versi baru; simpan original IDs/hash, provenance dan dua reviewer yang berbeda. Tidak membuat label dari model deep learning, tidak mengganti unknown menjadi0, tidak mengubah ROI perframe untuk memperbaiki statistik.

Unit split adalah komponen terhubung bukti antrean/identitas kendaraan yang mungkin sama, termasuk lintas kamera dan lokasi. Timestamp gap saja bukan bukti episode baru. Temporal holdout pada kamera tetap diperbolehkan dengan bukti pergantian antrean, embargo120s dan identity links tambahan; label metode ini tidak boleh menjadi unseen-camera/unseen-location evaluation. Kandidat split lama belumlocked. Tambahan17November harus dinilai sebelumlock, tanpa melihat skor model. Seluruh augmented crop/variant ikut source component, split ditentukan sebelumcrop.

Gates operasional pilot yang ditetapkan sebelum review akuisisi baru selesai (angka bukan standar statistik universal):
- Seluruh gambar dipakai memiliki source hash/dimensi/framing/ROI, anotasi dan audit reviewer berbeda yang hash-bound; nol issue terbuka.
- Tidak ada duplicate/source/event/likely-same-vehicle component melintasi split. Embargo dan quarantine eksplisit; tidak memaksa semua gambar masuk eksperimen.
- Training tiapkelas memiliki >=10 accepted instance dan >=5 encounter groups; validation dan test masing-masing >=2 independent positive encounter groups tiapkelas.
- MAE jumlah/cakupan hanya pada frame full-count; validation/test masing-masing >=10 full-count frame, >=2 kontrol kosong, dan tiapkelas >=2 positive full-count encounter groups.
- Kriteria ini menyiapkan pilot kecil dengan laporan jumlah aktual dan keterbatasan; bukan klaim estimasi akurasi stabil. Full-count group dan object identity group harus dibedakan dari hitunganfoto.
- Adapter, evaluator dan invariant tests lulus; semua path dapat dibuka dan tidak ada symlink keluarpaket yang tidaktercatat.
- Split lock mencakup hash manifest/annotations/ROIs/groups/protocol; tidak dibuat jika gates gagal. Gagal harus eksplisit, tidak dilewati dengan mengubahthreshold setelahlihatdata.

## Penggunaan label

- Positif training: accepted target instances dari frame full atau partial dalam TRAIN saja; auxiliary kendaraan bergerak hanya train, label peranterpisah dan metadata queue tidakdipalsukan. Default tidak menambang negatif pada framepartial.
- Negative sampling: hanya full-count TRAIN, seluruhcandidate rectangle dalam ROI; tidak beririsan semuaobjectbox (termasuk other/outside/context) atauignore region, dengan padding8pxnative. Simpan source/split/bbox/hash. Jangan mengambil outsideannotatedscope sebagai negatif.
- Val/test deteksi: accepted four-class instances; pending/unknown/nonaccepted target serta ignore regions menjadi neutral zones. Known other inside tetap non-target: prediksi kelas target di sana adalah false positive, bukan ignore.
- Pencocokan IoU>=0.5, satu-lawan-satu menurut confidence menurun. Utamakan matching GT sebelum ignore, salahkelas yang overlap GT tidak bolehdiserapignore. Duplicate prediction pada accepted GT tetap FP. Unmatchedpred hanya dinetralkan jika tidak overlap accepted GT>=0.5 dan prediction-area overlap ignore>=0.5.
- Evaluator menerima predicted ROI membership hasilpipeline (inside/outside/uncertain) terpisah dari GT; tidak meminjam manualGT untukmenentukanprediksi. Uncertainpredictions dianggap abstain dan tetapberkontribusi FN bila GTtidakterdeteksi; laporkan jumlahabstain/coverage. Prediksi invalid/processfailure wajib dilaporkan.
- Full-count MAE: hanya full frames; partial tidak masukdenominator dan tidakdianggap0. Prediksi gagal membuat evaluation_run_valid=false, bukan silentdrop/zero. Hitungan dan unioncoverage memakai daftar prediksi accepted yangsama. Partial metrics dilaporkan terpisah denganexposure counts; fullcount nulltetapnull.
- Group aggregation per encounter dan per camera, sertakan jumlahimage/object/group perclass. Janganmenterjemahkan dataauditpass menjadi hasilmodel.

Artefak minimal: DATASET_CARD.md, PROTOCOL.json, GROUPS.json, SPLITS.json/LOCK (hanya gatepass), train positives & negative manifest, eval GT/ignore/full-reference subsets, readiness gate report, galeri, checksum/source license attribution, code/tests/repro instructions. Media asli tidak diduplikasi kecuali paket finalterpilih.
