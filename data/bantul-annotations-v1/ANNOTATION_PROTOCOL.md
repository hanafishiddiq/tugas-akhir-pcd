# Anotasi Bantul 222 v1

Tanggal mulai 16 November 2026 WIB. Han mengotorisasi anotasi seluruh222 dan audit sampai selesai. Data input tetap Bantul_CCTV_v3, ROI locked bantul_road_roi_v1_20261116. Tidak ada training/split atau DL. Jangan mengubah sumber, gambar kurasi, ROI lock, atau anotasi26 historis.

## Cakupan dan prosedur

- Semua 222 gambar wajib punya record, termasuk16kontrol kosong. Native image dibuka untuk anotasi; overlay bbox buatan sendiri dibuka kembali untuk pemeriksaan. Gunakan grid koordinat dan crop detail bila objek kecil/rapat. Jangan membuat koordinat, kelas, titik kontak atau klaim review tanpa melihat buktinya.
- Satu kendaraan satu bbox; motor mencakup kendaraan+rider+penumpang, tidak menggabungkan dua motor. Bbox integer [xmin,ymin,xmax,ymax), kanan/bawah eksklusif; extent yang terlihat, tanpa menebak bagian tersembunyi/amodal. Jangan memotong bbox dengan ROI; atap/rider boleh di luar poligon.
- Anotasi semua kendaraan yang terlihat pada pendekat sasaran sepanjang ROI, serta konteks yang bbox-nya beririsan/dekat batas ROI. Di luar cakupan itu tidak wajib exhaustive dan harus diabaikan dalam negative mining. Bedakan opposite direction dari target approach berdasarkan bukti visual.
- Kelas mobil/motor/bus/truk/other/unknown. Passenger car/MPV/SUV=mobil; bus jelas/minicoach jelas=bus; truck kabin+bak/boks jelas=truk. Pickup/cargo van=other. Van/minibus ambigu tetapunknown; catat subtype/reason. Jangan memaksa kelas untuk meningkatkan dukungan.
- Membership manual dari posisi fisik/basis/kontak terlihat terhadap jalan, bukan bbox center/bottomcenter atau masker overlap. Gunakan inside/outside/boundary_review/unknown. Kendaraan straddle atau bukti kontak saling bertentangan tetapboundary_review bila tidak dapat diselesaikan. Titik kontak yang tidak terlihat=null; tidak boleh titik sintetis. Pita3px pada1280 atau4.5px pada1920 adalah bantuan audit, bukan rulehitung.
- Otomatisasi hanya boleh membantu format/proposal geometrik; seluruh bbox hasilnya harus dibandingkan dengan image. Tidak boleh propagasi satu template ke banyakframe tanpa review tiapframe. Label kendaraan tidak boleh berasal dari modeldeep learning.
- 25label historis hanya seed. Tinjau bbox, kelas, dan membership ulang terhadap ROI final. Anotasi lama tidak otomatis lulus.
- Ketidakpastian harus dilokalisasi keobject/ignore region jika memungkinkan. Annotation review bisa selesai dengan uncertainty yang terdokumentasi; jangan menyatakan frame siap acuan hitung penuh apabila label/membership target belum tuntas. Keputusan tampil di report sebagai partial, bukan counts0.

## Output per reviewer

Scope write hanya subfolder S/reviewers/<scope>/, dengan S=.scratch/bantul-annotation222-20261116. ANNOTATIONS.json adalah listrecord tepat satu per ASSIGNMENTS.json; jangan hilangkan frame. Simpan progres tiap5–10gambar. Progress.json mencatat jumlah annotated/selfreviewed dan masalah actual.

Record wajib:
```
{
 image_id: 'S001'|'W001'|'B001', source_image: <projectrelative>, camera_id: ...,
 native_reviewed: true, self_overlay_reviewed: true,
 annotation_scope: 'target_approach_and_near_boundary_context',
 review_notes: [specific notes], visual_evidence: [projectrelative files actually viewed],
 objects: [{id:'S001-01', bbox_xyxy:[...], class_name:..., subtype:...,
   confidence:'high'|'medium'|'low', role:'target_approach'|'context',
   occlusion:'none'|'partial'|'heavy', truncated: false,
   roi_membership:'inside'|'outside'|'boundary_review'|'unknown',
   membership_reason:<specific visual reason>, road_contact_xy:[x,y]|null,
   road_contact_note:<visible evidence or why null>,
   crosses_roi_boundary:true|false|'unknown', classification_note:...,
   ignore_reason:null|<specific reason>}],
 ignore_regions:[{bbox_xyxy:[...],reason:...,roi_relevance:'possible_target_in_roi'|'outside_target_roi'}],
 reviewer:'<scope>', source_legacy_annotation: false|true
}
```

Ignore region hanya area nyata dengan objek ambigu/crowd tidak dapat dipisahkan; jangan mengganti objek terbaca dengan region besar. Unknown/other tetap berbox sendiri jika satu kendaraan terlihat. Jangan menambahkan area ignore besar untuk menghapus kesalahan deteksi/kelas. Counts/coverage dihitung compiler utama, tidak diisi reviewer.

## Audit akhir

1. Struktur/integritas: tepat222source, SHA/dimensi tidakberubah, lock tetap; bboxinteger dalamimage,IDunik,noidenticalboxes,enum/point valid.
2. Self-review overlay seluruh222; cross-review oleh reviewer berbeda seluruh222 beserta source. Semua temuan terjawab dalam perubahan atau alasan berbukti. Kasus kelas langka/straddle/occlusion diperiksa secara khusus.
3. Counts dan coverage dari daftar accepted yang sama (targetclass+manualinside+confidence bukanlow+tanpaignore). Gabungan bbox clip pada roadmask, bukan penjumlahan area. Metode sweeprectangle/scanline independen memeriksa area union raster.
4. Ambiguous relevantobjects/ignore membuat fullcountreference=null dan interval/batasresolved tersedia terpisah. Jangan mengubah uncertainty menjadioutside/0. Lulus audit annotations bukan sertifikasihumanGT atau bukti kelayakan model.


Koreksi audit: ignore region wajib roi_relevance fisik eksplisit. Proyeksi helm/atap di luar poligon tidak membuktikan dasar kendaraan di luar. possible_target_in_roi membuat acuan hitungan penuh null bahkan jika overlap bbox nol; outside_target_roi hanya dengan bukti fisik dan alasan. Objek tunggal nontarget seperti becak tetap classother dengan bbox, bukan crowdignore.
