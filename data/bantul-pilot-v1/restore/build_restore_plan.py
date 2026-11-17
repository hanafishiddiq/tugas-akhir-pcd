"""Inventory the frozen pilot and write content-addressed Drive/support restore plans."""
from collections import Counter
import csv
from hashlib import file_digest
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PILOT = Path("Data/Eksperimen/Bantul_Pilot_v1")
BASE = Path("Data/Anotasi/Bantul_222_v1")
CURATED = Path("Data/Kurasi/Bantul_CCTV_v3")
SUPPORT_LIMIT = 88 * 1024**2
ARCHIVE_LIMIT = 90 * 1024**2

def digest(path):
    with path.open("rb") as source:
        return file_digest(source,"sha256").hexdigest()

def read(relative):
    return json.loads((ROOT/relative).read_text())

def write(name,value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")

class Gallery(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = set()
        self.cards = {}
        self.article = False
        self.in_heading = False
        self.key = None
    def handle_starttag(self,tag,items):
        attrs = dict(items)
        if tag=="article":
            self.article=True
            self.key=None
        if self.article and tag=="h2":
            self.in_heading=True
        for name in ("href","src"):
            link = attrs.get(name)
            if link:
                parts = urlsplit(link)
                if parts.scheme or parts.netloc:
                    continue
                path = (ROOT/PILOT/unquote(parts.path)).resolve(strict=True).relative_to(ROOT)
                self.refs.add(str(path))
                if self.key and tag=="img" and name=="src":
                    self.cards[self.key]=str(path)
    def handle_endtag(self,tag):
        if tag=="article":
            self.article=False
        if tag=="h2":
            self.in_heading=False
    def handle_data(self,body):
        if self.in_heading and body.strip():
            self.key=body.strip().split()[0]

def build():
    locks=read(PILOT/"SPLIT_LOCK.json")
    readiness=read(PILOT/"READINESS_REPORT.json")
    paths=read(PILOT/"INPUT_PATHS.json")
    assert locks["locked"] and readiness["ready"] and locks["paths"]==paths
    for key in ("input_sha256","verified_asset_sha256","code_sha256"):
        assert locks[key]==readiness[key],key

    payloads, destinations, aliases = {},{},{}
    def bind(name,reason,expected=None):
        source_path = (ROOT/name).resolve(strict=True)
        path = str(source_path.relative_to(ROOT))
        checksum = digest(source_path)
        if expected is not None:
            assert checksum==expected,(name,checksum,expected)
        if Path(name).is_absolute():
            aliases[str(name)]={"restore_path":path,"sha256":checksum}
        src=destinations.setdefault(path,{"sha256":checksum,"required_by":[]})
        assert src["sha256"]==checksum
        if reason not in src["required_by"]:
            src["required_by"].append(reason)
        item=payloads.setdefault(checksum,{"bytes":source_path.stat().st_size,"restore_paths":[]})
        if path not in item["restore_paths"]:
            item["restore_paths"].append(path)
        return checksum

    for name,expected in locks["verified_asset_sha256"].items():
        bind(name,"validator_evidence",expected)
    for key,path in paths.items():
        bind(path,"validator_input:"+key,locks["input_sha256"][key])
    for name,expected in locks["code_sha256"].items():
        path = BASE/"code/core.py" if name=="annotation_reference_core" else PILOT/"code"/name
        bind(path,"validator_code:"+name,expected)

    for name in (ROOT/PILOT).rglob("*"):
        if name.is_file() and "__pycache__" not in name.parts and not name.name.startswith("."):
            bind(name.relative_to(ROOT),"pilot_package_file")

    src_index=read(PILOT/"SOURCE_INDEX.json")
    for section in ("assembly_scripts","boundary_visual_paths"):
        for path,value in src_index[section].items():
            bind(path,section,value)
    for session,path in src_index["source_attribution"]["evidence"].items():
        bind(path,"source_license:"+session)
    for file in ("ANNOTATIONS.json","ROIS.json"):
        bind(BASE/file,"frozen_adapter_test_input")
    bind(CURATED/"DRIVE_MANIFEST.csv","existing_original_drive_receipt")
    bind(BASE/"DRIVE_RECEIPTS.json","existing_overlay_drive_receipt")

    gallery=Gallery()
    gallery.feed((ROOT/PILOT/"index.html").read_text())
    for path in gallery.refs:
        bind(path,"gallery")
    core=read(PILOT/"MANIFEST.json")
    aux=read(PILOT/"AUXILIARY_MANIFEST.json")
    assert len(core)==248 and len(aux)==12 and len(gallery.cards)==260
    with (ROOT/CURATED/"DRIVE_MANIFEST.csv").open() as handle:
        old_original_receipts={row["source_sha256"]:row for row in csv.DictReader(handle)}
    old_overlay_receipts={row["sha256"]:row for row in read(BASE/"DRIVE_RECEIPTS.json")["files"]
                          if row.get("mime_type")=="image/jpeg"}

    queue={r["sha256"]:r for r in read(HERE.relative_to(ROOT)/"UPLOAD_QUEUE_IMAGES.json")}
    assert len(queue)==71
    uploaded={r["sha256"]:r for r in read(HERE.relative_to(ROOT)/"DRIVE_UPLOAD_RECEIPTS.json")}
    assert set(uploaded)==set(queue)
    downloads={}
    def download(checksum, group, filename, image_id, receipt=None):
        if checksum in downloads:
            assert downloads[checksum]["filename"]==filename,(checksum,filename)
            return
        row={"kind":group,"filename":filename,"image_id":image_id,
             "sha256":checksum,"bytes":payloads[checksum]["bytes"]}
        if receipt:
            assert int(receipt["bytes"])==row["bytes"]
            row.update(drive_id=receipt.get("drive_id") or receipt["id"],
                       drive_url=receipt.get("drive_url") or receipt["url"],
                       receipt_status="existing_metadata_receipt_requires_download_sha256")
        else:
            queued=queue[checksum]
            assert queued["image_id"]==image_id and int(queued["bytes"])==row["bytes"]
            verified=uploaded[checksum]
            assert verified["success"] is True
            for field in ("group","image_id","file_name","bytes","sha256","mime_type"):
                assert queued[field]==verified[field],(checksum,field)
            row.update(filename=queued["file_name"],upload_group=queued["group"],
                       drive_id=verified["id"],drive_url=verified["url"],
                       receipt_status="new_publication_metadata_requires_download_sha256")
        downloads[checksum]=row

    for frame in core:
        key=frame["image_id"]
        path=frame["source_image"]
        checksum=bind(path,"core_original:"+key,frame["source_sha256"])
        if checksum in old_original_receipts:
            receipt=old_original_receipts[checksum]
            download(checksum,"existing_core_original",receipt["drive_filename"],key,receipt)
        else:
            download(checksum,"new_core_original",f'{key}{Path(path).suffix.lower()}',key)
        overlay=gallery.cards[key]
        oid=bind(overlay,"core_overlay:"+key)
        if oid in old_overlay_receipts:
            receipt=old_overlay_receipts[oid]
            download(oid,"existing_core_overlay",receipt["file_name"],key,receipt)
        else:
            download(oid,"new_or_revised_core_overlay",f"{key}.jpg",key)

    for a in aux:
        key=a["image_id"]
        bind(a["source_image"],"aux_source_original:"+key,a["source_sha256"])
        crop=gallery.cards[key]
        cid=bind(crop,"aux_classifier_crop:"+key)
        download(cid,"new_aux_crop",f"{key}.jpg",key)

    for checksum in (set(payloads)-set(downloads)) & set(old_overlay_receipts):
        receipt=old_overlay_receipts[checksum]
        download(checksum,"existing_audit_overlay",receipt["file_name"],Path(receipt["file_name"]).stem,receipt)

    categories=Counter(row["kind"] for row in downloads.values())
    expected=Counter(existing_core_original=222,existing_core_overlay=215,new_core_original=26,
                     new_or_revised_core_overlay=33,new_aux_crop=12)
    for kind,count in expected.items():
        assert categories[kind]==count,(kind,categories)
    # The package validator and its frozen BASE regression fixture read different
    # audit snapshots for some taxonomy corrections. Include those original files too.
    fixture=read(PILOT/"code/FIXTURE_222.json")
    originals={r["image_id"]:r for r in read(BASE/"ANNOTATIONS.json") if r["image_id"] in fixture["images"]}
    previous=set(payloads)
    for image_id,record in originals.items():
        bind(record["source_image"],"frozen_adapter_fixture:"+image_id,record["source_sha256"])
        for name in (*record["visual_evidence"],*record["cross_audit"]["visual_evidence"]):
            bind(name,"frozen_adapter_fixture:"+image_id)
    supplemental=set(payloads)-previous
    support=[(checksum,item) for checksum,item in payloads.items() if checksum not in downloads]
    # Fill sorted parts deterministically; the margin leaves room for ZIP central directory/file headers.
    parts=[]
    current=[]
    used=0
    for checksum,item in sorted(support,key=lambda p:(p[0] in supplemental,-p[1]["bytes"],p[0])):
        size=item["bytes"]
        if size>SUPPORT_LIMIT:
            raise ValueError("A single support asset exceeds the payload size limit")
        if current and used+size>SUPPORT_LIMIT:
            parts.append(current)
            current=[]
            used=0
        current.append(checksum)
        used+=size
    if current:
        parts.append(current)
    archives={}
    for i,contents in enumerate(parts,1):
        name=f"Bantul_Pilot_v1_Bukti_{i:02d}_of_{len(parts):02d}.zip"
        archives[name]={"object_sha256":contents,"uncompressed_bytes":sum(payloads[c]["bytes"] for c in contents)}
        for checksum in contents:
            payloads[checksum]["delivery"]={"kind":"support_archive","filename":name,
                                             "member":f"objects/{checksum}"}
    for checksum,row in downloads.items():
        payloads[checksum]["delivery"]=row
    document={
        "schema":1,"project_package":str(PILOT),
        "split_lock_file_sha256":digest(ROOT/PILOT/"SPLIT_LOCK.json"),
        "input_sha256":locks["input_sha256"],"validator_asset_sha256":locks["verified_asset_sha256"],
        "code_sha256":locks["code_sha256"],"legacy_absolute_aliases":aliases,
        "logical_destinations":destinations,"payloads":payloads,"archives":archives,
        "archive_max_bytes_exclusive":ARCHIVE_LIMIT,
        "note":"Existing Drive receipt status is metadata only. Every restored download must verify SHA-256 locally.",
    }
    write("RESTORE_MANIFEST.json",document)
    with (HERE/"DRIVE_DOWNLOADS.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["kind","image_id","filename","bytes","sha256","drive_id","drive_url","upload_group","receipt_status"])
        writer.writeheader()
        writer.writerows(sorted(downloads.values(),key=lambda row:(row["kind"],row["image_id"])))
    summary={
        "locked":True,"exact_legacy_absolute_alias_count":len(aliases),
        "validator_assets":len(locks["verified_asset_sha256"]),
        "all_logical_project_paths":len(destinations),"all_unique_payloads":len(payloads),
        "all_unique_bytes":sum(p["bytes"] for p in payloads.values()),
        "support_archive_count":len(archives),"support_payloads":len(support),
        "support_payload_bytes":sum(p["bytes"] for _,p in support),
        "publication_dependencies":{label:{"files":count,"bytes":sum(r["bytes"] for r in downloads.values() if r["kind"]==label)}
                                   for label,count in sorted(categories.items())},
        "archives":{k:{"payloads":len(a["object_sha256"]),"payload_bytes":a["uncompressed_bytes"]} for k,a in archives.items()},
        "full_raw_video_archives":0,
    }
    write("RESTORE_PLAN_SUMMARY.json",summary)
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    build()
