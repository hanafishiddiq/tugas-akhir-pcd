"""Verify published Bantul packages and reassemble split ZIPs; standard library only."""
import argparse, hashlib, json
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument("directory",type=Path);a=p.parse_args();root=a.directory.resolve()
    m=json.loads((root/"CHECKSUMS.json").read_text())
    for item in m["archives"]:
        f=root/item["file"]
        if f.parent!=root or not f.is_file() or f.stat().st_size!=item["bytes"] or digest(f)!=item["sha256"]:
            raise SystemExit("Missing or mismatched package: "+item["file"])
    for item in m.get("reassembly",[]):
        dst=root/item["file"]
        if dst.parent!=root:raise SystemExit("Unsafe destination path")
        if dst.exists():
            if digest(dst)==item["sha256"]:continue
            raise SystemExit("Existing ZIP differs; preserve it and inspect: "+str(dst))
        tmp=dst.with_name(dst.name+".reassembling")
        with tmp.open("xb") as out:
            for name in item["parts"]:
                src=root/name
                if src.parent!=root:raise SystemExit("Unsafe part path")
                with src.open("rb") as f:
                    for b in iter(lambda:f.read(8*1024*1024),b""):out.write(b)
        if tmp.stat().st_size!=item["bytes"] or digest(tmp)!=item["sha256"]:
            raise SystemExit("Reassembly mismatch; diagnostic file retained: "+str(tmp))
        tmp.rename(dst)
    print(f"Verified {len(m['archives'])} packages. Extract ZIPs into the same empty folder.")

if __name__=="__main__":main()
