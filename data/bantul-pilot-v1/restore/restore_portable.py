"""Pack selected evidence and restore the frozen Bantul pilot without editing locked input."""
import argparse
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from build_restore_plan import digest

HERE=Path(__file__).resolve().parent
DEFAULT=HERE/"RESTORE_MANIFEST.json"
BUFFER=1024*1024

def read(path):
    with path.open() as handle:
        return json.load(handle)

def path_in(root, name):
    part=PurePosixPath(name)
    if not name or part.is_absolute() or ".." in part.parts or "\\" in name or str(part)!=name or name==".":
        raise ValueError(f"Unsafe restore path: {name!r}")
    path=root/part
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes restore root: {name!r}")
    return path

def verify_payload(path, checksum, size):
    if not path.is_file() or path.stat().st_size!=size or digest(path)!=checksum:
        raise ValueError(f"Payload size or SHA-256 mismatch: {path}")

def pack(plan, workspace, out):
    out.mkdir(exist_ok=True,parents=True)
    archive_records={}
    for name,spec in plan["archives"].items():
        output=path_in(out,name)
        if output.exists():
            raise FileExistsError(output)
        staging=output.with_suffix(".zip.partial")
        with ZipFile(staging,"x",compression=ZIP_STORED,allowZip64=False) as z:
            for checksum in spec["object_sha256"]:
                item=plan["payloads"][checksum]
                assert item["delivery"]["filename"]==name
                source=path_in(workspace,item["restore_paths"][0])
                verify_payload(source,checksum,item["bytes"])
                info=ZipInfo(item["delivery"]["member"],date_time=(2026,11,17,0,0,0))
                info.create_system=3
                info.external_attr=0o100644<<16
                info.compress_type=ZIP_STORED
                with source.open("rb") as read_source, z.open(info,"w") as write_member:
                    shutil.copyfileobj(read_source,write_member,length=BUFFER)
        size=staging.stat().st_size
        if size>=plan["archive_max_bytes_exclusive"]:
            raise ValueError(f"Archive {name} exceeds transfer limit")
        with ZipFile(staging) as z:
            if z.testzip() is not None:
                raise ValueError(f"CRC failure: {name}")
        staging.rename(output)
        archive_records[name]={"bytes":size,"sha256":digest(output),
                               "entries":len(spec["object_sha256"]),"zip_crc_passed":True}
    return archive_records

def find_download(root, payload, checksum):
    delivery=payload["delivery"]
    filename=delivery["filename"]
    paths=[checksum,f'{delivery["kind"]}/{filename}',filename]
    if delivery.get("upload_group"):
        paths.insert(1,f'{delivery["upload_group"]}/{filename}')
    for name in paths:
        obj=path_in(root,name)
        if obj.is_file() and obj.stat().st_size==payload["bytes"] and digest(obj)==checksum:
            return obj
    raise FileNotFoundError(f'Missing verified {delivery["kind"]} download: {filename} ({checksum})')

def materialize(plan, source_dir, arch_dir, root, *, archive_integrity):
    from contextlib import ExitStack
    with ExitStack() as stack:
        zip_files={}
        for name, archive in plan["archives"].items():
            archive_path=path_in(arch_dir,name)
            spec=archive_integrity["files"][name]
            verify_payload(archive_path,spec["sha256"],spec["bytes"])
            if spec["bytes"]>=plan["archive_max_bytes_exclusive"]:
                raise ValueError(f"Archive exceeds transfer limit: {name}")
            zip_files[name]=stack.enter_context(ZipFile(archive_path))
            expect={f"objects/{c}" for c in archive["object_sha256"]}
            entries=zip_files[name].namelist()
            if len(entries)!=len(expect) or set(entries)!=expect:
                raise ValueError(f"Unexpected archive entries: {name}")

        for checksum,item in sorted(plan["payloads"].items()):
            delivery=item["delivery"]
            targets=[path_in(root,n) for n in item["restore_paths"]]
            primary,*copies=targets
            primary.parent.mkdir(parents=True,exist_ok=True)
            with primary.open("xb") as writer:
                if delivery["kind"]=="support_archive":
                    z=zip_files[delivery["filename"]]
                    member=delivery["member"]
                    if z.getinfo(member).file_size!=item["bytes"]:
                        raise ValueError(f"Wrong uncompressed asset size: {member}")
                    with z.open(member) as source:
                        shutil.copyfileobj(source,writer,length=BUFFER)
                else:
                    source=find_download(source_dir,item,checksum)
                    with source.open("rb") as downloaded:
                        shutil.copyfileobj(downloaded,writer,length=BUFFER)
            verify_payload(primary,checksum,item["bytes"])
            for target in copies:
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(primary,target)
                verify_payload(target,checksum,item["bytes"])

def checked_run(plan, root, *, run_tests=False):
    directory=path_in(root,plan["project_package"])
    lock_path=directory/"SPLIT_LOCK.json"
    if digest(lock_path)!=plan["split_lock_file_sha256"]:
        raise ValueError("Restored split lock hash mismatch")
    lock=read(lock_path)
    if not lock["locked"] or lock["input_sha256"]!=plan["input_sha256"] or lock["code_sha256"]!=plan["code_sha256"]:
        raise ValueError("Restore plan does not describe the frozen split")
    if lock["verified_asset_sha256"]!=plan["validator_asset_sha256"]:
        raise ValueError("Asset hashes disagree with split lock")
    if read(directory/"INPUT_PATHS.json")!=lock["paths"]:
        raise ValueError("Input paths disagree with split lock")
    for original,mapped in plan["legacy_absolute_aliases"].items():
        if not Path(original).is_absolute() or lock["verified_asset_sha256"].get(original)!=mapped["sha256"]:
            raise ValueError(f"Unexpected legacy alias {original!r}")
        data=plan["logical_destinations"][mapped["restore_path"]]
        if data["sha256"]!=mapped["sha256"]:
            raise ValueError("Restored legacy path manifest hash mismatch")
    for path,item in plan["logical_destinations"].items():
        payload=plan["payloads"][item["sha256"]]
        verify_payload(path_in(root,path),item["sha256"],payload["bytes"])

    # Restrict paths to the restored root even when the original workspace is available.
    os.environ["BANTUL_PROJECT_ROOT"]=str(root)
    os.environ.pop("BANTUL_REFERENCE_CORE",None)
    sys.path.insert(0,str(directory/"code"))
    validator=importlib.import_module("readiness")
    if Path(validator.__file__).resolve()!=directory/"code/readiness.py":
        raise RuntimeError("Validator was not imported from the restored package")
    import dataset
    if dataset.ROOT!=root:
        raise RuntimeError("Annotation core was not loaded in the restored project")
    original_resolver=validator.file_at
    alias_reads=set()
    def restored_file_at(target,name,external):
        if target==root and isinstance(name,str) and Path(name).is_absolute():
            mapped=plan["legacy_absolute_aliases"].get(name)
            if not mapped:
                raise ValueError(f"Missing documented legacy mapping: {name}")
            alias_reads.add(name)
            return original_resolver(target,mapped["restore_path"],external)
        return original_resolver(target,name,external)
    validator.file_at=restored_file_at
    result=validator.validate(root,lock["paths"])
    if not result["ready"] or result["errors"]:
        raise ValueError(f'Restored package gate failed: {result["errors"]}')
    for key in ("input_sha256","verified_asset_sha256","code_sha256","support","classifier_train_support"):
        if result[key]!=lock[key]:
            raise ValueError(f"Validator output differs from lock: {key}")
    groups=read(path_in(root,lock["paths"]["groups"]))["images"]
    if {i:g["split"] for i,g in groups.items()}!=lock["assignments"]:
        raise ValueError("Restored assignments differ from the lock")
    test_result={"run":False}
    if run_tests:
        suite=unittest.defaultTestLoader.loadTestsFromNames([Path(n).stem for n in validator.CORE_TESTS])
        tests=unittest.TextTestRunner(verbosity=1).run(suite)
        if not tests.wasSuccessful() or tests.skipped:
            raise ValueError("Restored package tests failed")
        test_result={"run":True,"passed":tests.testsRun}
    return {"ready":True,"inputs_match_locked_hash":len(lock["input_sha256"]),
            "assets_match_locked_hash":len(lock["verified_asset_sha256"]),
            "code_matches_locked_hash":len(lock["code_sha256"]),
            "aliases_exercised":len(alias_reads),"assignments_match":True,
            "locked_split_sha256":plan["split_lock_file_sha256"],"tests":test_result}

def restore(plan, sources, archives, destination, integrity, *, run_tests=False):
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Stage in the target parent to publish only a fully checked directory.
    with TemporaryDirectory(dir=destination.parent,prefix=".bantul-restore-") as tmp:
        stage=Path(tmp).resolve()/"project"
        stage.mkdir()
        materialize(plan,sources,archives,stage,archive_integrity=integrity)
        receipt=checked_run(plan,stage,run_tests=run_tests)
        stage.rename(destination)
    return receipt

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=DEFAULT)
    sub=parser.add_subparsers(dest="command",required=True)
    bundle=sub.add_parser("pack")
    bundle.add_argument("--workspace",type=Path,required=True)
    bundle.add_argument("--out",type=Path,required=True)
    unpack=sub.add_parser("restore")
    unpack.add_argument("--downloads",type=Path,required=True)
    unpack.add_argument("--archives",type=Path,required=True)
    unpack.add_argument("--integrity",type=Path,required=True)
    unpack.add_argument("--destination",type=Path,required=True)
    unpack.add_argument("--run-tests",action="store_true")
    check=sub.add_parser("verify")
    check.add_argument("--root",type=Path,required=True)
    check.add_argument("--run-tests",action="store_true")
    args=parser.parse_args()
    plan=read(args.manifest)
    if args.command=="pack":
        out=args.out.resolve()
        files=pack(plan,args.workspace.resolve(),out)
        info={"manifest_sha256":digest(args.manifest),"files":files}
        (out/"SUPPORT_ARCHIVE_CHECKSUMS.json").write_text(json.dumps(info,indent=2)+"\n")
    elif args.command=="verify":
        info=checked_run(plan,args.root.resolve(),run_tests=args.run_tests)
    else:
        integrity=read(args.integrity)
        if integrity["manifest_sha256"]!=digest(args.manifest):
            raise ValueError("Archive integrity index does not bind this restore manifest")
        info=restore(plan,args.downloads.resolve(),args.archives.resolve(),
                     args.destination.resolve(),integrity,run_tests=args.run_tests)
    print(json.dumps(info,indent=2))

if __name__=="__main__":
    main()
