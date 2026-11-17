"""Verify source integrity, group separation, and fixed pilot gates before locking."""
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import (
    AUX_RAW_KEYS, AUX_ROLE, CLASSES, EXCLUSIONS, RAW_KEYS, REFERENCE_CORE, SUBSETS,
    adapt, auxiliary_positive_manifest, canonical_hash, path_hash,
)

MINIMUM = {
    "train_instances_per_class": 10,
    "train_encounters_per_class": 5,
    "eval_encounters_per_class": 2,
    "eval_full_count_frames": 10,
    "eval_empty_controls": 2,
    "eval_full_count_encounters_per_class": 2,
    "temporal_embargo_each_side_seconds": 120,
}
FILES = ("manifest", "annotations", "rois", "groups", "protocol", "test_receipt")
CORE_TESTS = ("test_data_contract.py", "test_fixture.py")
CODE_FILES = ("dataset.py", "evaluate.py", "readiness.py", "run_tests.py", "FIXTURE_222.json", *CORE_TESTS)

def code_hashes():
    return {name:path_hash(Path(__file__).with_name(name)) for name in CODE_FILES} | {
        "annotation_reference_core":path_hash(REFERENCE_CORE),
    }

def file_at(root, relative, external):
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Expected a readable path: {relative!r}")
    path = root / relative
    actual = path.resolve(strict=True)
    if not actual.is_file():
        raise ValueError(f"Not a readable file: {relative}")
    tracked = external.get(relative)
    if not actual.is_relative_to(root) and (
        not tracked or str(actual) != tracked.get("resolved_path") or path_hash(actual) != tracked.get("sha256")
    ):
        raise ValueError(f"Undocumented external path: {relative}")
    with actual.open("rb") as handle:
        handle.read(1)
    return actual

def indexed(rows, field):
    result = {}
    if not isinstance(rows, list):
        raise ValueError(f"Expected list for {field} index")
    for item in rows:
        key = item[field]
        if not isinstance(key, str) or not key.strip() or key in result:
            raise ValueError(f"Missing or repeated {field}: {key!r}")
        result[key] = item
    return result

def load_mask(root, camera, data, external):
    path = data.get("mask_path", f"Data/ROI/Bantul_v1/masks/{camera}_road.png")
    target = file_at(root, path, external)
    if path_hash(target) != data["mask_sha256"]:
        raise ValueError(f"{camera} ROI mask hash mismatch")
    if data["geometry_valid"] is not True or data["status"] != "locked_for_annotation":
        raise ValueError(f"{camera} ROI geometry is unverified")
    with Image.open(target) as image:
        raw = np.array(image)
    if raw.shape != (data["height"], data["width"]) or not np.isin(raw, [0, 1, 255]).all():
        raise ValueError(f"{camera} ROI mask is not binary or dimensions mismatch")
    mask = raw > 0
    if not mask.any() or int(mask.sum()) != data["raster_area_px"]:
        raise ValueError(f"{camera} ROI mask area mismatch")
    return mask, path

def valid_audit(record, image_id, root, external, *, raw_keys=RAW_KEYS):
    audit = record["cross_audit"]
    raw = {key:record[key] for key in raw_keys}
    expected = canonical_hash(raw)
    reviewer, auditor = record["reviewer"], audit["reviewer"]
    if (
        record["annotation_sha256"] != expected or audit["annotation_sha256"] != expected
        or audit["image_id"] != image_id or reviewer == auditor
        or not all(isinstance(x,str) and x.strip() for x in (reviewer,auditor))
        or audit["source_reviewed"] is not True or audit["overlay_reviewed"] is not True
        or audit["status"] not in ("pass", "pass_with_uncertainty") or audit["open_issues"] != []
        or record["cross_audit_status"] != "passed"
    ):
        raise ValueError("Missing, stale or non-independent visual audit")
    for refs in (record["visual_evidence"], audit["visual_evidence"]):
        if not isinstance(refs,list) or not refs:
            raise ValueError("Visual evidence paths required")
        for path in refs:
            file_at(root,path,external)

def auxiliary_source_time(record):
    source_time = record["source_program_date_time"]
    if source_time is None:
        if record["timestamp_evidence"] != "native_camera_overlay":
            raise ValueError("Auxiliary source without HLS PDT needs audited native camera timestamp evidence")
        source_time = record["timestamp_camera_wib"]
    if not isinstance(source_time,str) or not source_time:
        raise ValueError("Missing auxiliary source timestamp")
    instant = datetime.fromisoformat(source_time)
    if instant.tzinfo is None:
        raise ValueError("Auxiliary source timestamp needs timezone")
    return instant

def protocol_config(protocol):
    thresholds = protocol["gates"]
    if not isinstance(thresholds,dict) or set(thresholds) != set(MINIMUM):
        raise ValueError("Protocol must explicitly specify all gate thresholds")
    for key, floor in MINIMUM.items():
        if type(thresholds[key]) is not int or thresholds[key] < floor:
            raise ValueError(f"Protocol weakens gate {key}: minimum {floor}")
    if protocol["evaluation_claim"] != "fixed_camera_temporal_holdout":
        raise ValueError("This protocol applies to fixed-camera temporal holdout")
    return thresholds

def support_table(rows):
    report = {}
    for subset in SUBSETS:
        sampled = [r for r in rows if r[1]["split"] == subset]
        report[subset] = {
            "images":len(sampled),
            "full_count_frames":sum(r[2].full_count for r in sampled),
            "full_empty_controls":sum(r[2].full_count and not r[2].positive for r in sampled),
            "by_class":{},
        }
        for cls in CLASSES:
            positives = [
                (frame["image_id"], group["object_encounter_ids"][obj["id"]], group.get("full_count_encounter_id"))
                for frame,group,labels in sampled for obj in labels.positive if obj["class_name"]==cls
            ]
            report[subset]["by_class"][cls] = {
                "accepted_instances":len(positives), "positive_images":len({x[0] for x in positives}),
                "positive_encounters":len({x[1] for x in positives}),
                "positive_full_count_scene_encounters":len({x[2] for x in positives if x[2] is not None}),
                "positive_full_count_encounters":len({x[1] for x in positives if x[2] is not None}),
            }
    return report

def support_failures(report, classifier, min_counts):
    errors = []
    for cls, values in classifier["by_class"].items():
        for metric, key in (("accepted_instances","train_instances_per_class"),("positive_encounters","train_encounters_per_class")):
            if values[metric] < min_counts[key]:
                errors.append(f'classifier train {cls} {metric}: {values[metric]} < {min_counts[key]}')
    for subset in ("validation","test"):
        for metric,key in (("full_count_frames","eval_full_count_frames"),("full_empty_controls","eval_empty_controls")):
            value = report[subset][metric]
            if value < min_counts[key]:
                errors.append(f'{subset} {metric}: {value} < {min_counts[key]}')
        for cls, values in report[subset]["by_class"].items():
            for metric,key in (("positive_encounters","eval_encounters_per_class"),
                               ("positive_full_count_encounters","eval_full_count_encounters_per_class"),
                               ("positive_full_count_scene_encounters","eval_full_count_encounters_per_class")):
                if values[metric] < min_counts[key]:
                    errors.append(f'{subset} {cls} {metric}: {values[metric]} < {min_counts[key]}')
    return errors

def check_shared_keys(seen, subset, physical_keys):
    for key,value in physical_keys:
        if not isinstance(value,str) or not value.strip():
            raise ValueError(f"Missing {key}")
        other = seen.setdefault((key,value),subset)
        if other != subset:
            raise ValueError(f"{key} crosses splits: {value} ({other}/{subset})")

def image_pixel_key(rgb):
    return canonical_hash({"shape":rgb.size,"pixels":sha256(rgb.tobytes()).hexdigest()})

def source_keys(image, digest, pixel_hash, entry):
    if not isinstance(entry["event_ids"],list) or not entry["event_ids"]:
        raise ValueError("Each used image requires an event_ids list")
    return [
        ("source_image",image["source_image"]), ("source_sha256",digest), ("decoded_rgb_sha256",pixel_hash),
        ("component_id",entry["component_id"]),
        *[("event_id",v) for v in entry["event_ids"]],
        *[("object_encounter_id",v) for v in entry["object_encounter_ids"].values()],
    ]

def classifier_support(rows, auxiliary, pixel_keys):
    """Deduplicate exact object crops even within train. Report core and aux inventory separately."""
    crops, used = [], {}
    for frame,group,labels in rows:
        if group["split"] == "train":
            crops.extend((pixel_keys[frame["image_id"]],o["bbox_xyxy"],o["class_name"],group["object_encounter_ids"][o["id"]])
                         for o in labels.positive)
    core_total = len(crops)
    crops.extend((pixel_keys[o["image_id"]],o["bbox_xyxy"],o["class_name"],o["encounter_id"]) for o in auxiliary)
    for digest,box,cls,encounter in crops:
        key = (digest,tuple(box))
        if used.setdefault(key,(cls,encounter)) != (cls,encounter):
            raise ValueError("Identical classifier crop has conflicting classes or encounter identities")
    return {
        "core_accepted_train_instances":core_total, "verified_auxiliary_crop_instances":len(auxiliary),
        "duplicate_crops_discounted":len(crops)-len(used),
        "by_class":{
            cls:{
                "accepted_instances":sum(c==cls for c,_ in used.values()),
                "positive_encounters":len({encounter for c,encounter in used.values() if c==cls}),
            } for cls in CLASSES
        },
    }

def validate(root, paths):
    """Return report; all hash/path reads are repeated before a lock is written."""
    root = Path(root).resolve()
    if not set(FILES) <= set(paths) or set(paths) - {*FILES,"auxiliary_manifest"}:
        raise ValueError(f"Paths must name {FILES}; auxiliary_manifest is optional")
    errors, hashes, verified_assets = [], {}, {}
    try:
        payloads = {}
        for key, value in paths.items():
            content = file_at(root,value,{}).read_bytes()
            hashes[key] = sha256(content).hexdigest()
            payloads[key] = json.loads(content)
        protocol = payloads["protocol"]
        thresholds = protocol_config(protocol)
        external = protocol.get("documented_external_assets", {})
        manifest = indexed(payloads["manifest"],"image_id")
        annotations = indexed(payloads["annotations"],"image_id")
        groups = payloads["groups"]
        assignments = groups["images"]
        aux = indexed(payloads.get("auxiliary_manifest",[]),"image_id")
        if (
            set(manifest) != set(annotations) or set(manifest) & set(aux)
            or set(manifest) | set(aux) != set(assignments) or not manifest
        ):
            raise ValueError("Grouping must cover exactly core manifest, annotations and optional disjoint auxiliary IDs")
        rois = payloads["rois"]
        masks = {}
        for camera, roi in rois.items():
            masks[camera], mask_path = load_mask(root,camera,roi,external)
            verified_assets[mask_path] = roi["mask_sha256"]
        receipt = payloads["test_receipt"]
        if receipt["code_hashes"] != code_hashes() or receipt["tests_failed"] != 0 or receipt["tests_run"] < 1 or receipt["suite"] != list(CORE_TESTS):
            errors.append("Adapter/evaluator test receipt missing required current successful code hashes")
    except (KeyError, TypeError, ValueError, OSError, AttributeError) as error:
        return {"ready":False,"errors":[str(error)],"input_sha256":hashes,"support":None}

    rows, seen, frame_times, pixel_keys = [], {}, {}, {}
    for image_id, image in sorted(manifest.items()):
        try:
            record = annotations[image_id]
            entry = assignments[image_id]
            subset = entry["split"]
            if subset in EXCLUSIONS:
                if not isinstance(entry.get("exclusion_reason"),str) or not entry["exclusion_reason"].strip():
                    raise ValueError("Embargo or quarantine needs an explicit reason")
                continue
            if subset not in SUBSETS:
                raise ValueError("Split must be train, validation, test, embargo or quarantine")
            if entry["sample_role"] not in ("queue_or_control","auxiliary_moving") or (
                entry["sample_role"]=="auxiliary_moving" and subset!="train"
            ):
                raise ValueError("Auxiliary moving frames are train-only")
            for key in ("source_image","source_sha256","width","height","camera_id","roi_version"):
                if image[key] != record[key]:
                    raise ValueError(f"Manifest and annotation mismatch for {key}")
            if image["native_checked"] is not True or image["roi_checked"] is not True:
                raise ValueError("Missing native framing or ROI review")
            framing = file_at(root,image["framing_evidence"],external)
            with Image.open(framing) as overlay:
                overlay.verify()
            verified_assets[image["framing_evidence"]] = path_hash(framing)
            source = file_at(root,image["source_image"],external)
            digest = path_hash(source)
            if digest != image["source_sha256"]:
                raise ValueError("Source SHA-256 mismatch")
            with Image.open(source) as photo:
                rgb = photo.convert("RGB")
            if rgb.size != (record["width"], record["height"]):
                raise ValueError("Native source dimensions differ")
            verified_assets[image["source_image"]] = digest
            camera = record["camera_id"]
            geometry = rois[camera]
            if record["roi_version"] != geometry["version"]:
                raise ValueError("Frame is bound to a different ROI version")
            valid_audit(record,image_id,root,external)
            for evidence in (*record["visual_evidence"], *record["cross_audit"]["visual_evidence"]):
                path = file_at(root,evidence,external)
                if evidence not in verified_assets:
                    verified_assets[evidence] = path_hash(path)
            labels = adapt(record, masks[camera])
            ids = entry["object_encounter_ids"]
            if set(ids) != {o["id"] for o in labels.positive}:
                raise ValueError("Each accepted instance requires exactly one encounter group")
            scene = entry.get("full_count_encounter_id")
            if labels.full_count != (isinstance(scene,str) and bool(scene.strip())):
                raise ValueError("Full-count labels require a scene encounter ID; partial labels must omit it")
            pixel_keys[image_id] = image_pixel_key(rgb)
            physical = source_keys(image,digest,pixel_keys[image_id],entry)
            if labels.full_count:
                physical.append(("full_count_encounter_id",scene))
            check_shared_keys(seen, subset, physical)
            stamp = entry.get("source_time_utc")
            if stamp is not None:
                instant = datetime.fromisoformat(stamp)
                if instant.tzinfo is None:
                    raise ValueError("Source time must be timezone-aware")
                original = image.get("source_program_date_time")
                if not original or datetime.fromisoformat(original) != instant:
                    raise ValueError("Source time does not match manifest HLS PDT")
                frame_times[image_id] = instant
            rows.append((record,entry,labels))
        except (KeyError, TypeError, ValueError, OSError, AttributeError) as error:
            errors.append(f"{image_id}: {error}")

    verified_aux = []
    for image_id, record in sorted(aux.items()):
        try:
            entry = assignments[image_id]
            crops = auxiliary_positive_manifest(record,entry)
            source = file_at(root,record["source_image"],external)
            digest = path_hash(source)
            if digest != record["source_sha256"]:
                raise ValueError("Auxiliary source SHA-256 mismatch")
            with Image.open(source) as photo:
                rgb = photo.convert("RGB")
            if rgb.size != (record["width"],record["height"]):
                raise ValueError("Auxiliary source dimensions mismatch")
            valid_audit(record,image_id,root,external,raw_keys=AUX_RAW_KEYS)
            frame_times[image_id] = auxiliary_source_time(record)
            for evidence in (record["source_image"],*record["visual_evidence"],*record["cross_audit"]["visual_evidence"]):
                path = file_at(root,evidence,external)
                if evidence not in verified_assets:
                    verified_assets[evidence] = path_hash(path)
            pixel_keys[image_id] = image_pixel_key(rgb)
            check_shared_keys(seen,"train",source_keys(record,digest,pixel_keys[image_id],entry))
            verified_aux.extend(crops)
        except (KeyError, TypeError, ValueError, OSError, AttributeError) as error:
            errors.append(f"{image_id}: {error}")

    try:
        link_checks(groups, frame_times, manifest | aux, thresholds, root, external, verified_assets)
    except (KeyError, TypeError, ValueError, OSError, AttributeError) as error:
        errors.append(f"group policy: {error}")
    report = support_table(rows)
    core_train = report["train"]
    if not any(r["accepted_instances"] for r in core_train["by_class"].values()):
        errors.append("Binary core detector has no accepted train vehicle positives")
    if not any(labels.full_count and group["split"]=="train" and group["sample_role"]=="queue_or_control"
               for _,group,labels in rows):
        errors.append("Binary core detector has no full-count train queue/control frame eligible for negative sampling")
    try:
        classifier = classifier_support(rows,verified_aux,pixel_keys)
        errors.extend(support_failures(report,classifier,thresholds))
    except ValueError as error:
        errors.append(str(error))
        classifier = None
    return {
        "ready":not errors,"errors":errors,"input_sha256":hashes,"support":report,
        "verified_asset_sha256":verified_assets, "code_sha256":code_hashes(),
        "classifier_train_support":classifier,"verified_auxiliary_train_crops":verified_aux,
        "binary_detector_background_note":"Full-count core train frames permit ROI negative filtering; actual proposed rectangles still require negative_manifest checks.",
    }

def link_checks(groups, frame_times, manifest, thresholds, root, external, verified_assets):
    assignments = groups["images"]
    edges = defaultdict(set)
    for edge in groups["must_link"]:
        image_ids = edge["image_ids"]
        if not isinstance(image_ids,list) or len(set(image_ids)) < 2 or not all(i in assignments for i in image_ids):
            raise ValueError("Each linkage needs distinct, declared IDs")
        evidence = file_at(root, edge["evidence"], external)
        verified_assets[edge["evidence"]] = path_hash(evidence)
        first, *rest = image_ids
        edges[first].update(rest)
        for item in rest:
            edges[item].add(first)
    visited = set()
    for image_id in edges:
        if image_id in visited:
            continue
        pending, active = [image_id], []
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            row = assignments[node]
            if row["split"] in SUBSETS:
                active.append(row)
            pending.extend(edges[node] - visited)
        if len({row["split"] for row in active}) > 1 or len({row["component_id"] for row in active}) > 1:
            raise ValueError(f"Identity component crosses subsets or components: {image_id}")
    active_windows = defaultdict(set)
    for image_id, row in assignments.items():
        if row["split"] in SUBSETS:
            source = manifest[image_id]
            active_windows[source["location_group"],source["session_id"]].add(row["split"])
    unresolved_windows = {window for window, subsets in active_windows.items() if len(subsets)>1}
    boundary_coverage = defaultdict(set)
    if not groups["boundary_evidence"] and len({r["split"] for r in assignments.values()} & set(SUBSETS)) > 1:
        raise ValueError("Holdout boundaries need explicit evidence")
    for boundary in groups["boundary_evidence"]:
        evidence = file_at(root,boundary["evidence"],external)
        verified_assets[boundary["evidence"]] = path_hash(evidence)
        if boundary["kind"] == "separate_acquisition_windows":
            continue
        if boundary["kind"] != "within_acquisition_reset":
            raise ValueError("Unrecognized boundary kind")
        cut = datetime.fromisoformat(boundary["cut_utc"])
        if cut.tzinfo is None:
            raise ValueError("Cut time must include timezone")
        margin = boundary["embargo_each_side_seconds"]
        if type(margin) is not int or margin < thresholds["temporal_embargo_each_side_seconds"]:
            raise ValueError("Insufficient embargo margin")
        window = (boundary["location_group"],boundary["session_id"])
        before, after = set(boundary["subsets_before"]), set(boundary["subsets_after"])
        if not before or not after or before & after or not (before | after) <= set(SUBSETS):
            raise ValueError("Boundary subset partitions must be nonempty and disjoint")
        for before_label in before:
            for after_label in after:
                boundary_coverage[window].add(frozenset((before_label,after_label)))
        for image_id, entry in assignments.items():
            frame = manifest[image_id]
            if entry["split"] not in SUBSETS or (frame["location_group"],frame["session_id"])!=window:
                continue
            if image_id not in frame_times:
                raise ValueError(f"{image_id} lacks auditable source time in temporal holdout window")
            delta = (frame_times[image_id]-cut).total_seconds()
            if abs(delta) <= margin:
                raise ValueError(f"{image_id} falls inside active embargo")
            if entry["split"] not in (before if delta < 0 else after):
                raise ValueError(f"{image_id} is on wrong side of chronological boundary")
    for window in unresolved_windows:
        required = {frozenset(pair) for pair in combinations(active_windows[window],2)}
        if not required <= boundary_coverage[window]:
            raise ValueError(f"Missing reset/embargo evidence for holdout split within {window}")

def write_split_lock(root, paths, output, *, expected_input_sha256):
    """Recompute every gate, verify expected frozen inputs, never overwrite."""
    report = validate(root, paths)
    if not report["ready"]:
        raise ValueError(f'Split lock refused: {len(report["errors"])} readiness errors')
    if expected_input_sha256 != report["input_sha256"]:
        raise ValueError("Split lock refused: input hashes differ from reviewed snapshot")
    root = Path(root).resolve()
    groups = json.loads(file_at(root,paths["groups"],{}).read_text())
    if path_hash(root/paths["groups"]) != report["input_sha256"]["groups"]:
        raise ValueError("Split lock refused: group content changed during validation")
    for key,expected in report["input_sha256"].items():
        if path_hash(file_at(root,paths[key],{})) != expected:
            raise ValueError(f"Split lock refused: {key} changed during validation")
    protocol = json.loads(file_at(root,paths["protocol"],{}).read_text())
    external = protocol.get("documented_external_assets", {})
    for asset, expected in report["verified_asset_sha256"].items():
        if path_hash(file_at(root,asset,external)) != expected:
            raise ValueError(f"Split lock refused: asset changed during validation: {asset}")
    if code_hashes() != report["code_sha256"]:
        raise ValueError("Split lock refused: helper code changed during validation")
    with Path(output).open("x",encoding="utf8") as handle:
        json.dump({
            "locked":True,"locked_at_utc":datetime.now(timezone.utc).isoformat(),
            "input_sha256":report["input_sha256"],"paths":paths,"support":report["support"],
            "verified_asset_sha256":report["verified_asset_sha256"],"code_sha256":report["code_sha256"],
            "classifier_train_support":report["classifier_train_support"],
            "assignments":{k:v["split"] for k,v in groups["images"].items()},
        },handle,indent=2,ensure_ascii=False)
        handle.write("\n")
    return report

def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--paths",type=Path,required=True,help="JSON mapping from input type to file path")
    parser.add_argument("--report",type=Path,help="Optional readiness report path")
    args = parser.parse_args()
    report = validate(args.root,json.loads(args.paths.read_text()))
    output = json.dumps(report,ensure_ascii=False,indent=2)+"\n"
    if args.report:
        with args.report.open("x",encoding="utf8") as handle:
            handle.write(output)
    else:
        print(output,end="")
    return 0 if report["ready"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
