"""Adapters for visible target vehicles and native-image rectangle geometry."""
from dataclasses import dataclass
from hashlib import file_digest, sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import os

import numpy as np

HERE = Path(__file__).resolve().parent
CORE_RELATIVE = Path("Data/Anotasi/Bantul_222_v1/code/core.py")

def reference_paths():
    """Use explicit project root/reference path, with discovery for a package inside its project."""
    configured_root = os.environ.get("BANTUL_PROJECT_ROOT")
    configured_core = os.environ.get("BANTUL_REFERENCE_CORE")
    root = Path(configured_root).resolve() if configured_root else next(
        (p for p in (HERE,*HERE.parents) if (p/CORE_RELATIVE).is_file()),None
    )
    bundled = HERE/"reference_core.py"
    if configured_core:
        return root, Path(configured_core).resolve(strict=True)
    if bundled.is_file():
        return root,bundled
    if root is not None and (root/CORE_RELATIVE).is_file():
        return root,root/CORE_RELATIVE
    raise RuntimeError("Set BANTUL_PROJECT_ROOT or BANTUL_REFERENCE_CORE, or include reference_core.py")

ROOT, REFERENCE_CORE = reference_paths()
spec = spec_from_file_location("bantul_annotation_reference", REFERENCE_CORE)
core = module_from_spec(spec)
exec(compile(spec.loader.get_source(spec.name), str(REFERENCE_CORE), "exec"), vars(core))
CLASSES = core.TARGETS
RAW_KEYS = (
    "image_id", "source_image", "camera_id", "native_reviewed", "self_overlay_reviewed",
    "annotation_scope", "review_notes", "visual_evidence", "objects", "ignore_regions",
    "reviewer", "source_legacy_annotation",
)
SUBSETS = ("train", "validation", "test")
EXCLUSIONS = ("embargo", "quarantine")
PAD = 8

def path_hash(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()

def canonical_hash(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def raw_record(record):
    return {key: record[key] for key in RAW_KEYS}

def accepted(obj):
    return (
        obj["class_name"] in CLASSES and obj["role"] == "target_approach"
        and obj["roi_membership"] == "inside" and obj["confidence"] in {"high", "medium"}
        and not obj["ignore_reason"]
    )

@dataclass(frozen=True)
class FrameLabels:
    positive: tuple
    neutral: tuple
    known_other: tuple
    known_non_target: tuple
    exclusion_boxes: tuple
    full_count: bool
    class_count_reference: dict
    class_reference_mask: dict

def adapt(record, roi):
    """Use canonical ROI metrics and independently verify the accepted label partition."""
    verify_mask(roi, record["width"], record["height"])
    errors = core.schema_errors(raw_record(record), record)
    if errors:
        raise ValueError(f'{record["image_id"]}: {errors}')
    positive = tuple(obj for obj in record["objects"] if accepted(obj))
    other = tuple(
        obj for obj in record["objects"]
        if obj["class_name"] == "other" and obj["roi_membership"] == "inside"
    )
    non_target = tuple(
        obj for obj in record["objects"]
        if obj["roi_membership"] == "outside"
        or (obj["role"] == "context" and obj["roi_membership"] == "inside" and not obj["ignore_reason"])
    )
    known_ids = {obj["id"] for obj in (*other,*non_target)}
    neutral = tuple(
        obj for obj in record["objects"]
        if obj["id"] not in known_ids and not accepted(obj) and (
            obj["class_name"] in (*CLASSES, "unknown")
            or obj["roi_membership"] in ("boundary_review", "unknown")
        )
    ) + tuple(record["ignore_regions"])
    cls_metrics = core.metrics(record["objects"], record["ignore_regions"], roi)
    if [obj["id"] for obj in positive] != cls_metrics["accepted_object_ids"]:
        raise ValueError("Accepted-object criteria disagree with annotation core")
    for key in (
        "accepted_object_ids", "count_reference_available", "counts_reference",
        "total_count_reference", "coverage_reference_percent", "counts_resolved_lower_bound",
    ):
        if cls_metrics[key] != record["metrics"][key]:
            raise ValueError(f'Stale reference metric {key} for {record["image_id"]}')
    candidates = set()
    pending = set(cls_metrics["pending_object_ids"])
    for obj in record["objects"]:
        if obj["id"] in pending:
            candidates.update(CLASSES if obj["class_name"] == "unknown" else (obj["class_name"],))
    for region in record["ignore_regions"]:
        possible = region.get("possible_classes", list(CLASSES))
        if not isinstance(possible, list) or not possible or not all(c in CLASSES for c in possible):
            raise ValueError("Invalid possible_classes for ignore region")
        if len(set(possible)) != len(possible):
            raise ValueError("Duplicate possible classes in ignore region")
        if "possible_classes" in region and not (
            isinstance(region.get("possible_classes_reason"), str) and region["possible_classes_reason"].strip()
        ):
            raise ValueError("Explicit possible_classes needs visual reason bound to the annotation audit")
        if region["roi_relevance"] == "possible_target_in_roi":
            candidates.update(possible)
    class_reference = {
        c: None if c in candidates else cls_metrics["counts_resolved_lower_bound"][c] for c in CLASSES
    }
    class_mask = {c: class_reference[c] is not None for c in CLASSES}
    excluded = tuple(obj["bbox_xyxy"] for obj in (*record["objects"], *record["ignore_regions"]))
    return FrameLabels(positive, neutral, other, non_target, excluded, cls_metrics["count_reference_available"], class_reference, class_mask)

def verify_mask(roi, width, height):
    if not isinstance(roi, np.ndarray) or roi.dtype != bool or roi.shape != (height, width) or not roi.any():
        raise ValueError("Expected nonempty boolean native ROI mask")

def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

def iou(a, b):
    shared = intersection(a, b)
    return shared / (area(a) + area(b) - shared)

def train_positives(record, roi, entry):
    if entry["split"] != "train":
        return []
    labels = adapt(record, roi)
    use = entry["sample_role"]
    if use not in ("queue_or_control", "auxiliary_moving"):
        raise ValueError(f"Unknown training sample role {use!r}")
    return [
        {
            "image_id": record["image_id"], "object_id": obj["id"], "source_image": record["source_image"],
            "class_name": obj["class_name"], "bbox_xyxy": list(obj["bbox_xyxy"]), "split": "train",
            "source_sha256": record["source_sha256"], "annotation_sha256": record["annotation_sha256"],
            "component_id": entry["component_id"], "encounter_id": entry["object_encounter_ids"][obj["id"]],
            "sample_role": use, "full_count_frame": labels.full_count,
        }
        for obj in labels.positive
    ]

def negative_manifest(record, roi, entry, candidates):
    """Filter proposed boxes. No window generation, image copies, crops or mining on test."""
    if entry["split"] != "train" or entry["sample_role"] != "queue_or_control":
        return []
    labels = adapt(record, roi)
    if not labels.full_count:
        return []
    width, height = record["width"], record["height"]
    padded = [
        (max(0, x1-PAD), max(0, y1-PAD), min(width, x2+PAD), min(height, y2+PAD))
        for x1,y1,x2,y2 in labels.exclusion_boxes
    ]
    selected = []
    seen = set()
    for box in candidates:
        box = list(box)
        if not core.bbox_valid(box, width, height):
            raise ValueError(f"Invalid negative candidate rectangle {box}")
        key = tuple(box)
        if key in seen:
            continue
        seen.add(key)
        x1,y1,x2,y2 = box
        if not roi[y1:y2, x1:x2].all() or any(intersection(box, blocked) for blocked in padded):
            continue
        item = {
            "image_id": record["image_id"], "source_image": record["source_image"], "split": "train",
            "bbox_xyxy": box, "source_sha256": record["source_sha256"],
            "annotation_sha256": record["annotation_sha256"], "component_id": entry["component_id"],
            "pad_px": PAD,
        }
        item["manifest_key_sha256"] = canonical_hash(item)
        selected.append(item)
    return selected

AUX_ROLE = "auxiliary_classification_train_only"
AUX_RAW_KEYS = (
    "image_id", "camera_id", "source_image", "source_sha256", "width", "height",
    "source_program_date_time", "timestamp_camera_wib", "timestamp_evidence",
    "session_id", "location_group", "sample_role",
    "reviewer", "visual_evidence", "objects",
)

def auxiliary_positive_manifest(record, entry):
    """Return clear vehicle classifier crops only, with no ROI or count semantics."""
    if entry["split"] != "train" or entry["sample_role"] != AUX_ROLE or record["sample_role"] != AUX_ROLE:
        raise ValueError("Auxiliary classifier images are train-only")
    if any(key in record for key in ("roi_version","metrics","reference_scope")):
        raise ValueError("Auxiliary classifier record must not emulate an ROI/count reference")
    width,height = record["width"],record["height"]
    ids = set()
    seen_boxes = set()
    positives = []
    for obj in record["objects"]:
        box = obj["bbox_xyxy"]
        oid = obj["id"]
        if (
            not isinstance(oid,str) or not oid.startswith(record["image_id"]+"-") or oid in ids
            or obj["class_name"] not in CLASSES or not core.bbox_valid(box,width,height)
            or tuple(box) in seen_boxes or obj["confidence"] != "high"
            or not isinstance(obj["reason"],str) or not obj["reason"].strip()
        ):
            raise ValueError(f"Invalid auxiliary classifier crop {oid!r}")
        ids.add(oid)
        seen_boxes.add(tuple(box))
        encounter = entry["object_encounter_ids"][oid]
        if not isinstance(encounter,str) or not encounter.strip():
            raise ValueError(f"Missing auxiliary encounter ID for {oid}")
        positives.append({
            "image_id":record["image_id"],"object_id":oid,"class_name":obj["class_name"],
            "bbox_xyxy":list(box),"source_image":record["source_image"],"split":"train",
            "source_sha256":record["source_sha256"],"annotation_sha256":record["annotation_sha256"],
            "camera_id":record["camera_id"],"encounter_id":encounter,"component_id":entry["component_id"],
            "sample_role":AUX_ROLE,"detection_or_count_support":False,
        })
    if not positives or set(entry["object_encounter_ids"]) != ids:
        raise ValueError("Auxiliary crops need an exhaustive encounter map")
    return positives
