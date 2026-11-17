"""Evaluate explicit pipeline outputs on fixed-camera Bantul holdout frames."""
import math
from collections import Counter, defaultdict

import numpy as np

from dataset import CLASSES, adapt, area, core, iou

MATCH_IOU = 0.5
IGNORE_FRACTION = 0.5

def unique_ids(items, key):
    if not all(isinstance(item,dict) for item in items):
        return False
    values = [item.get(key) for item in items]
    return all(isinstance(value, str) and value.strip() for value in values) and len(set(values)) == len(values)

def valid_score(value):
    try:
        return type(value) in (int,float) and math.isfinite(value)
    except OverflowError:
        return False

def predictions_or_error(result, frame):
    if not isinstance(result, dict) or result.get("status") != "ok":
        return None, f"Missing or failed pipeline result: {result!r}"[:240]
    predictions = result.get("predictions")
    if not isinstance(predictions, list) or not all(isinstance(p, dict) for p in predictions):
        return None, "Missing or invalid predictions list"
    if not unique_ids(predictions, "id"):
        return None, "Missing or duplicate prediction IDs"
    for pred in predictions:
        confidence = pred.get("confidence")
        valid = (
            pred.get("class_name") in CLASSES
            and core.bbox_valid(pred.get("bbox_xyxy"), frame["width"], frame["height"])
            and pred.get("roi_membership") in ("inside", "outside", "uncertain")
            and valid_score(confidence)
        )
        if not valid:
            return None, f'Invalid prediction {pred["id"]}'
    return predictions, None

def neutral_fraction(box, zones):
    """Fraction of the prediction covered by the clipped union of neutral boxes."""
    x1, y1, x2, y2 = box
    overlap = np.zeros((y2-y1, x2-x1), dtype=bool)
    for u1, v1, u2, v2 in zones:
        a, b, c, d = max(x1,u1), max(y1,v1), min(x2,u2), min(y2,v2)
        if a < c and b < d:
            overlap[b-y1:d-y1, a-x1:c-x1] = True
    return int(overlap.sum()) / area(box)

def evaluate_image(frame, roi, result):
    """Inside predictions are used for both counts and bbox coverage before matching."""
    labels = adapt(frame, roi)
    predictions, error = predictions_or_error(result, frame)
    if error:
        return {"image_id": frame["image_id"], "valid": False, "error": error}
    inside = sorted(
        (p for p in predictions if p["roi_membership"] == "inside"),
        key=lambda p: (-p["confidence"], p["id"]),
    )
    counts = Counter(p["class_name"] for p in inside)
    accepted_boxes = [p["bbox_xyxy"] for p in inside]
    covered_pixels = core.union_area(roi, accepted_boxes)
    coverage = 100 * covered_pixels / int(roi.sum())
    matches = {}
    decisions = []
    zones = [r["bbox_xyxy"] for r in labels.neutral]
    for pred in inside:
        box = pred["bbox_xyxy"]
        overlaps = [(iou(box, gt["bbox_xyxy"]), gt) for gt in labels.positive]
        potential = sorted(
            ((score, gt["id"]) for score, gt in overlaps
             if score >= MATCH_IOU and gt["class_name"] == pred["class_name"] and gt["id"] not in matches),
            key=lambda match: (-match[0], match[1]),
        )
        decision = {"id":pred["id"], "class_name":pred["class_name"]}
        if potential:
            score, gt_id = potential[0]
            matches[gt_id] = pred["id"]
            decision.update(outcome="tp", matched_object_id=gt_id, iou=score)
        elif any(score >= MATCH_IOU for score, _ in overlaps):
            reason = "duplicate" if any(
                score >= MATCH_IOU and gt["class_name"] == pred["class_name"] for score,gt in overlaps
            ) else "wrong_class"
            decision.update(outcome="fp", reason=reason)
        elif any(iou(box, gt["bbox_xyxy"]) >= MATCH_IOU for gt in labels.known_other):
            decision.update(outcome="fp", reason="known_other")
        elif any(iou(box, gt["bbox_xyxy"]) >= MATCH_IOU for gt in labels.known_non_target):
            decision.update(outcome="fp", reason="membership_error")
        elif neutral_fraction(box, zones) >= IGNORE_FRACTION:
            decision.update(outcome="neutral", reason="ignore_overlap")
        else:
            decision.update(outcome="fp", reason="unmatched")
        decisions.append(decision)
    counts = {c:counts[c] for c in CLASSES}
    gt_counts = frame["metrics"]["counts_reference"]
    count_errors = None if not labels.full_count else {
        **{c:abs(counts[c]-gt_counts[c]) for c in CLASSES},
        "total":abs(sum(counts.values())-sum(gt_counts.values())),
        "coverage_pp":abs(coverage-frame["metrics"]["coverage_reference_percent"]),
    }
    membership = Counter(p["roi_membership"] for p in predictions)
    return {
        "image_id":frame["image_id"], "valid":True, "full_count_frame":labels.full_count,
        "class_count_reference":labels.class_count_reference, "class_reference_mask":labels.class_reference_mask,
        "detected_object_ids":list(matches),
        "false_negative_object_ids":[obj["id"] for obj in labels.positive if obj["id"] not in matches],
        "detections":decisions,
        "abstain_prediction_ids":[p["id"] for p in predictions if p["roi_membership"]=="uncertain"],
        "membership_counts":dict(membership),
        "accepted_prediction_ids":[p["id"] for p in inside],
        "accepted_counts":counts, "accepted_coverage_percent":coverage,
        "accepted_covered_area_px":covered_pixels, "count_absolute_errors":count_errors,
    }

def aggregate(rows):
    """Report exposure beside micro metrics and per-object encounter support."""
    full = [row for row in rows if row[0]["full_count_frame"]]
    outcomes = {c:Counter() for c in CLASSES}
    encounter_outcomes = defaultdict(Counter)
    full_encounters = defaultdict(set)
    full_scene_encounters = defaultdict(set)
    membership = Counter()
    score_errors = Counter()
    for result, frame, group in rows:
        membership.update(result["membership_counts"])
        outcomes_for_frame = result["detections"]
        for det in outcomes_for_frame:
            outcomes[det["class_name"]][det["outcome"]] += 1
        object_by_id = {obj["id"]:obj for obj in frame["objects"]}
        for outcome,key in (("tp","detected_object_ids"),("fn","false_negative_object_ids")):
            for oid in result[key]:
                cls = object_by_id[oid]["class_name"]
                encounter = group["object_encounter_ids"][oid]
                outcomes[cls]["fn"] += outcome == "fn"
                encounter_outcomes[cls, encounter][outcome] += 1
                if result["full_count_frame"]:
                    full_encounters[cls].add(encounter)
                    full_scene_encounters[cls].add(group["full_count_encounter_id"])
        if result["count_absolute_errors"] is not None:
            score_errors.update(result["count_absolute_errors"])
    by_class = {}
    for cls in CLASSES:
        positive_image_count = sum(r[1]["metrics"]["counts_resolved_lower_bound"][cls] > 0 for r in rows)
        positive_full_images = sum(r[1]["metrics"]["counts_reference"][cls] > 0 for r in full)
        encounters = {
            encounter: {"tp": counts["tp"], "fn": counts["fn"]}
            for (classification, encounter), counts in encounter_outcomes.items() if classification == cls
        }
        tp, fp, fn, neutral = (outcomes[cls][k] for k in ("tp","fp","fn","neutral"))
        by_class[cls] = {
            "tp":tp, "fp":fp, "fn":fn, "neutral":neutral, "positive_objects":tp+fn,
            "positive_images":positive_image_count,
            "positive_full_count_images":positive_full_images,
            "positive_encounter_groups":len(encounters),
            "positive_full_count_encounter_groups":len(full_encounters[cls]),
            "positive_full_count_scene_groups":len(full_scene_encounters[cls]),
            "precision":tp/(tp+fp) if tp+fp else None,
            "recall":tp/(tp+fn) if tp+fn else None,
            "by_positive_encounter":encounters,
        }
    seen = sum(membership.values())
    return {
        "images":len(rows), "component_groups":len({r[2]["component_id"] for r in rows}),
        "full_count_frames":len(full), "partial_label_frames":len(rows)-len(full),
        "prediction_membership_counts":dict(membership),
        "membership_decision_coverage":(seen-membership["uncertain"])/seen if seen else None,
        "mae_full_only":{k:score_errors[k]/len(full) for k in (*CLASSES,"total","coverage_pp")} if full else None,
        "detection_by_class":by_class,
    }

def evaluate_holdout(frames, masks_by_camera, groups, results, *, subset):
    """Reject absent/extra/invalid results as one invalid run. No partial headline metrics."""
    if subset not in ("validation","test"):
        raise ValueError("Holdout evaluation requires validation or test")
    if not unique_ids(frames,"image_id"):
        return {"evaluation_run_valid":False,"errors":["Missing or duplicate frame IDs"],"summary":None}
    try:
        selected = [r for r in frames if groups[r["image_id"]]["split"]==subset]
    except (KeyError, TypeError):
        return {"evaluation_run_valid":False,"errors":["Missing or malformed image group"],"summary":None}
    if not selected:
        return {"evaluation_run_valid":False,"errors":["No frames in requested subset"],"summary":None}
    if not isinstance(results, dict):
        return {"evaluation_run_valid":False,"errors":["Results must map image ID to one output per selected frame"],"summary":None}
    extra = sorted(set(results) - {r["image_id"] for r in selected},key=repr)
    errors = [f"Unexpected image output {image_id}" for image_id in extra]
    rows, details = [], []
    for frame in selected:
        image_id = frame["image_id"]
        try:
            item = evaluate_image(frame, masks_by_camera[frame["camera_id"]], results.get(image_id))
            group = groups[image_id]
            accepted_ids = frame["metrics"]["accepted_object_ids"]
            if set(group["object_encounter_ids"]) != set(accepted_ids):
                raise ValueError("Incomplete or extra object encounter IDs")
            required_ids = [group["component_id"], *group["object_encounter_ids"].values()]
            if item.get("full_count_frame"):
                required_ids.append(group.get("full_count_encounter_id"))
            if not all(isinstance(g,str) and g.strip() for g in required_ids):
                raise ValueError("Missing or invalid component or encounter IDs")
        except (KeyError, TypeError, ValueError) as error:
            item = {"image_id":image_id,"valid":False,"error":str(error)}
        details.append(item)
        if item["valid"]:
            rows.append((item,frame,group))
        else:
            errors.append(f'{image_id}: {item["error"]}')
    if errors:
        return {"evaluation_run_valid":False,"errors":errors,"summary":None,"per_image":details}
    by_camera = defaultdict(list)
    for row in rows:
        by_camera[row[1]["camera_id"]].append(row)
    return {
        "evaluation_run_valid":True,"errors":[],"per_image":details,
        "summary":aggregate(rows),
        "by_camera":{k:aggregate(v) for k,v in sorted(by_camera.items())},
        "by_reference_scope":{
            "full_count":aggregate([r for r in rows if r[0]["full_count_frame"]]),
            "partial":aggregate([r for r in rows if not r[0]["full_count_frame"]]),
        },
    }
