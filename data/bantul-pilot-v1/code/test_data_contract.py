from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from dataset import AUX_RAW_KEYS, AUX_ROLE, CLASSES, adapt, canonical_hash, core, negative_manifest, path_hash, raw_record, train_positives
from evaluate import evaluate_holdout, evaluate_image
from readiness import CORE_TESTS, MINIMUM, auxiliary_source_time, code_hashes, file_at, link_checks, validate, write_split_lock

HERE = Path(__file__).parent

def obj(image_id, index, cls="mobil", bbox=(15,15,25,25), membership="inside"):
    return {
        "id":f"{image_id}-{index}","class_name":cls,"bbox_xyxy":list(bbox),"roi_membership":membership,
        "role":"target_approach","confidence":"high","subtype":"test","occlusion":"none","truncated":False,
        "membership_reason":"synthetic placement","road_contact_xy":None,"road_contact_note":"synthetic",
        "crosses_roi_boundary":False,"classification_note":"test vehicle",
        "ignore_reason":"synthetic uncertainty" if cls=="unknown" or membership in ("unknown","boundary_review") else None,
    }

def frame(image_id, mask, objects=(), regions=(), camera="camera"):
    return {
        "image_id":image_id,"camera_id":camera,"source_image":"photo.jpg","width":mask.shape[1],"height":mask.shape[0],
        "source_sha256":"synthetic","roi_version":"test-v1","reviewer":"annotator","native_reviewed":True,
        "self_overlay_reviewed":True,"annotation_scope":"target_approach_and_near_boundary_context",
        "review_notes":["synthetic fixture"],"visual_evidence":["evidence.png"],"source_legacy_annotation":False,
        "objects":list(objects),"ignore_regions":list(regions),"annotation_sha256":"synthetic",
        "metrics":core.metrics(objects,regions,mask),
    }

def region(bbox=(15,15,25,25), *, possible=None, roi_relevance="possible_target_in_roi"):
    base = {"bbox_xyxy":list(bbox),"reason":"synthetic uncertainty","roi_relevance":roi_relevance}
    if possible is not None:
        base.update(possible_classes=possible,possible_classes_reason="Visually established two-wheel outline")
    return base

def group(row, split="test", encounters=None):
    return {
        "split":split, "sample_role":"queue_or_control","component_id":"component-"+row["image_id"],
        "event_ids":["event-"+row["image_id"]],
        "full_count_encounter_id":"scene-"+row["image_id"] if row["metrics"]["count_reference_available"] else None,
        "object_encounter_ids":encounters if encounters is not None else {
            oid:"encounter-"+oid for oid in row["metrics"]["accepted_object_ids"]
        },
    }

def pred(id, cls="mobil", bbox=(15,15,25,25), conf=0.9, membership="inside"):
    return {"id":id,"class_name":cls,"bbox_xyxy":list(bbox),"confidence":conf,"roi_membership":membership}

def ok(*predictions):
    return {"status":"ok","predictions":list(predictions)}

class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.ones((90,100),dtype=bool)

    def test_matching_precedes_ignore_and_duplicates_and_wrong_class_stay_fp(self):
        f = frame("A",self.mask,[obj("A",1)], [region()])
        got = evaluate_image(f,self.mask,ok(pred("bad","truk",conf=.99),
            pred("correct",conf=.9),pred("duplicate",conf=.8)))
        self.assertEqual([(x["outcome"],x.get("reason")) for x in got["detections"]],
                         [("fp","wrong_class"),("tp",None),("fp","duplicate")])
        self.assertEqual(got["false_negative_object_ids"],[])

    def test_class_matching_is_one_to_one_and_scores_before_geometry(self):
        f = frame("A",self.mask,[obj("A",1,bbox=(0,0,10,10)),obj("A",2,bbox=(4,0,14,10))])
        p1 = pred("a",bbox=(2,0,12,10),conf=.95)
        p2 = pred("b",bbox=(4,0,14,10),conf=.7)
        got = evaluate_image(f,self.mask,ok(p2,p1))
        self.assertEqual([x["matched_object_id"] for x in got["detections"]],["A-1","A-2"])

    def test_union_ignore_and_other_inside_false_positive(self):
        f = frame("A",self.mask,[obj("A",1,"other",(30,15,40,25))],
                  [region((15,15,18,25)),region((17,15,20,25)),region((30,15,40,25))])
        got = evaluate_image(f,self.mask,ok(pred("neutral"),pred("other",bbox=(30,15,40,25))))
        self.assertEqual([i["outcome"] for i in got["detections"]],["neutral","fp"])
        self.assertEqual(got["detections"][1]["reason"],"known_other")

    def test_svm_margin_scores_keep_order_without_calibration(self):
        f = frame("A",self.mask,[obj("A",1)])
        got = evaluate_image(f,self.mask,ok(pred("negative-margin",conf=-3.5),pred("high-margin",conf=5.8)))
        self.assertEqual([(p["id"],p["outcome"]) for p in got["detections"]],
                         [("high-margin","tp"),("negative-margin","fp")])

    def test_confirmed_outside_inside_prediction_stays_membership_fp_under_ignore(self):
        outside = obj("A",1,"bus",membership="outside")
        uncertain = obj("A",2,"bus",(40,15,50,25),membership="boundary_review")
        f = frame("A",self.mask,[outside,uncertain],[region()])
        got = evaluate_image(f,self.mask,ok(pred("outside","bus"),pred("ambiguous","bus",bbox=(40,15,50,25))))
        self.assertEqual([(p["outcome"],p.get("reason")) for p in got["detections"]],
                         [("neutral","ignore_overlap"),("fp","membership_error")])
        self.assertEqual(got["accepted_counts"]["bus"],2)

    def test_neutral_needs_half_prediction_area_without_double_counting(self):
        f = frame("A",self.mask,regions=[region((15,15,19,25)),region((16,15,19,25))])
        got = evaluate_image(f,self.mask,ok(pred("overlap")))
        self.assertEqual(got["detections"][0]["outcome"],"fp")

    def test_abstention_creates_fn_and_goes_in_reporting_denominator(self):
        f = frame("A",self.mask,[obj("A",1)])
        scores = evaluate_holdout([f],{"camera":self.mask},{"A":group(f)},{"A":ok(
            pred("a",membership="uncertain"),pred("b",membership="outside"))},subset="test")
        self.assertTrue(scores["evaluation_run_valid"])
        summary = scores["summary"]
        self.assertEqual(summary["detection_by_class"]["mobil"]["fn"],1)
        self.assertEqual(summary["membership_decision_coverage"],0.5)
        self.assertEqual(summary["mae_full_only"]["mobil"],1)

    def test_undefined_partial_total_and_union_uses_same_pred_ids(self):
        mask = self.mask.copy()
        mask[:,0:18] = False
        full = frame("A",mask)
        partial = frame("B",mask,regions=[region()])
        results = {"A":ok(pred("x"),pred("y",bbox=(20,15,30,25)),pred("outside",membership="outside")),
                   "B":ok(pred("p"),pred("u",membership="uncertain"))}
        got = evaluate_holdout([full,partial],{"camera":mask},{"A":group(full),"B":group(partial)},results,subset="test")
        a,b = got["per_image"]
        self.assertEqual(a["accepted_prediction_ids"],["x","y"])
        self.assertEqual(a["accepted_covered_area_px"],120)
        self.assertEqual(got["summary"]["mae_full_only"]["mobil"],2)
        self.assertIsNone(b["count_absolute_errors"])
        self.assertEqual(got["by_reference_scope"]["partial"]["partial_label_frames"],1)

    def test_missing_invalid_or_process_failure_never_scores_run(self):
        f1,f2 = frame("A",self.mask),frame("B",self.mask)
        group_by_image = {"A":group(f1),"B":group(f2)}
        failures = [None, {"status":"error","error":"decode"},ok(pred("bad",conf=float("nan"))),
                    ok(pred("same"),pred("same")),ok(pred("bad",bbox=(-1,1,2,3)))]
        for failure in failures:
            with self.subTest(failure=failure):
                got = evaluate_holdout([f1,f2],{"camera":self.mask},group_by_image,
                                       {"A":ok(),"B":failure},subset="test")
                self.assertFalse(got["evaluation_run_valid"])
                self.assertIsNone(got["summary"])

class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.ones((90,100),dtype=bool)
        self.mask[0,0] = False

    def test_partition_and_class_reference_by_physical_uncertainty(self):
        objects = [obj("A",1,"bus"),obj("A",2,"motor",(8,1,9,2),"boundary_review"),
                   obj("A",3,"other",(30,30,35,35))]
        f = frame("A",self.mask,objects,[region((1,1,5,5),possible=["motor"])])
        labels = adapt(f,self.mask)
        self.assertEqual([o["id"] for o in labels.positive],["A-1"])
        self.assertEqual([o["id"] for o in labels.known_other],["A-3"])
        self.assertEqual(labels.class_count_reference,{"mobil":0,"motor":None,"bus":1,"truk":0})
        self.assertFalse(labels.class_reference_mask["motor"])
        self.assertFalse(labels.full_count)
        self.assertIsNone(f["metrics"]["total_count_reference"])
        self.assertIsNone(f["metrics"]["coverage_reference_percent"])
        outside = frame("A",self.mask,objects[:1],[region(roi_relevance="outside_target_roi")])
        self.assertEqual(adapt(outside,self.mask).class_count_reference["bus"],1)

    def test_untyped_region_or_unknown_object_invalidates_every_class(self):
        for objects,regions in (
            ([obj("A",1,"unknown")],[]),
            ([],[region()]),
        ):
            f = frame("A",self.mask,objects,regions)
            self.assertEqual(adapt(f,self.mask).class_count_reference,{c:None for c in CLASSES})

    def test_typed_ignore_requires_reason(self):
        r = region(possible=["motor"])
        del r["possible_classes_reason"]
        with self.assertRaisesRegex(ValueError,"visual reason"):
            adapt(frame("A",self.mask,regions=[r]),self.mask)

    def test_negative_rect_full_native_roi_and_padding_for_outside_other_and_ignore(self):
        objects = [obj("A",1,"other",(15,15,25,25)), obj("A",2,"bus",(40,15,50,25),"outside")]
        f = frame("A",self.mask,objects,[region((65,15,75,25),roi_relevance="outside_target_roi")])
        boxes = [(0,0,2,2),(5,15,8,18),(32,15,33,18),(58,15,59,18),(45,42,50,45),(45,42,50,45)]
        output = negative_manifest(f,self.mask,group(f,"train"),boxes)
        self.assertEqual([r["bbox_xyxy"] for r in output],[[45,42,50,45]])
        row = output[0].copy()
        digest = row.pop("manifest_key_sha256")
        self.assertEqual(digest,canonical_hash(row))
        self.assertEqual(negative_manifest(f,self.mask,group(f,"test"),boxes),[])

    def test_train_positives_allow_partial_and_auxiliary_role_but_no_partial_negatives(self):
        f = frame("A",self.mask,[obj("A",1)], [region()])
        entry = group(f,"train")
        entry["sample_role"] = "auxiliary_moving"
        self.assertEqual([r["object_id"] for r in train_positives(f,self.mask,entry)],["A-1"])
        self.assertEqual(negative_manifest(f,self.mask,entry,[(40,40,55,55)]),[])
        self.assertEqual(train_positives(f,self.mask,group(f,"validation")),[])

    def test_auxiliary_moving_full_count_still_has_no_eligible_negatives(self):
        f = frame("A",self.mask,[obj("A",1)])
        entry = group(f,"train")
        entry["sample_role"] = "auxiliary_moving"
        self.assertEqual(negative_manifest(f,self.mask,entry,[(40,40,55,55)]),[])


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mask = np.ones((90,100),dtype=bool)
        Image.fromarray(self.mask.astype("uint8")*255).save(self.root/"evidence.png")
        self.paths = {k:k+".json" for k in ("manifest","annotations","rois","groups","protocol","test_receipt")}
        manifest, annotations, entries = [],[],{}
        for subset in ("train","validation","test"):
            for number in range(10):
                image_id = f"{subset}-{number}"
                objects = []
                if subset=="train" or number<8:
                    for c,cls in enumerate(CLASSES):
                        for offset in (0,1):
                            index = 2*c+offset
                            objects.append(obj(image_id,index,cls,(index*10,10,index*10+8,18)))
                f = frame(image_id,self.mask,objects)
                f["source_image"] = image_id+".png"
                rgb = Image.new("RGB",(100,90),(len(entries),1,5))
                rgb.save(self.root/f["source_image"])
                f["source_sha256"] = path_hash(self.root/f["source_image"])
                digest = canonical_hash(raw_record(f))
                f["annotation_sha256"] = digest
                f["cross_audit_status"] = "passed"
                f["cross_audit"] = {"image_id":image_id,"reviewer":"checker","annotation_sha256":digest,
                                    "source_reviewed":True,"overlay_reviewed":True,"status":"pass",
                                    "open_issues":[],"visual_evidence":["evidence.png"]}
                annotations.append(f)
                m = {k:f[k] for k in ("image_id","source_image","source_sha256","camera_id","width","height","roi_version")}
                m.update(native_checked=True,roi_checked=True,framing_evidence="evidence.png",
                         location_group="location", session_id=subset+"-acquisition")
                manifest.append(m)
                entries[image_id] = group(f,subset,{
                        o["id"]:f'{subset}-{o["class_name"]}-{number%5}' for o in f["objects"]
                })
        self.write("manifest",manifest)
        self.write("annotations",annotations)
        self.write("groups",{"images":entries,"must_link":[],
                             "boundary_evidence":[{"kind":"separate_acquisition_windows","evidence":"evidence.png"}]})
        self.write("rois",{"camera":{
            "mask_path":"evidence.png","mask_sha256":path_hash(self.root/"evidence.png"),
            "height":90,"width":100,"raster_area_px":9000,"version":"test-v1","geometry_valid":True,
            "status":"locked_for_annotation"}})
        self.write("protocol",{"gates":MINIMUM,"evaluation_claim":"fixed_camera_temporal_holdout"})
        self.write("test_receipt",{"suite":list(CORE_TESTS),"tests_run":25,"tests_failed":0,"code_hashes":code_hashes()})

    def write(self,key,data):
        (self.root/self.paths[key]).write_text(json.dumps(data))

    def read(self,key):
        return json.loads((self.root/self.paths[key]).read_text())

    def test_operational_gate_requires_support_and_hashes_for_lock(self):
        report = validate(self.root,self.paths)
        self.assertTrue(report["ready"],report["errors"])
        self.assertEqual(report["support"]["validation"]["full_empty_controls"],2)
        out = self.root/"test-only.lock"
        with self.assertRaisesRegex(ValueError,"input hashes"):
            write_split_lock(self.root,self.paths,out,expected_input_sha256={})
        self.assertFalse(out.exists())
        write_split_lock(self.root,self.paths,out,expected_input_sha256=report["input_sha256"])
        self.assertTrue(json.loads(out.read_text())["locked"])
        with self.assertRaises(FileExistsError):
            write_split_lock(self.root,self.paths,out,expected_input_sha256=report["input_sha256"])

    def test_duplicate_content_and_stale_review_fail_even_when_gates_otherwise_met(self):
        manifest = self.read("manifest")
        annotations = self.read("annotations")
        ref = manifest[0]
        dest = manifest[10]
        duplicate = self.root/dest["source_image"]
        # Same pixels, different PNG encoding. Encoded hash cannot detect this leakage.
        with Image.open(self.root/ref["source_image"]) as original:
            original.save(duplicate,compress_level=0)
        self.assertNotEqual(path_hash(duplicate),ref["source_sha256"])
        dest["source_sha256"] = annotations[10]["source_sha256"] = path_hash(duplicate)
        annotations[11]["review_notes"] = ["stale hash after change"]
        self.write("manifest",manifest)
        self.write("annotations",annotations)
        report = validate(self.root,self.paths)
        self.assertFalse(report["ready"])
        self.assertTrue(any("decoded_rgb_sha256 crosses splits" in s for s in report["errors"]),report["errors"])
        self.assertTrue(any("stale or non-independent" in s for s in report["errors"]),report["errors"])

    def test_support_counts_encounters_not_photos_and_refuses_lock(self):
        groups = self.read("groups")
        for image_id,row in groups["images"].items():
            if row["split"]=="test":
                for oid,new in row["object_encounter_ids"].items():
                    if "-truk-" in new:
                        row["object_encounter_ids"][oid] = "test-same-truck"
        self.write("groups",groups)
        report = validate(self.root,self.paths)
        self.assertFalse(report["ready"],report["errors"])
        self.assertEqual(report["support"]["test"]["by_class"]["truk"]["positive_encounters"],1)
        self.assertEqual(report["support"]["test"]["by_class"]["truk"]["positive_full_count_encounters"],1)
        out = self.root/"forbidden.lock"
        with self.assertRaisesRegex(ValueError,"readiness errors"):
            write_split_lock(self.root,self.paths,out,expected_input_sha256=report["input_sha256"])
        self.assertFalse(out.exists())

    def test_transitive_must_link_through_excluded_bridge_catches_leakage(self):
        groups = self.read("groups")
        groups["images"]["validation-0"].update(split="quarantine",exclusion_reason="bridge")
        groups["must_link"] = [
            {"image_ids":["train-0","validation-0"],"evidence":"evidence.png"},
            {"image_ids":["validation-0","test-0"],"evidence":"evidence.png"},
        ]
        self.write("groups",groups)
        report = validate(self.root,self.paths)
        self.assertTrue(any("Identity component crosses" in s for s in report["errors"]),report["errors"])

    def test_within_window_needs_reset_evidence_and_embargo_handles_boundary(self):
        groups = self.read("groups")
        active = {k:groups["images"][k] for k in ("train-0","validation-0")}
        groups["images"] = active
        manifest = {k:{"session_id":"same","location_group":"location"} for k in active}
        with self.assertRaisesRegex(ValueError,"Missing reset"):
            link_checks(groups,{},manifest,MINIMUM,self.root,{}, {})
        groups["boundary_evidence"] = [{
            "kind":"within_acquisition_reset","evidence":"evidence.png","session_id":"same",
            "location_group":"location","cut_utc":"2026-11-15T07:00:00+00:00","embargo_each_side_seconds":120,
            "subsets_before":["train"],"subsets_after":["validation","test"],
        }]
        from datetime import datetime
        times = {"train-0":datetime.fromisoformat("2026-11-15T06:58:00+00:00"),
                 "validation-0":datetime.fromisoformat("2026-11-15T07:03:00+00:00")}
        with self.assertRaisesRegex(ValueError,"inside active embargo"):
            link_checks(groups,times,manifest,MINIMUM,self.root,{}, {})
        times["train-0"] = datetime.fromisoformat("2026-11-15T06:57:59+00:00")
        link_checks(groups,times,manifest,MINIMUM,self.root,{}, {})

    def test_thresholds_cannot_be_loosened_and_external_symlink_needs_hash(self):
        protocol = self.read("protocol")
        protocol["gates"]["train_instances_per_class"] = 9
        self.write("protocol",protocol)
        report = validate(self.root,self.paths)
        self.assertTrue(any("weakens gate" in s for s in report["errors"]))
        with TemporaryDirectory(dir=HERE) as neighboring:
            evidence = Path(neighboring)/"ref"
            evidence.write_text("diagnostic only")
            (self.root/"linked.txt").symlink_to(evidence)
            with self.assertRaisesRegex(ValueError,"Undocumented external"):
                file_at(self.root,"linked.txt",{})
            self.assertEqual(file_at(self.root,"linked.txt",{"linked.txt":{
                "resolved_path":str(evidence),"sha256":path_hash(evidence)}}),evidence)

    def test_auxiliary_only_fills_classifier_train_support_and_never_benchmark(self):
        annotations = self.read("annotations")
        assignments = self.read("groups")
        # Remove every core train truck and reconstruct the synthetic audit binding.
        train_truck_images = [f for f in annotations if f["image_id"].startswith("train-")]
        for f in train_truck_images:
            removed = {o["id"] for o in f["objects"] if o["class_name"] == "truk"}
            f["objects"] = [o for o in f["objects"] if o["id"] not in removed]
            f["metrics"] = core.metrics(f["objects"], [], self.mask)
            f["annotation_sha256"] = canonical_hash(raw_record(f))
            f["cross_audit"]["annotation_sha256"] = f["annotation_sha256"]
            encounter_ids = assignments["images"][f["image_id"]]["object_encounter_ids"]
            for oid in removed:
                del encounter_ids[oid]
        self.write("annotations",annotations)
        report = validate(self.root,self.paths)
        self.assertFalse(report["ready"])
        self.assertIn("classifier train truk accepted_instances: 0 < 10",report["errors"])

        self.paths["auxiliary_manifest"] = "aux.json"
        aux = []
        for idx in range(5):
            image_id = f"aux-{idx}"
            source = image_id+".png"
            Image.new("RGB",(100,90),(255,idx,255)).save(self.root/source)
            record = {
                "image_id":image_id,"camera_id":"classification-camera","source_image":source,
                "source_sha256":path_hash(self.root/source),"width":100,"height":90,
                "source_program_date_time":None if idx==0 else "2026-11-14T03:30:00+00:00",
                "timestamp_camera_wib":"2026-11-14T10:30:00+07:00" if idx==0 else None,
                "timestamp_evidence":"native_camera_overlay" if idx==0 else None,
                "session_id":"aux-acquisition","location_group":"another-camera","sample_role":AUX_ROLE,
                "reviewer":"crop-annotator","visual_evidence":["evidence.png"],"objects":[
                    {"id":f"{image_id}-{n}","bbox_xyxy":[10*n,10,10*n+8,18],"class_name":"truk",
                     "confidence":"high","reason":"Visible truck cab and cargo box shape"} for n in (1,2)
                ],
                "cross_audit_status":"passed",
            }
            digest = canonical_hash({k:record[k] for k in AUX_RAW_KEYS})
            record["annotation_sha256"] = digest
            record["cross_audit"] = {
                "image_id":image_id,"reviewer":"crop-checker","annotation_sha256":digest,
                "source_reviewed":True,"overlay_reviewed":True,"status":"pass","open_issues":[],
                "visual_evidence":["evidence.png"],
            }
            aux.append(record)
            assignments["images"][image_id] = {
                "split":"train","sample_role":AUX_ROLE,"component_id":"comp-"+image_id,
                "event_ids":["event-"+image_id],
                "object_encounter_ids":{o["id"]:"encounter-"+image_id for o in record["objects"]},
            }

        self.write("groups",assignments)
        self.write("auxiliary_manifest",aux)
        ok_report = validate(self.root,self.paths)
        self.assertTrue(ok_report["ready"],ok_report["errors"])
        self.assertEqual(ok_report["classifier_train_support"]["by_class"]["truk"],
                         {"accepted_instances":10,"positive_encounters":5})
        self.assertEqual(ok_report["support"]["train"]["by_class"]["truk"]["accepted_instances"],0)
        self.assertEqual(ok_report["support"]["validation"]["by_class"]["truk"]["positive_encounters"],5)
        self.assertTrue(all(not r["detection_or_count_support"] for r in ok_report["verified_auxiliary_train_crops"]))

        assignments["must_link"] = [{"image_ids":["aux-0","test-0"],"evidence":"evidence.png"}]
        self.write("groups",assignments)
        report = validate(self.root,self.paths)
        self.assertFalse(report["ready"])
        self.assertTrue(any("Identity component crosses" in s for s in report["errors"]),report["errors"])

        assignments["must_link"] = []
        assignments["images"]["aux-0"]["split"] = "validation"
        self.write("groups",assignments)
        report = validate(self.root,self.paths)
        self.assertFalse(report["ready"])
        self.assertTrue(any("train-only" in s for s in report["errors"]),report["errors"])

    def test_full_count_scene_support_cannot_be_padded_by_distinct_objects_in_one_scene(self):
        groups = self.read("groups")
        for image_id,row in groups["images"].items():
            if row["split"]=="test":
                row["full_count_encounter_id"]="one-test-scene"
        self.write("groups",groups)
        report = validate(self.root,self.paths)
        self.assertFalse(report["ready"])
        self.assertEqual(report["support"]["test"]["by_class"]["truk"]["positive_full_count_encounters"], 5)
        self.assertEqual(report["support"]["test"]["by_class"]["truk"]["positive_full_count_scene_encounters"], 1)

    def test_auxiliary_native_timestamp_validation(self):
        good = {"source_program_date_time":None,"timestamp_camera_wib":"2026-11-14T10:30:00+07:00",
                "timestamp_evidence":"native_camera_overlay"}
        self.assertEqual(auxiliary_source_time(good).hour,10)
        for update in ({"timestamp_evidence":None},{"timestamp_camera_wib":None},
                       {"timestamp_camera_wib":"invalid"},{"timestamp_camera_wib":"2026-11-14T10:30:00"}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                auxiliary_source_time({**good,**update})
