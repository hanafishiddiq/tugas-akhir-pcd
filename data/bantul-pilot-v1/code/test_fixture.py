"""Read-only fixture of five audited active frames. Test labels, never score a model."""
import json
from pathlib import Path
import unittest

from dataset import ROOT, adapt, path_hash
from readiness import load_mask, valid_audit

class FrozenPackageTests(unittest.TestCase):
    def test_real_222_acceptance_and_full_reference_binding(self):
        fixture = json.loads(Path(__file__).with_name("FIXTURE_222.json").read_text())
        package = ROOT/"Data/Anotasi/Bantul_222_v1"
        annotation_file = package/"ANNOTATIONS.json"
        roi_file = package/"ROIS.json"
        self.assertEqual(path_hash(annotation_file),fixture["annotations_sha256"])
        self.assertEqual(path_hash(roi_file),fixture["rois_sha256"])
        records = {r["image_id"]:r for r in json.loads(annotation_file.read_text())}
        rois = json.loads(roi_file.read_text())
        masks = {camera:load_mask(ROOT,camera,roi,{})[0] for camera,roi in rois.items()}
        for image_id,expected in fixture["images"].items():
            with self.subTest(image_id=image_id):
                record = records[image_id]
                self.assertEqual(path_hash(ROOT/record["source_image"]),expected["source_sha256"])
                valid_audit(record,image_id,ROOT,{})
                self.assertEqual(record["annotation_sha256"],expected["annotation_sha256"])
                labels = adapt(record,masks[record["camera_id"]])
                self.assertEqual([o["id"] for o in labels.positive],expected["positive_ids"])
                self.assertEqual(labels.full_count,expected["full_count"])
                self.assertEqual(record["metrics"]["counts_reference"],expected["counts_reference"])
                self.assertEqual(record["metrics"]["coverage_reference_percent"],expected["coverage_reference_percent"])
        self.assertIn("W072-01",[o["id"] for o in adapt(records["W072"],masks["02_paseban_barat"]).known_other])
        self.assertTrue(all(x is None for x in adapt(records["S034"],masks["01_paseban_selatan"]).class_count_reference.values()))
