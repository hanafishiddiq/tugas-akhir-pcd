import unittest
import numpy as np
from core import metrics,union_area,bbox_valid,point_inside
def obj(id='x',cls='mobil',membership='inside',ignore=None,box=None):
 return dict(id=id,class_name=cls,roi_membership=membership,role='target_approach',confidence='high',ignore_reason=ignore,bbox_xyxy=box or [0,0,5,5])
class Contracts(unittest.TestCase):
 def setUp(self):self.mask=np.zeros((8,8),bool);self.mask[2:6,2:6]=True
 def test_clipped_overlap_not_double_counted(self):self.assertEqual(union_area(self.mask,[[0,0,5,5],[3,3,8,8]]),14)
 def test_unknown_is_not_zero_or_background(self):
  m=metrics([obj(cls='unknown',ignore='ambiguous')],[],self.mask);self.assertIsNone(m['counts_reference']);self.assertEqual(m['total_count_upper_bound'],1)
 def test_outside_unknown_does_not_invent_inside_count(self):
  m=metrics([obj(cls='unknown',membership='outside',ignore='class unresolved outside')],[],self.mask);self.assertEqual(m['total_count_reference'],0)
 def test_straddling_stays_partial(self):self.assertIsNone(metrics([obj(membership='boundary_review',ignore='wheel straddles')],[],self.mask)['total_count_reference'])
 def test_ambiguous_region_has_no_fabricated_count_upper(self):
  m=metrics([obj()], [{'bbox_xyxy':[3,3,7,7],'reason':'crowd'}],self.mask);self.assertIsNone(m['total_count_upper_bound']);self.assertIsNone(m['counts_reference'])
 def test_count_and_area_use_same_accepted_ids(self):
  m=metrics([obj('a'),obj('b',membership='outside',box=[3,3,8,8])],[],self.mask);self.assertEqual(m['accepted_object_ids'],['a']);self.assertEqual(m['covered_area_lower_px'],9)
 def test_other_not_forced_into_truck(self):self.assertEqual(metrics([obj(cls='other')],[],self.mask)['total_count_reference'],0)
 def test_bounds_halfopen(self):self.assertTrue(bbox_valid([0,0,8,8],8,8));self.assertFalse(bbox_valid([0,0,9,8],8,8))
 def test_projected_roof_does_not_define_membership(self):
  p=[[0,10],[20,10],[20,30],[0,30]];self.assertFalse(point_inside([10,5],p));self.assertTrue(point_inside([10,20],p))
 def test_helmet_region_outside_projection_still_blocks_full_count(self):
  r={'bbox_xyxy':[0,0,2,2],'roi_relevance':'possible_target_in_roi','reason':'helmet visible; hidden vehicle base might be inside'}
  m=metrics([], [r],self.mask);self.assertIsNone(m['counts_reference']);self.assertIsNone(m['total_count_upper_bound']);self.assertIsNone(m['covered_area_upper_px']);self.assertIsNone(m['coverage_upper_percent']);self.assertEqual(m['observed_envelope_area_px'],0)
 def test_explicit_outside_region_does_not_block_full_count(self):
  r={'bbox_xyxy':[0,0,5,5],'roi_relevance':'outside_target_roi','reason':'opposing lane body projects into ROI'}
  m=metrics([], [r],self.mask);self.assertEqual(m['total_count_reference'],0)
if __name__=='__main__':unittest.main()
