from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_detector_supervision import invalid_masks


class SupervisionTests(unittest.TestCase):
    def test_geometric_triangle_can_rasterize_to_empty(self):
        row=dict(width=640,height=640,annotations=[dict(id=343,
            segmentation=[[287.,293.,286.,293.,286.,294.]])])
        self.assertEqual(invalid_masks(row)[0]['annotation_id'],343)

    def test_valid_polygon_and_explicit_empty_target(self):
        self.assertEqual(invalid_masks(dict(width=10,height=10,annotations=[])),[])
        row=dict(width=10,height=10,annotations=[dict(id=1,segmentation=[[2,2,8,2,8,8,2,8]])])
        self.assertEqual(invalid_masks(row),[])


if __name__=='__main__':
    unittest.main()
