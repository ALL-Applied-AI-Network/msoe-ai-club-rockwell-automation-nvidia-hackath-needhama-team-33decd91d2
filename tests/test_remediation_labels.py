import unittest
import numpy as np
from src.remediation.labels import visible_labels


class LabelTests(unittest.TestCase):
    def test_visible_extent_and_occlusion_rejection(self):
        ids=np.zeros((100,200),dtype=np.uint32); ids[20:30,40:60]=7
        report={'objects':[{'path':'/cap','class_name':'crooked_cap'},{'path':'/hidden','class_name':'good_cap'}],
                'inventory':{'slots':{}}}
        labels,rejected,text=visible_labels(report,{'data':ids,'info':{'idToLabels':{'7':'/cap/Flange'}}},None)
        self.assertEqual(labels[0]['bbox_xyxy'],[40,20,60,30])
        self.assertEqual(labels[0]['visible_pixels'],200)
        self.assertEqual(rejected[0]['path'],'/hidden')
        self.assertEqual(text.strip(),'1 0.25000000 0.25000000 0.10000000 0.10000000')


if __name__=='__main__': unittest.main()
