import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from contact_control import ContactControl

MAPPING = {"arm":(0,7), "gripper":(7,8), "body":(8,11), "base":(11,13)}
CFG = dict(conservative_scale=.7, smoothing_alpha=.5, force_soft_threshold=50., minimum_scale=.2, cumulative_limit=5000.)

class Controls(unittest.TestCase):
    def test_unmodified_baseline_and_input(self):
        a=np.linspace(-1,1,13,dtype=np.float32); original=a.copy()
        b,_=ContactControl("sac",MAPPING,CFG)(a,100,4000,20)
        expected=a.copy(); expected[8:10]=0
        np.testing.assert_array_equal(a,original); np.testing.assert_array_equal(b,expected)
    def test_gripper_changes_are_immediate(self):
        for method in ("conservative","force_rule"):
            ctrl=ContactControl(method,MAPPING,CFG)
            for i,g in enumerate([-1.,1.,-1.]):
                a=np.ones(13,dtype=np.float32); a[7]=g
                b,_=ctrl(a,500,4999,i+1)
                self.assertEqual(b[7],g); self.assertTrue(np.all(b[8:10]==0))
    def test_known_force_budget_case(self):
        a=np.ones(13,dtype=np.float32)
        b,info=ContactControl("force_rule",MAPPING,CFG)(a,100,0,1)
        self.assertEqual(info['scale'],.25)
        np.testing.assert_array_equal(b[[0,6,10,11,12]], np.full(5,.25))
    def test_smoothing_and_reset(self):
        ctrl=ContactControl("conservative",MAPPING,CFG)
        a=np.ones(13,dtype=np.float32)
        ctrl(a,0,0,1); b,_=ctrl(-a,0,0,2)
        np.testing.assert_array_equal(b[[0,6,10,11,12]],np.zeros(5))
        fresh,_=ContactControl("conservative",MAPPING,CFG)(-a,0,0,1)
        self.assertAlmostEqual(float(fresh[0]),-.7,places=6)
    def test_invalid_mapping_fails_closed(self):
        with self.assertRaises(AssertionError): ContactControl('sac',{},CFG)

if __name__=='__main__': unittest.main()
