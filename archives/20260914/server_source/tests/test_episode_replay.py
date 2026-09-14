import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from episode_replay import EpisodeReplay

class ReplayBoundaries(unittest.TestCase):
    def test_windows_keep_transition_alignment_and_episode_boundaries(self):
        r=EpisodeReplay(128,3,device='cpu')
        for e in range(3):
            obs=(1000*e+np.arange(201)).reshape(-1,1)
            actions=(1000*e+np.arange(200)).reshape(-1,1)
            rewards=1000*e+np.arange(1,201)
            r.add(obs,actions,rewards)
        o,a,reward,task=r.sample()
        np.testing.assert_array_equal(o[:-1].numpy(),a.numpy())
        np.testing.assert_array_equal(o[1:].numpy(),reward.numpy())
        self.assertTrue(((o[1:]-o[:-1])==1).all())
        self.assertIsNone(task)
    def test_incomplete_transition_rejected(self):
        r=EpisodeReplay(2,3,'cpu')
        with self.assertRaises(AssertionError):r.add(np.zeros((3,1)),np.zeros((3,1)),np.zeros(3))

if __name__=='__main__':unittest.main()
