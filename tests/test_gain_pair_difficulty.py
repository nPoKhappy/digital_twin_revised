import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from scripts.analysis.analyze_gain_pair_difficulty import load_probe_accuracy


class GainPairClusterTests(unittest.TestCase):
    def test_distributions_and_chunks_keep_all36_comparisons_together(self):
        edges=pd.DataFrame([dict(distribution=d,pair='air2 → H₂S',q25_abs_normalized_gain=.5,q50_abs_normalized_gain=1.,q75_abs_normalized_gain=2.) for d in ['in_range','soft_ood']])
        scales=pd.Series({'air2_SP':2.,'B35_H2S':4.})
        with TemporaryDirectory() as temporary:
            paths=[]
            for distribution in ['in_range','soft_ood']:
                path=Path(temporary)/f'{distribution}.csv'
                pd.DataFrame([dict(distribution=distribution,probe_index=0,target='B35_H2S',mv='air2_SP',teacher_gain=2.,teacher_sign=1,valid_teacher_gain=True,agreement='correct' if i<18 else 'wrong') for i in range(36)]).to_csv(path,index=False)
                paths.append(path)
            probe=load_probe_accuracy(paths,scales,edges,chunk_size=7)
        self.assertEqual(len(probe),2)
        self.assertTrue(probe.total.eq(36).all())
        self.assertTrue(probe.correct.eq(18).all())
        self.assertTrue(probe.probe_kci_percent.eq(50).all())
        self.assertTrue(probe.magnitude_bin.eq('Q3').all())


if __name__=='__main__':unittest.main()
