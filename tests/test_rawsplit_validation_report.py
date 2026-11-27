"""Exercise report assembly and the common seed-42 baseline on known inputs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.analysis import build_rawsplit_validation_report as report


class RawSplitReportTests(unittest.TestCase):
    def test_assembly_uses_seed42_baseline_and_actual_parameter_labels(self):
        old=report.OLD
        jobs=json.loads((report.ROOT/'configs/pgin/retrain_lhs10_23_full_80_10_10/config_manifest.json').read_text(encoding='utf8'))
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            (out/'evaluation_validation.json').write_text(json.dumps({'passed':True,'model_count':63,'dynamic_metrics':2268,'kci_components':504}),encoding='utf8')
            (out/'independent_npz_audit.json').write_text(json.dumps({'passed':True,'model_count':63,'rmse_checks':2268,'kci_pair_checks':504,'teacher_arrays_consistent':True}),encoding='utf8')
            (out/'run_protocol.json').write_text(json.dumps({'jobs':jobs,'sources_sha256':{}}),encoding='utf8')
            dependency='scripts/training/train_pgin_stepwise_gain.py'
            (out/'dependency_snapshot.json').write_text(json.dumps({'path':dependency,'sha256':report.sha(report.ROOT/dependency)}),encoding='utf8')
            old_metrics=pd.read_csv(old/'relative_rmse_components.csv')
            old_metrics[['seed','model','partition','case','target','evaluation','n','RMSE']].to_csv(out/'validation_metrics.csv',index=False)
            pd.read_csv(old/'kci_components.csv').to_csv(out/'kci_components.csv',index=False)
            def draw(_data,_meta,families,*_args):
                return [None]*(len(families)*36)
            with patch.object(report,'OUT',out), patch.object(report.journal,'draw',side_effect=draw):
                report.main()
            coords=pd.read_csv(out/'regional_coordinates_63_models.csv')
            baseline=coords[(coords.model=='no_gain')&(coords.seed==42)]
            self.assertEqual(len(coords),126)
            self.assertTrue((baseline.AvgRelRMSE-1).abs().lt(1e-12).all())
            metadata=pd.read_csv(out/'model_parameter_metadata.csv').set_index('model')
            self.assertIn('0.05',metadata.loc['gain005','full'])
            self.assertIn('0.05',metadata.loc['ss005_only','full'])
            self.assertTrue(json.loads((out/'analysis_validation.json').read_text())['passed'])


if __name__=='__main__': unittest.main()
