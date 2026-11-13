import unittest

import pandas as pd

from scripts.plotting.plot_best_models_test_tables import delta_percent, signed, grouped_horizon_rows


class TestTableCalculations(unittest.TestCase):
    def test_delta_direction_and_no_unit_conversion_effect(self):
        self.assertAlmostEqual(delta_percent(.008,.01),-20)
        self.assertAlmostEqual(delta_percent(.012,.01),20)
        self.assertAlmostEqual(delta_percent(8,10),delta_percent(.008,.01))
        with self.assertRaises(ValueError):delta_percent(.01,0)

    def test_display_does_not_imply_a_sign_for_rounded_zero(self):
        self.assertEqual(signed(-.049),'0.0')
        self.assertEqual(signed(.049),'0.0')
        self.assertEqual(signed(-.051),'-0.1')
        self.assertEqual(signed(.051),'+0.1')

    def test_grouped_table_keeps_horizon_order_and_uses_raw_minimum(self):
        rows=[]
        for evaluation,baseline in [('TS',.03),('H18',.02),('H1',.01)]:
            for model,ratio in [('no_gain',1),('gain',1.1),('ss_only',.8),('gain_ss',.9)]:
                rows.append(dict(partition='internal',case='R5',target='B35_H2S',evaluation=evaluation,model=model,RMSE=baseline*ratio))
        table=grouped_horizon_rows(pd.DataFrame(rows),'internal','B35_H2S',['R5'])
        self.assertEqual(table.evaluation.tolist(),['H1','H18','TS'])
        self.assertEqual(table.NoGain_RMSE_x1e3.tolist(),[10,20,30])
        self.assertTrue(table.best_models.eq('ss_only').all())
        self.assertTrue((table.ss_only_delta_percent+20).abs().lt(1e-10).all())


if __name__=='__main__':unittest.main()
