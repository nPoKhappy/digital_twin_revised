from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.analysis.analyze_gain_pair_difficulty import configure_style, plot_teacher_landscape, plot_conditional_kci, finish_analysis, QUARTILES, SIGN_ORDER
if __name__=='__main__':
    import json
    p=Path(__file__).resolve().parent
    teacher=pd.read_csv(p/'teacher_gain_landscape.csv')
    summary=pd.read_csv(p/'teacher_gain_summary.csv')
    magnitude=pd.read_csv(p/'conditional_kci_by_gain_magnitude.csv')
    sign=pd.read_csv(p/'conditional_kci_by_teacher_sign.csv')
    configure_style()
    plot_teacher_landscape(teacher,summary,p)
    plot_conditional_kci(magnitude,'magnitude_bin',QUARTILES,'All18 KCI conditioned on teacher Gain magnitude','conditional_kci_by_gain_magnitude',p)
    plot_conditional_kci(sign,'teacher_sign_label',SIGN_ORDER,'All18 KCI conditioned on teacher Gain sign','conditional_kci_by_teacher_sign',p)
    finish_analysis(p,json.loads((p/'gain_pair_difficulty_protocol.json').read_text(encoding='utf-8')),summary,magnitude,sign)
