"""Rebuild all five test comparison tables from CSVs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.plotting.plot_best_models_test_tables import build_report
if __name__=='__main__':build_report(Path(__file__).resolve().parent)
