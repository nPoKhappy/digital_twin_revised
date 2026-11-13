"""Rebuild the Gain+SS report from saved validation CSVs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.prediction.validation_comparison_report import build_report
if __name__=='__main__':
    build_report(Path(__file__).resolve().parent)
