"""Redraw the validation journal figures with room above the first row."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from matplotlib.figure import Figure
from scripts.analysis import plot_jtice_validation as figures


def main():
    original=Figure.add_gridspec
    def spaced(self, nrows=1, ncols=1, **kwargs):
        if nrows in (2,4) and ncols==3 and kwargs.get('top')==.94:
            kwargs['top']=.87 if nrows==2 else .90
        return original(self,nrows,ncols,**kwargs)
    Figure.add_gridspec=spaced
    try:
        figures.main()
    finally:
        Figure.add_gridspec=original
    path=figures.OUT/'figure_manifest.json'
    manifest=json.loads(path.read_text(encoding='utf8'))
    manifest['layout_override_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['layout_override']='Raised space above subplot titles to separate the top legend.'
    path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
