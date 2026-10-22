"""Статичные картинки для README → docs/img/*.png (карта — скриншот отчёта, kaleido не грузит тайлы).

    python scripts/readme_images.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import plotly
import plotly.io as pio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", category=FutureWarning)

from dvf import analysis as A  # noqa: E402
from dvf import figures as F   # noqa: E402
from dvf.clean import load     # noqa: E402

_js = os.path.join(tempfile.gettempdir(), f"plotly-{plotly.__version__}.min.js")
shutil.copy(os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js"), _js)
pio.kaleido.scope.plotlyjs = _js


def main() -> None:
    R = A.run_all(load()[0])
    out = ROOT / "docs" / "img"
    out.mkdir(parents=True, exist_ok=True)
    figs = {"map": F.price_grid_static(R["map"]["cells"]), "index": F.hedonic_index(R["index"]["table"]), "size": F.size_premium(R["size"]),
            "new_build": F.new_build(R["new"]), "olympics": F.olympics_placebo(R["oly"]["placebo"], R["oly"]["effect"],
                                                                                R["oly"]["p"]),
            "round": F.threshold_ratio(R["round"]["r100"], R["round"]["r10"]),
            "arr": F.arr_level_change(R["arr"], R["arr_rho"][0])}
    for k, f in figs.items():
        f.write_image(out / f"{k}.png", width=1000, scale=1.5)
        print(k)


if __name__ == "__main__":
    main()
