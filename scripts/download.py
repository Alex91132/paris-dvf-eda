"""Скачать DVF géolocalisé (Etalab) по Парижу и ближнему пригороду в data/raw/.

    python scripts/download.py

Источник: https://files.data.gouv.fr/geo-dvf/ — «Demandes de valeurs foncières géolocalisées»,
Licence Ouverte / Open Licence 2.0. Около 23 МБ.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YEARS = (2021, 2022, 2023, 2024)
DEPS = ("75", "92", "93", "94")
URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{y}/departements/{d}.csv.gz"


def main() -> None:
    out = ROOT / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for y in YEARS:
        for d in DEPS:
            f = out / f"dvf_{y}_{d}.csv.gz"
            if f.exists():
                continue
            print("→", f.name)
            urllib.request.urlretrieve(URL.format(y=y, d=d), f)


if __name__ == "__main__":
    main()
