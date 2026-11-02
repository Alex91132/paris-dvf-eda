"""Скачать DVF géolocalisé (Etalab) по Парижу и ближнему пригороду в data/raw/.

    python scripts/download.py

Источник: https://files.data.gouv.fr/geo-dvf/ — «Demandes de valeurs foncières géolocalisées»,
Licence Ouverte / Open Licence 2.0. Около 23 МБ.
"""
from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YEARS = (2021, 2022, 2023, 2024)
DEPS = ("75", "92", "93", "94")
URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/{y}/departements/{d}.csv.gz"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # иначе «→» роняет скрипт в консоли Windows
    out = ROOT / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for y in YEARS:
        for d in DEPS:
            f = out / f"dvf_{y}_{d}.csv.gz"
            if f.exists():
                continue
            url = URL.format(y=y, d=d)
            print("→", f.name)
            try:
                urllib.request.urlretrieve(url, f)
            except urllib.error.URLError as e:
                # на части сборок Python нет CA-сертификатов: SSL: CERTIFICATE_VERIFY_FAILED
                raise SystemExit(f"не удалось скачать {f.name}: {e}\n"
                                 f"если это ошибка SSL — качайте curl: curl -L -o {f} {url}")


if __name__ == "__main__":
    main()
