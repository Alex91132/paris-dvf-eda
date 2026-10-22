"""Выводы отчёта на реальных данных (пропускаются, если data/ не скачана)."""
import pytest

from dvf import analysis as A
from dvf.clean import CACHE, RAW

pytestmark = pytest.mark.skipif(not CACHE.exists() and not any(RAW.glob("dvf_*.csv.gz")),
                                reason="нет данных: python scripts/download.py")


@pytest.fixture(scope="module")
def s():
    from dvf.clean import load
    return A.prepare(load()[0])


def test_prices_fell_from_peak(s):
    summ = A.hedonic_index(s)["summary"]
    for z in ("Париж", "пригород"):
        assert summ[z]["from_peak"] < -0.05 and summ[z]["last_hi"] < 1.0


def test_new_build_premium_hidden_by_location(s):
    nb = A.new_build_premium(s)
    assert nb["within"][1] > 0.15 and nb["raw"] < nb["within"][1]


def test_simpson_large_flats_in_suburbs(s):
    t = A.size_premium(s).set_index(["zone", "bin"])
    assert t.loc[("пригород", "120+"), "raw"] > 0 > t.loc[("пригород", "120+"), "hi"]


def test_round_threshold_bunching(s):
    r = A.round_prices(s)
    assert r["excess100"] > r["excess10"] and r["p"] < 0.05
