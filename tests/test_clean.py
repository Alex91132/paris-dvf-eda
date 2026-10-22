"""Очистка DVF: одна строка на продажу жилья, цена не делится между помещениями."""
import numpy as np
import pandas as pd
import pytest

from dvf.clean import COMMERCIAL, clean, haversine_km

BASE = dict(date_mutation="2023-05-10", numero_disposition=1, nature_mutation="Vente", code_commune="75111",
            nom_commune="Paris 11e Arrondissement", code_departement="75", id_parcelle="75111000AB0001",
            nombre_pieces_principales=2.0, lot1_surface_carrez=np.nan, surface_terrain=np.nan,
            longitude=2.38, latitude=48.86)


def rows(*specs) -> pd.DataFrame:
    return pd.DataFrame([{**BASE, **s} for s in specs])


def flat(mid, price, surf, **kw):
    return dict(id_mutation=mid, valeur_fonciere=price, type_local="Appartement", surface_reelle_bati=surf, **kw)


def dep(mid, price):
    return dict(id_mutation=mid, valeur_fonciere=price, type_local="Dépendance", surface_reelle_bati=np.nan,
                nombre_pieces_principales=0.0)


def _many_ok(n=30, start=100):
    """Фон: обычные квартиры ~10 000 €/м², чтобы робастный фильтр по коммуне имел с чем сравнивать."""
    rng = np.random.default_rng(0)
    return [flat(f"ok{i}", 500_000 * float(np.exp(rng.normal(0, 0.1))), 50.0) for i in range(start, start + n)]


def test_flat_with_parking_counts_once_with_full_price():
    s, _ = clean(rows(flat("A", 600_000, 50.0), dep("A", 600_000), dep("A", 600_000), *_many_ok()))
    a = s[s["id_mutation"] == "A"]
    assert len(a) == 1 and a["price"].iloc[0] == 600_000 and a["ppm2"].iloc[0] == pytest.approx(12_000)


def test_two_flats_or_commercial_in_one_sale_are_dropped():
    s, _ = clean(rows(flat("B", 900_000, 40.0), flat("B", 900_000, 45.0, id_parcelle="X"),
                      flat("C", 700_000, 50.0), dict(id_mutation="C", valeur_fonciere=700_000, type_local=COMMERCIAL,
                                                     surface_reelle_bati=30.0), *_many_ok()))
    assert not s["id_mutation"].isin(["B", "C"]).any()


def test_house_repeated_by_land_culture_is_one_sale():
    h = dict(id_mutation="H", valeur_fonciere=800_000, type_local="Maison", surface_reelle_bati=100.0,
             code_commune="92050", code_departement="92", longitude=2.2, latitude=48.89)
    s, _ = clean(rows({**h, "surface_terrain": 300.0}, {**h, "surface_terrain": 200.0}, *_many_ok()))
    hh = s[s["id_mutation"] == "H"]
    assert len(hh) == 1 and hh["surface_terrain"].iloc[0] == 500.0 and hh["house"].iloc[0]


def test_bounds_and_robust_outlier_filter():
    bad = [flat("tiny", 100_000, 5.0), flat("cheap", 20_000, 50.0), flat("far", 500_000, 50.0, latitude=83.5),
           flat("odd", 1_900_000, 50.0)]           # 38 000 €/м² — в границах, но выброс внутри коммуны
    s, funnel = clean(rows(*bad, *_many_ok()))
    assert not s["id_mutation"].isin(["tiny", "cheap", "far", "odd"]).any()
    assert len(s) == 30 and funnel[-1][1] == 30
    assert [n for _, n in funnel[2:]] == sorted([n for _, n in funnel[2:]], reverse=True)


def test_vefa_is_new_build_and_exchange_is_dropped():
    s, _ = clean(rows(flat("V", 600_000, 50.0, nature_mutation="Vente en l'état futur d'achèvement"),
                      flat("E", 600_000, 50.0, nature_mutation="Echange"), *_many_ok()))
    assert s.set_index("id_mutation").loc["V", "new_build"] and "E" not in set(s["id_mutation"])


def test_arrondissement_and_distance():
    s, _ = clean(rows(*_many_ok()))
    assert (s["arr"] == 11).all()
    assert haversine_km(48.85296, 2.34990) == pytest.approx(0, abs=1e-9)
    assert haversine_km(48.85296 + 1 / 111.2, 2.34990) == pytest.approx(1.0, rel=2e-3)
