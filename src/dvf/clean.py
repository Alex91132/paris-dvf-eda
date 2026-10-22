"""Из сырого DVF (строка = мутация × участок × помещение × культура) — одна строка на продажу жилья.

DVF повторяет полную цену сделки (valeur_fonciere) в каждой строке мутации: квартира, её парковка
и погреб — три строки с одной и той же ценой. Цена за м² корректна только там, где в сделке ровно
одно жилое помещение и нет коммерческих: иначе цена относится к нескольким объектам сразу.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "processed" / "sales.parquet"

NOTRE_DAME = (48.85296, 2.34990)          # «нулевой километр» дорог Франции
DEP_RU = {"75": "Париж", "92": "О-де-Сен", "93": "Сен-Сен-Дени", "94": "Валь-де-Марн"}
NATURES = {"Vente": "вторичка", "Vente en l'état futur d'achèvement": "новостройка"}
RESIDENTIAL = ("Appartement", "Maison")
COMMERCIAL = "Local industriel. commercial ou assimilé"

COLS = ["id_mutation", "date_mutation", "numero_disposition", "nature_mutation", "valeur_fonciere",
        "code_commune", "nom_commune", "code_departement", "id_parcelle", "type_local",
        "surface_reelle_bati", "nombre_pieces_principales", "lot1_surface_carrez", "surface_terrain",
        "longitude", "latitude"]

# Границы правдоподобия: всё за ними — ошибки ввода, доли, сделки с обременением и т. п.
SURF = (9.0, 400.0)          # м²
PPM2 = (1_500.0, 40_000.0)   # €/м²
MAD_Z = 4.0                  # робастный z внутри коммуны и типа жилья
MAX_KM = 25.0                # дальше от центра — ошибка геокодирования (вся зона в пределах ~20 км)


def read_raw(raw: Path = RAW) -> pd.DataFrame:
    files = sorted(raw.glob("dvf_*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"нет данных в {raw} — запустите scripts/download.py")
    dtype = {"code_commune": str, "code_departement": str, "id_parcelle": str}
    return pd.concat([pd.read_csv(f, usecols=COLS, dtype=dtype, low_memory=False) for f in files],
                     ignore_index=True)


def haversine_km(lat, lon, lat0=NOTRE_DAME[0], lon0=NOTRE_DAME[1]):
    p1, p2 = np.radians(lat), np.radians(lat0)
    dphi, dl = p2 - p1, np.radians(lon0) - np.radians(lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Продажи квартир и домов: (таблица, воронка очистки [(шаг, число сделок)])."""
    funnel = [("строк в сыром DVF", len(raw)), ("сделок (мутаций)", raw["id_mutation"].nunique())]
    df = raw[raw["nature_mutation"].isin(NATURES)]
    funnel.append(("продажи (вторичка и новостройки)", df["id_mutation"].nunique()))

    g = df.groupby("id_mutation")
    one_price = g["valeur_fonciere"].nunique() == 1
    one_disp = g["numero_disposition"].nunique() == 1
    no_comm = ~df["type_local"].eq(COMMERCIAL).groupby(df["id_mutation"]).any()
    # одно и то же помещение повторяется по культурам участка (сад, земля) — схлопываем
    res = df[df["type_local"].isin(RESIDENTIAL)].drop_duplicates(
        ["id_mutation", "id_parcelle", "type_local", "surface_reelle_bati", "nombre_pieces_principales"])
    n_res = res.groupby("id_mutation").size()
    ok = (one_price & one_disp & no_comm).reindex(n_res.index) & n_res.eq(1)
    res = res[res["id_mutation"].isin(ok[ok].index)]
    funnel.append(("ровно одно жильё, без коммерции, одна цена", len(res)))

    land = df.groupby("id_mutation")["surface_terrain"].sum(min_count=1)
    s = res.assign(surface_terrain=res["id_mutation"].map(land)).copy()
    s = s[s["valeur_fonciere"].notna() & s["surface_reelle_bati"].between(*SURF) & s["longitude"].notna()]
    s = s[haversine_km(s["latitude"].to_numpy(), s["longitude"].to_numpy()) <= MAX_KM]
    funnel.append((f"площадь {SURF[0]:.0f}–{SURF[1]:.0f} м², есть цена и координаты", len(s)))

    s["ppm2"] = s["valeur_fonciere"] / s["surface_reelle_bati"]
    s = s[s["ppm2"].between(*PPM2)]
    funnel.append((f"цена {PPM2[0]:,.0f}–{PPM2[1]:,.0f} €/м²".replace(",", " "), len(s)))

    lp = np.log(s["ppm2"])
    grp = [s["code_commune"], s["type_local"]]
    med = lp.groupby(grp).transform("median")
    mad = (lp - med).abs().groupby(grp).transform("median") * 1.4826
    s = s[((lp - med).abs() <= MAD_Z * mad.clip(lower=0.05)) | mad.isna()]
    funnel.append((f"без выбросов внутри коммуны (|z| ≤ {MAD_Z:g} по MAD)", len(s)))

    s = s.rename(columns={"valeur_fonciere": "price", "surface_reelle_bati": "surface",
                          "nombre_pieces_principales": "rooms", "longitude": "lon", "latitude": "lat"})
    s["date"] = pd.to_datetime(s["date_mutation"])
    s["year"] = s["date"].dt.year
    s["quarter"] = s["date"].dt.to_period("Q").astype(str)
    s["month"] = s["date"].dt.month
    s["dep"] = s["code_departement"]
    s["paris"] = s["dep"].eq("75")
    s["new_build"] = s["nature_mutation"].ne("Vente")
    s["house"] = s["type_local"].eq("Maison")
    s["arr"] = np.where(s["paris"], pd.to_numeric(s["code_commune"]) - 75100, 0)   # 75101 → 1-й округ
    s["dist_km"] = haversine_km(s["lat"].to_numpy(), s["lon"].to_numpy())
    s["log_ppm2"] = np.log(s["ppm2"])
    keep = ["id_mutation", "date", "year", "quarter", "month", "dep", "paris", "code_commune", "nom_commune",
            "arr", "type_local", "house", "new_build", "price", "surface", "rooms", "surface_terrain",
            "ppm2", "log_ppm2", "lat", "lon", "dist_km"]
    return s[keep].sort_values("date").reset_index(drop=True), funnel


def load(rebuild: bool = False) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Очищенные продажи с кэшем в data/processed."""
    fun = CACHE.with_suffix(".funnel.csv")
    if CACHE.exists() and fun.exists() and not rebuild:
        f = pd.read_csv(fun)
        return pd.read_parquet(CACHE), list(zip(f["step"], f["n"]))
    s, funnel = clean(read_raw())
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    s.to_parquet(CACHE, index=False)
    pd.DataFrame(funnel, columns=["step", "n"]).to_csv(fun, index=False)
    return s, funnel
