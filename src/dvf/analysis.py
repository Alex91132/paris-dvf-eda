"""Гипотезы EDA. Каждая функция возвращает словарь: таблицы для графиков и числа для выводов."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .stats import boot_median, boot_spearman_ci, cell_center, design, fe_ols, grid_cell, permutation_did, side_ratio

SIZE_BINS = [9, 20, 30, 40, 60, 80, 120, 400]
SIZE_LABELS = ["9–20", "20–30", "30–40", "40–60", "60–80", "80–120", "120+"]
OLYMPIC = {"93066": "Сен-Дени", "93070": "Сент-Уан", "93039": "Иль-Сен-Дени"}   # Олимпийская деревня
HEDONIC = ["ls", "ls2"]


def prepare(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["cell"] = grid_cell(s["lat"], s["lon"])
    s["ls"] = np.log(s["surface"])
    s["ls2"] = s["ls"] ** 2
    s["rooms_c"] = s["rooms"].clip(1, 5).astype(int)
    s["size_bin"] = pd.cut(s["surface"], SIZE_BINS, labels=SIZE_LABELS)
    s["zone"] = np.where(s["paris"], "Париж", "пригород")
    s["period"] = np.where(s["year"] <= 2022, "2021–22", "2023–24")
    return s


def flats(s: pd.DataFrame) -> pd.DataFrame:
    return s[~s["house"]]


def _attrs(d: pd.DataFrame, extra: dict | None = None, numeric=HEDONIC, new_build=True) -> pd.DataFrame:
    X = design(d, list(numeric) + (["new_build"] if new_build else []), {"rooms_c": 2, **(extra or {})})
    return X


# ----------------------------------------------------------------- обзор и карта
def overview(s: pd.DataFrame) -> dict:
    by = s.groupby(["year", "zone"]).agg(n=("ppm2", "size"), med=("ppm2", "median")).reset_index()
    return {"by_year": by, "n": len(s), "n_flats": int((~s["house"]).sum()), "n_communes": s["code_commune"].nunique(),
            "med_paris": float(s.loc[s["paris"] & ~s["house"], "ppm2"].median()),
            "med_sub": float(s.loc[~s["paris"] & ~s["house"], "ppm2"].median())}


def price_map(s: pd.DataFrame, min_n: int = 10) -> dict:
    f = flats(s)
    g = f.groupby("cell").agg(med=("ppm2", "median"), n=("ppm2", "size"), commune=("nom_commune", "first"))
    g = g[g["n"] >= min_n].reset_index()
    g["lat"], g["lon"] = cell_center(g["cell"].to_numpy())
    p = f.groupby(["cell", "period"])["ppm2"].agg(["median", "size"]).unstack()
    ok = (p[("size", "2021–22")] >= min_n) & (p[("size", "2023–24")] >= min_n)
    ch = (p.loc[ok, ("median", "2023–24")] / p.loc[ok, ("median", "2021–22")] - 1).rename("change").reset_index()
    ch["lat"], ch["lon"] = cell_center(ch["cell"].to_numpy())
    ch = ch.merge(g[["cell", "commune"]], on="cell", how="left")
    return {"cells": g, "change": ch, "share_down": float((ch["change"] < 0).mean())}


def arrondissements(s: pd.DataFrame) -> pd.DataFrame:
    """Медиана €/м² по округам с бутстреп-ДИ и изменение 2021 → 2024 (медиан)."""
    f = flats(s)
    f = f[f["paris"]]
    rows = []
    for a, d in f.groupby("arr"):
        m, lo, hi = boot_median(d["ppm2"], seed=a)
        m21, m24 = d.loc[d["year"] == 2021, "ppm2"].median(), d.loc[d["year"] == 2024, "ppm2"].median()
        rows.append({"arr": a, "med": m, "lo": lo, "hi": hi, "n": len(d), "change": m24 / m21 - 1})
    return pd.DataFrame(rows).sort_values("med")


def location_share(s: pd.DataFrame) -> dict:
    """Какая доля разброса цены за м² — локация: коммуна, ячейка 300 м, ячейка + характеристики."""
    f = flats(s)
    y = f["log_ppm2"]
    r_comm = fe_ols(y, design(f, ["ls"]), f["code_commune"]).r2_fe_only
    full = fe_ols(y, _attrs(f, {"quarter": "2021Q1"}), f["cell"])
    return {"commune": r_comm, "cell": full.r2_fe_only, "cell_attrs": full.r2,
            "n_cells": full.n_groups}


# ----------------------------------------------------------------- H1: цены после ставок
def hedonic_index(s: pd.DataFrame) -> dict:
    """Квартальный гедонический индекс (ячейка 300 м + площадь + комнаты + новостройка) против наивной медианы."""
    out, rows = {}, []
    for zone, d in flats(s).groupby("zone"):
        m = fe_ols(d["log_ppm2"], _attrs(d, {"quarter": "2021Q1"}), d["cell"])
        med = d.groupby("quarter")["ppm2"].median()
        for q in med.index:
            b, lo, hi = (0.0, 0.0, 0.0) if q == "2021Q1" else m.ci(f"quarter={q}")
            rows.append({"zone": zone, "quarter": q, "idx": np.exp(b), "lo": np.exp(lo), "hi": np.exp(hi),
                         "naive": med[q] / med.iloc[0], "n": int((d["quarter"] == q).sum())})
        out[zone] = m
    t = pd.DataFrame(rows)
    summ = {}
    for zone, d in t.groupby("zone"):
        pk = d.loc[d["idx"].idxmax()]
        last = d.iloc[-1]
        summ[zone] = {"peak_q": pk["quarter"], "peak": pk["idx"], "last": last["idx"], "last_lo": last["lo"],
                      "last_hi": last["hi"], "from_peak": last["idx"] / pk["idx"] - 1,
                      "naive_last": last["naive"], "r2": out[zone].r2}
    return {"table": t, "summary": summ}


def size_mix(s: pd.DataFrame) -> pd.DataFrame:
    """Состав сделок по годам — проверка, не сдвинулся ли «микс» (он и ломает наивную медиану)."""
    f = flats(s)
    return f.groupby(["zone", "year"]).agg(small=("surface", lambda x: (x < 30).mean()),
                                           med_surface=("surface", "median"),
                                           new=("new_build", "mean"), dist=("dist_km", "median")).reset_index()


# ----------------------------------------------------------------- H2: объёмы против цен
def volumes(s: pd.DataFrame, min_n: int = 60) -> dict:
    q = s.groupby(["quarter", "zone"]).size().rename("n").reset_index()
    y = s.groupby(["year", "zone"]).size().unstack()
    # по коммунам: изменение числа сделок и «чистой» цены (средний остаток гедонической модели без времени)
    f = flats(s)
    f = f[~f["new_build"]]           # без новостроек: их объём задаёт стройка, а не спрос
    m = fe_ols(f["log_ppm2"], _attrs(f, new_build=False), f["cell"])
    f = f.assign(res=m.resid)        # эффект ячейки постоянен во времени: сдвиг среднего остатка = сдвиг цены
    g = f.groupby(["code_commune", "period"]).agg(n=("res", "size"), r=("res", "mean")).unstack()
    g = g[(g[("n", "2021–22")] >= min_n) & (g[("n", "2023–24")] >= min_n)]
    c = pd.DataFrame({"vol": g[("n", "2023–24")] / g[("n", "2021–22")] - 1,
                      "price": np.expm1(g[("r", "2023–24")] - g[("r", "2021–22")])})
    c["commune"] = f.groupby("code_commune")["nom_commune"].first().reindex(c.index)
    c["paris"] = c.index.str.startswith("75")
    rho, p = stats.spearmanr(c["vol"], c["price"])
    kind = s.groupby(["year", "new_build"]).size().unstack()
    return {"quarterly": q, "yearly": y, "communes": c.reset_index(), "rho": float(rho), "p": float(p),
            "kind_change": (kind.loc[2024] / kind.loc[2021] - 1).rename({False: "вторичка", True: "новостройки"}).to_dict(),
            "vol_change": (y.loc[2024] / y.loc[2021] - 1).to_dict(),
            "vol_peak_change": (y.loc[2024] / y.max() - 1).to_dict()}


# ----------------------------------------------------------------- H3: «пончик»
def donut(s: pd.DataFrame) -> dict:
    """Разрыв пригород/Париж по кварталам при той же ячейке и характеристиках; градиент от центра по годам."""
    f = flats(s)
    X = _attrs(f, {"quarter": "2021Q1"})
    qs = pd.get_dummies(f["quarter"], dtype=float).mul((~f["paris"]).astype(float), axis=0)
    qs = qs.drop(columns="2021Q1").add_prefix("sub×")
    m = fe_ols(f["log_ppm2"], pd.concat([X, qs], axis=1), f["cell"])
    rows = [{"quarter": "2021Q1", "gap": 1.0, "lo": 1.0, "hi": 1.0}]
    for c in qs.columns:
        b, lo, hi = m.ci(c)
        rows.append({"quarter": c[4:], "gap": np.exp(b), "lo": np.exp(lo), "hi": np.exp(hi)})
    gap = pd.DataFrame(rows)
    grad = []
    for (zone, y), d in list(f.groupby(["zone", "year"])) + [(("всё", y), d) for y, d in f.groupby("year")]:
        g = fe_ols(d["log_ppm2"], design(d, ["dist_km"] + HEDONIC), np.zeros(len(d)), d["code_commune"])
        b, lo, hi = g.ci("dist_km")
        grad.append({"zone": zone, "year": y, "slope": np.expm1(b), "lo": np.expm1(lo), "hi": np.expm1(hi)})
    bins = pd.cut(f["dist_km"], np.arange(0, 21, 1.0))
    prof = f.groupby([bins, "year"], observed=True)["ppm2"].median().rename("med").reset_index()
    prof["km"] = prof["dist_km"].map(lambda iv: iv.mid).astype(float)
    pk = gap.loc[gap["gap"].idxmax()]
    ring = f[f["dist_km"].between(4, 7)]                 # пояс, где по обе стороны границы — одно расстояние
    border = ring.groupby("zone")["ppm2"].median()
    bm = fe_ols(ring["log_ppm2"], design(ring, ["paris", "dist_km"] + HEDONIC, {"quarter": "2021Q1"}),
                np.zeros(len(ring)), ring["code_commune"])
    pk = gap.loc[gap["gap"].idxmax()]
    return {"border_med": border.to_dict(), "border": bm.pct("paris"),"gap": gap, "gradient": pd.DataFrame(grad), "profile": prof.drop(columns="dist_km"),
            "peak_q": pk["quarter"], "peak": pk["gap"], "peak_lo": pk["lo"], "peak_hi": pk["hi"],
            "last": gap.iloc[-1].to_dict()}


# ----------------------------------------------------------------- H4: студии дороже за м²?
def size_premium(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for zone, d in flats(s).groupby("zone"):
        raw = d.groupby("size_bin", observed=True)["ppm2"].median()
        m = fe_ols(d["log_ppm2"], design(d, ["new_build"], {"size_bin": "9–20", "quarter": "2021Q1"}), d["cell"])
        for b in SIZE_LABELS:
            r = raw[b] / raw["9–20"] - 1
            w = (0.0, 0.0, 0.0) if b == "9–20" else m.pct(f"size_bin={b}")
            rows.append({"zone": zone, "bin": b, "raw": r, "within": w[0], "lo": w[1], "hi": w[2],
                         "n": int((d["size_bin"] == b).sum())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- H5: новостройки
def new_build_premium(s: pd.DataFrame) -> dict:
    d = flats(s)
    d = d[~d["paris"]]                                   # в Париже новостроек 0,2 %
    raw = d.groupby("new_build")["ppm2"].median()
    m = fe_ols(d["log_ppm2"], _attrs(d, {"quarter": "2021Q1"}), d["cell"])
    by_dep = {}
    for dep, x in d.groupby("dep"):
        mm = fe_ols(x["log_ppm2"], _attrs(x, {"quarter": "2021Q1"}), x["cell"])
        by_dep[dep] = mm.pct("new_build") + (int(x["new_build"].sum()),)
    return {"raw": float(raw[True] / raw[False] - 1), "within": m.pct("new_build"), "share": float(d["new_build"].mean()),
            "by_dep": by_dep, "n_new": int(d["new_build"].sum())}


# ----------------------------------------------------------------- H6: Олимпиада
def olympics(s: pd.DataFrame, pre: int = 2021, post: int = 2024, min_n: int = 30) -> dict:
    """Сен-Дени, Сент-Уан, Иль-Сен-Дени против остальных коммун департамента 93."""
    d = flats(s)
    d = d[d["dep"] == "93"]
    m = fe_ols(d["log_ppm2"], _attrs(d, {"year": 2021}), d["cell"])
    d = d.assign(res=m.resid)
    g = d.groupby(["code_commune", "year"])["res"].agg(["mean", "size"]).unstack()
    ok = (g["size"] >= min_n).all(axis=1)
    means, counts = g["mean"][ok], g["size"][ok]
    treated = set(OLYMPIC) & set(means.index)
    eff, p, placebo = permutation_did(means, counts, treated, pre, post)
    ev = []
    for y in sorted(means.columns):
        e, py, pl = permutation_did(means, counts, treated, pre, y) if y != pre else (0.0, 1.0, np.zeros(1))
        ev.append({"year": y, "effect": np.expm1(e), "p": py, "lo": np.expm1(np.percentile(pl, 2.5)),
                   "hi": np.expm1(np.percentile(pl, 97.5))})
    # то же с «обычными» кластерными ошибками — для сравнения
    d = d[d["code_commune"].isin(means.index)]
    dd = d[d["year"].isin([pre, post])]
    X = _attrs(dd, {"year": pre})
    X["oly×post"] = (dd["code_commune"].isin(treated) & (dd["year"] == post)).astype(float)
    mc = fe_ols(dd["log_ppm2"], X, dd["cell"], dd["code_commune"])
    raw = d.groupby([d["code_commune"].isin(treated), "year"])["ppm2"].median().unstack()
    return {"effect": float(np.expm1(eff)), "p": p, "placebo": np.expm1(placebo), "event": pd.DataFrame(ev),
            "n_units": len(means), "n_placebo": len(placebo), "cluster": mc.pct("oly×post"),
            "raw_treated": raw.loc[True].to_dict(), "raw_control": raw.loc[False].to_dict()}


# ----------------------------------------------------------------- H7: круглые цены
def round_prices(s: pd.DataFrame) -> dict:
    pr = s["price"].to_numpy()
    shares = {k: float((pr % k == 0).mean()) for k in (1_000, 5_000, 10_000, 50_000, 100_000)}
    r100 = [(x, *side_ratio(pr, x)) for x in range(200_000, 1_000_001, 100_000)]
    r10 = [(x, *side_ratio(pr, x)) for x in range(150_000, 1_000_001, 10_000) if x % 50_000]
    r100 = [t for t in r100 if np.isfinite(t[1])]   # порог без сделок с одной стороны в сравнение не идёт
    r10 = [t for t in r10 if np.isfinite(t[1])]
    a, b = np.array([r[1] for r in r100]), np.array([r[1] for r in r10])
    u = stats.mannwhitneyu(a, b, alternative="greater")
    hist = pd.Series(pr[(pr >= 150_000) & (pr < 450_000)] // 1000 * 1000).value_counts().sort_index()
    return {"shares": shares, "r100": pd.DataFrame(r100, columns=["x", "lr", "below", "above"]),
            "r10": pd.DataFrame(r10, columns=["x", "lr", "below", "above"]), "p": float(u.pvalue),
            "excess100": float(np.expm1(a.mean())), "excess10": float(np.expm1(b.mean())),
            "hist": hist.rename_axis("price").rename("n").reset_index()}


# ----------------------------------------------------------------- H8: сезонность
DAYS = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def seasonality(s: pd.DataFrame) -> dict:
    """Сделки по месяцам (на день) и сезонность цены.

    Помесячные дамми с годовыми эффектами путают сезон с трендом: в 2023–24 цены падали внутри
    года, и декабрь «дешевле» января просто потому, что позже. Поэтому цена — классическим
    разложением: помесячный гедонический индекс / центрированное скользящее среднее 2×12.
    """
    cnt = s.groupby("month").size()
    exp = DAYS / DAYS.sum() * cnt.sum()
    chi = stats.chisquare(cnt.to_numpy(), exp)
    per_day = cnt / DAYS / (cnt.sum() / DAYS.sum())
    f = flats(s).assign(ym=lambda d: d["date"].dt.strftime("%Y-%m"))
    m = fe_ols(f["log_ppm2"], _attrs(f, {"ym": "2021-01"}), f["cell"])
    months = sorted(f["ym"].unique())
    li = pd.Series([0.0] + [float(m.params[f"ym={x}"]) for x in months[1:]], index=pd.PeriodIndex(months, freq="M"))
    w = np.r_[0.5, np.ones(11), 0.5] / 12
    ma = pd.Series(np.convolve(li.to_numpy(), w, mode="same"), index=li.index)
    ma.iloc[:6], ma.iloc[-6:] = np.nan, np.nan
    sf = (li - ma).dropna()
    sea = pd.DataFrame({"month": sf.index.month, "year": sf.index.year, "eff": np.expm1(sf.to_numpy())})
    g = sea.groupby("month")["eff"]
    spread = sea.groupby("month")["eff"].agg(["mean", "min", "max", "size"]).reset_index()
    # амплитуда сезонности против шума: F-тест «месяц» на отклонениях от тренда
    fstat = stats.f_oneway(*[x.to_numpy() for _, x in g])
    # то же наивно — годовые эффекты вместо тренда
    nv = fe_ols(f["log_ppm2"], _attrs(f, {"year": 2021, "month": 1}), f["cell"])
    naive = pd.DataFrame([{"month": 1, "eff": 0.0}] + [{"month": k, "eff": nv.pct(f"month={k}")[0]} for k in range(2, 13)])
    return {"counts": cnt, "per_day": per_day, "chi2": float(chi.statistic), "p": float(chi.pvalue),
            "index": np.expm1(li), "trend": np.expm1(ma), "season": sea, "spread": spread,
            "f": float(fstat.statistic), "f_p": float(fstat.pvalue), "naive": naive}


def run_all(s: pd.DataFrame) -> dict:
    s = prepare(s)
    arr = arrondissements(s)
    rho, p = stats.spearmanr(arr["med"], arr["change"])
    rho_lo, rho_hi = boot_spearman_ci(arr["med"], arr["change"])
    return {"s": s, "overview": overview(s), "map": price_map(s), "arr": arr,
            "arr_rho": (float(rho), float(p), rho_lo, rho_hi),
            "location": location_share(s), "index": hedonic_index(s), "mix": size_mix(s), "volumes": volumes(s),
            "donut": donut(s), "size": size_premium(s), "new": new_build_premium(s), "oly": olympics(s),
            "round": round_prices(s), "season": seasonality(s)}
