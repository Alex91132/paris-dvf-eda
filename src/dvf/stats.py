"""Статистика для EDA: регрессия с фиксированными эффектами, бутстреп, перестановочный тест, «кучкование» цен.

Регрессия с фиксированным эффектом локации считается через внутригрупповое преобразование
(вычитание средних по ячейке), а не через тысячи дамми-столбцов: 300 тыс. сделок × 1 000 ячеек
в statsmodels требуют ~10 ГБ памяти, здесь — секунды.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

Z95 = 1.959963984540054
CELL_KM = 0.3          # сторона ячейки сетки локаций
_LAT0 = 48.85


def grid_cell(lat, lon, km: float = CELL_KM) -> np.ndarray:
    """Ячейка квадратной сетки со стороной km (равноугольная аппроксимация на широте Парижа)."""
    ky, kx = 111.2, 111.2 * np.cos(np.radians(_LAT0))
    gx = np.floor(np.asarray(lon) * kx / km).astype(np.int64)
    gy = np.floor(np.asarray(lat) * ky / km).astype(np.int64)
    return gx * 100_000 + gy


def cell_center(cell: np.ndarray, km: float = CELL_KM) -> tuple[np.ndarray, np.ndarray]:
    ky, kx = 111.2, 111.2 * np.cos(np.radians(_LAT0))
    gx, gy = np.asarray(cell) // 100_000, np.asarray(cell) % 100_000
    return (gy + 0.5) * km / ky, (gx + 0.5) * km / kx


@dataclass
class FE:
    """Результат регрессии с одним фиксированным эффектом и кластерными ошибками."""
    params: pd.Series
    se: pd.Series
    r2: float            # доля дисперсии y, объяснённая моделью вместе с фиксированным эффектом
    r2_fe_only: float    # только фиксированный эффект (локация)
    n: int
    n_groups: int
    n_clusters: int
    resid: np.ndarray    # остатки после внутригруппового преобразования

    def ci(self, name: str) -> tuple[float, float, float]:
        b, s = float(self.params[name]), float(self.se[name])
        return b, b - Z95 * s, b + Z95 * s

    def pct(self, name: str) -> tuple[float, float, float]:
        """Коэффициент лог-модели как процент: exp(b) − 1 и 95 % ДИ."""
        return tuple(float(np.expm1(v)) for v in self.ci(name))


def design(df: pd.DataFrame, numeric: list[str] = (), dummies: dict[str, object] = None) -> pd.DataFrame:
    """Матрица признаков: числовые столбцы + дамми категорий (база — указанное значение)."""
    X = pd.DataFrame({c: df[c].astype(float).to_numpy() for c in numeric}, index=df.index)
    for col, base in (dummies or {}).items():
        d = pd.get_dummies(df[col], prefix=col, prefix_sep="=", dtype=float)
        X = pd.concat([X, d.drop(columns=f"{col}={base}")], axis=1)
    return X


def fe_ols(y: pd.Series, X: pd.DataFrame, groups, clusters=None) -> FE:
    """МНК y ~ X + FE(groups); ковариация — кластерная (по clusters, по умолчанию — по groups)."""
    groups = pd.Series(pd.factorize(np.asarray(groups))[0], index=y.index)
    clusters = groups if clusters is None else pd.Series(pd.factorize(np.asarray(clusters))[0], index=y.index)
    yd = y - y.groupby(groups).transform("mean")
    Xd = X - X.groupby(groups).transform("mean")
    keep = Xd.abs().sum() > 1e-9                      # столбцы, полностью поглощённые эффектом
    Xd = Xd.loc[:, keep]
    A, b = Xd.to_numpy(), yd.to_numpy()
    xtx_inv = np.linalg.pinv(A.T @ A)
    beta = xtx_inv @ (A.T @ b)
    e = b - A @ beta
    n, k = A.shape
    g = clusters.nunique()
    scores = pd.DataFrame(A * e[:, None]).groupby(clusters.to_numpy()).sum().to_numpy()
    meat = scores.T @ scores
    corr = g / (g - 1) * (n - 1) / (n - k - groups.nunique())
    cov = corr * xtx_inv @ meat @ xtx_inv
    sst = float(((y - y.mean()) ** 2).sum())
    return FE(params=pd.Series(beta, index=Xd.columns), se=pd.Series(np.sqrt(np.diag(cov)), index=Xd.columns),
              r2=1 - float(e @ e) / sst, r2_fe_only=1 - float((yd ** 2).sum()) / sst, n=n,
              n_groups=int(groups.nunique()), n_clusters=int(g), resid=e)


def boot_median(x, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Медиана и 95 % перцентильный бутстреп-интервал."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(x, (n_boot, len(x))), axis=1) if len(x) * n_boot <= 3e7 else np.array(
        [np.median(rng.choice(x, len(x))) for _ in range(n_boot)])
    return float(np.median(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def permutation_did(means: pd.DataFrame, counts: pd.DataFrame, treated: set, pre, post):
    """Разность разностей на средних по единицам и точный перестановочный тест.

    means / counts: индекс — единица (коммуна), столбцы — периоды. Эффект = изменение pre→post у
    treated минус у остальных (веса — число сделок). Плацебо — все наборы из стольких же единиц:
    p = доля плацебо с |эффектом| ≥ наблюдаемого. Не опирается на асимптотику кластерных ошибок,
    которая при трёх «леченых» кластерах не работает.
    """
    units = list(means.index)
    ch = (means[post] - means[pre]).to_numpy()
    w = np.minimum(counts[pre], counts[post]).to_numpy().astype(float)

    def effect(mask: np.ndarray) -> float:
        return np.average(ch[mask], weights=w[mask]) - np.average(ch[~mask], weights=w[~mask])

    obs = effect(np.array([u in treated for u in units]))
    placebo = []
    for combo in combinations(range(len(units)), len(treated)):
        m = np.zeros(len(units), bool)
        m[list(combo)] = True
        placebo.append(effect(m))
    placebo = np.array(placebo)
    p = float((np.abs(placebo) >= abs(obs) - 1e-12).mean())
    return float(obs), p, placebo


def side_ratio(price: np.ndarray, x: float, w: float = 10_000) -> tuple[float, int, int]:
    """log(число сделок в [x−w, x) / в (x, x+w]) — перевес «чуть ниже порога»; сам порог не считается."""
    below = int(((price >= x - w) & (price < x)).sum())
    above = int(((price > x) & (price <= x + w)).sum())
    return float(np.log(below / above)), below, above
