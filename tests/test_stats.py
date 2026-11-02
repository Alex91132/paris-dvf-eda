"""Статистика: FE-регрессия против statsmodels, перестановочный тест, бутстреп, пороги."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from dvf.stats import (boot_median, boot_spearman_ci, cell_center, design, fe_ols, grid_cell,
                       permutation_did, side_ratio, _spearman)


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(1)
    n, g = 3000, 40
    d = pd.DataFrame({"g": rng.integers(0, g, n), "x": rng.normal(size=n), "q": rng.choice(["a", "b", "c"], n)})
    d["y"] = 0.5 * d["x"] + d["q"].map({"a": 0, "b": 0.1, "c": -0.2}) + rng.normal(size=g)[d["g"]] + rng.normal(0, .3, n)
    return d


def test_fe_matches_dummy_regression_with_cluster_se(panel):
    m = fe_ols(panel["y"], design(panel, ["x"], {"q": "a"}), panel["g"])
    ref = smf.ols("y ~ x + C(q) + C(g)", panel).fit(cov_type="cluster", cov_kwds={"groups": panel["g"]})
    assert m.params["x"] == pytest.approx(ref.params["x"], rel=1e-10)
    assert m.params["q=c"] == pytest.approx(ref.params["C(q)[T.c]"], rel=1e-10)
    assert m.r2 == pytest.approx(ref.rsquared, rel=1e-10)
    # поправка на число эффектов в знаменателе отличается от statsmodels — ошибки совпадают до неё
    assert m.se["x"] == pytest.approx(ref.bse["x"], rel=0.02)
    lo, hi = m.ci("x")[1:]
    assert lo < 0.5 < hi


def test_fe_only_r2(panel):
    m = fe_ols(panel["y"], design(panel, ["x"]), panel["g"])
    ref = smf.ols("y ~ C(g)", panel).fit()
    assert m.r2_fe_only == pytest.approx(ref.rsquared, rel=1e-10)


def test_permutation_did_finds_planted_effect_and_not_noise():
    rng = np.random.default_rng(2)
    units = [f"u{i}" for i in range(20)]
    pre = rng.normal(0, 0.01, 20)
    post = pre + rng.normal(0, 0.01, 20)
    counts = pd.DataFrame({0: 100, 1: 100}, index=units)
    planted = post.copy()
    planted[:3] += 0.1
    eff, p, plc = permutation_did(pd.DataFrame({0: pre, 1: planted}, index=units), counts, set(units[:3]), 0, 1)
    assert eff == pytest.approx(0.1, abs=0.02) and p == pytest.approx(1 / len(plc)) and len(plc) == 1140
    _, p0, _ = permutation_did(pd.DataFrame({0: pre, 1: post}, index=units), counts, set(units[:3]), 0, 1)
    assert p0 > 0.05


def test_boot_median_covers():
    x = np.random.default_rng(3).lognormal(9, 0.4, 2000)
    m, lo, hi = boot_median(x)
    assert lo < m < hi and lo < np.exp(9) < hi


def test_side_ratio_excludes_threshold():
    p = np.array([295_000, 299_000, 300_000, 300_000, 305_000])
    lr, b, a = side_ratio(p, 300_000, 10_000)
    assert (b, a) == (2, 1) and lr == pytest.approx(np.log(2))


def test_grid_cell_roundtrip():
    lat, lon = np.array([48.8530, 48.9]), np.array([2.3499, 2.2])
    c = grid_cell(lat, lon)
    clat, clon = cell_center(c)
    assert np.all(np.abs(clat - lat) < 0.3 / 111.2) and np.all(np.abs(clon - lon) < 0.3 / 73)
    assert grid_cell(lat + 1e-6, lon + 1e-6)[0] == c[0]


def test_spearman_known_values_and_ties():
    x = np.arange(20.0)
    assert _spearman(x, 3 * x + 7) == pytest.approx(1.0)
    assert _spearman(x, -x) == pytest.approx(-1.0)
    y = x.copy()
    y[9] = y[10] = 5.0                    # связь: ранг считается средним
    assert 0.9 < _spearman(x, y) < 1.0


def test_boot_spearman_ci_brackets_estimate():
    rng = np.random.default_rng(4)
    x = np.arange(20.0)
    y = x + rng.normal(0, 3, 20)
    lo, hi = boot_spearman_ci(x, y)
    rho = _spearman(x, y)
    assert lo <= rho <= hi
    lo2, hi2 = boot_spearman_ci(x, -x)     # строго обратная связь
    assert hi2 <= -0.99


def test_guards_on_empty_side_and_missing_base():
    lr, below, above = side_ratio(np.array([290_000.0, 295_000.0]), 300_000, 10_000)
    assert np.isnan(lr) and above == 0 and below == 2
    X = design(pd.DataFrame({"q": ["b"], "x": [1.0]}), ["x"], {"q": "a"})
    assert list(X.columns) == ["x", "q=b"]
