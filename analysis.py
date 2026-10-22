# %% [markdown]
# # Рынок жилья Парижа и ближнего пригорода, 2021–2024 — EDA
#
# Нотариальные сделки из открытого реестра DVF (департаменты 75, 92, 93, 94). Анализ построен на
# гипотезах: каждая проверяется регрессией с фиксированным эффектом локации (ячейка 300 × 300 м),
# бутстрепом или перестановочным тестом. Интерактивная версия с картами — `docs/index.html`.
#
# Исходник ноутбука — `analysis.py` (ячейки `# %%`), сборка — `python make_notebook.py`.

# %%
import os
import shutil
import sys
import tempfile
import warnings

import pandas as pd
import plotly
import plotly.io as pio
from IPython.display import Image, display

sys.path.insert(0, "src")
warnings.filterwarnings("ignore", category=FutureWarning)
from dvf import analysis as A   # noqa: E402
from dvf import figures as F    # noqa: E402
from dvf.clean import load      # noqa: E402

# kaleido не понимает не-ASCII пути к plotly.js — кладём копию во временную папку
_js = os.path.join(tempfile.gettempdir(), f"plotly-{plotly.__version__}.min.js")
shutil.copy(os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js"), _js)
pio.kaleido.scope.plotlyjs = _js
pd.set_option("display.precision", 3)


def show(fig, width=1000):
    """Статичная картинка: GitHub не рендерит интерактивный Plotly в ноутбуках."""
    display(Image(fig.to_image("png", width=width, scale=1)))


# %% [markdown]
# ## 1. Данные: что такое одна сделка
#
# Строка DVF — сделка × участок × помещение × вид угодий, и полная цена сделки повторена в каждой.
# Оставляем сделки ровно с одним жильём, без коммерческих помещений и с одной ценой; подсобные
# помещения (парковки, погреба) остаются в цене.

# %%
s, steps = load()
R = A.run_all(s)
s = R["s"]
show(F.funnel(steps))
pd.DataFrame(steps, columns=["шаг", "число"])

# %%
R["overview"]["by_year"].pivot(index="year", columns="zone", values=["n", "med"]).round(0)

# %% [markdown]
# ## 2. Локация решает почти всё
#
# Доля разброса log(€/м²) квартир, которую объясняет локация, — отсюда фиксированный эффект ячейки
# 300 м во всех дальнейших сравнениях.

# %%
pd.Series(R["location"]).round(3)

# %%
show(F.arrondissements(R["arr"]), 900)
print("Спирмен «уровень округа — изменение 2024 к 2021»: ρ = %.2f, p = %.1e" % R["arr_rho"])
show(F.arr_level_change(R["arr"], R["arr_rho"][0]))

# %%
dn = R["donut"]
print("Эффект границы Парижа (4–7 км от центра, с поправкой на расстояние): %+.1f%% [%+.1f; %+.1f]"
      % tuple(100 * x for x in dn["border"]))
dn["border_med"]

# %% [markdown]
# ## 3. Гипотеза: после подъёма ставок цены упали
#
# Гедонический индекс: log(€/м²) ~ дамми кварталов + FE ячейки + площадь (log, квадрат) + комнаты +
# новостройка; ошибки кластерные по ячейкам. Пунктир — наивная медиана.

# %%
show(F.hedonic_index(R["index"]["table"]))
pd.DataFrame(R["index"]["summary"]).T

# %%
R["mix"]   # состав сделок по годам — проверка наивной медианы

# %% [markdown]
# ## 4. Гипотеза: объём падает сильнее цены

# %%
v = R["volumes"]
show(F.volumes(v["quarterly"]))
print("Число сделок 2024 к 2021:", {k: round(x, 3) for k, x in v["vol_change"].items()})
print("По типу:", {k: round(x, 3) for k, x in v["kind_change"].items()})
print("Коммуны, вторичка: Спирмен(Δобъёма, Δцены) = %.3f, p = %.4f, n = %d" % (v["rho"], v["p"], len(v["communes"])))
show(F.volume_vs_price(v["communes"], v["rho"], v["p"]))

# %% [markdown]
# ## 5. Гипотеза «пончика»: пригород выиграл у Парижа
#
# Одна регрессия: дамми кварталов + дамми «квартал × пригород» при FE ячейки.

# %%
show(F.donut_gap(dn["gap"]))
show(F.distance_profile(dn["profile"]))
show(F.gradient(dn["gradient"]))
dn["gradient"].pivot(index="year", columns="zone", values="slope")

# %% [markdown]
# ## 6. Гипотеза: маленькие квартиры дороже за м²
#
# «Как есть» (медианы) против «в той же ячейке 300 м». В пригороде знак премии больших квартир
# меняется — парадокс Симпсона.

# %%
show(F.size_premium(R["size"]))
R["size"].pivot(index="bin", columns="zone", values=["raw", "within"]).loc[A.SIZE_LABELS]

# %% [markdown]
# ## 7. Сколько стоит новизна (пригород)

# %%
nb = R["new"]
show(F.new_build(nb))
print("Медианы как есть: %+.1f%%; в той же ячейке: %+.1f%% [%+.1f; %+.1f]"
      % (100 * nb["raw"], *(100 * x for x in nb["within"])))

# %% [markdown]
# ## 8. Гипотеза: Олимпиада подняла цены у Олимпийской деревни
#
# Разность разностей на средних остатках по коммунам 93; основной тест — перестановочный
# (все наборы из трёх коммун), потому что кластерные ошибки при трёх «экспериментальных» коммунах ненадёжны.

# %%
o = R["oly"]
show(F.olympics_event(o["event"]))
show(F.olympics_placebo(o["placebo"], o["effect"], o["p"]))
print("Эффект 2024 к 2021: %+.1f%%, перестановочный p = %.3f (%d наборов); кластерный ДИ [%+.1f; %+.1f]"
      % (100 * o["effect"], o["p"], o["n_placebo"], 100 * o["cluster"][1], 100 * o["cluster"][2]))

# %% [markdown]
# ## 9. Круглые числа и «чуть ниже»

# %%
rd = R["round"]
print({k: round(x, 3) for k, x in rd["shares"].items()})
show(F.price_hist(rd["hist"]))
show(F.threshold_ratio(rd["r100"], rd["r10"]))
print("Перевес «ниже»: круглые %+.1f%%, прочие %+.1f%%, Манн — Уитни p = %.4f"
      % (100 * rd["excess100"], 100 * rd["excess10"], rd["p"]))

# %% [markdown]
# ## 10. Сезонность
#
# Объём — сделки в день по месяцам; цена — отклонение помесячного гедонического индекса от
# скользящего среднего 2×12 (дамми месяцев с годовыми эффектами путают сезон с внутригодовым трендом).

# %%
se = R["season"]
show(F.season_volume(se["per_day"]))
naive = se["naive"].assign(eff=lambda d: d["eff"] - d["eff"].mean())
show(F.season_price(se["season"], se["spread"], naive))
print("χ² объёма = %.0f; F-тест сезонности цены: F = %.1f, p = %.1e" % (se["chi2"], se["f"], se["f_p"]))
se["spread"]
