"""Графики Plotly для отчёта и ноутбука. Палитра — одна на весь отчёт: Париж синий, пригород оранжевый."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
PARIS, SUB, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
ZONE = {"Париж": PARIS, "пригород": SUB}
BLUES = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
YEARS = {2021: "#9ec5f4", 2022: "#5598e7", 2023: "#256abf", 2024: "#0d366b"}   # упорядоченные годы — одна гамма
DIVERGING = [[0.0, "#b3261e"], [0.25, "#e88a7d"], [0.5, "#f0efec"], [0.75, "#86b6ef"], [1.0, "#1c5cab"]]
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
CONFIG = {"displaylogo": False, "locale": "ru", "responsive": True,
          "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}


def rgba(hex_: str, a: float) -> str:
    h = hex_.lstrip("#")
    return f"rgba({int(h[:2], 16)},{int(h[2:4], 16)},{int(h[4:], 16)},{a})"


def base(fig: go.Figure, title: str, h: int = 380, y: str = "", yfmt: str | None = None, x: str = "",
         hover: str = "closest") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK), x=0.01, xanchor="left", y=0.985, yanchor="top"),
        height=h, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, separators=", ",
        font=dict(family=FONT, size=12, color=INK2), margin=dict(l=56, r=16, t=84, b=48),
        legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0, xanchor="left", font=dict(size=12),
                    bgcolor="rgba(0,0,0,0)"),
        hovermode=hover, hoverlabel=dict(bgcolor="#ffffff", bordercolor=AXIS, font=dict(color=INK, family=FONT)))
    fig.update_xaxes(title=x, showgrid=False, linecolor=AXIS, tickcolor=AXIS, ticks="outside", ticklen=4,
                     zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(title=y, gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED),
                     tickformat=yfmt)
    return fig


def _band(fig, x, lo, hi, color, name, legend: bool = False, **kw):
    fig.add_trace(go.Scatter(x=list(x) + list(x)[::-1], y=list(hi) + list(lo)[::-1], fill="toself", mode="lines",
                             fillcolor=rgba(color, 0.16), line=dict(width=0), hoverinfo="skip", showlegend=legend,
                             name=name, **kw))


def _zero(fig, y=0.0, **kw):
    fig.add_hline(y=y, line=dict(color=AXIS, width=1), **kw)


# ----------------------------------------------------------------- данные
def funnel(steps: list[tuple[str, int]]) -> go.Figure:
    lab = [a for a, _ in steps][::-1]
    val = [b for _, b in steps][::-1]
    fig = go.Figure(go.Bar(x=val, y=lab, orientation="h", marker=dict(color=PARIS, cornerradius=4),
                           text=[f"{v:,}".replace(",", " ") for v in val], textposition="outside",
                           textfont=dict(color=INK2), cliponaxis=False,
                           hovertemplate="%{y}: %{x:,}<extra></extra>"))
    base(fig, "От сырых строк DVF до продаж жилья", h=330)
    fig.update_xaxes(visible=False, range=[0, max(val) * 1.18])
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK2))
    fig.update_layout(margin=dict(l=8, r=8))
    fig.update_yaxes(automargin=True)
    return fig


# ----------------------------------------------------------------- карта
def _map(fig, h, zoom=10.4):
    fig.update_layout(map=dict(style="carto-positron", center=dict(lat=48.856, lon=2.36), zoom=zoom),
                      height=h, margin=dict(l=0, r=0, t=56, b=0), paper_bgcolor=SURFACE,
                      font=dict(family=FONT, size=12, color=INK2))
    return fig


def price_map(cells: pd.DataFrame) -> go.Figure:
    hover = [f"<b>{c}</b><br>{m:,.0f} €/м² · {n} сделок".replace(",", " ")
             for c, m, n in zip(cells["commune"], cells["med"], cells["n"])]
    fig = go.Figure(go.Scattermap(
        lat=cells["lat"], lon=cells["lon"], mode="markers", text=hover, hoverinfo="text",
        marker=dict(size=7, color=cells["med"], colorscale=[[i / 6, c] for i, c in enumerate(BLUES)],
                    cmin=3_000, cmax=15_000, opacity=0.9,
                    colorbar=dict(title=dict(text="€/м²", side="top"), thickness=10, len=0.6, x=0.99,
                                  tickformat=",.0f", outlinewidth=0))))
    fig.update_layout(title=dict(text="Медиана цены квартир, €/м², ячейки 300 м (2021–2024, от 10 сделок)",
                                 font=dict(size=15, color=INK), x=0, xanchor="left"))
    return _map(fig, 600)


def price_grid_static(cells: pd.DataFrame) -> go.Figure:
    """Та же карта без подложки — для статичных картинок (kaleido не загружает тайлы)."""
    fig = go.Figure(go.Scatter(
        x=cells["lon"], y=cells["lat"], mode="markers",
        marker=dict(symbol="square", size=5.2, color=cells["med"], colorscale=[[i / 6, c] for i, c in enumerate(BLUES)],
                    cmin=3_000, cmax=15_000, colorbar=dict(title=dict(text="€/м²", side="top"), thickness=10, len=0.7,
                                                           tickformat=",.0f", outlinewidth=0))))
    base(fig, "Медиана цены квартир, €/м², ячейки 300 м (2021–2024)", h=640)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1 / np.cos(np.radians(48.85)))
    fig.update_layout(margin=dict(l=8, r=8, t=48, b=8))
    return fig


def change_map(ch: pd.DataFrame) -> go.Figure:
    hover = [f"<b>{c}</b><br>{v:+.1%}".replace(".", ",") for c, v in zip(ch["commune"], ch["change"])]
    fig = go.Figure(go.Scattermap(
        lat=ch["lat"], lon=ch["lon"], mode="markers", text=hover, hoverinfo="text",
        marker=dict(size=7, color=ch["change"], colorscale=DIVERGING, cmin=-0.25, cmax=0.25, opacity=0.95,
                    colorbar=dict(title=dict(text="изменение", side="top"), thickness=10, len=0.6, x=0.99,
                                  tickformat="+.0%", outlinewidth=0))))
    fig.update_layout(title=dict(text="Изменение медианы €/м²: 2023–24 к 2021–22 (ячейки с ≥ 10 сделок в обоих периодах)",
                                 font=dict(size=15, color=INK), x=0, xanchor="left"))
    return _map(fig, 600)


def arrondissements(a: pd.DataFrame) -> go.Figure:
    a = a.sort_values("med")
    lab = [f"{x}-й" for x in a["arr"]]
    fig = go.Figure(go.Bar(
        x=a["med"], y=lab, orientation="h", marker=dict(color=PARIS, cornerradius=4),
        error_x=dict(type="data", symmetric=False, array=a["hi"] - a["med"], arrayminus=a["med"] - a["lo"],
                     color=INK2, thickness=1.2, width=3),
        customdata=np.c_[a["lo"], a["hi"], a["n"]],
        hovertemplate="%{y} округ: %{x:,.0f} €/м²<br>95 % ДИ %{customdata[0]:,.0f}–%{customdata[1]:,.0f}"
                      "<br>%{customdata[2]:,} сделок<extra></extra>"))
    base(fig, "Медиана €/м² по округам Парижа, 2021–2024 (95 % ДИ — бутстреп)", h=560, x="€/м²")
    fig.update_xaxes(showgrid=True, gridcolor=GRID, tickformat=",.0f")
    fig.update_yaxes(showgrid=False, tickfont=dict(color=INK2), automargin=True)
    return fig


def arr_level_change(a: pd.DataFrame, rho: float) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=a["med"], y=a["change"], mode="markers+text", text=a["arr"].astype(str), textposition="top center",
        textfont=dict(size=11, color=INK2), marker=dict(size=10, color=PARIS, line=dict(color=SURFACE, width=2)),
        hovertemplate="%{text}-й округ<br>уровень %{x:,.0f} €/м²<br>изменение %{y:+.1%}<extra></extra>"))
    base(fig, f"Чем дороже округ, тем меньше упал (ρ Спирмена = {rho:.2f})".replace(".", ","), h=400,
         x="медиана €/м², 2021–2024", y="медиана 2024 к 2021", yfmt="+.0%")
    fig.update_xaxes(tickformat=",.0f")
    return fig


# ----------------------------------------------------------------- индекс
def hedonic_index(t: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for zone, d in t.groupby("zone"):
        c = ZONE[zone]
        _band(fig, d["quarter"], d["lo"], d["hi"], c, zone)
        fig.add_trace(go.Scatter(x=d["quarter"], y=d["idx"], name=f"{zone}: гедонический", mode="lines+markers",
                                 line=dict(color=c, width=2), marker=dict(size=6),
                                 hovertemplate="%{y:.3f}<extra>" + zone + ", гедонический</extra>"))
        fig.add_trace(go.Scatter(x=d["quarter"], y=d["naive"], name=f"{zone}: медиана", mode="lines",
                                 line=dict(color=c, width=1.5, dash="dot"),
                                 hovertemplate="%{y:.3f}<extra>" + zone + ", медиана</extra>"))
    fig.add_vline(x="2022Q3", line=dict(color=MUTED, width=1, dash="dash"))
    fig.add_annotation(x="2022Q3", y=1.06, text="ЕЦБ начинает повышать ставку (21.07.2022)", showarrow=False,
                       xanchor="left", xshift=4, font=dict(size=11, color=INK2))
    _zero(fig, 1.0)
    base(fig, "Индекс цены квартир за м², 2021Q1 = 1 (полоса — 95 % ДИ)", h=420, y="индекс", hover="x unified")
    return fig


# ----------------------------------------------------------------- объёмы
def volumes(q: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for zone, d in q.groupby("zone"):
        fig.add_trace(go.Scatter(x=d["quarter"], y=d["n"], name=zone, mode="lines+markers",
                                 line=dict(color=ZONE[zone], width=2), marker=dict(size=6),
                                 hovertemplate="%{y:,} сделок<extra>" + zone + "</extra>"))
    base(fig, "Число продаж жилья по кварталам", h=360, y="сделок", hover="x unified")
    fig.update_yaxes(rangemode="tozero", tickformat=",.0f")
    return fig


def volume_vs_price(c: pd.DataFrame, rho: float, p: float) -> go.Figure:
    fig = go.Figure()
    for paris, d in c.groupby("paris"):
        zone = "Париж" if paris else "пригород"
        fig.add_trace(go.Scatter(
            x=d["vol"], y=d["price"], mode="markers", name=("округа Парижа" if paris else "коммуны пригорода"),
            text=d["commune"], marker=dict(size=9, color=ZONE[zone], line=dict(color=SURFACE, width=1.5)),
            hovertemplate="<b>%{text}</b><br>сделок %{x:+.0%}<br>цена %{y:+.1%}<extra></extra>"))
    _zero(fig)
    base(fig, f"Коммуны: падение объёма и цены, 2023–24 к 2021–22 (вторичка; ρ = {rho:.2f}, p = {p:.3f})"
         .replace(".", ","), h=420, x="изменение числа сделок", y="изменение цены (в той же ячейке 300 м)",
         yfmt="+.0%")
    fig.update_xaxes(tickformat="+.0%")
    return fig


# ----------------------------------------------------------------- пончик
def donut_gap(g: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    _band(fig, g["quarter"], g["lo"], g["hi"], SUB, "ДИ")
    fig.add_trace(go.Scatter(x=g["quarter"], y=g["gap"], mode="lines+markers", line=dict(color=SUB, width=2),
                             marker=dict(size=6), name="пригород / Париж",
                             hovertemplate="%{y:.3f}<extra>пригород / Париж</extra>"))
    _zero(fig, 1.0)
    base(fig, "Цена пригорода относительно Парижа, 2021Q1 = 1 (те же ячейки и характеристики)", h=360,
         y="отношение индексов", hover="x unified")
    return fig


def distance_profile(prof: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for y, d in prof.groupby("year"):
        fig.add_trace(go.Scatter(x=d["km"], y=d["med"], name=str(y), mode="lines+markers",
                                 line=dict(color=YEARS[y], width=2), marker=dict(size=5),
                                 hovertemplate="%{y:,.0f} €/м²<extra>" + str(y) + "</extra>"))
    fig.add_vrect(x0=4, x1=6, fillcolor=rgba(MUTED, 0.10), line_width=0)
    fig.add_annotation(x=5, y=0.02, yref="paper", text="граница<br>Парижа", showarrow=False,
                       font=dict(size=11, color=INK2), yanchor="bottom")
    base(fig, "Медиана €/м² квартир по расстоянию от Нотр-Дама", h=400, x="км от центра", y="€/м²",
         hover="x unified")
    fig.update_yaxes(tickformat=",.0f")
    return fig


def gradient(gr: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for zone, d in gr[gr["zone"] != "всё"].groupby("zone"):
        fig.add_trace(go.Scatter(
            x=d["year"], y=d["slope"], name=zone, mode="lines+markers", line=dict(color=ZONE[zone], width=2),
            marker=dict(size=8), error_y=dict(type="data", symmetric=False, array=d["hi"] - d["slope"],
                                              arrayminus=d["slope"] - d["lo"], color=ZONE[zone], thickness=1.2),
            hovertemplate="%{y:.1%} на км<extra>" + zone + "</extra>"))
    base(fig, "Градиент цены: % на каждый км от центра (95 % ДИ, кластеры — коммуны)", h=360,
         y="% цены на км", yfmt=".0%", hover="x unified")
    fig.update_xaxes(dtick=1)
    return fig


# ----------------------------------------------------------------- размер
def size_premium(t: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Париж", "пригород"], shared_yaxes=True,
                        horizontal_spacing=0.06)
    for i, zone in enumerate(["Париж", "пригород"], start=1):
        d = t[t["zone"] == zone]
        c = ZONE[zone]
        fig.add_trace(go.Scatter(x=d["bin"], y=d["raw"], mode="lines+markers", name="медианы как есть",
                                 line=dict(color=MUTED, width=1.5, dash="dot"), marker=dict(size=7, color=MUTED),
                                 showlegend=i == 1, legendgroup="raw",
                                 hovertemplate="%{y:+.1%}<extra>как есть</extra>"), row=1, col=i)
        fig.add_trace(go.Scatter(
            x=d["bin"], y=d["within"], mode="lines+markers", name=f"{zone}: в той же ячейке 300 м",
            line=dict(color=c, width=2), marker=dict(size=8),
            error_y=dict(type="data", symmetric=False, array=d["hi"] - d["within"], arrayminus=d["within"] - d["lo"],
                         color=c, thickness=1.2),
            legendgroup=zone, hovertemplate="%{y:+.1%}<extra>в той же ячейке</extra>"),
            row=1, col=i)
        fig.add_hline(y=0, line=dict(color=AXIS, width=1), row=1, col=i)
    base(fig, "Цена за м² относительно студий 9–20 м²", h=420, y="разница с 9–20 м²", yfmt="+.0%",
         hover="x unified")
    fig.update_xaxes(title="площадь, м²")
    fig.for_each_annotation(lambda a: a.update(font=dict(size=13, color=INK2)))
    fig.update_layout(margin=dict(t=110), legend=dict(y=1.1))
    return fig


def new_build(nb: dict) -> go.Figure:
    dep_ru = {"92": "О-де-Сен (92)", "93": "Сен-Сен-Дени (93)", "94": "Валь-де-Марн (94)"}
    rows = [("медианы как есть", nb["raw"], nb["raw"], nb["raw"], MUTED),
            ("в той же ячейке 300 м", *nb["within"], SUB)]
    rows += [(f"{dep_ru[k]}", v[0], v[1], v[2], SUB) for k, v in nb["by_dep"].items()]
    lab = [r[0] for r in rows][::-1]
    fig = go.Figure()
    for (l, m, lo, hi, c) in rows[::-1]:
        fig.add_trace(go.Scatter(x=[m], y=[l], mode="markers", marker=dict(size=11, color=c),
                                 error_x=dict(type="data", symmetric=False, array=[hi - m], arrayminus=[m - lo],
                                              color=c, thickness=1.5, width=4),
                                 showlegend=False, hovertemplate=f"{l}: %{{x:+.1%}}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1))
    base(fig, "Премия новостройки к вторичке за м², пригород (95 % ДИ; департаменты — в той же ячейке)", h=320,
         x="премия", yfmt=None)
    fig.update_layout(margin=dict(l=170))
    fig.update_xaxes(tickformat="+.0%", showgrid=True, gridcolor=GRID, range=[-0.03, 0.35])
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=lab, tickfont=dict(color=INK2),
                     automargin=True)
    return fig


# ----------------------------------------------------------------- Олимпиада
def olympics_event(ev: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    _band(fig, ev["year"], ev["lo"], ev["hi"], MUTED, "95 % плацебо-наборов", legend=True)
    fig.add_trace(go.Scatter(x=ev["year"], y=ev["effect"], mode="lines+markers", line=dict(color=THIRD, width=2),
                             marker=dict(size=9), customdata=ev["p"], name="олимпийские коммуны",
                             hovertemplate="%{y:+.1%}, перестановочный p = %{customdata:.2f}<extra></extra>"))
    _zero(fig)
    base(fig, "Олимпийские коммуны минус остальные коммуны 93, к 2021 году (Игры — лето 2024)", h=360,
         y="разность изменений", yfmt="+.0%", hover="x unified")
    fig.update_xaxes(dtick=1)
    return fig


def olympics_placebo(placebo: np.ndarray, obs: float, p: float) -> go.Figure:
    h, e = np.histogram(placebo, bins=60)
    fig = go.Figure(go.Bar(x=(e[:-1] + e[1:]) / 2, y=h, marker=dict(color=AXIS), width=np.diff(e) * 0.85,
                           hovertemplate="%{x:+.1%}: %{y} наборов<extra></extra>", name="плацебо"))
    fig.add_vline(x=obs, line=dict(color=THIRD, width=2.5))
    fig.add_annotation(x=obs, y=1, yref="paper", text=f"Олимпийские коммуны: {obs:+.1%}, p = {p:.2f}"
                       .replace(".", ","), showarrow=False, xanchor="left", xshift=6, yanchor="top",
                       font=dict(size=12, color=INK))
    base(fig, f"{len(placebo):,} «плацебо-олимпиад»: все наборы из 3 коммун 93 (2024 к 2021)".replace(",", " "),
         h=340, x="разность изменений", y="наборов")
    fig.update_xaxes(tickformat="+.0%")
    return fig


# ----------------------------------------------------------------- круглые цены
def price_hist(h: pd.DataFrame, lo: int = 250_000, hi: int = 350_000) -> go.Figure:
    d = h[(h["price"] >= lo) & (h["price"] < hi)]
    rnd = d["price"] % 10_000 == 0
    fig = go.Figure()
    for flag, name, c in [(False, "прочие тысячи", BLUES[2]), (True, "кратные 10 000 €", SUB)]:
        x = d[rnd == flag]
        fig.add_trace(go.Bar(x=x["price"], y=x["n"], name=name, marker=dict(color=c, cornerradius=2), width=800,
                             hovertemplate="%{x:,.0f} €: %{y} сделок<extra></extra>"))
    fig.add_vline(x=300_000, line=dict(color=INK2, width=1, dash="dot"))
    base(fig, "Сколько сделок по каждой цене (шаг 1 000 €)", h=380, x="цена сделки, €", y="сделок")
    fig.update_xaxes(tickformat=",.0f")
    fig.update_layout(barmode="overlay", bargap=0)
    return fig


def threshold_ratio(r100: pd.DataFrame, r10: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r10["x"], y=np.expm1(r10["lr"]), mode="markers", name="прочие пороги кратные 10 000",
                             marker=dict(size=7, color=AXIS), hovertemplate="%{x:,.0f} €: %{y:+.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=r100["x"], y=np.expm1(r100["lr"]), mode="markers", name="круглые: 200 000 … 1 000 000",
                             marker=dict(size=11, color=PARIS, line=dict(color=SURFACE, width=2)),
                             hovertemplate="%{x:,.0f} €: %{y:+.1%}<extra></extra>"))
    _zero(fig)
    base(fig, "Перевес сделок «чуть ниже» порога: [x − 10 000; x) против (x; x + 10 000]", h=380,
         x="порог x, €", y="ниже / выше − 1", yfmt="+.0%")
    fig.update_xaxes(tickformat=",.0f")
    return fig


# ----------------------------------------------------------------- сезонность
MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]


def season_volume(per_day: pd.Series) -> go.Figure:
    fig = go.Figure(go.Bar(x=MONTHS, y=per_day.to_numpy(), marker=dict(color=PARIS, cornerradius=4),
                           hovertemplate="%{x}: %{y:.2f} от среднего<extra></extra>"))
    _zero(fig, 1.0)
    base(fig, "Сделок в день по месяцам подписания, среднее = 1", h=340, y="к среднему")
    return fig


def season_price(sea: pd.DataFrame, spread: pd.DataFrame, naive: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for y, d in sea.groupby("year"):
        fig.add_trace(go.Scatter(x=[MONTHS[m - 1] for m in d["month"]], y=d["eff"], mode="markers", name=str(y),
                                 marker=dict(size=7, color=YEARS[int(y)], opacity=0.8),
                                 hovertemplate="%{y:+.1%}<extra>" + str(y) + "</extra>"))
    fig.add_trace(go.Scatter(x=MONTHS, y=spread["mean"], mode="lines", name="среднее (без тренда)",
                             line=dict(color=INK, width=2), hovertemplate="%{y:+.1%}<extra>среднее</extra>"))
    fig.add_trace(go.Scatter(x=MONTHS, y=naive["eff"], mode="lines", name="наивно: дамми месяца + год",
                             line=dict(color=SUB, width=1.5, dash="dot"),
                             hovertemplate="%{y:+.1%}<extra>наивно</extra>"))
    _zero(fig)
    fig.update_xaxes(categoryorder="array", categoryarray=MONTHS)
    base(fig, "Сезонность цены за м²: отклонение от тренда", h=380, y="к тренду", yfmt="+.1%", hover="x unified")
    return fig
