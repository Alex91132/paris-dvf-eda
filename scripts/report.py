"""Интерактивный отчёт → docs/index.html (GitHub Pages) и docs/report.json (ключевые числа).

    python scripts/report.py

Все числа в тексте считаются из данных при сборке: поменяются данные — поменяются выводы.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", category=FutureWarning)

from dvf import analysis as A  # noqa: E402
from dvf import figures as F   # noqa: E402
from dvf.clean import load     # noqa: E402

NB = " "


def fmt(x, d=0) -> str:
    return f"{x:,.{d}f}".replace(",", NB).replace(".", ",").replace("-", "−")


def pct(x, d=1, sign=True) -> str:
    s = f"{100 * x:+.{d}f}" if sign else f"{100 * x:.{d}f}"
    return s.replace(".", ",").replace("-", "−") + NB + "%"


def ci(lo, hi, d=1) -> str:
    return f"[{pct(lo, d)}; {pct(hi, d)}]"


def pval(p) -> str:
    return "p &lt; 0,001" if p < 0.001 else f"p = {p:.3f}".replace(".", ",")


def main() -> None:
    s, steps = load()
    R = A.run_all(s)
    ov, idx, vol, dn, nb, oly, rd, se = (R[k] for k in ("overview", "index", "volumes", "donut", "new", "oly",
                                                      "round", "season"))
    loc, arr, mix, sz = R["location"], R["arr"], R["mix"], R["size"]
    P, S = idx["summary"]["Париж"], idx["summary"]["пригород"]
    rho_arr, p_arr = R["arr_rho"]
    grad = dn["gradient"].set_index(["zone", "year"])
    m = mix.set_index(["zone", "year"])
    szi = sz.set_index(["zone", "bin"])
    naive = se["naive"].assign(eff=lambda d: d["eff"] - d["eff"].mean())

    figs: dict[str, object] = {
        "funnel": F.funnel(steps), "map": F.price_map(R["map"]["cells"]), "change": F.change_map(R["map"]["change"]),
        "arr": F.arrondissements(arr), "arr2": F.arr_level_change(arr, rho_arr),
        "index": F.hedonic_index(idx["table"]), "vol": F.volumes(vol["quarterly"]),
        "vol2": F.volume_vs_price(vol["communes"], vol["rho"], vol["p"]), "gap": F.donut_gap(dn["gap"]),
        "prof": F.distance_profile(dn["profile"]), "grad": F.gradient(dn["gradient"]), "size": F.size_premium(sz),
        "new": F.new_build(nb), "oly": F.olympics_event(oly["event"]),
        "oly2": F.olympics_placebo(oly["placebo"], oly["effect"], oly["p"]), "hist": F.price_hist(rd["hist"]),
        "thr": F.threshold_ratio(rd["r100"], rd["r10"]), "sv": F.season_volume(se["per_day"]),
        "sp": F.season_price(se["season"], se["spread"], naive)}

    def chart(key: str) -> str:
        return '<div class="chart">' + figs[key].to_html(full_html=False, include_plotlyjs=False, config=F.CONFIG,
                                                          div_id=f"fig-{key}") + "</div>"

    nd = float(se["naive"].set_index("month").loc[[11, 12], "eff"].mean())
    pk_sum = max(se["spread"]["mean"]) - min(se["spread"]["mean"])
    jul_aug = se["spread"].set_index("month").loc[[7, 8], "mean"].mean()
    vc, vpk, kc = vol["vol_change"], vol["vol_peak_change"], vol["kind_change"]
    paris_arr = arr.set_index("arr")

    facts = {
        "built": dt.date.today().isoformat(), "sales": ov["n"], "flats": ov["n_flats"], "communes": ov["n_communes"],
        "median_ppm2": {"paris": ov["med_paris"], "suburbs": ov["med_sub"]},
        "index": {z: {k: (float(v) if not isinstance(v, str) else v) for k, v in d.items()}
                  for z, d in idx["summary"].items()},
        "volume_change_2024_vs_2021": vc, "volume_change_by_kind": kc,
        "commune_volume_price_spearman": [vol["rho"], vol["p"]],
        "suburb_vs_paris_gap": {"peak_q": dn["peak_q"], "peak": dn["peak"], "last": dn["last"]["gap"]},
        "arr_level_change_spearman": [rho_arr, p_arr], "location_r2": loc,
        "new_build_premium": {"raw": nb["raw"], "within_cell": list(nb["within"])},
        "olympics": {"effect": oly["effect"], "perm_p": oly["p"], "cluster_ci": list(oly["cluster"])},
        "round_prices": {"shares": rd["shares"], "excess_below_100k": rd["excess100"],
                         "excess_below_10k": rd["excess10"], "p": rd["p"]},
        "seasonality": {"chi2": se["chi2"], "price_f": se["f"], "price_f_p": se["f_p"], "jul_aug": jul_aug}}

    sections = []

    def sec(id_: str, title: str, body: str):
        sections.append((id_, title, body))

    # ------------------------------------------------------------------ 1. данные
    sec("data", "1. Данные: что такое одна сделка", f"""
<p>DVF — реестр всех нотариальных сделок с недвижимостью во Франции. Строка в нём — не сделка, а
<em>сделка × участок × помещение × вид угодий</em>: квартира с парковкой и погребом — три строки, и в каждой
повторена <strong>полная</strong> цена сделки. Поэтому наивное «цена / площадь» по строкам даёт мусор.
Правило очистки: в сделке ровно одно жильё (квартира или дом), нет коммерческих помещений и цена одна.
Подсобные помещения (парковки, погреба) остаются — они продаются вместе с квартирой и входят в цену.</p>
{chart("funnel")}
<p>Дальше — границы правдоподобия (площадь 9–400 м², 1 500–40 000 €/м², координаты не дальше 25 км от центра:
есть геокодирование в Арктику) и робастный фильтр выбросов внутри коммуны и типа жилья: |z| ≤ 4 по медианному
абсолютному отклонению. Осталось <strong>{fmt(ov["n"])}</strong> продаж в {ov["n_communes"]} коммунах
(Париж считается по 20 округам), из них {fmt(ov["n_flats"])} — квартиры. Дальше анализ — по квартирам: дома
есть почти только в пригороде, и смешивать их — сравнивать разные рынки.</p>""")

    # ------------------------------------------------------------------ 2. карта
    best, worst = paris_arr["med"].idxmax(), paris_arr["med"].idxmin()
    sec("map", "2. Карта цен: локация решает почти всё", f"""
<div class="tiles">
 <div class="tile"><b>{pct(loc["commune"], 0, False)}</b><span>разброса log(€/м²) объясняет одна коммуна или округ</span></div>
 <div class="tile"><b>{pct(loc["cell"], 0, False)}</b><span>— ячейка 300 × 300 м ({fmt(loc["n_cells"])} ячеек)</span></div>
 <div class="tile"><b>{pct(loc["cell_attrs"], 0, False)}</b><span>ячейка + площадь, комнаты, новостройка, квартал</span></div>
</div>
<p>Площадь, комнаты и время добавляют к ячейке всего {fmt(100 * (loc["cell_attrs"] - loc["cell"]), 1)} п.п.:
цена квадратного метра — это прежде всего адрес. Поэтому дальше каждое сравнение делается <em>внутри</em>
ячейки 300 м (фиксированный эффект локации), иначе любые различия тонут в разнице районов.</p>
{chart("map")}
<p>Самый дорогой округ — {best}-й ({fmt(paris_arr.loc[best, "med"])} €/м²), самый доступный — {worst}-й
({fmt(paris_arr.loc[worst, "med"])} €/м²). Граница Парижа видна на карте как обрыв цвета. На одном и том же
расстоянии от центра (4–7 км) медиана в Париже {fmt(dn["border_med"]["Париж"])} €/м², за кольцевой дорогой
(périphérique) — {fmt(dn["border_med"]["пригород"])}; с поправкой на расстояние, площадь и время Париж дороже на
<strong>{pct(dn["border"][0], 0)}</strong> (95 % ДИ {ci(dn["border"][1], dn["border"][2], 0)}, кластеры — коммуны).
Административная граница стоит дороже километров.</p>
{chart("arr")}
{chart("change")}
<p>В {pct(R["map"]["share_down"], 0, False)} ячеек медиана 2023–24 ниже, чем 2021–22. Но падали округа
по-разному: <strong>чем дороже округ, тем меньше он потерял</strong> — ранговая корреляция уровня и изменения
ρ = {rho_arr:.2f} ({pval(p_arr)}, 20 округов). Центр (4-й, 6-й: {pct(paris_arr.loc[4, "change"], 0)},
{pct(paris_arr.loc[6, "change"], 0)}) почти устоял, северо-восток (18-й, 19-й:
{pct(paris_arr.loc[18, "change"], 0)}, {pct(paris_arr.loc[19, "change"], 0)}) потерял больше всего.</p>
{chart("arr2")}""".replace("ρ = 0.", "ρ = 0,"))

    # ------------------------------------------------------------------ 3. индекс
    sec("rates", "3. Гипотеза: после подъёма ставок цены упали", f"""
<p><strong>Проблема наивной медианы.</strong> Если в 2024 году продают больше студий или больше квартир в
дешёвых районах, медиана €/м² сдвинется без всякого изменения цен. Поэтому основной показатель —
<em>гедонический индекс</em>: регрессия log(€/м²) на дамми кварталов при фиксированной ячейке 300 м,
площади (log и квадрат), числе комнат и признаке новостройки. Коэффициент квартала — изменение цены
«той же квартиры в том же месте». Ошибки — кластерные по ячейкам.</p>
{chart("index")}
<div class="check"><b>Результат.</b> Париж: пик — {P["peak_q"]}, к 2024Q4 индекс
{fmt(P["last"], 3)} (95 % ДИ {fmt(P["last_lo"], 3)}–{fmt(P["last_hi"], 3)}), <strong>{pct(P["from_peak"])} от пика</strong>.
Пригород: рос до {S["peak_q"]} (+{fmt(100 * (S["peak"] - 1), 1)}{NB}%), затем
<strong>{pct(S["from_peak"])} от пика</strong>. Пригород развернулся на год позже Парижа — с началом
повышения ставок ЕЦБ.</div>
<p><strong>Проверка составом.</strong> Наивная медиана здесь почти совпала с гедоническим индексом
(Париж {fmt(P["naive_last"], 3)} против {fmt(P["last"], 3)}). Состав сделок менялся, но слабо: доля
квартир до 30 м² в Париже {pct(m.loc[("Париж", 2021), "small"], 0, False)} → {pct(m.loc[("Париж", 2024), "small"], 0, False)},
доля новостроек в пригороде {pct(m.loc[("пригород", 2021), "new"], 1, False)} → {pct(m.loc[("пригород", 2024), "new"], 1, False)}.
Сдвиги разнонаправленные и почти гасят друг друга. Вывод «цены упали» устойчив, но это мы проверили, а не
предположили.</p>""")

    # ------------------------------------------------------------------ 4. объёмы
    sec("volume", "4. Гипотеза: объём падает сильнее цены", f"""
<p>Продавцы неохотно снижают цену («липкие цены»), поэтому рынок охлаждается в первую очередь числом сделок.</p>
{chart("vol")}
<div class="check"><b>Результат.</b> Число продаж 2024 к 2021: Париж <strong>{pct(vc["Париж"], 0)}</strong>
(от пика 2022 — {pct(vpk["Париж"], 0)}), пригород <strong>{pct(vc["пригород"], 0)}</strong> — при падении цены
от пика на {fmt(-100 * S["from_peak"], 0)}–{fmt(-100 * P["from_peak"], 0)}{NB}%. Объём упал в 2–3 раза сильнее цены. В пригороде на это наложился обвал новостроек:
вторичка {pct(kc["вторичка"], 0)}, новостройки {pct(kc["новостройки"], 0)} (стройка, а не только спрос).</div>
<p>Если цены «липкие», то там, где продавцы держали цену, сделок должно стать меньше. По коммунам (только
вторичка; цена — средний сдвиг остатков гедонической модели внутри ячейки) связь действительно
отрицательная: <strong>ρ = {fmt(vol["rho"], 2)}</strong> ({pval(vol["p"])}, {len(vol["communes"])} коммун с ≥ 60
сделками в каждом периоде). Где цена падала меньше, сильнее падал объём. Связь умеренная и наблюдательная:
это согласуется с липкостью цен, но не доказывает механизм.</p>
{chart("vol2")}""")

    # ------------------------------------------------------------------ 5. пончик
    g21, g24 = grad.loc[("Париж", 2021)], grad.loc[("Париж", 2024)]
    g_hike = float(dn["gap"].set_index("quarter").loc["2022Q3", "gap"])
    sec("donut", "5. Гипотеза «пончика»: пригород выиграл у Парижа", f"""
<p>После пандемии во многих мегаполисах спрос сместился из центра на окраины — за площадью и зеленью
(«donut effect»). Сравниваем Париж и пригород <em>одной</em> регрессией: дамми кварталов и дамми
«квартал × пригород» при фиксированных ячейках и характеристиках.</p>
{chart("gap")}
<div class="check"><b>Результат.</b> Пригород подорожал относительно Парижа на
<strong>{pct(dn["peak"] - 1)}</strong> к {dn["peak_q"]} (95 % ДИ {ci(dn["peak_lo"] - 1, dn["peak_hi"] - 1)}) и
удержал разрыв до конца 2024 ({pct(dn["last"]["gap"] - 1)}). Эффект пончика есть, но небольшой, и набран
он в основном до ставок ({pct(g_hike - 1)} к 2022Q3); с 2023 года Париж и пригород падают вместе.</div>
{chart("prof")}
<p>Внутри Парижа — обратная картина: центр держится, окраинные округа падают сильнее (раздел 2). Градиент
«% цены на км от центра» в Париже: {pct(g21["slope"])} в 2021 и {pct(g24["slope"])} в 2024. Направление
согласуется с ранговой корреляцией по округам, но интервалы по кластерам-коммунам (всего 20 округов)
перекрываются ({ci(g21["lo"], g21["hi"])} и {ci(g24["lo"], g24["hi"])}) — как самостоятельное доказательство
градиент не годится. Рынок «бежит в качество»: дорогие адреса Парижа и просторный пригород теряют меньше,
дешёвый Париж — больше.</p>
{chart("grad")}""")

    # ------------------------------------------------------------------ 6. размер
    sec("size", "6. Гипотеза: маленькие квартиры дороже за м²", f"""
<p>Популярное правило: метр в студии дороже, чем в большой квартире. Но большие квартиры в Париже стоят в
дорогих округах, а в пригороде — в дорогих коммунах, так что сравнение медиан смешивает размер с адресом.</p>
{chart("size")}
<div class="check"><b>Результат.</b> <strong>Пригород</strong> — правило верно: в той же ячейке 60–80 м² на
{pct(-szi.loc[("пригород", "60–80"), "within"], 0, False)} дешевле студии за м² (медианы «как есть» преувеличивают
до {pct(-szi.loc[("пригород", "60–80"), "raw"], 0, False)}). А для квартир от 120 м² сравнение «как есть»
переворачивает знак: медианы показывают <em>премию</em> {pct(szi.loc[("пригород", "120+"), "raw"], 0)}, хотя в
той же ячейке такие квартиры на {pct(-szi.loc[("пригород", "120+"), "within"], 0, False)} дешевле студии —
большие квартиры сосредоточены в дорогих коммунах вроде Нёйи. Парадокс Симпсона в чистом виде.
<strong>Париж</strong> — правило почти не работает: разница внутри ячейки в пределах ±{fmt(100 * szi.loc["Париж", "within"].abs().max(), 0)}{NB}%,
а большие квартиры (от 80 м²) даже дороже за метр: {pct(szi.loc[("Париж", "120+"), "within"], 1)} — в
парижских кварталах это квартиры в лучших домах. Медианы «как есть» завышают эту премию в
{fmt(szi.loc[("Париж", "120+"), "raw"] / szi.loc[("Париж", "120+"), "within"], 0)} раза.</div>""")

    # ------------------------------------------------------------------ 7. новостройки
    lo, hi = nb["within"][1], nb["within"][2]
    sec("new", "7. Сколько стоит новизна", f"""
<p>В пригороде новостройки (VEFA — продажа «на стадии строительства») — {pct(nb["share"], 0, False)} сделок
с квартирами ({fmt(nb["n_new"])}). В Париже их почти нет (0,2{NB}%), поэтому сравнение — только по пригороду.</p>
{chart("new")}
<div class="check"><b>Результат.</b> Медианы «как есть»: новостройка дороже вторички всего на
{pct(nb["raw"])}. В той же ячейке 300 м при той же площади и комнатах премия —
<strong>{pct(nb["within"][0])}</strong> (95 % ДИ {ci(lo, hi)}), и она устойчива по всем трём департаментам.
Наивное сравнение занижает премию в {fmt(nb["within"][0] / nb["raw"], 0)} раз: новые дома строят там, где
дешевле земля, — в дешёвых районах.</div>""")

    # ------------------------------------------------------------------ 8. Олимпиада
    rt, rc = oly["raw_treated"], oly["raw_control"]
    sec("olympics", "8. Гипотеза: Олимпиада-2024 подняла цены у Олимпийской деревни", f"""
<p>Олимпийская деревня построена в Сен-Дени, Сент-Уане и Иль-Сен-Дени (департамент 93). Разность разностей:
изменение цены (остатки гедонической модели внутри ячейки) у этих трёх коммун минус у остальных
{oly["n_units"] - 3} коммун департамента с ≥ 30 сделками в каждом году.</p>
<p><strong>Ловушка с ошибками.</strong> В «экспериментальной» группе всего три коммуны, а кластерные
стандартные ошибки рассчитаны на много кластеров в каждой группе — при трёх они ненадёжны. Поэтому основной тест —
перестановочный: та же разность для <em>всех</em> {fmt(oly["n_placebo"])} наборов из трёх коммун 93.
Если Олимпиада что-то дала, наш набор должен выделяться среди плацебо.</p>
{chart("oly")}
{chart("oly2")}
<div class="check"><b>Результат: не подтвердилось.</b> Разность 2024 к 2021: {pct(oly["effect"])},
перестановочный {pval(oly["p"])} — таких «эффектов» полно среди случайных наборов коммун. Кластерный
интервал тоже накрывает ноль: {ci(*oly["cluster"][1:])}. Медианы: олимпийские коммуны
{fmt(rt[2021])} → {fmt(rt[2024])} €/м², остальные {fmt(rc[2021])} → {fmt(rc[2024])}. Вероятное объяснение —
Игры объявлены в 2017-м, и ожидания были в цене задолго до 2021 года, с которого начинаются наши данные.</div>""")

    # ------------------------------------------------------------------ 9. круглые цены
    sh = rd["shares"]
    sec("round", "9. Психология цены: круглые числа и «чуть ниже»", f"""
<p>Цены сделок — не непрерывная величина. {pct(sh[1000], 0, False)} цен кратны 1 000 €,
{pct(sh[10_000], 0, False)} — 10 000 €, {pct(sh[100_000], 1, False)} — 100 000 €.</p>
{chart("hist")}
<p><strong>Вторая гипотеза — «цена-приманка»:</strong> квартиру выставляют за 299 000, а не за 300 000, и
продают чуть ниже круглого порога. Но сравнивать «ниже/выше порога» в лоб нельзя: плотность сделок выше ~300
тыс. € падает с ценой, и «ниже» всегда больше. Контроль — те же окна у некруглых порогов (210 000, 220 000 …),
где плотность падает так же.</p>
{chart("thr")}
<div class="check"><b>Результат.</b> У круглых порогов (кратных 100 000) сделок в окне «на 10 000 ниже»
больше, чем «на 10 000 выше», в среднем на <strong>{pct(rd["excess100"])}</strong>; у прочих порогов,
кратных 10 000, — на {pct(rd["excess10"])} (Манн — Уитни, односторонний, {pval(rd["p"])}). Перевес «чуть
ниже» у круглых чисел — не артефакт формы распределения. Исключение — 200 000, где плотность ещё растёт.</div>""")

    # ------------------------------------------------------------------ 10. сезонность
    pdv = se["per_day"]
    sec("season", "10. Сезонность: июль подписывают, август отдыхает", f"""
{chart("sv")}
<p>Число сделок в день в июле — {fmt(pdv[7], 2)} от среднего, в августе — {fmt(pdv[8], 2)} (χ² против
равномерного по дням = {fmt(se["chi2"])}, {pval(se["p"])}). Дата в DVF — подписание у нотариуса, обычно через
~3 месяца после предварительного договора: июльский пик — это весенние договорённости, август — отпуск.</p>
<p><strong>Ловушка с ценой.</strong> Наивная модель «дамми месяца + год» показывает, что в ноябре–декабре
метр дешевле января на {fmt(-100 * nd, 1)}{NB}%. Но в 2023–24 цены падали <em>внутри</em> года, и декабрь «дешевле» просто потому,
что он позже. Правильно — классическое разложение: помесячный гедонический индекс делим на центрированное
скользящее среднее 2×12 и смотрим отклонения от тренда.</p>
{chart("sp")}
<div class="check"><b>Результат.</b> После снятия тренда сезонность цены есть, но маленькая: размах
{fmt(100 * pk_sum, 1)}{NB}%, сделки июля–августа на {pct(jul_aug)} выше тренда (F = {fmt(se["f"], 1)},
{pval(se["f_p"])}; по три года на месяц). «Дешёвый декабрь» наивной модели — в основном тренд.</div>""")

    # ------------------------------------------------------------------ итог
    sec("lessons", "Чему учит этот EDA", f"""
<ol>
<li><strong>Сначала понять, что такое строка.</strong> В DVF цена повторяется по строкам сделки; без
агрегации до сделки «цена за м²» бессмысленна.</li>
<li><strong>Сравнивать внутри локации.</strong> Ячейка 300 м объясняет {pct(loc["cell"], 0, False)}
разброса цены. Без неё новостройки «дороже на {pct(nb["raw"], 0, False)}» вместо {pct(nb["within"][0], 0, False)},
а большие квартиры в пригороде «дороже» студий вместо «дешевле» — знак меняется.</li>
<li><strong>Проверять состав, даже если он не помешал.</strong> Наивная медиана цены здесь почти совпала с
гедоническим индексом — но это результат проверки, а не допущение.</li>
<li><strong>Тренд — не сезонность.</strong> Дамми месяцев при падающем рынке приписывают сезону падение цен.</li>
<li><strong>Мало «экспериментальных» единиц — перестановки.</strong> Три олимпийские коммуны — не основание для асимптотических
кластерных ошибок; точный перестановочный тест честнее и показал отсутствие эффекта.</li>
<li><strong>Нулевой результат — тоже результат.</strong> Олимпийского эффекта в 2021–2024 не видно, а пончик
оказался скромным (+{fmt(100 * (dn["peak"] - 1), 0)}{NB}%) и ограниченным 2021–22 годами.</li>
</ol>""")

    sec("method", "Данные, методика, ограничения", """
<ul>
<li><strong>Источник</strong>: «Demandes de valeurs foncières géolocalisées» (DGFiP, геокодирование Etalab),
<a href="https://files.data.gouv.fr/geo-dvf/">files.data.gouv.fr/geo-dvf</a>,
<a href="https://www.etalab.gouv.fr/licence-ouverte-open-licence/">Licence Ouverte 2.0</a>. Департаменты 75,
92, 93, 94; 2021–2024. В репозиторий данные не входят — их скачивает <code>scripts/download.py</code>.</li>
<li><strong>Площадь</strong> — surface_reelle_bati (кадастровая), не площадь Каррез. Этажа, состояния,
вида и года постройки в DVF нет — они уходят в остаток модели; ячейка 300 м частично их поглощает (тип застройки).</li>
<li><strong>Модель</strong>: МНК log(€/м²) с фиксированным эффектом ячейки 300 м (внутригрупповое
преобразование), кластерные ошибки по ячейкам; для градиента — по коммунам.</li>
<li><strong>Повторных продаж</strong> нет: DVF не даёт сквозного идентификатора квартиры, поэтому индекс —
гедонический, а не repeat-sales.</li>
<li>Дата сделки — подписание у нотариуса, цена договорена на 2–4 месяца раньше: все сдвиги во времени
запаздывают на квартал.</li>
</ul>""")

    toc = "".join(f'<a href="#{i}">{t}</a>' for i, t, _ in sections)
    body = "".join(f'<section id="{i}"><h2>{t}</h2>{b}</section>' for i, t, b in sections)
    html = f"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Рынок жилья Парижа</title>
<meta name="description" content="EDA 317 тыс. сделок с жильём в Париже и ближнем пригороде 2021–2024 (DVF): гедонический индекс, эффект ставок, пончик, новостройки, Олимпиада, круглые цены.">
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/plotly.js@2.35.2/dist/plotly-locale-ru.js"></script>
<style>
:root {{ color-scheme: light; --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --line: #c3c2b7; --paris: #2a78d6; --sub: #eb6834; --link: #1c5cab; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--page); color: var(--ink); font: 16px/1.6 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
main {{ max-width: 1040px; margin: 0 auto; padding: 32px 16px 72px; }}
h1 {{ font-size: clamp(26px, 4vw, 34px); line-height: 1.2; margin: 0 0 8px; letter-spacing: -0.01em; }}
h2 {{ font-size: 22px; line-height: 1.3; margin: 0 0 12px; }}
section {{ margin-top: 40px; padding-top: 28px; border-top: 1px solid var(--grid); }}
p, li {{ color: var(--ink); max-width: 76ch; }} .lead {{ color: var(--ink2); font-size: 17px; }}
a {{ color: var(--link); }} code {{ background: #efeee9; padding: 1px 5px; border-radius: 4px; font-size: 14px; }}
.chart {{ background: var(--surface); border: 1px solid rgba(11,11,11,0.08); border-radius: 10px; padding: 6px; margin: 16px 0; overflow: hidden; }}
.check {{ background: var(--surface); border-left: 3px solid var(--paris); border-radius: 6px; padding: 12px 16px; margin: 16px 0; max-width: 80ch; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 16px 0; }}
.tile {{ background: var(--surface); border: 1px solid rgba(11,11,11,0.08); border-radius: 10px; padding: 14px 16px; }}
.tile b {{ display: block; font-size: 28px; line-height: 1.15; font-weight: 650; }}
.tile span {{ color: var(--ink2); font-size: 14px; }}
nav.toc {{ display: flex; flex-wrap: wrap; gap: 6px 14px; margin: 20px 0 0; font-size: 14px; }}
nav.toc a {{ text-decoration: none; }} nav.toc a:hover {{ text-decoration: underline; }}
footer {{ margin-top: 48px; color: var(--muted); font-size: 13px; }}
.ctitle {{ font-size: 14px; font-weight: 600; color: var(--ink); padding: 6px 8px 0; }}
</style></head><body><main>
<h1>Рынок жилья Парижа и ближнего пригорода, 2021–2024</h1>
<p class="lead">Разведочный анализ {fmt(ov["n"])} нотариальных сделок из открытого реестра DVF: Париж и
департаменты 92, 93, 94. Анализ построен на гипотезах: каждая проверяется статистически — регрессией с
фиксированным эффектом локации, бутстрепом, перестановочным тестом. Числа в тексте пересчитываются из данных
при сборке отчёта ({dt.date.today().strftime("%d.%m.%Y")}). Код — <a href="https://github.com/Alex91132/paris-dvf-eda">GitHub</a>.</p>
<div class="tiles">
 <div class="tile"><b>{fmt(ov["med_paris"])} €</b><span>медиана за м² квартиры в Париже</span></div>
 <div class="tile"><b>{fmt(ov["med_sub"])} €</b><span>в пригороде (92, 93, 94)</span></div>
 <div class="tile"><b>{pct(P["from_peak"], 0)}</b><span>Париж от пика 2021Q3 (гедонический индекс)</span></div>
 <div class="tile"><b>{pct(vc["пригород"], 0)}</b><span>сделок в пригороде, 2024 к 2021 (Париж {pct(vc["Париж"], 0)})</span></div>
</div>
<nav class="toc">{toc}</nav>
{body}
<footer>Данные: DGFiP, Demandes de valeurs foncières; геокодирование — Etalab; Licence Ouverte 2.0.
Карта — © OpenStreetMap, © CARTO.</footer>
</main>
<script>
// Узкий экран: заголовок графика — HTML-строкой над ним (Plotly не переносит строки), легенда — под графиком.
window.addEventListener("load", () => {{
  if (window.innerWidth >= 640) return;
  document.querySelectorAll(".chart .js-plotly-plot").forEach(g => {{
    const t = g.layout && g.layout.title && g.layout.title.text;
    if (t) {{ const d = document.createElement("div"); d.className = "ctitle"; d.textContent = t;
             g.closest(".chart").prepend(d); }}
    const upd = {{"title.text": "", "margin.t": 16}};
    if (!g._fullLayout.map) Object.assign(upd, {{"legend.y": -0.32, "legend.yanchor": "top", "margin.b": 150,
                                                  "margin.l": 48}});
    Plotly.relayout(g, upd);
  }});
}});
</script>
</body></html>"""
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "report.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"docs/index.html: {len(html) / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
