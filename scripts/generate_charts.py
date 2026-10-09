#!/usr/bin/env python3
"""generate_charts.py

Genera i dati di dettaglio grafico (data/charts/{TICKER}.json + index.json)
per il sottoinsieme di strumenti in BUY, WATCHLIST (timing fresco / trend
maturo) o con flag ANTEPRIMA in data/etf_scores.json.

Logica indicatori (KAMA/SAR/AO/RSI/ER/Baffetti/segnale/Renko/ml_exit)
riportata IDENTICA da raptor-one/raptor_chart_fetch.py per restare
coerenti con lo standard grafico condiviso tra i repo. Il motore di
segnale qui dentro (BUY1/BUY2/BUY3/EXIT1/EXIT2/MEAN REV/WATCH) è quello
"nativo" del grafico standard — indipendente dallo Score/Segnale di
calculate_scores.py usato per lo screening dell'universo core (vocabolari
diversi per design, come da standard).

Esegue DOPO calculate_scores.py nel workflow (legge il suo output).
"""

import json
import math
import os
import sys
import time
import datetime
import urllib.parse

import yfinance as yf

# ── Modello ML per suggerimento uscita (allenato offline su raptor-one, solo inferenza qui) ──
try:
    import joblib
    _ML_EXIT = joblib.load(os.path.join(os.path.dirname(__file__), "models_exit.joblib"))
except Exception as _e:
    print(f"ATTENZIONE: modello ML uscita non caricato ({_e}) — suggerimento uscita disattivato")
    _ML_EXIT = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORES_PATH = os.path.join(BASE_DIR, "data", "etf_scores.json")
CHARTS_DIR = os.path.join(BASE_DIR, "data", "charts")
REGOLE_DIR = os.path.join(BASE_DIR, "regole")
REGOLE_TEMPLATE_PATH = os.path.join(BASE_DIR, "regole_template.html")
with open(REGOLE_TEMPLATE_PATH, encoding="utf-8") as _f:
    REGOLE_TEMPLATE = _f.read()

# Batching per limitare tempi/rate-limit — lista qui è molto più piccola
# dell'universo completo (solo BUY/WATCHLIST/ANTEPRIMA), non serve batch
# grande, ma manteniamo comunque una pausa per sicurezza.
SLEEP_BETWEEN_TICKERS = 0.3

# Segnali che qualificano un ticker per la generazione del grafico dettaglio
QUALIFYING_SIGNALS = {"BUY", "WATCHLIST — timing fresco", "WATCHLIST — trend maturo"}

# Filtro borsa: solo Borsa Italiana (.MI). Il modulo "Posizioni Aperte" copre
# solo questo segmento dell'universo core, per scelta esplicita — Xetra (.DE)
# resta escluso da questa pipeline (non dallo screening generale di core).
EXCHANGE_SUFFIX = ".MI"


def select_tickers():
    """Legge etf_scores.json e seleziona i ticker qualificati (BUY, WATCHLIST, ANTEPRIMA)
    limitati a Borsa Italiana (.MI). Porta con sé anche i campi anagrafici/di score
    già calcolati da calculate_scores.py (name, asset_class, adx, close) per evitare
    di ricalcolarli qui."""
    if not os.path.exists(SCORES_PATH):
        print(f"[ERROR] {SCORES_PATH} non trovato — esegui prima calculate_scores.py", file=sys.stderr)
        sys.exit(1)

    with open(SCORES_PATH, "r", encoding="utf-8") as f:
        scores_data = json.load(f)

    selected = []
    for item in scores_data.get("scores", []):
        signal = item.get("signal")
        anteprima = item.get("anteprima", False)
        ticker_yf = item.get("ticker_yf", "")
        if not ticker_yf.endswith(EXCHANGE_SUFFIX):
            continue
        if signal in QUALIFYING_SIGNALS or anteprima:
            selected.append({
                "y": ticker_yf,
                "t": ticker_yf.split(".")[0],
                "name": item.get("name"),
                "asset_class": item.get("asset_class"),
                "adx": item.get("adx"),
                "isin": item.get("isin"),
                "score": item.get("score_operativo"),
            })

    # Dedup su ticker yahoo
    seen = set()
    unique = []
    for t in selected:
        if t["y"] not in seen:
            seen.add(t["y"])
            unique.append(t)
    return unique


# ═══════════════════════════════════════════════════════
#  INDICATORI — riportati identici da raptor-one/raptor_chart_fetch.py
#  per restare coerenti con lo standard grafico condiviso tra i repo.
# ═══════════════════════════════════════════════════════

def calc_kama(close, n=10, fast=2, slow=30):
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    kama = [None] * len(close)
    if len(close) <= n:
        return kama
    kama[n] = close[n]
    for i in range(n + 1, len(close)):
        direction = abs(close[i] - close[i - n])
        volatility = sum(abs(close[j] - close[j - 1]) for j in range(i - n + 1, i + 1))
        er = direction / volatility if volatility != 0 else 0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    return kama


def calc_sar_array(high, low, af0=0.02, af_max=0.20):
    n = len(high)
    sar_arr = [None] * n
    bull_arr = [None] * n
    if n < 5:
        return sar_arr, bull_arr
    sar = low[0]
    ep = high[0]
    af = af0
    bull = True
    sar_arr[0] = round(sar, 4)
    bull_arr[0] = bull
    for i in range(1, n):
        if bull:
            new_sar = sar + af * (ep - sar)
            new_sar = min(new_sar, low[max(0, i - 1)], low[max(0, i - 2)])
            if low[i] < new_sar:
                bull = False
                new_sar = ep
                ep = low[i]
                af = af0
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af0, af_max)
        else:
            new_sar = sar + af * (ep - sar)
            new_sar = max(new_sar, high[max(0, i - 1)], high[max(0, i - 2)])
            if high[i] > new_sar:
                bull = True
                new_sar = ep
                ep = high[i]
                af = af0
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af0, af_max)
        sar = new_sar
        sar_arr[i] = round(sar, 4)
        bull_arr[i] = bull
    return sar_arr, bull_arr


def calc_ao_array(high, low):
    mid = [(h + l) / 2 for h, l in zip(high, low)]
    result = [None] * len(mid)
    for i in range(33, len(mid)):
        sma5 = sum(mid[i - 4:i + 1]) / 5
        sma34 = sum(mid[i - 33:i + 1]) / 34
        result[i] = round(sma5 - sma34, 4)
    return result


def calc_rsi_array(close, n=14):
    result = [None] * len(close)
    if len(close) < n + 2:
        return result
    for i in range(n, len(close)):
        gains = 0.0
        losses = 0.0
        for j in range(i - n + 1, i + 1):
            d = close[j] - close[j - 1]
            if d > 0:
                gains += d
            else:
                losses += -d
        ag = gains / n
        al = losses / n
        result[i] = round(100 - 100 / (1 + ag / al), 2) if al > 0 else 100.0
    return result


def calc_trend_array(close, kama, n=20):
    """Etichetta di trend SOLO INFORMATIVA (non modifica i segnali).
    R = KAMA in calo su n barre e prezzo sotto KAMA; U = KAMA in salita e prezzo sopra; L = altrimenti."""
    result = [None] * len(close)
    for i in range(n, len(close)):
        k, k0 = kama[i], kama[i - n]
        if k is None or k0 is None:
            continue
        result[i] = "R" if (k < k0 and close[i] < k) else "U" if (k > k0 and close[i] > k) else "L"
    return result


def calc_er_array(close, n=10):
    result = [0] * len(close)
    for i in range(n, len(close)):
        direction = abs(close[i] - close[i - n])
        volatility = sum(abs(close[j] - close[j - 1]) for j in range(i - n + 1, i + 1))
        result[i] = round(direction / volatility, 4) if volatility != 0 else 0
    return result


def calc_baffetti_array(high, low):
    """Barre consecutive con mid-price in salita."""
    mid = [(h + l) / 2 for h, l in zip(high, low)]
    result = [0] * len(mid)
    streak = 0
    for i in range(1, len(mid)):
        streak = streak + 1 if mid[i] > mid[i - 1] else 0
        result[i] = streak
    return result


def calc_mm_align_array(close):
    n = len(close)
    result = [False] * n
    cum = [0.0] * (n + 1)
    for i in range(n):
        cum[i + 1] = cum[i] + close[i]

    def avg(i, w):
        return (cum[i + 1] - cum[i + 1 - w]) / w if i + 1 >= w else None

    for i in range(n):
        mm20, mm50, mm100 = avg(i, 20), avg(i, 50), avg(i, 100)
        if mm20 is not None and mm50 is not None and mm100 is not None:
            result[i] = close[i] > mm20 > mm50 > mm100
    return result


def calc_cross_days_array(close, kama):
    n = len(close)
    result = [999] * n
    last_flip = None
    prev_above = None
    for i in range(n):
        if kama[i] is None:
            continue
        above = close[i] > kama[i]
        if prev_above is None:
            prev_above = above
            last_flip = i
            result[i] = 0
            continue
        if above != prev_above:
            last_flip = i
            prev_above = above
        result[i] = i - last_flip
    return result


def calc_ao_improving_array(ao):
    n = len(ao)
    result = [False] * n
    for i in range(1, n):
        if ao[i] is not None and ao[i - 1] is not None and ao[i] > ao[i - 1]:
            result[i] = True
    return result


def calc_segnale_array(close, kama, er_arr, baff_arr, ao_imp_arr, sar_bull_arr, cross_arr, mm_arr, rsi_arr):
    """Motore di segnale nativo del grafico (BUY1/BUY2/BUY3/EXIT1/EXIT2/MEAN REV/WATCH)."""
    n = len(close)
    result = [None] * n
    for i in range(n):
        if kama[i] is None or sar_bull_arr[i] is None:
            continue
        lk = kama[i]
        lc = close[i]
        above_kama = lc > lk if lk else False
        sar_bull = sar_bull_arr[i]
        cross = cross_arr[i]
        ao_imp = ao_imp_arr[i]
        baff = baff_arr[i]
        er = er_arr[i]
        mm_align = mm_arr[i]
        rsi = rsi_arr[i] if rsi_arr[i] is not None else 50
        if sar_bull and cross <= 3 and ao_imp:
            result[i] = "BUY1"
        elif above_kama and baff >= 2:
            result[i] = "BUY2"
        elif above_kama and er >= 0.50 and baff >= 3 and mm_align:
            result[i] = "BUY3"
        elif not above_kama and not sar_bull:
            result[i] = "EXIT2"
        elif not sar_bull:
            result[i] = "EXIT1"
        else:
            near_kama = abs(lc - lk) / lk < 0.03 if lk and lk > 0 else False
            if er < 0.30 and rsi < 30 and ao_imp and (near_kama or not above_kama):
                result[i] = "MEAN REV"
            else:
                result[i] = "WATCH"
    return result


# ═══════════════════════════════════════════════════════
#  RENKO — brick adattivo su ATR(14)
# ═══════════════════════════════════════════════════════

def calc_atr(high, low, close, n=14):
    trs = []
    for i in range(len(close)):
        if i == 0:
            trs.append(high[i] - low[i])
        else:
            trs.append(max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])))
    if len(trs) < n + 1:
        return None
    atr = sum(trs[1:n + 1]) / n
    for i in range(n + 1, len(trs)):
        atr = (atr * (n - 1) + trs[i]) / n
    return atr


def calc_renko(dates, close, brick_size):
    if not brick_size or brick_size <= 0 or len(close) < 2:
        return []
    bricks = []
    level = close[0]
    direction = 0
    for i in range(1, len(close)):
        price = close[i]
        d = dates[i] if i < len(dates) else None
        while True:
            if direction >= 0 and price >= level + brick_size:
                new_level = level + brick_size
                bricks.append({"o": round(level, 4), "c": round(new_level, 4), "dir": 1, "d": d})
                level = new_level
                direction = 1
                continue
            if direction <= 0 and price <= level - brick_size:
                new_level = level - brick_size
                bricks.append({"o": round(level, 4), "c": round(new_level, 4), "dir": -1, "d": d})
                level = new_level
                direction = -1
                continue
            if direction == 1 and price <= level - 2 * brick_size:
                new_level = level - brick_size
                bricks.append({"o": round(level, 4), "c": round(new_level, 4), "dir": -1, "d": d})
                level = new_level
                direction = -1
                continue
            if direction == -1 and price >= level + 2 * brick_size:
                new_level = level + brick_size
                bricks.append({"o": round(level, 4), "c": round(new_level, 4), "dir": 1, "d": d})
                level = new_level
                direction = 1
                continue
            break
    return bricks[-200:]


def calc_sar_streak_array(sarBull_arr):
    n = len(sarBull_arr)
    streak = [0] * n
    for i in range(1, n):
        if sarBull_arr[i] is None or sarBull_arr[i - 1] is None:
            continue
        streak[i] = streak[i - 1] + 1 if sarBull_arr[i] == sarBull_arr[i - 1] else 0
    return streak


def sanitize_nan(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_nan(v) for v in obj]
    return obj


def fmt(arr):
    return [round(v, 4) if isinstance(v, (int, float)) else v for v in arr]


# ═══════════════════════════════════════════════════════
#  TRADE SIMULATION — "Posizioni Aperte"
#  Porting adattato da chart/fetch.py (simulate_trades/perf_stats),
#  riscritto sul vocabolario nativo del motore core:
#  entrata: BUY1/BUY2/BUY3 — uscita: EXIT1/EXIT2.
#  WATCH e MEAN REV non aprono né chiudono (il concetto di
#  mean-reversion è già implicito nelle condizioni di BUY, come da
#  scelta esplicita — nessun motore MR separato).
# ═══════════════════════════════════════════════════════

ENTRY_SIGNALS = ("BUY1", "BUY2", "BUY3")
EXIT_SIGNALS = ("EXIT1", "EXIT2")

# Note a template (v1) — associate al codice segnale di entrata/uscita.
# Non leggono i valori esatti degli indicatori in quel giorno (versione
# "narrativa" dinamica, più costosa, valutabile in seguito): sono frasi
# fisse ma correttamente informative sul perché del segnale.
ENTRY_NOTES = {
    "BUY1": "Entrata: inversione SAR rialzista con momentum in miglioramento (AO) e incrocio KAMA recente.",
    "BUY2": "Entrata: prezzo sopra KAMA con pattern Baffetti forte.",
    "BUY3": "Entrata: prezzo sopra KAMA, Efficiency Ratio elevato e medie allineate (trend maturo).",
}
EXIT_NOTES = {
    "EXIT1": "Uscita: inversione SAR ribassista (perdita di trend).",
    "EXIT2": "Uscita: prezzo sotto KAMA e SAR ribassista (trend invertito).",
    "OPEN": "Posizione ancora aperta — nessun segnale di uscita ricevuto.",
}


def simulate_trades(dates, closes, segnale_arr):
    """Deriva i trade storici (chiusi + eventuale aperto) dalla sequenza segnali.
    Ogni trade: data/prezzo entrata e uscita, %var, giorni, segnali, nota."""
    trades = []
    in_trade = False
    ent_i = -1
    ent_sig = None
    n = len(closes)
    for i in range(n):
        sig = segnale_arr[i]
        if not in_trade:
            if sig in ENTRY_SIGNALS:
                in_trade = True
                ent_i = i
                ent_sig = sig
        else:
            is_last = (i == n - 1)
            if sig in EXIT_SIGNALS or is_last:
                exit_sig = sig if sig in EXIT_SIGNALS else "OPEN"
                is_open = exit_sig == "OPEN"
                pnl = round((closes[i] - closes[ent_i]) / closes[ent_i] * 100, 2) if closes[ent_i] else 0
                trades.append({
                    "dataEntrata": dates[ent_i], "dataUscita": None if is_open else dates[i],
                    "prezzoEntrata": closes[ent_i], "prezzoUscita": None if is_open else closes[i],
                    "entSig": ent_sig, "exitSig": exit_sig,
                    "pnlPct": pnl, "giorni": i - ent_i, "isOpen": is_open,
                    "note": ENTRY_NOTES.get(ent_sig, "") + (" " + EXIT_NOTES.get(exit_sig, "") if exit_sig in EXIT_NOTES else ""),
                })
                if not is_open:
                    in_trade = False
    return trades


def perf_stats(trades):
    closed = [t for t in trades if not t["isOpen"]]
    if not closed:
        return {"trades": len(trades), "closed": 0, "wins": 0, "wr": 0, "totalPnl": 0, "best": 0, "worst": 0, "avg": 0, "avgWin": 0, "avgLoss": 0, "dd": 0}
    wins = [t for t in closed if t["pnlPct"] > 0]
    losses = [t for t in closed if t["pnlPct"] <= 0]
    pnls = [t["pnlPct"] for t in closed]
    win_pnls = [t["pnlPct"] for t in wins]
    loss_pnls = [t["pnlPct"] for t in losses]
    total_pnl = sum(pnls)
    peak = eq = dd = 0
    for p in pnls:
        eq += p
        if eq > peak:
            peak = eq
        if peak - eq > dd:
            dd = peak - eq
    return {
        "trades": len(trades), "closed": len(closed), "wins": len(wins),
        "wr": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "totalPnl": round(total_pnl, 2),
        "best": round(max(pnls), 2), "worst": round(min(pnls), 2),
        "avg": round(total_pnl / len(closed), 2) if closed else 0,
        "avgWin": round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0,
        "avgLoss": round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0,
        "dd": round(dd, 2),
    }


# ═══════════════════════════════════════════════════════
#  PROCESS TICKER
# ═══════════════════════════════════════════════════════
def process_ticker(info):
    symbol = info["y"]
    try:
        tk = yf.Ticker(symbol)
        hist_d = tk.history(period="2y", interval="1d", timeout=20)
        if hist_d.empty or len(hist_d) < 60:
            return None

        opens = [round(float(x), 4) for x in hist_d["Open"].values]
        highs = [round(float(x), 4) for x in hist_d["High"].values]
        lows = [round(float(x), 4) for x in hist_d["Low"].values]
        closes = [round(float(x), 4) for x in hist_d["Close"].values]
        vols = [int(x) for x in hist_d["Volume"].values]
        dates = [ts.strftime("%Y-%m-%d") for ts in hist_d.index]
        ts_d = [int(ts.timestamp()) for ts in hist_d.index]
        d_bars = [[ts_d[i], opens[i], highs[i], lows[i], closes[i], vols[i]] for i in range(len(closes))]

        time.sleep(SLEEP_BETWEEN_TICKERS)
        h_bars = []
        try:
            hist_h = tk.history(period="5d", interval="1h", timeout=20)
            if not hist_h.empty:
                ho = [round(float(x), 4) for x in hist_h["Open"].values]
                hh = [round(float(x), 4) for x in hist_h["High"].values]
                hl = [round(float(x), 4) for x in hist_h["Low"].values]
                hc = [round(float(x), 4) for x in hist_h["Close"].values]
                hv = [int(x) for x in hist_h["Volume"].values]
                ht = [int(ts.timestamp()) for ts in hist_h.index]
                h_bars = [[ht[i], ho[i], hh[i], hl[i], hc[i], hv[i]] for i in range(len(hc))]
        except Exception:
            # Dati orari spesso lacunosi su ETP europei minori — non blocca il grafico daily
            pass

        kama_arr = calc_kama(closes)
        sar_arr, sarBull_arr = calc_sar_array(highs, lows)
        ao_arr = calc_ao_array(highs, lows)
        rsi_arr = calc_rsi_array(closes)
        rsi5_arr = calc_rsi_array(closes, n=5)
        er_arr = calc_er_array(closes)
        baff_arr = calc_baffetti_array(highs, lows)
        mm_arr = calc_mm_align_array(closes)
        cross_arr = calc_cross_days_array(closes, kama_arr)
        ao_imp_arr = calc_ao_improving_array(ao_arr)
        segnale_arr = calc_segnale_array(
            closes, kama_arr, er_arr, baff_arr, ao_imp_arr, sarBull_arr, cross_arr, mm_arr, rsi_arr
        )
        sarStreak_arr = calc_sar_streak_array(sarBull_arr)

        kama_h, sar_h, sarBull_h = [], [], []
        if len(h_bars) > 12:
            hc = [b[4] for b in h_bars]
            hh = [b[2] for b in h_bars]
            hl = [b[3] for b in h_bars]
            kama_h = calc_kama(hc)
            sar_h, sarBull_h = calc_sar_array(hh, hl)

        atr = calc_atr(highs, lows, closes, 14)
        brick = round(atr, 4) if atr else None
        renko = calc_renko(dates, closes, brick) if brick else []

        ml_exit = None
        last_seg = segnale_arr[-1] if segnale_arr else None
        if _ML_EXIT is not None and last_seg in ("BUY1", "BUY2"):
            try:
                i = len(closes) - 1
                vol_avg20 = sum(vols[max(0, i - 20):i]) / max(1, min(20, i)) if i > 0 else 1
                feat = {
                    "er": er_arr[i], "baff": baff_arr[i],
                    "rsi": rsi_arr[i] if rsi_arr[i] is not None else 50,
                    "ao": ao_arr[i] if ao_arr[i] is not None else 0,
                    "cross_days": cross_arr[i], "mm_align": int(mm_arr[i]),
                    "atr_pct": (atr / closes[i] * 100) if atr and closes[i] else 0,
                    "vol_ratio": (vols[i] / vol_avg20) if vol_avg20 else 1,
                    "tier_buy1": 1 if last_seg == "BUY1" else 0,
                }
                X = [[feat[f] for f in _ML_EXIT["features"]]]
                peak_pct = float(_ML_EXIT["reg_peak"].predict(X)[0])
                days_peak = float(_ML_EXIT["reg_days"].predict(X)[0])
                ml_exit = {"peak_return_pct": round(peak_pct, 2), "days_to_peak": round(days_peak, 1)}
            except Exception as e:
                print(f"  ATTENZIONE ML uscita {symbol}: {e}")

        trades = simulate_trades(dates, closes, segnale_arr)
        perf = perf_stats(trades)
        open_trade = next((t for t in trades if t["isOpen"]), None)

        result = {
            "ticker": info["t"], "yahoo": symbol,
            "name": info.get("name"), "asset_class": info.get("asset_class"), "isin": info.get("isin"),
            "d": d_bars, "h": h_bars,
            "kama_d": fmt(kama_arr), "sar_d": fmt(sar_arr), "sarBull_d": sarBull_arr,
            "ao_d": fmt(ao_arr), "rsi_d": fmt(rsi_arr), "rsi5_d": fmt(rsi5_arr), "baff_d": baff_arr,
            "segnale_d": segnale_arr,
            "er_d": fmt(er_arr), "trend_d": calc_trend_array(closes, kama_arr), "crossDays_d": cross_arr, "mmAlign_d": mm_arr,
            "sarStreak_d": sarStreak_arr,
            "kama_h": fmt(kama_h), "sar_h": fmt(sar_h), "sarBull_h": sarBull_h,
            "atr": round(atr, 4) if atr else None,
            "renko_brick": brick, "renko": renko,
            "ml_exit": ml_exit,
            "trades": trades[-30:], "perf": perf,
            "open_trade": open_trade,
            "aoImp": bool(ao_imp_arr[-1]) if ao_imp_arr else None,
        }
        return sanitize_nan(result)
    except Exception as e:
        print(f"  ERR {symbol}: {e}")
        return None




def _n(v, nd=2, suf=""):
    return "—" if v is None else f"{v:.{nd}f}{suf}"


def _ok(cond):
    return "—" if cond is None else ("✅" if cond else "❌")


def _tabella(rows):
    out = []
    for cond, soglia, valore, esito, signif in rows:
        cls = "" if esito is None else ("si" if esito else "no")
        out.append(f'<tr><td>{cond}</td><td>{soglia}</td><td class="val">{valore}</td>'
                   f'<td class="esito {cls}">{_ok(esito)}</td><td class="sig">{signif}</td></tr>')
    return ('<table><thead><tr><th>Condizione</th><th>Soglia</th><th>Valore oggi</th><th></th>'
            '<th>Significato</th></tr></thead><tbody>' + "".join(out) + '</tbody></table>')


def _riepilogo(rows):
    validi = [r[3] for r in rows if r[3] is not None]
    n_ok = sum(1 for v in validi if v)
    return f'<p class="riep">Condizioni soddisfatte: <strong>{n_ok}/{len(validi)}</strong></p>'


def build_regole_html(result: dict, info: dict) -> str:
    """Scheda regole per singolo ticker, costruita sul motore NATIVO del grafico
    (BUY1/BUY2/BUY3/EXIT1/EXIT2/MEAN REV/WATCH di calc_segnale_array), non sullo
    Score/Segnale di calculate_scores.py (vocabolario diverso per design)."""
    now = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    ticker = result["ticker"]
    yahoo = result["yahoo"]
    d_bars = result["d"]
    prezzo = d_bars[-1][4] if d_bars else None
    prezzo_prec = d_bars[-2][4] if len(d_bars) > 1 else None
    oggi_pct = ((prezzo / prezzo_prec - 1) * 100) if prezzo and prezzo_prec else None

    kama = result["kama_d"][-1] if result.get("kama_d") else None
    sar_bull = result["sarBull_d"][-1] if result.get("sarBull_d") else None
    cross = result["crossDays_d"][-1] if result.get("crossDays_d") else None
    ao_imp = result.get("aoImp")
    baff = result["baff_d"][-1] if result.get("baff_d") else None
    er = result["er_d"][-1] if result.get("er_d") else None
    mm_align = result["mmAlign_d"][-1] if result.get("mmAlign_d") else None
    rsi = result["rsi_d"][-1] if result.get("rsi_d") else None
    rsi5 = result["rsi5_d"][-1] if result.get("rsi5_d") else None
    segnale = result["segnale_d"][-1] if result.get("segnale_d") else None
    above_kama = None if (prezzo is None or kama is None) else prezzo > kama
    near_kama = None if (prezzo is None or kama is None or not kama) else abs(prezzo - kama) / kama < 0.03

    sez = []
    b1 = [
        ("SAR", "Rialzista", "Rialzista" if sar_bull else ("Ribassista" if sar_bull is not None else "—"), sar_bull, "Inversione di trend in corso"),
        ("Incrocio KAMA recente", "≤ 3 barre fa", f"{cross} barre" if cross is not None else "—", None if cross is None else cross <= 3, "Il prezzo ha appena riattraversato la sua media"),
        ("AO in miglioramento", "Sì", "Sì" if ao_imp else ("No" if ao_imp is not None else "—"), ao_imp, "Momentum in accelerazione"),
    ]
    sez.append(("🟢 BUY1 — Inversione fresca", "Pattern indipendente da BUY2/BUY3. Attivo oggi: " + ("<strong>SÌ</strong>" if segnale == "BUY1" else "no"), b1))

    b2 = [
        ("Prezzo &gt; KAMA", f"&gt; {_n(kama, 4)}", _n(prezzo, 4), above_kama, "Prezzo sopra la propria media mobile adattiva"),
        ("Baffetti", "≥ 2", str(baff) if baff is not None else "—", None if baff is None else baff >= 2, "Barre consecutive sopra KAMA con corpo pieno"),
    ]
    sez.append(("🔵 BUY2 — Pattern Baffetti", "Pattern indipendente da BUY1/BUY3. Attivo oggi: " + ("<strong>SÌ</strong>" if segnale == "BUY2" else "no"), b2))

    b3 = [
        ("Prezzo &gt; KAMA", f"&gt; {_n(kama, 4)}", _n(prezzo, 4), above_kama, "Prezzo sopra la propria media mobile adattiva"),
        ("ER (Efficiency Ratio)", "≥ 0.50", _n(er, 3), None if er is None else er >= 0.50, "Mercato molto direzionale, poco rumore"),
        ("Baffetti", "≥ 3", str(baff) if baff is not None else "—", None if baff is None else baff >= 3, "Momentum continuativo"),
        ("Medie allineate", "Sì", "Sì" if mm_align else ("No" if mm_align is not None else "—"), mm_align, "Trend maturo, non solo appena iniziato"),
    ]
    sez.append(("🟣 BUY3 — Trend maturo", "Pattern indipendente da BUY1/BUY2, il più severo dei tre. Attivo oggi: " + ("<strong>SÌ</strong>" if segnale == "BUY3" else "no"), b3))

    e2 = [
        ("Prezzo &lt; KAMA", f"&lt; {_n(kama, 4)}", _n(prezzo, 4), None if above_kama is None else not above_kama, "Trend invertito"),
        ("SAR", "Ribassista", "Ribassista" if sar_bull is False else ("Rialzista" if sar_bull is not None else "—"), None if sar_bull is None else not sar_bull, "Conferma l'inversione"),
    ]
    sez.append(("🔴 EXIT2 — Uscita forte", "Priorità su EXIT1 quando entrambe le condizioni sono vere. Attivo oggi: " + ("<strong>SÌ</strong>" if segnale == "EXIT2" else "no"), e2))

    e1 = [
        ("SAR", "Ribassista", "Ribassista" if sar_bull is False else ("Rialzista" if sar_bull is not None else "—"), None if sar_bull is None else not sar_bull, "Perdita di trend, prezzo ancora sopra KAMA"),
    ]
    sez.append(("🟠 EXIT1 — Uscita per perdita di trend", "Scatta solo se EXIT2 non è già vera (prezzo ancora sopra KAMA). Attivo oggi: " + ("<strong>SÌ</strong>" if segnale == "EXIT1" else "no"), e1))

    mr = [
        ("ER basso", "&lt; 0.30", _n(er, 3), None if er is None else er < 0.30, "Mercato laterale, non direzionale"),
        ("RSI14 ipervenduto", "&lt; 30", _n(rsi, 1), None if rsi is None else rsi < 30, "Ipervenduto"),
        ("AO in miglioramento", "Sì", "Sì" if ao_imp else ("No" if ao_imp is not None else "—"), ao_imp, "informativo"),
        ("Vicino a KAMA o sotto", "Sì", "Sì" if (near_kama or (above_kama is False)) else "No", bool(near_kama or (above_kama is False)), "Prezzo nella zona di rimbalzo"),
    ]
    sez.append(("🎯 MEAN REV — Solo informativo", "Non apre né chiude posizioni: segnala un possibile rimbalzo dentro un trade già aperto da BUY1/2/3, o nessuna azione se non si è in posizione. Attivo oggi: " + ("<strong>SÌ</strong>" if segnale == "MEAN REV" else "no"), mr))

    tr_last = result["trend_d"][-1] if result.get("trend_d") else None
    k20 = result["kama_d"][-21] if result.get("kama_d") and len(result["kama_d"]) > 20 else None
    trw = [
        ("KAMA in calo su 20 barre", "KAMA oggi &lt; KAMA 20 barre fa", f"{_n(kama, 4)} vs {_n(k20, 4)}", None if (kama is None or k20 is None) else kama < k20, "Direzione di fondo"),
        ("Prezzo sotto KAMA", f"&lt; {_n(kama, 4)}", _n(prezzo, 4), None if above_kama is None else not above_kama, "Conferma della direzione"),
    ]
    sez.append(("📉 TREND — Solo informativo", "Ribassista se entrambe vere; Rialzista se KAMA in salita e prezzo sopra; Laterale negli altri casi. Non modifica i segnali. Oggi: <strong>" + {"R": "Ribassista", "U": "Rialzista", "L": "Laterale"}.get(tr_last, "—") + "</strong>", trw))

    sezioni_html = "".join(f'<h2>{tit}</h2><p class="nota">{nota}</p>{_tabella(rows)}{_riepilogo(rows)}' for tit, nota, rows in sez)

    ml_exit = result.get("ml_exit")
    if ml_exit:
        ml_html = (
            '<h2>🤖 Suggerimento ML uscita</h2>'
            '<p class="nota">Stima di un modello allenato offline (non una regola, una previsione statistica) su posizioni BUY1/BUY2 aperte: quanto potrebbe salire ancora e in quanti giorni, prima di un possibile massimo.</p>'
            '<table><tbody>'
            f'<tr><td>Rendimento di picco stimato</td><td class="val">{_n(ml_exit.get("peak_return_pct"), 2, "%")}</td></tr>'
            f'<tr><td>Giorni stimati al picco</td><td class="val">{_n(ml_exit.get("days_to_peak"), 1, " gg")}</td></tr>'
            '</tbody></table>')
    else:
        ml_html = ""

    perf = result.get("perf") or {}
    open_trade = result.get("open_trade")
    storico_html = (
        '<h2>📈 Storico dei trade su questo titolo</h2>'
        '<p class="nota">Backtest sullo stesso storico scaricato, senza correzione per survivorship: indicativo, non una previsione.</p>'
        '<table><tbody>'
        f'<tr><td>Trade totali / chiusi</td><td class="val">{perf.get("trades", 0)} / {perf.get("closed", 0)}</td></tr>'
        f'<tr><td>% vincenti</td><td class="val">{_n(perf.get("wr"), 1, "%")}</td></tr>'
        f'<tr><td>P&amp;L medio per trade</td><td class="val">{_n(perf.get("avg"), 2, "%")}</td></tr>'
        f'<tr><td>Media vincenti / perdenti</td><td class="val">{_n(perf.get("avgWin"), 2, "%")} / {_n(perf.get("avgLoss"), 2, "%")}</td></tr>'
        f'<tr><td>Migliore / peggiore trade</td><td class="val">{_n(perf.get("best"), 2, "%")} / {_n(perf.get("worst"), 2, "%")}</td></tr>'
        f'<tr><td>P&amp;L cumulato / drawdown max</td><td class="val">{_n(perf.get("totalPnl"), 2, "%")} / {_n(perf.get("dd"), 2, "%")}</td></tr>'
        + (f'<tr><td>Posizione aperta</td><td class="val">dal {open_trade["dataEntrata"]} ({open_trade["giorni"]}gg) · {_n(open_trade["pnlPct"], 2, "%")}</td></tr>' if open_trade else '<tr><td>Posizione aperta</td><td class="val">nessuna</td></tr>')
        + '</tbody></table>')

    suffix_map = {".MI": "MIL:", ".DE": "XETR:", ".PA": "EURONEXT:", ".AS": "EURONEXT:", ".L": "LSE:"}
    tv_symbol = ticker
    for suf, pref in suffix_map.items():
        if yahoo.endswith(suf):
            tv_symbol = pref + ticker
            break
    links = [
        f'<a href="https://www.tradingview.com/chart/?symbol={urllib.parse.quote(tv_symbol, safe="")}" target="_blank">📈 TradingView</a>',
        f'<a href="https://finance.yahoo.com/quote/{urllib.parse.quote(yahoo, safe="")}" target="_blank">🟣 Yahoo Finance</a>',
        '<a href="../index.html" target="_blank">🏠 Dashboard</a>',
        '<a href="../posizioni-aperte.html" target="_blank">📂 Posizioni Aperte</a>',
    ]

    repl = {
        "{{NOME}}": result.get("name") or ticker,
        "{{TICKER}}": yahoo,
        "{{GENERATO}}": now,
        "{{AGGIORNATO}}": d_bars[-1][0] if d_bars else "—",
        "{{LINKS}}": " ".join(links),
        "{{PREZZO}}": _n(prezzo, 4),
        "{{OGGI_PCT}}": _n(oggi_pct, 2, "%"),
        "{{RSI14}}": _n(rsi, 1),
        "{{RSI5}}": _n(rsi5, 1),
        "{{ADX}}": _n(info.get("adx"), 1),
        "{{SEGNALE}}": segnale or "—",
        "{{SCORE}}": str(info.get("score")) if info.get("score") is not None else "—",
        "{{SEZIONI}}": sezioni_html,
        "{{ML_EXIT}}": ml_html,
        "{{STORICO}}": storico_html,
    }
    html = REGOLE_TEMPLATE
    for k, v in repl.items():
        html = html.replace(k, str(v))
    return html


def main():
    now = datetime.datetime.now()
    tickers = select_tickers()
    print(f"generate_charts.py — {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"Ticker qualificati Borsa Italiana (BUY/WATCHLIST/ANTEPRIMA): {len(tickers)}")

    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(REGOLE_DIR, exist_ok=True)

    ok = 0
    errors = 0
    index = []
    for i, info in enumerate(tickers):
        result = process_ticker(info)
        if result:
            fname = info["y"].replace(".", "_") + ".json"
            with open(os.path.join(CHARTS_DIR, fname), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

            try:
                regole_html = build_regole_html(result, info)
                regole_fname = fname.replace(".json", "_Regole.html")
                with open(os.path.join(REGOLE_DIR, regole_fname), "w", encoding="utf-8") as f:
                    f.write(regole_html)
            except Exception as e:
                print(f"  ATTENZIONE regole {info['y']}: {e}")

            last_close = result["d"][-1][4] if result["d"] else None
            prev_close = result["d"][-2][4] if len(result["d"]) > 1 else None
            today_chg = round((last_close - prev_close) / prev_close * 100, 2) if last_close and prev_close else None
            open_trade = result.get("open_trade")

            index.append({
                "t": info["t"], "y": info["y"], "f": fname,
                "name": info.get("name"), "asset_class": info.get("asset_class"),
                "adx": info.get("adx"), "score": info.get("score"),
                "close": last_close, "today_chg": today_chg,
                "signal": (result["segnale_d"][-1] if result["segnale_d"] else None),
                "rsi": (result["rsi_d"][-1] if result["rsi_d"] else None),
                "isOpen": bool(open_trade),
                "daysOpen": open_trade["giorni"] if open_trade else None,
                "entryDate": open_trade["dataEntrata"] if open_trade else None,
                "entryPrice": open_trade["prezzoEntrata"] if open_trade else None,
                "perf": result.get("perf"),
            })
            ok += 1
        else:
            errors += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(tickers)} — ok:{ok} errori:{errors}")
        time.sleep(SLEEP_BETWEEN_TICKERS)

    meta = {
        "timestamp": now.isoformat(),
        "timestamp_it": now.strftime("%d/%m/%Y %H:%M"),
        "ok": ok, "errors": errors,
        "index": index,
    }
    with open(os.path.join(CHARTS_DIR, "index.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))

    print(f"\nSalvati {ok} file in data/charts/ — {errors} errori")


if __name__ == "__main__":
    main()
