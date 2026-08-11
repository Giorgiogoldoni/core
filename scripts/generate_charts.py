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

# Batching per limitare tempi/rate-limit — lista qui è molto più piccola
# dell'universo completo (solo BUY/WATCHLIST/ANTEPRIMA), non serve batch
# grande, ma manteniamo comunque una pausa per sicurezza.
SLEEP_BETWEEN_TICKERS = 0.3

# Segnali che qualificano un ticker per la generazione del grafico dettaglio
QUALIFYING_SIGNALS = {"BUY", "WATCHLIST — timing fresco", "WATCHLIST — trend maturo"}


def select_tickers():
    """Legge etf_scores.json e seleziona i ticker qualificati (BUY, WATCHLIST, ANTEPRIMA)."""
    if not os.path.exists(SCORES_PATH):
        print(f"[ERROR] {SCORES_PATH} non trovato — esegui prima calculate_scores.py", file=sys.stderr)
        sys.exit(1)

    with open(SCORES_PATH, "r", encoding="utf-8") as f:
        scores_data = json.load(f)

    selected = []
    for item in scores_data.get("scores", []):
        signal = item.get("signal")
        anteprima = item.get("anteprima", False)
        if signal in QUALIFYING_SIGNALS or anteprima:
            selected.append({"y": item["ticker_yf"], "t": item["ticker_yf"].split(".")[0]})

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
#  PROCESS TICKER
# ═══════════════════════════════════════════════════════
def process_ticker(info):
    symbol = info["y"]
    try:
        tk = yf.Ticker(symbol)
        hist_d = tk.history(period="1y", interval="1d", timeout=20)
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

        result = {
            "ticker": info["t"], "yahoo": symbol,
            "d": d_bars, "h": h_bars,
            "kama_d": fmt(kama_arr), "sar_d": fmt(sar_arr), "sarBull_d": sarBull_arr,
            "ao_d": fmt(ao_arr), "rsi_d": fmt(rsi_arr), "rsi5_d": fmt(rsi5_arr), "baff_d": baff_arr,
            "segnale_d": segnale_arr,
            "er_d": fmt(er_arr), "crossDays_d": cross_arr, "mmAlign_d": mm_arr,
            "sarStreak_d": sarStreak_arr,
            "kama_h": fmt(kama_h), "sar_h": fmt(sar_h), "sarBull_h": sarBull_h,
            "atr": round(atr, 4) if atr else None,
            "renko_brick": brick, "renko": renko,
            "ml_exit": ml_exit,
        }
        return sanitize_nan(result)
    except Exception as e:
        print(f"  ERR {symbol}: {e}")
        return None


def main():
    now = datetime.datetime.now()
    tickers = select_tickers()
    print(f"generate_charts.py — {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"Ticker qualificati (BUY/WATCHLIST/ANTEPRIMA): {len(tickers)}")

    os.makedirs(CHARTS_DIR, exist_ok=True)

    ok = 0
    errors = 0
    index = []
    for i, info in enumerate(tickers):
        result = process_ticker(info)
        if result:
            fname = info["y"].replace(".", "_") + ".json"
            with open(os.path.join(CHARTS_DIR, fname), "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            index.append({"t": info["t"], "y": info["y"], "f": fname})
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
