#!/usr/bin/env python3
"""calculate_scores.py

Legge data/tickers_universe.json, scarica i dati storici tramite yfinance,
calcola lo Score Operativo (Vento + Onda SAR + Hard Gate) e scrive
data/etf_scores.json per la dashboard web.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

# Batching per evitare timeout Actions (90 min) e rate-limit yfinance
# su run sequenziali di migliaia di ticker. Stesso pattern del fetch giornaliero.
BATCH_SIZE = 40
BATCH_PAUSE_SECONDS = 3

# Percorsi dei file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIVERSE_PATH = os.path.join(BASE_DIR, "data", "tickers_universe.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "etf_scores.json")


def calculate_kama(series: pd.Series, n=10, pow1=2, pow2=30) -> pd.Series:
    """Calcola la Kaufman's Adaptive Moving Average (KAMA)."""
    change = (series - series.shift(n)).abs()
    volatility = (series - series.shift(1)).abs().rolling(n).sum()
    er = np.where(volatility == 0, 0, change / volatility)

    sc_fast = 2 / (pow1 + 1)
    sc_slow = 2 / (pow2 + 1)
    sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2

    kama = pd.Series(index=series.index, dtype=float)
    first_valid = series.first_valid_index()
    if first_valid is None:
        return kama

    idx_start = series.index.get_loc(first_valid) + n
    if idx_start >= len(series):
        return kama

    kama.iloc[idx_start] = series.iloc[idx_start]
    for i in range(idx_start + 1, len(series)):
        kama.iloc[i] = kama.iloc[i - 1] + sc[i] * (
            series.iloc[i] - kama.iloc[i - 1]
        )

    return kama


def calculate_adx(
    df: pd.DataFrame, n=14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calcola ADX, +DI e -DI."""
    high, low, close = df["High"], df["Low"], df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smoothed = tr.rolling(n).sum()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(n).sum() / tr_smoothed)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(n).sum() / tr_smoothed)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(n).mean()

    return adx, plus_di, minus_di


def calculate_psar(
    df: pd.DataFrame, iaf=0.02, maxaf=0.2
) -> tuple[pd.Series, pd.Series]:
    """Calcola il Parabolic SAR e restituisce (psar, trend_is_up)."""
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    length = len(df)

    psar = np.zeros(length)
    bull = True
    af = iaf
    hp = high[0]
    lp = low[0]
    psar[0] = low[0]

    for i in range(1, length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
            psar[i] = min(psar[i], low[i - 1], low[i - 2] if i > 1 else low[i - 1])
            if low[i] < psar[i]:
                bull = False
                psar[i] = hp
                lp = low[i]
                af = iaf
            else:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + iaf, maxaf)
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
            psar[i] = max(psar[i], high[i - 1], high[i - 2] if i > 1 else high[i - 1])
            if high[i] > psar[i]:
                bull = True
                psar[i] = lp
                hp = high[i]
                af = iaf
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + iaf, maxaf)

    sar_series = pd.Series(psar, index=df.index)
    trend_up = pd.Series(close > psar, index=df.index)
    return sar_series, trend_up


def compute_score_for_ticker(ticker_yf: str) -> dict | None:
    """Scarica i dati storici ed elabora il Composite Score."""
    try:
        df = yf.download(ticker_yf, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 50:
            return None

        # Appiattisci MultiIndex colonne se presente
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"]
        volume = df["Volume"]

        # Indicatori
        kama = calculate_kama(close, n=10)
        adx, plus_di, minus_di = calculate_adx(df, n=14)
        psar, sar_is_up = calculate_psar(df)

        # Ultimo e penultimo candle
        c_last = close.iloc[-1]
        k_last = kama.iloc[-1]
        k_prev = kama.iloc[-4] if len(kama) >= 4 else k_last
        adx_last = adx.iloc[-1]
        pdi_last = plus_di.iloc[-1]
        mdi_last = minus_di.iloc[-1]
        vol_last = volume.iloc[-1]
        vol_sma = volume.rolling(20).mean().iloc[-1]

        # --- 1. HARD GATE (Filtro Bloccante) ---
        # Il prezzo deve essere sopra la KAMA
        hard_gate = 1 if c_last > k_last else 0

        # --- 2. IL VENTO (Max 70 Punti) ---
        # Direzione (25 pt): Prezzo > KAMA e pendenza KAMA positiva
        score_dir = 25 if (c_last > k_last and k_last > k_prev) else 0

        # Intensità (20 pt): ADX > 22 e +DI > -DI (penalità se ADX > 48 per esaurimento)
        if adx_last > 48:
            score_int = 10
        elif adx_last > 22 and pdi_last > mdi_last:
            score_int = 20
        else:
            score_int = 0

        # Stato del Mare / Volumi (10 pt): Volumi superiori alla media 20p
        score_sea = 10 if vol_last > vol_sma else 0

        # Forza Relativa di fondo (15 pt): Prezzo > Prezzo di 50 giorni fa
        score_rs = 15 if (len(close) >= 50 and c_last > close.iloc[-50]) else 0

        score_vento = score_dir + score_int + score_sea + score_rs

        # --- 3. L'ONDA / TIMING (Max 30 Punti) ---
        # Primo pallino SAR UP = 30 pt; 2° o 3° pallino = 15 pt; altrimenti = 0 pt
        first_sar_up = sar_is_up.iloc[-1] and not sar_is_up.iloc[-2]
        first_sar_down = (not sar_is_up.iloc[-1]) and sar_is_up.iloc[-2]

        # Calcolo età pallino
        sar_age = 0
        if sar_is_up.iloc[-1]:
            for val in reversed(sar_is_up.values):
                if val:
                    sar_age += 1
                else:
                    break

        if first_sar_up:
            score_onda = 30
        elif sar_is_up.iloc[-1] and 2 <= sar_age <= 3:
            score_onda = 15
        else:
            score_onda = 0

        # Data della barra in cui è scattato l'attuale streak SAR UP (se in corso)
        signal_date = None
        if sar_is_up.iloc[-1] and sar_age > 0:
            flip_idx = close.index[-sar_age]
            signal_date = flip_idx.strftime("%Y-%m-%d")

        # --- 4. SCORE FINALE ---
        score_totale = score_vento + score_onda
        score_operativo = score_totale * hard_gate

        if score_operativo >= 80:
            signal = "BUY"
        elif score_operativo >= 65:
            # Sotto-classificazione WATCHLIST:
            # timing fresco = SAR appena scattato (Onda piena) ma Vento non ancora forte
            # trend maturo  = Vento forte ma il timing d'ingresso fresco è già passato
            if score_onda >= 30:
                signal = "WATCHLIST — timing fresco"
            else:
                signal = "WATCHLIST — trend maturo"
        else:
            signal = "NO TRADE"

        # Segnale anticipatorio ANTEPRIMA: solo SAR al 1°/2° pallino UP,
        # SENZA hard gate (prezzo>KAMA) né requisito di Vento — early warning
        # non confermato, pensato per anticipare, non per operare direttamente.
        # Non sostituisce BUY/WATCHLIST/NO TRADE: è un flag aggiuntivo.
        anteprima = bool(sar_is_up.iloc[-1] and sar_age in (1, 2))

        # UP/DOWN: segnala il primo pallino SAR in assoluto, in entrambe le
        # direzioni, indipendente da hard gate/Vento/Onda — non è un segnale
        # operativo BUY/SELL, è un flag grezzo "il SAR ha appena flippato qui".
        sar_flip = "UP" if first_sar_up else ("DOWN" if first_sar_down else None)

        return {
            "close": round(float(c_last), 2),
            "score_operativo": int(score_operativo),
            "score_vento": int(score_vento),
            "score_onda": int(score_onda),
            "hard_gate": int(hard_gate),
            "signal": signal,
            "signal_date": signal_date,
            "anteprima": anteprima,
            "sar_flip": sar_flip,
            "sar_age": int(sar_age),
            "adx": round(float(adx_last), 1) if not np.isnan(adx_last) else 0.0,
            "sar_is_up": bool(sar_is_up.iloc[-1]),
            "sar_first_dot": bool(first_sar_up),
        }
    except Exception as e:
        print(f"[ERROR] {ticker_yf}: {e}", file=sys.stderr)
        return None


def main():
    if not os.path.exists(UNIVERSE_PATH):
        print(f"[ERROR] {UNIVERSE_PATH} non trovato.", file=sys.stderr)
        sys.exit(1)

    with open(UNIVERSE_PATH, "r", encoding="utf-8") as f:
        universe_data = json.load(f)

    instruments = universe_data.get("instruments", [])
    results = []

    print(
        f"Inizio calcolo Score su {len(instruments)} strumenti dall'universo..."
    )

    total = len(instruments)
    for batch_start in range(0, total, BATCH_SIZE):
        batch = instruments[batch_start : batch_start + BATCH_SIZE]
        for item in batch:
            ticker = item["ticker_yf"]
            score_data = compute_score_for_ticker(ticker)
            if score_data:
                combined = {**item, **score_data}
                if item.get("is_money_market"):
                    # Il prezzo di questi strumenti sale in modo quasi lineare per
                    # capitalizzazione interesse (EONIA/€STR/SONIA), non per un vero
                    # trend di mercato: KAMA/SAR/ADX lo leggono come trend perfetto e
                    # generano BUY strutturalmente falsi. Sterilizzato: prezzo resta
                    # visibile, Score/Segnale/ANTEPRIMA/SAR Flip azzerati con etichetta.
                    combined["score_operativo"] = 0
                    combined["score_vento"] = 0
                    combined["score_onda"] = 0
                    combined["signal"] = "NO TRADE — money market"
                    combined["signal_date"] = None
                    combined["anteprima"] = False
                    combined["sar_flip"] = None
                results.append(combined)

        done = min(batch_start + BATCH_SIZE, total)
        print(f"Batch completato: {done}/{total} strumenti processati.")

        if done < total:
            time.sleep(BATCH_PAUSE_SECONDS)

    # Ordina per score_operativo decrescente
    results_sorted = sorted(
        results, key=lambda x: x["score_operativo"], reverse=True
    )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(results_sorted),
        "scores": results_sorted,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"Calcolati con successo {len(results_sorted)} score. Salvati in {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
