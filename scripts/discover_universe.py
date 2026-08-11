#!/usr/bin/env python3
"""
discover_universe.py
Ricostruisce l'universo ETF/ETP (Xetra + Borsa Italiana), deduplica per ISIN
e scrive data/tickers_universe.json.

STATO ATTUALE:
- fetch_xetra(): OPERATIVA. Legge data/raw/t7-xetr-allTradableInstruments.csv
  (file ufficiale Xetra T7, aggiornamento manuale periodico a cura dell'utente).
  Filtro: Instrument Type in (ETF, ETN, ETC), Settlement Currency == EUR.
  asset_class affidabile solo per ~23% delle righe (nomi molto abbreviati),
  il resto è default_fallback da rivedere.
- fetch_borsa_italiana(): OPERATIVA. Legge data/raw/euronext_trackers.xlsx
  (file Euronext, aggiornamento manuale periodico), filtra Market == "ETF Plus".
  asset_class affidabile per ~80% delle righe.

Regola dedup: se lo stesso ISIN compare su entrambe le borse, si tiene la riga
Borsa Italiana (priorità a Borsa Italiana) — copre ~1.933 strumenti su 3.406 Xetra EUR.
"""

import json
import os
import sys
from datetime import datetime, timezone

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tickers_universe.json")

# File ufficiale Xetra T7 "all tradable instruments" caricato manualmente dall'utente
# (aggiornamento manuale periodico, non scaricabile via Actions).
XETRA_SOURCE = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "t7-xetr-allTradableInstruments.csv"
)

# File Euronext caricato manualmente dall'utente (aggiornamento manuale periodico).
# Contiene tutti i mercati Euronext; filtriamo solo Market == "ETF Plus" (Borsa Italiana).
BORSA_ITALIANA_SOURCE = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "euronext_trackers.xlsx"
)

# Classificazione euristica asset_class basata su keyword nel nome esteso.
# NON è affidabile al 100%: keyword non previste (es. altcoin non elencate, strutture
# di credito non comuni) finiscono nel fallback "equity" con confidence "default_fallback".
BOND_KEYWORDS = [
    "BOND", "GOVT", "GOVERNMENT", "TREASURY", "CORPORATE", "GILT", "BUND", "BTP",
    "SOVEREIGN", "FIXED INCOME", "DURATION", "CLO", "CDO", "ABS", "MBS", "CREDIT",
    "INFL-LINKED", "INFLATION", "SHORT MAT",
]
COMMODITY_KEYWORDS = [
    "GOLD", "SILVER", "OIL", "GAS", "NATURAL GAS", "WHEAT", "SUGAR", "COFFEE",
    "COTTON", "COPPER", "PALLADIUM", "PLATINUM", "COMMODIT", "BRENT", "WTI",
]
CRYPTO_KEYWORDS = [
    "BITCOIN", "BTC", "ETHEREUM", "XRP", "SOLANA", "CARDANO", "DOGE", "LITECOIN",
    "CRYPTO", "STAKING", "SUI", "POLKADOT", "CHAINLINK", "AVALANCHE", "POLYGON",
    "TRON", "TON ", "BNB", "RIPPLE", "SHIBA", "UNISWAP", "AAVE", "COSMOS", "ALGORAND",
]
EQUITY_KEYWORDS = [
    "MSCI", "S&P", "DOW JONES", "NASDAQ", "STOXX", "FTSE", "RUSSELL", "DAX",
    "CAC 40", "MIB", "TOPIX", "NIKKEI", "IBEX", "EQUITY", "SMALL CAP", "LARGE CAP",
    "MID CAP", "WORLD", "GLOBAL", "EMERGING MARKETS", "EUROPE", "JAPAN", "CHINA",
    "INDIA", "KOREA", "NORTH AM", "HANG SENG", "ALLCAP",
]


def classify_instrument(fullname):
    """
    Ritorna (asset_class, is_leveraged, confidence, is_money_market).
    confidence è "keyword_match" se una regola ha trovato corrispondenza esplicita,
    "default_fallback" se è stato assunto equity per esclusione (da rivedere a mano).

    is_money_market: True per ETF/ETC money-market/tasso overnight (EONIA/€STR/SONIA) —
    il loro prezzo è una salita quasi lineare per capitalizzazione interesse, non un vero
    trend di mercato: generano falsi BUY strutturali nello Score tecnico (KAMA/SAR/ADX
    li leggono come trend perfetto). Score/Segnale vengono neutralizzati a valle in
    calculate_scores.py, non qui — qui solo la classificazione.
    """
    import re

    fn = (fullname or "").upper()
    is_leveraged = (
        bool(re.search(r"[+-]?\d+X\b", fn))
        or "LEVERAGED" in fn
        or "INVERSE" in fn
        or "LEVERAGE SHARES" in fn  # brand con pattern tipo "3XCRM"/"-1AAPL" senza confine di parola dopo la X
    )
    is_money_market = bool(
        re.search(r"\b(OVERNIGHT|OVERNGHT|OVNI|EONIA|ESTR|SONIA|MONEY\s*MARKET|CASH|FED\s*FUNDS)\b", fn)
    ) and "BITCOIN" not in fn  # "Bitcoin Cash" (cripto) è un falso positivo del match su CASH

    if is_leveraged:
        return "leva_short", True, "keyword_match", is_money_market
    if any(k in fn for k in CRYPTO_KEYWORDS):
        return "crypto", False, "keyword_match", is_money_market
    if any(k in fn for k in COMMODITY_KEYWORDS):
        return "commodity", False, "keyword_match", is_money_market
    if any(k in fn for k in BOND_KEYWORDS):
        return "bond", False, "keyword_match", is_money_market
    if any(k in fn for k in EQUITY_KEYWORDS):
        return "equity", False, "keyword_match", is_money_market
    return "equity", False, "default_fallback", is_money_market


def fetch_xetra():
    """
    Legge data/raw/t7-xetr-allTradableInstruments.csv (file ufficiale Xetra T7,
    aggiornato manualmente dall'utente), filtra Instrument Type in (ETF, ETN, ETC)
    e Settlement Currency == EUR (elimina da solo i ~150 doppioni multi-valuta),
    poi classifica con le stesse regole euristiche di fetch_borsa_italiana().

    NOTA IMPORTANTE: il campo "Instrument" qui è molto più abbreviato/criptico
    di "Instrument Fullname" su Euronext (es. "I2-EOST.50EQUWE EOA"). Il bucket
    default_fallback risulta quindi molto più ampio (~77% contro ~20% su Borsa
    Italiana) — da tenere presente, non è un bug del parser ma un limite della fonte.

    product_type: qui la fonte ufficiale è affidabile (campo "Instrument Type"),
    a differenza di Borsa Italiana dove è dedotto dal nome. Il valore ETN viene
    mappato su "ETP" per restare coerente con lo schema (ETF/ETC/ETP) già usato
    per Borsa Italiana.
    """
    if not os.path.exists(XETRA_SOURCE):
        print(
            f"[WARN] File Xetra non trovato ({XETRA_SOURCE}). "
            "Nessun dato Xetra caricato in questo run.",
            file=sys.stderr,
        )
        return []

    import pandas as pd

    df = pd.read_csv(XETRA_SOURCE, sep=";", skiprows=2, low_memory=False)
    subset = df[
        df["Instrument Type"].isin(["ETF", "ETN", "ETC"])
        & (df["Settlement Currency"] == "EUR")
    ]

    product_type_map = {"ETF": "ETF", "ETC": "ETC", "ETN": "ETP"}

    rows = []
    for _, row in subset.iterrows():
        isin = str(row.get("ISIN") or "").strip()
        symbol = str(row.get("Mnemonic") or "").strip()
        name = str(row.get("Instrument") or "").strip()
        if not isin or not symbol or symbol.lower() == "nan":
            continue

        asset_class, is_leveraged, confidence, is_money_market = classify_instrument(name)

        import re

        name_upper = name.upper()
        if re.search(r"\(ACC\)|\bACC\b", name_upper):
            distribution_policy = "accumulating"
        elif re.search(r"\(DIST\)|\bDIST\b", name_upper):
            distribution_policy = "distributing"
        else:
            distribution_policy = None

        rows.append(
            {
                "isin": isin,
                "ticker_yf": f"{symbol}.DE",
                "name": name,
                "currency": str(row.get("Settlement Currency") or "EUR").strip(),
                "asset_class": asset_class,
                "is_leveraged": is_leveraged,
                "product_type": product_type_map.get(row.get("Instrument Type"), "ETF"),
                "distribution_policy": distribution_policy,
                "asset_class_confidence": confidence,
            }
        )
    return rows


def fetch_borsa_italiana():
    """
    Legge data/raw/euronext_trackers.xlsx (aggiornato manualmente dall'utente),
    filtra Market == "ETF Plus" (= Borsa Italiana) e classifica ogni strumento
    con regole euristiche su Instrument Fullname.

    NOTA: il file contiene TUTTI i mercati Euronext (Paris, Amsterdam, Brussels,
    Oslo compresi) — qui prendiamo SOLO ETF Plus, per scelta di scope confermata.
    """
    if not os.path.exists(BORSA_ITALIANA_SOURCE):
        print(
            f"[WARN] File Euronext non trovato ({BORSA_ITALIANA_SOURCE}). "
            "Nessun dato Borsa Italiana caricato in questo run.",
            file=sys.stderr,
        )
        return []

    import pandas as pd

    df = pd.read_excel(
        BORSA_ITALIANA_SOURCE, sheet_name="Simple", header=0, skiprows=[1, 2, 3]
    )
    etf_plus = df[df["Market"] == "ETF Plus"]

    rows = []
    for _, row in etf_plus.iterrows():
        isin = str(row.get("ISIN") or "").strip()
        symbol = str(row.get("Symbol") or "").strip()
        fullname = str(row.get("Instrument Fullname") or "").strip()
        if not isin or not symbol:
            continue

        asset_class, is_leveraged, confidence, is_money_market = classify_instrument(fullname)

        fn_upper = fullname.upper()
        if "ETC" in fn_upper.split():
            product_type = "ETC"
        elif "ETP" in fn_upper.split():
            product_type = "ETP"
        else:
            product_type = "ETF"

        import re

        if re.search(r"\(ACC\)|\bACC\b", fn_upper):
            distribution_policy = "accumulating"
        elif re.search(r"\(DIST\)|\bDIST\b|\bDIS\b", fn_upper):
            distribution_policy = "distributing"
        else:
            distribution_policy = None

        rows.append(
            {
                "isin": isin,
                "ticker_yf": f"{symbol}.MI",
                "name": fullname,
                "currency": str(row.get("Currency") or "EUR").strip(),
                "asset_class": asset_class,
                "is_leveraged": is_leveraged,
                "product_type": product_type,
                "distribution_policy": distribution_policy,
                "asset_class_confidence": confidence,
                "is_money_market": is_money_market,
            }
        )
    return rows


def dedup_by_isin(xetra_rows, borsa_italiana_rows):
    """
    Deduplica per ISIN. In caso di conflitto, priorità a Borsa Italiana.
    """
    merged = {}
    for row in xetra_rows:
        merged[row["isin"]] = row
    for row in borsa_italiana_rows:
        merged[row["isin"]] = row  # sovrascrive eventuale riga Xetra con stesso ISIN
    return list(merged.values())


def write_universe(rows):
    rows_sorted = sorted(rows, key=lambda r: r["isin"])
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(rows_sorted),
        "instruments": rows_sorted,
    }

    tmp_path = OUTPUT_PATH + ".tmp"
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, OUTPUT_PATH)  # scrittura atomica
    print(f"Scritti {len(rows_sorted)} strumenti in {OUTPUT_PATH}")


def main():
    xetra_rows = fetch_xetra()
    borsa_italiana_rows = fetch_borsa_italiana()
    merged = dedup_by_isin(xetra_rows, borsa_italiana_rows)
    write_universe(merged)


if __name__ == "__main__":
    main()
