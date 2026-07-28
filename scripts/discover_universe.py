#!/usr/bin/env python3
"""
discover_universe.py
Ricostruisce l'universo ETF/ETP (Xetra + Borsa Italiana), deduplica per ISIN
e scrive data/tickers_universe.json.

STATO ATTUALE (da verificare/completare):
- fetch_xetra(): fonte indicata https://live.deutsche-boerse.com/en/etfs/search
  Quella pagina è una UI di ricerca dinamica, NON un export diretto.
  Qui sotto è implementato un PLACEHOLDER che assume un CSV con colonne tipiche
  Deutsche Börse (ISIN, Name, Trading Symbol, Currency, Asset Class, TER).
  Va sostituito con la chiamata reale (endpoint XHR individuato oppure file
  esportato manualmente) al primo run.
- fetch_borsa_italiana(): STUB VUOTO — nessuna fonte ancora fornita.
  TODO: implementare quando disponibile URL o file di esempio ETFplus.

Regola dedup: se lo stesso ISIN compare su entrambe le borse, si tiene la riga
Borsa Italiana (priorità a Borsa Italiana).
"""

import json
import os
import sys
from datetime import datetime, timezone

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tickers_universe.json")

# Se in futuro trovi un CSV/Excel di export reale, punta qui il path o l'URL.
XETRA_SOURCE_PLACEHOLDER = "xetra_etfs.csv"  # TODO: sostituire con fonte reale


def fetch_xetra():
    """
    PLACEHOLDER. Assume un CSV con colonne:
    ISIN, Name, Trading Symbol, Currency, Asset Class, TER

    Ritorna una lista di dict con lo schema:
    {"isin": ..., "ticker_yf": ..., "name": ..., "currency": ..., "asset_class": ...}
    """
    if not os.path.exists(XETRA_SOURCE_PLACEHOLDER):
        print(
            f"[WARN] Fonte Xetra non trovata ({XETRA_SOURCE_PLACEHOLDER}). "
            "Nessun dato Xetra caricato in questo run. "
            "TODO: sostituire con la fonte reale (endpoint o file export).",
            file=sys.stderr,
        )
        return []

    import csv

    rows = []
    with open(XETRA_SOURCE_PLACEHOLDER, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            isin = (row.get("ISIN") or "").strip()
            symbol = (row.get("Trading Symbol") or "").strip()
            if not isin or not symbol:
                continue
            rows.append(
                {
                    "isin": isin,
                    "ticker_yf": f"{symbol}.DE",
                    "name": (row.get("Name") or "").strip(),
                    "currency": (row.get("Currency") or "EUR").strip(),
                    "asset_class": (row.get("Asset Class") or None) or None,
                }
            )
    return rows


def fetch_borsa_italiana():
    """
    STUB VUOTO.
    TODO: implementare quando sarà disponibile una fonte (URL export ETFplus
    o file di esempio caricato dall'utente). Deve ritornare lista di dict
    con lo stesso schema di fetch_xetra(), con ticker_yf che termina in ".MI".
    """
    print(
        "[INFO] fetch_borsa_italiana() è uno stub vuoto: nessuna fonte "
        "ancora fornita. TODO: implementare quando disponibile.",
        file=sys.stderr,
    )
    return []


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
