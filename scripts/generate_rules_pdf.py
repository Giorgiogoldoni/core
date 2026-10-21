#!/usr/bin/env python3
"""generate_rules_pdf.py

Genera regole_generali.pdf (root del repo) con le regole generali di CORE,
la data/ora dell'aggiornamento e alcune statistiche dell'ultimo run.
Collegato al bottone "Regole (PDF)" di index.html.

Va eseguito dal workflow DOPO calculate_scores.py e generate_charts.py.

ATTENZIONE - manutenzione: i testi delle regole sono scritti a mano a partire da
scripts/calculate_scores.py (compute_score_for_ticker) e scripts/generate_charts.py
(calc_segnale_array, simulate_trades). Se si cambiano soglie o logiche in quei file,
va aggiornato anche questo testo. Per non lasciare un PDF silenziosamente sbagliato,
CHECKS verifica che le soglie citate esistano ancora nel codice: se una manca, il PDF
mostra un avviso rosso in prima pagina e il run stampa un WARNING.
"""

import datetime
import json
import os
import sys
from zoneinfo import ZoneInfo

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORES_PATH = os.path.join(BASE_DIR, "data", "etf_scores.json")
UNIVERSE_PATH = os.path.join(BASE_DIR, "data", "tickers_universe.json")
CHARTS_INDEX_PATH = os.path.join(BASE_DIR, "data", "charts", "index.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "regole_generali.pdf")

STALE_BUSINESS_DAYS = 3  # deve coincidere con index.html (STALE_BUSINESS_DAYS)

# (file, frammento di codice che deve esistere ancora)
CHECKS = [
    ("scripts/calculate_scores.py", "score_operativo >= 80"),
    ("scripts/calculate_scores.py", "score_operativo >= 65"),
    ("scripts/calculate_scores.py", "score_dir = 25"),
    ("scripts/calculate_scores.py", "adx_last > 48"),
    ("scripts/calculate_scores.py", "adx_last > 22 and pdi_last > mdi_last"),
    ("scripts/calculate_scores.py", "score_sea = 10"),
    ("scripts/calculate_scores.py", "score_rs = 15"),
    ("scripts/calculate_scores.py", "score_onda = 30"),
    ("scripts/calculate_scores.py", "score_onda = 15"),
    ("scripts/calculate_scores.py", "2 <= sar_age <= 3"),
    ("scripts/calculate_scores.py", "len(df) < 50"),
    ("scripts/calculate_scores.py", 'period="1y"'),
    ("scripts/generate_charts.py", "sar_bull and cross <= 3 and ao_imp"),
    ("scripts/generate_charts.py", "above_kama and baff >= 2"),
    ("scripts/generate_charts.py", "above_kama and er >= 0.50 and baff >= 3 and mm_align"),
    ("scripts/generate_charts.py", "not above_kama and not sar_bull"),
    ("scripts/generate_charts.py", "er < 0.30 and rsi < 30 and ao_imp"),
    ("scripts/generate_charts.py", "< 0.03"),
    ("scripts/generate_charts.py", 'EXCHANGE_SUFFIX = None'),
    ("scripts/generate_charts.py", 'period="2y"'),
    ("scripts/generate_charts.py", 'last_seg in ("BUY1", "BUY2")'),
]


def check_code_alignment():
    missing = []
    for rel, frag in CHECKS:
        path = os.path.join(BASE_DIR, rel)
        try:
            with open(path, encoding="utf-8") as f:
                if frag not in f.read():
                    missing.append(f"{rel}: {frag}")
        except OSError:
            missing.append(f"{rel}: file non leggibile")
    return missing


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def business_days_between(d_from, d_to):
    n, d = 0, d_from
    while d < d_to:
        d += datetime.timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def collect_stats():
    st = {"n_universe": None, "scores_at": None, "signals": {}, "ref_date": None,
          "n_with_date": 0, "n_stale": None, "n_charts": None}
    uni = load_json(UNIVERSE_PATH)
    if uni:
        st["n_universe"] = uni.get("count") or len(uni.get("instruments", []))
    sc = load_json(SCORES_PATH)
    if sc:
        st["scores_at"] = sc.get("generated_at")
        rows = sc.get("scores", [])
        for r in rows:
            k = r.get("signal") or "n.d."
            st["signals"][k] = st["signals"].get(k, 0) + 1
        dates = []
        for r in rows:
            ld = r.get("last_date")
            if ld:
                try:
                    dates.append(datetime.date.fromisoformat(ld))
                except ValueError:
                    pass
        st["n_with_date"] = len(dates)
        if dates:
            ref = max(dates)
            st["ref_date"] = ref
            st["n_stale"] = sum(1 for d in dates if business_days_between(d, ref) > STALE_BUSINESS_DAYS)
    ci = load_json(CHARTS_INDEX_PATH)
    if ci:
        st["n_charts"] = len(ci.get("index", []))
    return st


# ── Stili ──────────────────────────────────────────────────────────────
ss = getSampleStyleSheet()
INK = colors.HexColor("#1a1a1a")
SOFT = colors.HexColor("#555555")
ACCENT = colors.HexColor("#0b3d6e")
LINE = colors.HexColor("#c9c9c9")
HEAD_BG = colors.HexColor("#eef2f6")

H1 = ParagraphStyle("H1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20,
                    textColor=ACCENT, alignment=TA_LEFT, spaceAfter=2)
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontName="Helvetica", fontSize=10, textColor=SOFT, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5,
                    textColor=ACCENT, spaceBefore=12, spaceAfter=4)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10.5,
                    textColor=INK, spaceBefore=8, spaceAfter=3)
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontName="Helvetica", fontSize=9.3, leading=12.8,
                      textColor=INK, spaceAfter=4)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.6, leading=11, spaceAfter=0)
CELLB = ParagraphStyle("CELLB", parent=CELL, fontName="Helvetica-Bold")
WARN = ParagraphStyle("WARN", parent=BODY, textColor=colors.HexColor("#b42318"), fontName="Helvetica-Bold")


def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph("&bull;&nbsp;" + t, ParagraphStyle("B", parent=BODY, leftIndent=11, firstLineIndent=-9, spaceAfter=2))
            for t in items]


def table(rows, widths, header=True):
    data = []
    for i, r in enumerate(rows):
        st = CELLB if (header and i == 0) else CELL
        data.append([Paragraph(str(c), st) for c in r])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), HEAD_BG))
    t.setStyle(TableStyle(style))
    return t


def fmt_it(dt):
    return dt.strftime("%d/%m/%Y alle %H:%M")


def build():
    now = datetime.datetime.now(ZoneInfo("Europe/Rome"))
    st = collect_stats()
    missing = check_code_alignment()
    W = A4[0] - 36 * mm  # larghezza utile
    story = []

    # ── Testata ──
    story.append(P("CORE &mdash; Regole generali", H1))
    story.append(P(f"Documento aggiornato il <b>{fmt_it(now)}</b> (ora italiana) &middot; generato automaticamente a ogni aggiornamento", SUB))

    if missing:
        story.append(P("ATTENZIONE: il codice dei motori non contiene pi&ugrave; alcune soglie citate in questo documento. "
                       "Le regole qui sotto potrebbero non essere allineate. Controllare: "
                       + "; ".join(m.replace("<", "&lt;").replace(">", "&gt;") for m in missing[:6])
                       + ("..." if len(missing) > 6 else ""), WARN))
        print("WARNING: regole PDF potenzialmente non allineate al codice:", missing, file=sys.stderr)

    # Stato dati
    scores_at = st["scores_at"]
    try:
        scores_txt = fmt_it(datetime.datetime.fromisoformat(scores_at).astimezone(ZoneInfo("Europe/Rome"))) if scores_at else "n.d."
    except Exception:
        scores_txt = str(scores_at)
    if st["ref_date"]:
        ref_txt = st["ref_date"].strftime("%d/%m/%Y")
        stale_txt = (f"{st['n_stale']} su {st['n_with_date']} strumenti con data disponibile "
                     f"(ultimo dato fermo da oltre {STALE_BUSINESS_DAYS} giorni di borsa)")
    else:
        ref_txt = "n.d."
        stale_txt = "n.d. (la data dell'ultimo dato per singolo ETF sar&agrave; disponibile dal prossimo aggiornamento)"
    sig = st["signals"]
    sig_txt = " &middot; ".join(f"{k}: {v}" for k, v in sorted(sig.items(), key=lambda x: -x[1])) or "n.d."
    rows = [
        ["Stato all'ultimo aggiornamento", ""],
        ["Score calcolati il", scores_txt],
        ["Strumenti nell'universo", str(st["n_universe"]) if st["n_universe"] is not None else "n.d."],
        ["Ultimo dato disponibile (pi&ugrave; recente dell'universo)", ref_txt],
        ["ETF con dato non aggiornato", stale_txt],
        ["Segnali (Score)", sig_txt],
        ["Grafici di dettaglio disponibili", str(st["n_charts"]) if st["n_charts"] is not None else "n.d."],
    ]
    t = table(rows, [W * 0.38, W * 0.62])
    t.setStyle(TableStyle([("SPAN", (0, 0), (1, 0))]))
    story.append(t)

    story.append(P("Questo documento descrive come CORE seleziona gli strumenti e come produce i segnali. "
                   "&Egrave; uno strumento di screening: non costituisce consulenza, raccomandazione personalizzata "
                   "o garanzia di risultato.", BODY))

    # ── 1. Universo ──
    story.append(P("1. Universo e dati", H2))
    story += bullets([
        "<b>Cosa include:</b> ETF, ETC ed ETN in euro quotati su Borsa Italiana (segmento ETF Plus) e su Xetra (Deutsche B&ouml;rse T7: tipi ETF, ETN, ETC, valuta di regolamento EUR).",
        "<b>Doppia quotazione:</b> uno stesso ISIN quotato su entrambe le borse conta una volta sola; si tiene la riga di Borsa Italiana (ticker .MI).",
        "<b>Fonti:</b> due file ufficiali caricati a mano nel repository (elenco Euronext e file T7 Xetra). Un nuovo ETF entra nell'universo solo dopo l'aggiornamento di questi file, che va fatto periodicamente (consigliato ogni mese). I prezzi giornalieri arrivano da Yahoo Finance.",
        "<b>Classificazione:</b> asset class, leva e tipo prodotto sono dedotti dal nome dello strumento con parole chiave, non da una fonte ufficiale. Le righe marcate <i>da rivedere</i> (default_fallback) sono classificazioni incerte.",
        "<b>Fondi monetari:</b> sono esclusi dai segnali (Score a 0, segnale &laquo;NO TRADE &mdash; money market&raquo;), perch&eacute; il loro andamento produrrebbe segnali di acquisto privi di significato.",
        "<b>Aggiornamento:</b> da luned&igrave; a venerd&igrave; alle 18:00 UTC, dopo la chiusura dei mercati europei.",
    ])

    story.append(P("2. Ultimo dato disponibile per ETF", H2))
    story += bullets([
        "La colonna <b>Ultimo dato</b> mostra la data dell'ultima barra di prezzo ricevuta per quello strumento. Compare anche nell'Excel scaricabile e nell'intestazione del grafico.",
        f"Uno strumento &egrave; segnalato <b>non aggiornato</b> (data in rosso con segno di avviso nella pagina, filtro dedicato) se la sua ultima barra dista pi&ugrave; di {STALE_BUSINESS_DAYS} giorni di borsa aperti (luned&igrave;-venerd&igrave;) dalla data pi&ugrave; recente presente nell'universo.",
        "Il confronto &egrave; con l'ultimo dato dell'universo, non con la data di oggi: nei weekend e nei giorni di chiusura non risulta &laquo;non aggiornato&raquo; l'intero universo. Le festivit&agrave; non sono considerate nel conteggio.",
        "Un ETF poco scambiato pu&ograve; avere barre con prezzo piatto e volume zero: il dato c'&egrave; ma &egrave; poco informativo. Un segnale su uno strumento non aggiornato va verificato prima di essere usato.",
    ])

    # ── 3. Score ──
    story.append(P("3. Score operativo e segnale di screening (colonne Score / Segnale)", H2))
    story.append(P("Calcolato su 1 anno di prezzi giornalieri (servono almeno 50 barre, altrimenti lo strumento resta senza score). "
                   "Lo Score va da 0 a 100: <b>Vento</b> (massimo 70) + <b>Onda</b> (massimo 30), moltiplicato per l'<b>Hard Gate</b>.", BODY))
    story.append(table([
        ["Componente", "Regola", "Punti"],
        ["Hard Gate", "Il prezzo di chiusura deve essere sopra la KAMA. Se non lo &egrave;, lo Score finale &egrave; 0.", "blocca tutto"],
        ["Vento - Direzione", "Prezzo sopra KAMA e KAMA in salita rispetto a 3 barre prima.", "25"],
        ["Vento - Intensit&agrave;", "ADX(14) sopra 22 con +DI maggiore di -DI = trend direzionale. ADX sopra 48 = trend a rischio esaurimento (punteggio ridotto).", "20 (10 se ADX &gt; 48)"],
        ["Vento - Volumi", "Volume dell'ultima barra superiore alla media a 20 barre.", "10"],
        ["Vento - Forza relativa", "Prezzo superiore alla chiusura di 50 sedute fa.", "15"],
        ["Onda - Timing", "Primo pallino SAR rialzista dopo un SAR ribassista = timing fresco. Secondo o terzo pallino consecutivo = timing ancora valido.", "30 / 15"],
    ], [W * 0.2, W * 0.62, W * 0.18]))
    story.append(Spacer(1, 5))
    story.append(table([
        ["Segnale", "Condizione"],
        ["BUY", "Score operativo di almeno 80."],
        ["WATCHLIST - timing fresco", "Score tra 65 e 79 con Onda a 30 (SAR appena scattato, ma Vento non ancora forte)."],
        ["WATCHLIST - trend maturo", "Score tra 65 e 79 con Onda inferiore a 30 (trend forte, ingresso fresco gi&agrave; passato)."],
        ["NO TRADE", "Score sotto 65, oppure prezzo sotto la KAMA (Hard Gate)."],
        ["NO TRADE - money market", "Fondo monetario: sempre escluso dai segnali."],
        ["ANTEPRIMA (flag)", "SAR rialzista al primo o secondo pallino, senza richiedere Hard Gate n&eacute; Vento. &Egrave; un allarme anticipato non confermato, non un segnale operativo."],
        ["SAR Flip UP / DOWN (flag)", "Segnala il primo pallino SAR in assoluto, rialzista o ribassista, indipendentemente da Score e Hard Gate. Non &egrave; un segnale di acquisto o vendita."],
        ["Segnale dal", "Data della barra in cui &egrave; iniziata la serie rialzista del SAR in corso (vuota se il SAR &egrave; ribassista)."],
    ], [W * 0.27, W * 0.73]))

    # ── 4. Motore del grafico ──
    story.append(P("4. Motore dei segnali del grafico (BUY1 / BUY2 / BUY3 / EXIT / MEAN REV / WATCH)", H2))
    story.append(P("Nella pagina di dettaglio ogni barra riceve un codice di segnale, calcolato su 2 anni di prezzi. "
                   "&Egrave; un motore <b>indipendente</b> dallo Score della sezione 3: i due usano vocabolari diversi, "
                   "e un &laquo;BUY&raquo; dello Score non coincide con un &laquo;BUY1/2/3&raquo; del grafico.", BODY))
    story.append(P("Indicatori usati", H3))
    story.append(table([
        ["Indicatore", "Definizione"],
        ["KAMA", "Media mobile adattiva di Kaufman, periodo 10, costanti veloce 2 e lenta 30."],
        ["SAR", "Parabolic SAR, passo 0,02, massimo 0,20. Rialzista se i pallini sono sotto i prezzi."],
        ["AO", "Awesome Oscillator: media a 5 barre meno media a 34 barre del prezzo medio (massimo+minimo)/2. &laquo;AO in miglioramento&raquo; = AO di oggi maggiore di quello di ieri."],
        ["RSI(14)", "Indice di forza relativa a 14 barre, calcolato con medie semplici."],
        ["ER", "Efficiency Ratio a 10 barre: movimento netto diviso movimento totale (da 0 a 1; alto = trend pulito)."],
        ["Baffetti", "Numero di barre consecutive in cui il prezzo medio (massimo+minimo)/2 sale rispetto alla barra precedente."],
        ["Medie allineate", "Chiusura sopra la MM20, MM20 sopra la MM50 e MM50 sopra la MM100 (medie semplici su prezzi di chiusura)."],
        ["Giorni dall'incrocio", "Barre trascorse dall'ultimo passaggio del prezzo sopra o sotto la KAMA."],
        ["Trend (U/L/R)", "Solo informativo, non modifica i segnali. Rialzista se KAMA in salita su 20 barre e prezzo sopra; ribassista se KAMA in calo e prezzo sotto; laterale altrimenti."],
        ["ATR(14) / Renko", "Volatilit&agrave; media a 14 barre; i mattoni Renko usano l'ATR come dimensione."],
    ], [W * 0.2, W * 0.8]))
    story.append(P("Codici di segnale (in ordine di priorit&agrave;: vale il primo che si verifica)", H3))
    story.append(table([
        ["Codice", "Condizione", "Effetto"],
        ["BUY1", "SAR rialzista, incrocio prezzo/KAMA avvenuto da non pi&ugrave; di 3 barre e AO in miglioramento.", "Apre posizione"],
        ["BUY2", "Prezzo sopra la KAMA e almeno 2 Baffetti consecutivi.", "Apre posizione"],
        ["BUY3", "Prezzo sopra la KAMA, ER di almeno 0,50, almeno 3 Baffetti e medie allineate.", "Apre posizione"],
        ["EXIT2", "Prezzo sotto la KAMA e SAR ribassista.", "Chiude posizione"],
        ["EXIT1", "SAR ribassista con prezzo ancora sopra la KAMA (e nessuna condizione di BUY verificata).", "Chiude posizione"],
        ["MEAN REV", "SAR rialzista, ER sotto 0,30, RSI sotto 30, AO in miglioramento e prezzo entro il 3% dalla KAMA oppure sotto di essa.", "Nessun effetto"],
        ["WATCH", "Tutti gli altri casi.", "Nessun effetto"],
    ], [W * 0.13, W * 0.67, W * 0.2]))
    story.append(P("Posizione aperta e storia dei segnali", H3))
    story += bullets([
        "Si apre una posizione al primo BUY1/BUY2/BUY3 quando non ce n'&egrave; una aperta; i BUY successivi a posizione aperta vengono ignorati.",
        "La posizione resta aperta durante WATCH e MEAN REV e si chiude al primo EXIT1 o EXIT2. Se all'ultima barra non c'&egrave; stata uscita, il trade &egrave; <i>OPEN</i> e il risultato non &egrave; realizzato.",
        "Prezzi di entrata e uscita: chiusura della barra in cui compare il segnale. Variazione % = (uscita - entrata) / entrata.",
        "La storia dei segnali &egrave; una simulazione teorica: <b>non include commissioni, spread, slippage n&eacute; tasse</b> e presuppone di poter eseguire alla chiusura.",
        "<b>Win rate</b> = % di trade chiusi con risultato positivo. <b>P&amp;L totale</b> = somma semplice (non composta) delle variazioni % dei trade chiusi. <b>Drawdown</b> = massimo calo della somma cumulata, in punti percentuali.",
        "<b>Uscita ML</b>: sui segnali BUY1 e BUY2 dell'ultima barra, un modello statistico stima il rendimento massimo atteso e i giorni al picco. &Egrave; solo un'indicazione: non chiude posizioni e non modifica i segnali.",
    ])
    story.append(P("Quali ETF hanno il grafico", H3))
    story.append(P("Il grafico di dettaglio e la scheda &laquo;Regole operative&raquo; del singolo ETF vengono generati solo per gli strumenti (Borsa Italiana e Xetra) che nell'ultimo aggiornamento risultano BUY, WATCHLIST o ANTEPRIMA nello Score. "
                   "Per gli altri la colonna Grafico mostra un trattino.", BODY))

    # ── 5. Limiti ──
    story.append(P("5. Limiti noti da tenere presenti", H2))
    story += bullets([
        "<b>Due motori diversi.</b> Score/Segnale (sezione 3) e codici BUY1/2/3, EXIT, MEAN REV (sezione 4) possono dare indicazioni diverse sullo stesso ETF nello stesso giorno.",
        "<b>BUY2 e BUY3 non richiedono il SAR rialzista.</b> Un codice BUY pu&ograve; quindi comparire con SAR ribassista; l'EXIT scatta solo quando nessuna condizione di BUY &egrave; vera.",
        "<b>Dati Yahoo Finance.</b> Possono avere ritardi, buchi o barre non aggiornate; per questo &egrave; mostrata la data dell'ultimo dato. Gli ETF a bassa liquidit&agrave; possono avere prezzi piatti e volumi nulli.",
        "<b>Storia limitata.</b> Gli strumenti quotati da poco hanno meno barre: indicatori e statistiche dei trade sono meno affidabili.",
        "<b>Classificazione per parole chiave.</b> Asset class, leva ed emittente sono dedotti dal nome; per Xetra il riconoscimento dell'emittente &egrave; molto meno affidabile che per Borsa Italiana.",
        "<b>Festivit&agrave;.</b> Il controllo dell'ultimo dato conta solo luned&igrave;-venerd&igrave;: un ETF pu&ograve; risultare non aggiornato con un giorno di margine in meno del reale dopo una festivit&agrave;.",
    ])

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(SOFT)
        canvas.drawString(18 * mm, 10 * mm, f"CORE - Regole generali - aggiornato il {now.strftime('%d/%m/%Y %H:%M')}")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Pagina {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(OUTPUT_PATH, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="CORE - Regole generali", author="CORE")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Scritto {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
