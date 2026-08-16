#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAPTOR Portafoglio — Scan Nuove Partenze (email mattutina)
Gira una volta al giorno la mattina. NON fa alcun fetch Yahoo Finance:
legge etf_scores.json già generato ogni sera dal workflow di 'core'
(github.com/Giorgiogoldoni/core, aggiornamento 18:00 UTC), e filtra
gli strumenti con un ingresso "fresco" (SAR passato rialzista da 1-2
giorni, motore già calcolato da core) tra equity/commodity/leva.
Invia un'unica email con al massimo 15 nomi, ordinati per punteggio.
"""

import json, os, smtplib, urllib.request, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SCORES_URL   = 'https://raw.githubusercontent.com/Giorgiogoldoni/core/main/data/etf_scores.json'
ASSET_CLASSI = {'equity', 'commodity', 'leva_short'}  # esclude bond/crypto per scelta esplicita
SAR_AGE_MAX  = 2     # "nuova partenza" = SAR flip rialzista da al massimo 2 giorni
MAX_RESULTS  = 15    # tetto email, per non spammare


def fetch_scores():
    with urllib.request.urlopen(SCORES_URL, timeout=30) as r:
        return json.loads(r.read().decode())


def filtra_nuove_partenze(scores):
    items = scores.get('scores', [])
    fresh = [
        i for i in items
        if i.get('signal') == 'BUY'
        and i.get('sar_age') is not None and i.get('sar_age') <= SAR_AGE_MAX
        and i.get('asset_class') in ASSET_CLASSI
        and i.get('sar_is_up')
    ]
    fresh.sort(key=lambda x: -(x.get('score_operativo') or 0))
    return fresh[:MAX_RESULTS]


def build_email_html(items, generated_at):
    rows = ''
    for i in items:
        tv_map = {'.MI': 'MIL:', '.DE': 'XETR:', '.PA': 'EURONEXT:', '.L': 'LSE:'}
        yahoo = i['ticker_yf']
        tv_pref, tv_sym = 'MIL:', yahoo.split('.')[0]
        for suf, pref in tv_map.items():
            if yahoo.endswith(suf):
                tv_pref = pref
                break
        tv_url = f"https://www.tradingview.com/chart/?symbol={tv_pref}{tv_sym}"
        rows += f"""<tr>
          <td style="padding:6px 10px;font-family:monospace;font-weight:700">{yahoo}</td>
          <td style="padding:6px 10px;font-size:12px;color:#444">{i.get('name','')[:45]}</td>
          <td style="padding:6px 10px;font-size:11px;color:#888">{i.get('asset_class','')}</td>
          <td style="padding:6px 10px;text-align:right;font-weight:700">{i.get('score_operativo')}</td>
          <td style="padding:6px 10px;text-align:center">{i.get('sar_age')}g</td>
          <td style="padding:6px 10px;text-align:right;font-family:monospace">{i.get('close','—')}</td>
          <td style="padding:6px 10px"><a href="{tv_url}" style="color:#1a6fcf">TradingView</a></td>
        </tr>"""

    return f"""<html><body style="font-family:Arial,sans-serif;background:#f6f8fa;padding:20px">
    <div style="max-width:760px;margin:0 auto;background:#fff;border-radius:8px;overflow:hidden;border:1px solid #e1e4e8">
      <div style="background:#1a1816;color:#fff;padding:16px 20px">
        <h2 style="margin:0;font-size:18px">🟢 Nuove Partenze — {len(items)} strumenti</h2>
        <div style="font-size:11px;opacity:.7">Dati core aggiornati: {generated_at}</div>
      </div>
      <table style="width:100%;border-collapse:collapse">
        <thead><tr style="background:#f6f8fa;font-size:11px;color:#666;text-align:left">
          <th style="padding:6px 10px">Ticker</th><th style="padding:6px 10px">Nome</th>
          <th style="padding:6px 10px">Classe</th><th style="padding:6px 10px;text-align:right">Score</th>
          <th style="padding:6px 10px">SAR età</th><th style="padding:6px 10px;text-align:right">Prezzo</th>
          <th style="padding:6px 10px">Grafico</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <div style="padding:12px 20px;font-size:10px;color:#999">
        Filtro: segnale BUY, SAR passato rialzista da ≤{SAR_AGE_MAX} giorni, classe equity/commodity/leva.
        Fonte dati: <a href="https://giorgiogoldoni.github.io/core/">core</a> (non è un consiglio di investimento).
      </div>
    </div>
    </body></html>"""


def send_email(subject, html):
    EMAIL_USER = os.environ.get('EMAIL_USER', '')
    EMAIL_PASS = os.environ.get('EMAIL_PASS', '')
    if not EMAIL_USER or not EMAIL_PASS:
        print("EMAIL non configurata — skip")
        return
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = EMAIL_USER
    msg['To']      = EMAIL_USER
    msg.attach(MIMEText(html, 'html'))
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as srv:
        srv.login(EMAIL_USER, EMAIL_PASS)
        srv.sendmail(EMAIL_USER, EMAIL_USER, msg.as_string())
    print(f"✅ Email inviata: {subject}")


def check_freschezza(scores, max_ore_feriale=36):
    """Verifica che i dati di core non siano troppo vecchi. Soglia più larga il lunedì
    (core gira solo lun-ven: da venerdì 18:00 UTC a lunedì mattina passano ~60h di norma,
    non è un'anomalia)."""
    gen = scores.get('generated_at')
    if not gen:
        return False, None, "⚠️ etf_scores.json non ha il campo generated_at — impossibile verificare l'età dei dati"
    try:
        gen_dt = datetime.datetime.fromisoformat(gen)
        if gen_dt.tzinfo is not None:
            gen_dt = gen_dt.replace(tzinfo=None)
        now_utc = datetime.datetime.utcnow()
        eta_ore = (now_utc - gen_dt).total_seconds() / 3600
    except Exception as e:
        return False, None, f"⚠️ Impossibile interpretare generated_at ('{gen}'): {e}"

    max_ore = max_ore_feriale + 24 if now_utc.weekday() == 0 else max_ore_feriale  # lunedì: +24h di margine weekend
    if eta_ore > max_ore:
        return False, eta_ore, (f"⚠️ Dati di core vecchi di {eta_ore:.0f} ore (soglia {max_ore}h) — "
                                 f"il workflow di aggiornamento di core potrebbe essersi fermato")
    return True, eta_ore, None


def build_warning_email_html(msg, eta_ore, generated_at):
    return f"""<html><body style="font-family:Arial,sans-serif;background:#f6f8fa;padding:20px">
    <div style="max-width:600px;margin:0 auto;background:#fff;border-radius:8px;overflow:hidden;border:1px solid #e1e4e8">
      <div style="background:#862e2e;color:#fff;padding:16px 20px">
        <h2 style="margin:0;font-size:16px">⚠️ Dati core non aggiornati</h2>
      </div>
      <div style="padding:16px 20px;font-size:13px;color:#333">
        <p>{msg}</p>
        <p style="font-size:11px;color:#888">generated_at riportato: {generated_at or '—'}
        {f' · età: {eta_ore:.0f}h' if eta_ore is not None else ''}</p>
        <p style="font-size:11px;color:#888">Nessuna scansione "nuove partenze" eseguita oggi — verifica il workflow
        di <a href="https://github.com/Giorgiogoldoni/core/actions">core</a>.</p>
      </div>
    </div>
    </body></html>"""


def main():
    now = datetime.datetime.now()
    print(f"Scan Nuove Partenze — {now.strftime('%Y-%m-%d %H:%M')}")
    try:
        scores = fetch_scores()
    except Exception as e:
        print(f"❌ Errore lettura etf_scores.json: {e}")
        return

    ok, eta_ore, msg = check_freschezza(scores)
    if not ok:
        print(msg)
        html = build_warning_email_html(msg, eta_ore, scores.get('generated_at'))
        send_email(f"⚠️ RAPTOR — dati core non aggiornati · {now.strftime('%d/%m %H:%M')}", html)
        return
    print(f"✅ Dati core freschi (età: {eta_ore:.1f}h)")

    top = filtra_nuove_partenze(scores)
    print(f"Trovati {len(top)} strumenti (max {MAX_RESULTS})")

    # Salva sempre il JSON per il bottone "Nuove Partenze" in pagina, anche se vuoto oggi
    output = {
        'generated_at': now.isoformat(),
        'core_generated_at': scores.get('generated_at'),
        'count': len(top),
        'items': [{
            'ticker': i['ticker_yf'], 'nome': i.get('name', ''),
            'asset_class': i.get('asset_class'), 'score': i.get('score_operativo'),
            'sar_age': i.get('sar_age'), 'prezzo': i.get('close'),
        } for i in top],
    }
    with open('nuove_partenze.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)

    if not top:
        print("Nessuna nuova partenza oggi — nessuna email inviata")
        return

    html = build_email_html(top, scores.get('generated_at', '—'))
    subj = f"🟢 Nuove Partenze — {len(top)} strumenti · {now.strftime('%d/%m')}"
    send_email(subj, html)


if __name__ == '__main__':
    main()
