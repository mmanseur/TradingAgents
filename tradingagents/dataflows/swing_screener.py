"""
Swing Screener — TSX focused
Filtre et score les meilleures opportunités swing sur le TSX.
Données: Yahoo Finance Canada (.TO suffix)
Scoring: 100 pts — Tendance + RSI + MACD + Volume + ATR
"""

import logging
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from .stockstats_utils import yf_retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Univers S&P/TSX 60
# ---------------------------------------------------------------------------
TSX60_TICKERS = [
    # Banques
    "RY.TO", "TD.TO", "BNS.TO", "BMO.TO", "CM.TO", "NA.TO",
    # Assurances / Finances
    "MFC.TO", "SLF.TO", "GWO.TO", "IFC.TO", "FFH.TO", "POW.TO",
    # Énergie
    "ENB.TO", "TRP.TO", "CNQ.TO", "SU.TO", "CVE.TO", "PPL.TO",
    # Télécommunications
    "BCE.TO", "T.TO", "RCI-B.TO", "QBR-B.TO",
    # Services publics
    "FTS.TO", "EMA.TO", "H.TO", "AQN.TO",
    # Matériaux / Mines
    "ABX.TO", "AEM.TO", "WPM.TO", "K.TO", "TECK-B.TO",
    "FM.TO", "CCO.TO", "NTR.TO", "CS.TO", "AGI.TO", "HBM.TO",
    # Technologie
    "SHOP.TO", "CSU.TO", "OTEX.TO", "GIB-A.TO",
    # Consommation
    "ATD.TO", "L.TO", "MRU.TO", "DOL.TO", "EMP-A.TO",
    # Transport / Industriel
    "CNR.TO", "CP.TO", "TFI.TO", "CAE.TO", "MG.TO",
    # Diversifié / Holdings
    "BN.TO", "BAM.TO", "WN.TO", "TRI.TO",
    # Ingénierie
    "WSP.TO", "STN.TO",
]


# ---------------------------------------------------------------------------
# Fetch OHLCV (cache local via yfinance)
# ---------------------------------------------------------------------------

def _fetch_ohlcv(ticker: str, period_days: int = 300) -> pd.DataFrame:
    """Télécharge OHLCV pour un ticker TSX. Retourne DataFrame vide si échec."""
    try:
        obj = yf.Ticker(ticker)
        df = yf_retry(lambda: obj.history(period=f"{period_days}d", auto_adjust=True))
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c.replace(" ", "_") for c in df.columns]
        df = df.rename(columns={"Date": "Date", "Open": "Open", "High": "High",
                                 "Low": "Low", "Close": "Close", "Volume": "Volume"})
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.sort_values("Date").reset_index(drop=True)
        return df
    except Exception as e:
        logger.debug(f"{ticker}: fetch error — {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Scoring swing
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_swing_score(df: pd.DataFrame) -> dict:
    """
    Score swing trading 0–100 pour un titre.

    Retourne un dict avec :
      score       — total (int)
      signal      — BUY STRONG / BUY / NEUTRAL / AVOID
      close, rsi_value, atr_pct, volume_ratio
      trend, rsi_score, macd_score, volume_score, atr_score (composantes)
    """
    if df.empty or len(df) < 55:
        return {"score": 0, "signal": "NO DATA", "error": "insufficient_data"}

    close  = df["Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # --- Indicateurs ---
    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    rsi = _rsi(close)

    ema12       = close.ewm(span=12, adjust=False).mean()
    ema26       = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist   = macd_line - signal_line

    atr     = _atr(high, low, close)
    vol_avg = volume.rolling(20).mean()

    # --- Valeurs actuelles ---
    c   = float(close.iloc[-1])
    s20 = float(sma20.iloc[-1])
    s50 = float(sma50.iloc[-1])
    s200 = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else c
    r   = float(rsi.iloc[-1])
    ml  = float(macd_line.iloc[-1])
    sl  = float(signal_line.iloc[-1])
    mh  = float(macd_hist.iloc[-1])
    pmh = float(macd_hist.iloc[-2])
    a   = float(atr.iloc[-1])
    v   = float(volume.iloc[-1])
    va  = float(vol_avg.iloc[-1])

    atr_pct   = (a / c) * 100 if c else 0
    vol_ratio = (v / va) if va else 0

    # ---- 1. TENDANCE — 30 pts ----
    trend_score = 0
    if c > s20:          trend_score += 15
    if s20 > s50:        trend_score += 10
    if c > s200:         trend_score += 5

    # ---- 2. RSI zone swing — 25 pts ----
    rsi_score = 0
    if 45 <= r <= 60:    rsi_score = 25   # Zone idéale
    elif 40 <= r < 45:   rsi_score = 18   # Rebond oversold
    elif 60 < r <= 65:   rsi_score = 15   # Momentum mais attention
    elif 35 <= r < 40:   rsi_score = 8    # Setup oversold possible
    elif r > 70:         rsi_score = 3    # Overbought

    # ---- 3. MACD — 20 pts ----
    macd_score = 0
    if ml > sl:          macd_score += 10  # Croisement haussier
    if mh > pmh:         macd_score += 10  # Histogramme croissant

    # ---- 4. VOLUME — 15 pts ----
    volume_score = 0
    if vol_ratio >= 1.5:   volume_score = 15
    elif vol_ratio >= 1.2: volume_score = 10
    elif vol_ratio >= 1.0: volume_score = 5

    # ---- 5. ATR% optimal swing — 10 pts ----
    atr_score = 0
    if 1.5 <= atr_pct <= 4.0:             atr_score = 10
    elif 1.0 <= atr_pct < 1.5 or 4.0 < atr_pct <= 6.0: atr_score = 5

    total = trend_score + rsi_score + macd_score + volume_score + atr_score

    # Signal qualitatif
    if total >= 75:    signal = "ACHAT FORT"
    elif total >= 55:  signal = "ACHAT"
    elif total >= 35:  signal = "NEUTRE"
    else:              signal = "EVITER"

    return {
        "score":        total,
        "signal":       signal,
        "close":        round(c, 2),
        "sma20":        round(s20, 2),
        "sma50":        round(s50, 2),
        "rsi_value":    round(r, 1),
        "atr_pct":      round(atr_pct, 2),
        "volume_ratio": round(vol_ratio, 2),
        # Composantes
        "trend":        trend_score,
        "rsi_score":    rsi_score,
        "macd_score":   macd_score,
        "volume_score": volume_score,
        "atr_score":    atr_score,
    }


# ---------------------------------------------------------------------------
# Screener principal
# ---------------------------------------------------------------------------

def run_screener(
    tickers: Optional[list] = None,
    min_score: int = 0,
    top_n: int = 20,
    curr_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Lance le screener sur la liste de tickers (défaut: TSX 60).

    Returns:
        DataFrame classé par score décroissant, colonnes:
        ticker, signal, score, close, rsi_value, atr_pct,
        volume_ratio, trend, rsi_score, macd_score, volume_score, atr_score
    """
    universe = tickers or TSX60_TICKERS
    results = []

    for i, ticker in enumerate(universe, 1):
        logger.info(f"[{i}/{len(universe)}] Screening {ticker}...")
        df = _fetch_ohlcv(ticker)
        if df.empty:
            logger.debug(f"{ticker}: no data")
            continue

        row = compute_swing_score(df)
        if "error" in row:
            continue

        row["ticker"] = ticker
        results.append(row)

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out = out[out["score"] >= min_score]
    out = out.sort_values("score", ascending=False).head(top_n)
    out = out.reset_index(drop=True)

    # Ordre colonnes
    cols = [
        "ticker", "signal", "score",
        "close", "rsi_value", "atr_pct", "volume_ratio",
        "trend", "rsi_score", "macd_score", "volume_score", "atr_score",
        "sma20", "sma50",
    ]
    return out[[c for c in cols if c in out.columns]]
