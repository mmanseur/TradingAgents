"""
scan_positions.py — Screener swing TSX
Usage:
    python scan_positions.py                    # Scan TSX 60 complet
    python scan_positions.py RY.TO TD.TO SU.TO  # Tickers custom
    python scan_positions.py --top 10           # Limiter le top N
    python scan_positions.py --min-score 50     # Seuil minimum
"""

import argparse
import sys
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from tradingagents.dataflows.swing_screener import (
    run_screener,
    TSX60_TICKERS,
)

console = Console()

# Couleurs par signal
SIGNAL_STYLE = {
    "ACHAT FORT": "bold green",
    "ACHAT":      "green",
    "NEUTRE":     "yellow",
    "EVITER":     "red",
    "NO DATA":    "dim",
}

SCORE_COLOR = {
    (75, 100): "bold green",
    (55,  74): "green",
    (35,  54): "yellow",
    (0,   34): "red",
}


def score_color(score: int) -> str:
    for (lo, hi), style in SCORE_COLOR.items():
        if lo <= score <= hi:
            return style
    return "white"


def rsi_color(rsi: float) -> str:
    if 45 <= rsi <= 60:   return "bold green"
    if 40 <= rsi < 45:    return "green"
    if 60 < rsi <= 70:    return "yellow"
    if rsi > 70:          return "red"
    if rsi < 35:          return "cyan"
    return "white"


def vol_color(ratio: float) -> str:
    if ratio >= 1.5:   return "bold green"
    if ratio >= 1.2:   return "green"
    if ratio >= 1.0:   return "yellow"
    return "dim"


def build_table(df) -> Table:
    table = Table(
        title=f"[bold cyan]Swing Screener TSX[/bold cyan]  --  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        row_styles=["", "dim"],
        expand=True,
    )

    table.add_column("#",          style="dim",     width=3,  justify="right")
    table.add_column("Ticker",     style="bold",    width=10)
    table.add_column("Signal",     width=12,        justify="center")
    table.add_column("Score",      width=7,         justify="center")
    table.add_column("Cloture",     width=9,         justify="right")
    table.add_column("RSI",        width=7,         justify="center")
    table.add_column("Vol x",       width=7,         justify="center")
    table.add_column("ATR%",       width=7,         justify="center")
    table.add_column("Tendance",   width=9,         justify="center")
    table.add_column("RSI pts",    width=8,         justify="center")
    table.add_column("MACD pts",   width=9,         justify="center")

    for i, row in df.iterrows():
        sig   = row.get("signal", "—")
        sc    = int(row.get("score", 0))
        close = row.get("close", 0)
        rsi   = row.get("rsi_value", 0)
        vr    = row.get("volume_ratio", 0)
        atr   = row.get("atr_pct", 0)
        trend = int(row.get("trend", 0))
        rsi_s = int(row.get("rsi_score", 0))
        macd  = int(row.get("macd_score", 0))

        table.add_row(
            str(i + 1),
            row["ticker"],
            Text(sig, style=SIGNAL_STYLE.get(sig, "white")),
            Text(str(sc), style=score_color(sc)),
            f"{close:.2f}$",
            Text(f"{rsi:.1f}", style=rsi_color(rsi)),
            Text(f"{vr:.1f}x", style=vol_color(vr)),
            f"{atr:.2f}%",
            f"{trend}/30",
            f"{rsi_s}/25",
            f"{macd}/20",
        )

    return table


def print_legend():
    console.print()
    console.print(Rule("[dim]Legende scoring[/dim]", style="dim"))
    legend = (
        "[bold green]ACHAT FORT[/bold green] 75+  "
        "[green]ACHAT[/green] 55-74  "
        "[yellow]NEUTRE[/yellow] 35-54  "
        "[red]EVITER[/red] <35\n"
        "Score = Tendance[dim](30)[/dim] + RSI[dim](25)[/dim] + "
        "MACD[dim](20)[/dim] + Volume[dim](15)[/dim] + ATR%[dim](10)[/dim]"
    )
    console.print(Panel(legend, border_style="dim", padding=(0, 2)))


def main():
    parser = argparse.ArgumentParser(description="Swing Screener TSX")
    parser.add_argument("tickers", nargs="*", help="Tickers custom (ex: RY.TO TD.TO)")
    parser.add_argument("--top",       type=int, default=20,  help="Nombre de résultats (défaut: 20)")
    parser.add_argument("--min-score", type=int, default=0,   help="Score minimum (défaut: 0)")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers] if args.tickers else None
    universe_label = (
        f"[dim]{len(tickers)} tickers custom[/dim]"
        if tickers
        else f"[dim]S&P/TSX 60 - {len(TSX60_TICKERS)} titres[/dim]"
    )

    console.print()
    console.print(Panel(
        f"[bold cyan]TradingAgents - Swing Screener[/bold cyan]\n"
        f"Univers: {universe_label}  |  Top {args.top}  |  Score min: {args.min_score}",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    with console.status("[cyan]Telechargement donnees Yahoo Finance...[/cyan]", spinner="dots"):
        df = run_screener(
            tickers=tickers,
            min_score=args.min_score,
            top_n=args.top,
        )

    if df.empty:
        console.print("[red]Aucun resultat - verifier connexion ou tickers.[/red]")
        sys.exit(1)

    console.print(build_table(df))
    print_legend()

    # Resume rapide
    n_buy_strong = len(df[df["signal"] == "ACHAT FORT"])
    n_buy        = len(df[df["signal"] == "ACHAT"])
    console.print()
    console.print(
        f"[dim]Resultats: [bold green]{n_buy_strong} ACHAT FORT[/bold green]  "
        f"[green]{n_buy} ACHAT[/green]  "
        f"sur {len(df)} scannes[/dim]"
    )
    console.print()


if __name__ == "__main__":
    main()
