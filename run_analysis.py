"""Script non-interactif pour lancer TradingAgents analysis via l'API directe."""
import sys
import json
import os
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

def analyze_ticker(ticker: str, date: str = None, output_language: str = "French"):
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env.enterprise"), override=False)

    if date is None:
        from datetime import datetime
        date = datetime.now().strftime("%Y-%m-%d")

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "anthropic"
    config["deep_think_llm"] = "claude-sonnet-4-6"
    config["quick_think_llm"] = "claude-haiku-4-5"
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["output_language"] = output_language
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }

    # Tous les analysts
    selected_analysts = ["market", "social", "news", "fundamentals"]

    print(f"\n{'='*60}")
    print(f"Analyzing {ticker} as of {date}")
    print(f"{'='*60}")

    try:
        graph = TradingAgentsGraph(
            selected_analysts,
            config=config,
            debug=False,
        )
        _, decision = graph.propagate(ticker, date)
        print(f"\n{'='*60}")
        print(f"RESULT for {ticker}:")
        print(f"{'='*60}")
        print(json.dumps(decision, indent=2, default=str))
        return decision
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["RY"]
    results = {}
    for t in tickers:
        results[t] = analyze_ticker(t)
    # Save combined results
    output_path = os.path.join(os.path.dirname(__file__), "reports", "ta_analysis.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
