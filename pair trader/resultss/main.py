
import sys
sys.path.insert(0, "/home/user/pairs_trader")

from data.generate import generate_universe
from cointegration import scan_pairs
from backtest import walk_forward_backtest
from tearsheet import compute_metrics, plot_tearsheet


def main():
    print("=" * 60)
    print("  STATISTICAL ARBITRAGE — PAIRS TRADING")
    print("  Walk-Forward Backtest System")
    print("=" * 60)

    print("\n[1/4] Generating synthetic price universe...")
    prices, true_pairs = generate_universe(n_days=756, seed=42)
    print(f"      {prices.shape[1]} tickers, {prices.shape[0]} trading days (~3 years)")
    for a, b, beta in true_pairs:
        print(f"        TRUE PAIR: {a} / {b}  (β={beta:.3f})")

    print("\n[2/4] Scanning for cointegrated pairs...")
    found = scan_pairs(prices)
    if not found.empty:
        print(f"      Found {len(found)} pairs (p<0.05, half-life 5-60 days):")
        true_set = {(a, b) for a, b, _ in true_pairs}
        for _, row in found.iterrows():
            tag = "✓ TRUE" if (row.ticker_a, row.ticker_b) in true_set else "  noise"
            print(f"        {tag}  {row.ticker_a}/{row.ticker_b}  "
                  f"p={row.p_value:.3f}  HL={row.half_life_days:.1f}d")

    print("\n[3/4] Running walk-forward backtest...")
    print("      (train=252d, test=63d, rolling every quarter)")
    results = walk_forward_backtest(prices, train_days=252, test_days=63)
    print(f"      Pair-windows traded: {results['total_pairs_traded']}")

    print("\n[4/4] Computing metrics and saving tearsheet...")
    metrics = compute_metrics(results["daily_pnl"], results["equity_curve"])
    print("\n" + "─" * 40)
    for k, v in metrics.items():
        if not k.startswith("_"): print(f"  {k:<22} {v}")
    print("─" * 40)
    plot_tearsheet(results["daily_pnl"], results["equity_curve"], results["pair_log"])
    print("\nDone. See results/tearsheet.png")


if __name__ == "__main__":
    main()