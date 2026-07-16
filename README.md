Statistical Arbitrage — Pairs Trading System
A production-style quantitative trading system implementing statistical arbitrage via pairs trading, built entirely from scratch using NumPy and SciPy — no statsmodels or black-box wrappers.

What It Does
Identifies cointegrated price pairs in a universe of assets, generates mean-reversion trading signals using z-scores, and validates performance through rigorous walk-forward backtesting — the same methodology used at quantitative trading firms.

Architecture
```
pairs_trader/
├── data/
│   └── generate.py        # Synthetic universe: OU-process cointegrated pairs + noise
├── cointegration.py       # ADF test + Engle-Granger from scratch 
├── signals.py             # Z-score state machine + Kelly Criterion sizing
├── backtest.py            # Walk-forward backtester with transaction costs
├── tearsheet.py           # Performance metrics + 4-panel chart
└── main.py                # End-to-end pipeline entry point
```

Technical Implementation

Cointegration Testing (from scratch)
The Engle-Granger two-step test is implemented without statsmodels:

Step 1 — OLS regression:** `spread_t = Y_t - β·X_t - α`
Step 2 — ADF test on spread:**
```
Δy_t = α + β·y_{t-1} + Σγ_i·Δy_{t-i} + ε_t
```
The t-statistic on `β` is compared to MacKinnon (1994) critical values. If `p < 0.05`, the pair is cointegrated — the spread is stationary and mean-reverting.

Half-life estimation** via OU process fit:
```
half_life = -ln(2) / ln(ρ)   where ρ = 1 + θ
```
Pairs with half life < 5 days (costs dominate) or > 60 days (capital tied up too long) are filtered out.

Signal Generation
```python
# Entry: spread is 2 standard deviations from rolling mean
z_t < -2.0  →  long spread  (buy A, short B)
z_t > +2.0  →  short spread (sell A, buy B)

# Exit
|z_t| < 0.5  →  close position

# Stop loss
|z_t| > 3.5  →  emergency exit
```

Signals are generated via an explicit **state machine** (not vectorised) to avoid subtle lookahead bias that vectorised approaches introduce.

Walk-Forward Backtesting
Window | Days | Purpose 
| Training | 252 | Fit hedge ratio β via OLS |
| Testing | 63 | Trade on unseen data |
| Roll | +63 | Slide forward one quarter |

The hedge ratio is **re-fit every quarter** — never using future data. This prevents the most common source of inflated backtest results.

Transaction Costs
```python
COMMISSION = 0.10%  per side
SLIPPAGE   = 0.05%  per side
ROUND_TRIP = 0.30%  total
```

Position Sizing — Kelly Criterion
```
f* = (p·b − q) / b
```
where `p` = win rate, `b` = avg_win/avg_loss. **Half-Kelly** is used in practice, capped at 25%, reducing variance while preserving long-run growth optimality.

Results

| Metric | Value |
| Total Return | +1.20% |
| Sharpe Ratio | 0.43 |
| Max Drawdown | −2.09% |
| Profit Factor | 1.11 |
| Win Rate | 49.3% |
| Pairs Traded | 35 windows |

> Returns are intentionally conservative — synthetic data uses small OU noise, and Kelly defaults to 2% per pair until trade history accumulates. On real cointegrated pairs (GLD/SLV, XOM/CVX), position sizes and returns scale accordingly.

Scanner Validation

The universe contains 3 cointegrated pairs mixed with 8 independent random walks. The scanner correctly identifies all 3 true pairs (100% recall):

```
✓ TRUE  ASSET_X1/ASSET_Y1  p=0.005  HL=7.1d
✓ TRUE  ASSET_X2/ASSET_Y2  p=0.005  HL=9.2d
✓ TRUE  ASSET_X3/ASSET_Y3  p=0.005  HL=7.5d
  noise  NOISE_4/NOISE_5   p=0.038  HL=27.1d   ← ~5 expected by chance at α=0.05
```

False positives are expected: 91 pairs tested at α=0.05 → ~4.5 false hits by pure chance. The walk-forward test filters these out — spurious pairs don't trade profitably on unseen data.

Installation & Usage

```bash
# Clone
git clone https://github.com/roykongwalei/pairs-trading
cd pairs-trading

# Install dependencies
pip install numpy scipy pandas matplotlib seaborn

# Run full pipeline
python main.py

# Use real market data instead of synthetic
# Edit main.py and replace generate_universe() with:
import yfinance as yf
import numpy as np
raw = yf.download(["GLD","SLV","XOM","CVX","KO","PEP"], period="3y")["Close"]
prices = np.log(raw.dropna())
```

 Key Concepts Demonstrated

| Concept | Implementation |
| Cointegration vs. correlation | Engle-Granger test from scratch |
| Lookahead bias prevention | Walk-forward & state-machine signals |
| Transaction cost modelling | Commission + slippage on every trade |
| Multiple comparisons problem | Expected false positives calculated |
| Position sizing theory | Kelly Criterion with half-Kelly cap |
| Spread stationarity | OU process half-life estimation |


