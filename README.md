# Local_Monte_Carlo_Equity_Sim

A correlated Monte Carlo simulation engine for portfolio Value-at-Risk, with Historical Simulation and Parametric benchmarks. Built on real market data pulled live from Yahoo Finance. Runs as a native desktop application.

<img width="1127" height="838" alt="monte_carlo_engine" src="https://github.com/user-attachments/assets/491b00a3-99f8-4b9b-99d8-e84349145406" />


---

## The problem

Standard VaR implementations make a silent, damaging assumption: that asset returns are independent. A two-asset portfolio of VWRP and XMWX is not two independent bets — they share factor exposures (global equity beta, currency, rate sensitivity). Ignoring that correlation causes you to systematically underestimate joint drawdown risk, precisely when it matters most: correlated assets tend to fall together in stress scenarios.

This engine addresses that directly. The Monte Carlo path generator preserves the full empirical cross-asset correlation structure via Cholesky decomposition of the covariance matrix, producing a joint return distribution that reflects how these assets actually co-move.

---

## Algorithms

### 1. Monte Carlo — Correlated GBM

The core method. For an N-asset portfolio:

**Step 1 — Estimate parameters from historical data**
```
μ  = E[r]          (N×1 vector of daily log-return means)
Σ  = Cov[r]        (N×N covariance matrix)
```

**Step 2 — Cholesky factorisation**
```
Σ = LLᵀ    where L is lower-triangular
```
This decomposes the covariance structure so correlated draws can be generated from independent normals. For a well-conditioned covariance matrix this is O(N³) once and then O(N) per simulation path.

**Step 3 — Simulate paths**
```
For each day d in horizon h, simulation s:
    Z[d,s]  ~ N(0, I)          (N independent standard normals)
    r[d,s]  = μ + L @ Z[d,s]  (correlated, mean-shifted returns)
```

**Step 4 — Aggregate to portfolio P&L**
```
R_portfolio[s] = Σ_d  (w · r[d,s])    (sum over horizon days)
```
where `w` is the weight vector. This gives a simulated distribution of portfolio log-returns over the full horizon.

**Step 5 — Extract risk measures**
```
VaR(c)  = -percentile(R_portfolio, 1-c)
CVaR(c) = -E[R_portfolio | R_portfolio ≤ -VaR(c)]
```

Default: 10,000 paths. Increasing to 100,000 tightens the CVaR estimate significantly for the deep tail (99th percentile).

---

### 2. Parametric (Variance-Covariance)

Analytic closed form under the assumption that portfolio returns are normally distributed. No simulation required.

**Portfolio variance:**
```
σ²_p = wᵀ Σ w
```

**Scaled to horizon h (square-root-of-time):**
```
σ_p(h) = σ_p × √h
```

**VaR and CVaR:**
```
VaR(c)  = -z_c × σ_p(h)                    where z_c = Φ⁻¹(1-c)
CVaR(c) = φ(z_c) / (1-c) × σ_p(h)         where φ is the standard normal PDF
```

The CVaR formula is analytic for the normal distribution. This method is fast and interpretable but will underestimate tail risk for any asset with meaningful skewness or excess kurtosis — both of which are empirically present in equity return series.

**When to trust it:** as a quick sanity check and as a lower bound. If Monte Carlo produces a materially higher number, that gap is the cost of the normality assumption.

---

### 3. Historical Simulation

Distribution-free. No parametric assumptions. Uses the empirical return distribution directly.

```
port_returns[t] = w · r[t]          (weighted portfolio return each day)
VaR(c)          = -percentile(port_returns, 1-c)
CVaR(c)         = -E[port_returns | port_returns ≤ -VaR(c)]
```

Horizon scaling uses the square-root-of-time approximation:
```
VaR(c, h) = VaR(c, 1) × √h
```

This is the most honest method for capturing empirical tail behaviour — it will reflect the actual fat tails and negative skewness observed in historical equity returns. Its limitation is sample dependency: with 3 years of daily data (~756 observations), the 99th percentile estimate rests on roughly 7–8 observations. The Monte Carlo method sidesteps this by generating a much larger synthetic sample from the same estimated distribution.

---

## Why the three methods diverge

In normal market regimes, all three methods converge. The interesting case is when they don't:

- **Historical > Parametric** → the empirical distribution has fatter tails than the normal assumption. The assets have experienced drawdowns that a Gaussian model would assign near-zero probability.
- **Monte Carlo ≈ Parametric** → the simulated distribution is close to normal. The Cholesky-transformed GBM produces normal marginals; if the historical data is also approximately normal, the two methods will agree.
- **Monte Carlo > Historical** → unusual but possible if the historical window was unusually calm. The MC model is extrapolating from estimated parameters into a larger synthetic sample.

Running all three simultaneously and comparing the spread is itself a diagnostic.

---

## Portfolio risk measures output

| Measure | Formula | Interpretation |
|---|---|---|
| VaR(c, h) | -Q_{1-c}(R) | Maximum expected loss at confidence c over h days |
| CVaR(c, h) | -E[R \| R ≤ -VaR] | Expected loss given breach — coherent risk measure |
| Ann. Volatility | σ × √252 | Annualised standard deviation of portfolio returns |
| Sharpe Ratio | (μ × 252) / (σ × √252) | Return per unit of risk (assumes Rf = 0) |
| Skewness | E[(r-μ)³] / σ³ | Negative = left tail heavier — bad for long-only portfolios |
| Excess Kurtosis | E[(r-μ)⁴] / σ⁴ - 3 | Positive = fatter tails than normal |

---

## Stack

- **Python 3.12**
- `numpy` — matrix ops, Cholesky, simulation
- `scipy.stats` — normal quantiles, analytic CVaR, descriptive stats
- `pandas` — return series construction, covariance estimation
- `yfinance` — live market data with in-session cache
- `customtkinter` — native desktop GUI, threaded to keep UI responsive

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/equity-var-sim
cd equity-var-sim
pip install -r requirements.txt
python main.py
```

---

## Known limitations

**Square-root-of-time scaling** holds exactly only under i.i.d. normality. Equity returns exhibit volatility clustering (GARCH effects) and autocorrelation, both of which cause the true multi-day VaR to exceed the scaled estimate. A proper multi-day Monte Carlo (simulating the full path day-by-day) is implemented here; the SRT approximation is only used in the Historical method.

**Covariance estimation** uses the full sample mean and covariance. This is sensitive to the lookback window and treats all observations equally. Exponentially weighted covariance (giving more weight to recent observations) would produce more responsive risk estimates.

**Normal marginals in Monte Carlo.** The Cholesky GBM produces normally distributed marginal returns. This correctly captures correlation but misses the empirical fat tails of individual assets. A copula-based approach (e.g. a Student-t copula) would preserve both the correlation structure and the tail behaviour.

## Planned extensions

- Backtesting with Kupiec proportion-of-failures test (regulatory VaR validation)
- Component and marginal VaR — per-asset risk contribution decomposition
- Return distribution plot with VaR/CVaR threshold marked
- Exponentially weighted covariance estimation
- Stress scenarios — GFC 2008, COVID March 2020, 2022 rate shock
