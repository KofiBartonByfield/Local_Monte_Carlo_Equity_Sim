"""
core/var_engine.py
Object-oriented VaR engine supporting three methods:
  - Historical Simulation
  - Parametric (variance-covariance, assumes normality)
  - Monte Carlo (GBM, configurable paths & horizon)

All methods return a VaRResult dataclass for clean downstream consumption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy import stats
from typing import Literal

VaRMethod = Literal["historical", "parametric", "montecarlo"]


@dataclass
class VaRResult:
    method: str
    confidence: float          # e.g. 0.95
    horizon_days: int
    portfolio_value: float

    var_pct: float             # VaR as % of portfolio
    var_abs: float             # VaR in £/$ terms
    cvar_pct: float            # Conditional VaR (Expected Shortfall)
    cvar_abs: float

    # Monte Carlo extras (None for other methods)
    simulated_pnl: np.ndarray | None = field(default=None, repr=False)

    @property
    def confidence_label(self) -> str:
        return f"{self.confidence*100:.0f}%"

    def __str__(self) -> str:
        return (
            f"[{self.method.upper()}] {self.confidence_label} VaR "
            f"({self.horizon_days}d): {self.var_pct:.2%} | "
            f"£{self.var_abs:,.0f}  |  CVaR: {self.cvar_pct:.2%}"
        )


class VaREngine:
    """
    Compute Value-at-Risk for a weighted portfolio of assets.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns, shape (T, N).
    weights : np.ndarray
        Portfolio weights, shape (N,). Must sum to 1.
    portfolio_value : float
        Total portfolio value in base currency.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        portfolio_value: float = 100_000.0,
    ) -> None:
        if len(weights) != returns.shape[1]:
            raise ValueError("weights length must match number of assets")
        weights = np.asarray(weights, dtype=float)
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(f"weights must sum to 1, got {weights.sum():.4f}")

        self.returns = returns
        self.weights = weights
        self.portfolio_value = portfolio_value

        # Pre-compute portfolio return series (weighted sum of log returns)
        self.port_returns: np.ndarray = returns.values @ weights

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        method: VaRMethod = "historical",
        confidence: float = 0.95,
        horizon_days: int = 1,
        n_simulations: int = 10_000,
    ) -> VaRResult:
        """Dispatch to the chosen method and return a VaRResult."""
        if method == "historical":
            return self._historical(confidence, horizon_days)
        elif method == "parametric":
            return self._parametric(confidence, horizon_days)
        elif method == "montecarlo":
            return self._montecarlo(confidence, horizon_days, n_simulations)
        else:
            raise ValueError(f"Unknown method: {method!r}")

    # ------------------------------------------------------------------
    # Method implementations
    # ------------------------------------------------------------------

    def _historical(self, confidence: float, horizon_days: int) -> VaRResult:
        """
        Historical simulation. Scale single-day losses to horizon via
        square-root-of-time rule (standard approximation).
        """
        scaled = self.port_returns * np.sqrt(horizon_days)
        var_pct = float(-np.percentile(scaled, (1 - confidence) * 100))
        # CVaR = mean of losses beyond VaR threshold
        tail = scaled[scaled <= -var_pct]
        cvar_pct = float(-tail.mean()) if len(tail) > 0 else var_pct

        return VaRResult(
            method="historical",
            confidence=confidence,
            horizon_days=horizon_days,
            portfolio_value=self.portfolio_value,
            var_pct=var_pct,
            var_abs=var_pct * self.portfolio_value,
            cvar_pct=cvar_pct,
            cvar_abs=cvar_pct * self.portfolio_value,
        )

    def _parametric(self, confidence: float, horizon_days: int) -> VaRResult:
        """
        Parametric (variance-covariance). Assumes normally distributed returns.
        Uses full covariance matrix for the portfolio.
        """
        cov = np.cov(self.returns.values, rowvar=False)
        port_var = float(self.weights @ cov @ self.weights)
        port_vol = np.sqrt(port_var * horizon_days)  # scale to horizon

        z = stats.norm.ppf(1 - confidence)
        var_pct = float(-z * port_vol)  # positive number = loss

        # CVaR for normal: E[X | X < -VaR] = -phi(z) / (1-c) * sigma
        phi_z = stats.norm.pdf(z)
        cvar_pct = float(phi_z / (1 - confidence) * port_vol)

        return VaRResult(
            method="parametric",
            confidence=confidence,
            horizon_days=horizon_days,
            portfolio_value=self.portfolio_value,
            var_pct=var_pct,
            var_abs=var_pct * self.portfolio_value,
            cvar_pct=cvar_pct,
            cvar_abs=cvar_pct * self.portfolio_value,
        )

    def _montecarlo(
        self,
        confidence: float,
        horizon_days: int,
        n_simulations: int,
    ) -> VaRResult:
        """
        Monte Carlo simulation via Geometric Brownian Motion with Cholesky
        decomposition to preserve asset correlations.
        """
        n_assets = self.returns.shape[1]
        mu = self.returns.mean().values          # shape (N,)
        cov = np.cov(self.returns.values, rowvar=False)

        # Cholesky factor for correlated draws
        L = np.linalg.cholesky(cov)

        rng = np.random.default_rng(seed=42)
        # Draw iid standard normals: (horizon, sims, assets)
        Z = rng.standard_normal((horizon_days, n_simulations, n_assets))

        # Correlated returns per day: Z @ L.T
        correlated = Z @ L.T + mu  # broadcast mu over (horizon, sims)

        # Cumulative portfolio log-return over horizon
        cum_port_return = (correlated @ self.weights).sum(axis=0)  # (sims,)

        var_pct = float(-np.percentile(cum_port_return, (1 - confidence) * 100))
        tail = cum_port_return[cum_port_return <= -var_pct]
        cvar_pct = float(-tail.mean()) if len(tail) > 0 else var_pct

        return VaRResult(
            method="montecarlo",
            confidence=confidence,
            horizon_days=horizon_days,
            portfolio_value=self.portfolio_value,
            var_pct=var_pct,
            var_abs=var_pct * self.portfolio_value,
            cvar_pct=cvar_pct,
            cvar_abs=cvar_pct * self.portfolio_value,
            simulated_pnl=cum_port_return * self.portfolio_value,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary_stats(self) -> dict:
        """Basic portfolio-level descriptive statistics."""
        r = self.port_returns
        return {
            "ann_return": float(r.mean() * 252),
            "ann_vol": float(r.std() * np.sqrt(252)),
            "sharpe": float((r.mean() * 252) / (r.std() * np.sqrt(252))),
            "skewness": float(stats.skew(r)),
            "kurtosis": float(stats.kurtosis(r)),
            "min_daily": float(r.min()),
            "max_daily": float(r.max()),
            "obs": len(r),
        }
