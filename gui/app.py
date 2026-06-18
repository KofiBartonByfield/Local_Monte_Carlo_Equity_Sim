"""
gui/app.py
customtkinter desktop GUI for the VaR engine.

Layout
------
Left panel  : Portfolio editor (tickers, weights, value)
Right panel : Controls (method, confidence, horizon, simulations)
Bottom      : Results display + status bar
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import numpy as np

from data.fetcher import fetch_prices, log_returns
from core.portfolio import Portfolio
from core.var_engine import VaREngine, VaRResult


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

METHODS = ["historical", "parametric", "montecarlo"]
CONFIDENCES = ["0.90", "0.95", "0.99"]
HORIZONS = ["1", "5", "10", "21"]  # days


class VaRApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Local VaR Engine")
        self.geometry("900x640")
        self.resizable(True, True)

        self._portfolio = Portfolio.default()
        self._result: VaRResult | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        self._build_left_panel()
        self._build_right_panel()
        self._build_results_panel()
        self._build_status_bar()

    def _build_left_panel(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="Portfolio", font=("", 15, "bold")).grid(
            row=0, column=0, columnspan=3, pady=(10, 5)
        )

        # Header row
        for col, txt in enumerate(["Ticker", "Weight", ""]):
            ctk.CTkLabel(frame, text=txt, font=("", 11)).grid(
                row=1, column=col, padx=5
            )

        self._holding_rows: list[tuple] = []
        self._holdings_frame = frame
        self._next_row = 2

        for ticker, weight in self._portfolio.holdings.items():
            self._add_holding_row(ticker, weight)

        ctk.CTkButton(frame, text="+ Add holding", width=120,
                      command=self._add_holding_row).grid(
            row=99, column=0, columnspan=3, pady=8
        )

        ctk.CTkLabel(frame, text="Portfolio value (£)").grid(
            row=100, column=0, sticky="w", padx=10
        )
        self._value_var = tk.StringVar(value=str(int(self._portfolio.value)))
        ctk.CTkEntry(frame, textvariable=self._value_var, width=120).grid(
            row=100, column=1, padx=5, pady=5
        )

        ctk.CTkLabel(frame, text="Lookback (years)").grid(
            row=101, column=0, sticky="w", padx=10
        )
        self._lookback_var = tk.StringVar(value="3")
        ctk.CTkEntry(frame, textvariable=self._lookback_var, width=80).grid(
            row=101, column=1, padx=5, pady=5
        )

    def _add_holding_row(self, ticker: str = "", weight: float = 0.0):
        row = self._next_row
        t_var = tk.StringVar(value=ticker)
        w_var = tk.StringVar(value=str(round(weight, 4)) if weight else "")

        t_entry = ctk.CTkEntry(self._holdings_frame, textvariable=t_var, width=90)
        w_entry = ctk.CTkEntry(self._holdings_frame, textvariable=w_var, width=70)
        rm_btn = ctk.CTkButton(
            self._holdings_frame, text="✕", width=28, fg_color="transparent",
            command=lambda r=row: self._remove_holding_row(r)
        )

        t_entry.grid(row=row, column=0, padx=5, pady=2)
        w_entry.grid(row=row, column=1, padx=5, pady=2)
        rm_btn.grid(row=row, column=2, padx=2)

        self._holding_rows.append((row, t_var, w_var, t_entry, w_entry, rm_btn))
        self._next_row += 1

    def _remove_holding_row(self, target_row: int):
        self._holding_rows = [
            r for r in self._holding_rows if r[0] != target_row
        ]
        for widget in self._holdings_frame.grid_slaves():
            info = widget.grid_info()
            if int(info.get("row", -1)) == target_row:
                widget.grid_forget()
                widget.destroy()

    def _build_right_panel(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")

        ctk.CTkLabel(frame, text="Parameters", font=("", 15, "bold")).pack(pady=(10, 5))

        def labeled_option(label, values, default):
            ctk.CTkLabel(frame, text=label).pack(anchor="w", padx=15)
            var = tk.StringVar(value=default)
            ctk.CTkOptionMenu(frame, values=values, variable=var).pack(
                padx=15, pady=(0, 8), fill="x"
            )
            return var

        self._method_var = labeled_option("Method", METHODS, "montecarlo")
        self._confidence_var = labeled_option("Confidence", CONFIDENCES, "0.95")
        self._horizon_var = labeled_option("Horizon (days)", HORIZONS, "1")

        ctk.CTkLabel(frame, text="MC Simulations").pack(anchor="w", padx=15)
        self._sims_var = tk.StringVar(value="10000")
        ctk.CTkEntry(frame, textvariable=self._sims_var).pack(
            padx=15, pady=(0, 15), fill="x"
        )

        ctk.CTkButton(
            frame, text="Run VaR", height=40, command=self._run
        ).pack(padx=15, pady=5, fill="x")

        ctk.CTkButton(
            frame, text="Run All Methods", height=32,
            fg_color="transparent", border_width=1,
            command=self._run_all
        ).pack(padx=15, pady=5, fill="x")

    def _build_results_panel(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="nsew")
        frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        ctk.CTkLabel(frame, text="Results", font=("", 13, "bold")).grid(
            row=0, column=0, columnspan=5, pady=(6, 4)
        )

        labels = ["Method", "Confidence", "Horizon", "VaR %", "VaR £", "CVaR %", "CVaR £"]
        self._result_vars: list[tk.StringVar] = []

        for col, lbl in enumerate(labels):
            ctk.CTkLabel(frame, text=lbl, font=("", 10), text_color="gray").grid(
                row=1, column=col, padx=8
            )
            var = tk.StringVar(value="—")
            ctk.CTkLabel(frame, textvariable=var, font=("", 12, "bold")).grid(
                row=2, column=col, padx=8, pady=(0, 8)
            )
            self._result_vars.append(var)

        # Stats row
        self._stats_label = ctk.CTkLabel(
            frame, text="", font=("", 10), text_color="gray"
        )
        self._stats_label.grid(row=3, column=0, columnspan=7, pady=(0, 6))

    def _build_status_bar(self):
        self._status_var = tk.StringVar(value="Ready.")
        ctk.CTkLabel(self, textvariable=self._status_var, font=("", 10),
                     text_color="gray").grid(
            row=2, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 4)
        )

    # ------------------------------------------------------------------
    # Runtime logic
    # ------------------------------------------------------------------

    def _parse_portfolio(self) -> Portfolio | None:
        try:
            holdings = {}
            for _, t_var, w_var, *_ in self._holding_rows:
                ticker = t_var.get().strip().upper()
                weight_str = w_var.get().strip()
                if ticker and weight_str:
                    holdings[ticker] = float(weight_str)
            value = float(self._value_var.get())
            p = Portfolio(name="Custom", holdings=holdings, value=value)
            return p
        except Exception as e:
            messagebox.showerror("Portfolio Error", str(e))
            return None

    def _run(self):
        threading.Thread(target=self._compute, daemon=True).start()

    def _run_all(self):
        threading.Thread(target=self._compute_all, daemon=True).start()

    def _set_status(self, msg: str):
        self._status_var.set(msg)

    def _compute(self):
        portfolio = self._parse_portfolio()
        if not portfolio:
            return
        method = self._method_var.get()
        confidence = float(self._confidence_var.get())
        horizon = int(self._horizon_var.get())
        n_sims = int(self._sims_var.get())
        lookback = float(self._lookback_var.get())

        self._set_status(f"Fetching data for {portfolio.tickers}…")
        try:
            prices = fetch_prices(portfolio.tickers, period_years=lookback)
            rets = log_returns(prices)
        except Exception as e:
            self._set_status(f"Data error: {e}")
            return

        self._set_status(f"Running {method} VaR…")
        try:
            engine = VaREngine(rets, portfolio.weights, portfolio.value)
            result = engine.compute(method, confidence, horizon, n_sims)
            stats = engine.summary_stats()
        except Exception as e:
            self._set_status(f"Compute error: {e}")
            return

        self._display_result(result, stats)
        self._set_status(
            f"Done. {stats['obs']} obs used. "
            f"Ann.Vol={stats['ann_vol']:.1%}  Sharpe={stats['sharpe']:.2f}"
        )

    def _compute_all(self):
        portfolio = self._parse_portfolio()
        if not portfolio:
            return
        confidence = float(self._confidence_var.get())
        horizon = int(self._horizon_var.get())
        n_sims = int(self._sims_var.get())
        lookback = float(self._lookback_var.get())

        self._set_status(f"Fetching data…")
        try:
            prices = fetch_prices(portfolio.tickers, period_years=lookback)
            rets = log_returns(prices)
        except Exception as e:
            self._set_status(f"Data error: {e}")
            return

        engine = VaREngine(rets, portfolio.weights, portfolio.value)
        results = []
        for m in METHODS:
            self._set_status(f"Running {m}…")
            try:
                r = engine.compute(m, confidence, horizon, n_sims)
                results.append(r)
                print(r)  # also log to terminal
            except Exception as e:
                self._set_status(f"{m} failed: {e}")
                return

        # Show last result in panel; print all to terminal
        stats = engine.summary_stats()
        self._display_result(results[-1], stats)
        summary = " | ".join(f"{r.method}: {r.var_pct:.2%}" for r in results)
        self._set_status(f"All methods done → {summary}")

    def _display_result(self, r: VaRResult, stats: dict):
        values = [
            r.method,
            r.confidence_label,
            f"{r.horizon_days}d",
            f"{r.var_pct:.2%}",
            f"£{r.var_abs:,.0f}",
            f"{r.cvar_pct:.2%}",
            f"£{r.cvar_abs:,.0f}",
        ]
        for var, val in zip(self._result_vars, values):
            var.set(val)

        self._stats_label.configure(
            text=(
                f"Ann. Return {stats['ann_return']:.1%}  |  "
                f"Ann. Vol {stats['ann_vol']:.1%}  |  "
                f"Sharpe {stats['sharpe']:.2f}  |  "
                f"Skew {stats['skewness']:.2f}  |  "
                f"Kurt {stats['kurtosis']:.2f}  |  "
                f"Min daily {stats['min_daily']:.2%}  |  "
                f"Obs {stats['obs']}"
            )
        )


def run():
    app = VaRApp()
    app.mainloop()
