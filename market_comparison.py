"""
Market Quote Comparison: Brulant v1.2 digital prices vs Deribit implied.
========================================================================
Fetches BTC option quotes from Deribit public API, converts vanilla call
prices to digital option prices via call-spread approximation, and compares
against Brulant v1.2 Monte Carlo digital prices.

This anchors the model to reality -- moment matching is internal consistency,
but market comparison is external validity.
"""
from __future__ import annotations

import json
import time
import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

from experiment_v12 import simulate_v12
from benchmark_v12 import BRULANT_V12

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False


def fetch_deribit_options(currency: str = "BTC") -> List[Dict]:
    """Fetch all active option book summaries from Deribit public API."""
    if not _HAS_REQUESTS:
        raise RuntimeError("requests library required for Deribit API access")

    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
    resp = requests.get(url, params={"currency": currency, "kind": "option"}, timeout=30)
    data = resp.json()
    if "result" not in data:
        raise RuntimeError(f"Deribit API error: {data}")
    return data["result"]


def parse_instrument_name(name: str) -> Dict[str, Any]:
    """Parse Deribit instrument name like 'BTC-28MAR25-90000-C'."""
    parts = name.split("-")
    if len(parts) != 4:
        return {}
    return {
        "underlying": parts[0],
        "expiry_str": parts[1],
        "strike": float(parts[2]),
        "option_type": parts[3],  # C or P
    }


def deribit_expiry_to_datetime(expiry_str: str) -> datetime.datetime:
    """Convert '28MAR25' to datetime (expiry at 08:00 UTC on Deribit)."""
    months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
              "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
    day = int(expiry_str[:len(expiry_str)-5])
    month_str = expiry_str[len(expiry_str)-5:len(expiry_str)-2]
    year = 2000 + int(expiry_str[-2:])
    month = months.get(month_str.upper(), 1)
    return datetime.datetime(year, month, day, 8, 0, 0, tzinfo=datetime.timezone.utc)


def digital_from_calls(strikes: np.ndarray, call_prices: np.ndarray,
                       delta_k: float = 500.0) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate digital call prices from vanilla calls via call-spread.

    Digital(K) ~ [C(K - delta) - C(K + delta)] / (2 * delta)

    Returns digital prices and the strikes where computation is valid.
    """
    from scipy.interpolate import interp1d

    # Fit a smooth interpolation to call prices
    if len(strikes) < 3:
        return np.array([]), np.array([])

    sorted_idx = np.argsort(strikes)
    strikes_sorted = strikes[sorted_idx]
    prices_sorted = call_prices[sorted_idx]

    f = interp1d(strikes_sorted, prices_sorted, kind='linear', fill_value='extrapolate')

    # Compute digital prices at each strike
    valid_strikes = strikes_sorted[(strikes_sorted > strikes_sorted[0] + delta_k) &
                                    (strikes_sorted < strikes_sorted[-1] - delta_k)]

    digital_prices = np.array([
        (f(K - delta_k) - f(K + delta_k)) / (2 * delta_k)
        for K in valid_strikes
    ])

    # Clip to [0, 1] range
    digital_prices = np.clip(digital_prices, 0.0, 1.0)

    return valid_strikes, digital_prices


def price_v12_digital(S0: float, strikes: np.ndarray, hours: float,
                      num_paths: int = 200000, seed: int = 42) -> np.ndarray:
    """Price digital calls using Brulant v1.2."""
    T = hours / (24.0 * 365.0)
    n_steps = max(1, int(hours * 60))
    dt = T / n_steps
    _, S_T = simulate_v12(n_steps, dt, num_paths, seed=seed, S0=S0, **BRULANT_V12)
    return np.array([np.mean(S_T >= K) for K in strikes])


def main():
    print("=" * 70)
    print("  MARKET COMPARISON: Brulant v1.2 vs Deribit Option Quotes")
    print("=" * 70)

    # Fetch current BTC spot
    try:
        resp = requests.get("https://api.binance.com/api/v3/ticker/price",
                            params={"symbol": "BTCUSDT"}, timeout=10)
        S0 = float(resp.json()["price"])
    except Exception:
        S0 = 85000.0
    print(f"\n  Spot: ${S0:,.2f}")

    # Fetch Deribit options
    print("  Fetching Deribit option quotes...")
    try:
        all_options = fetch_deribit_options("BTC")
    except Exception as e:
        print(f"  ERROR: Could not fetch Deribit data: {e}")
        print("  Deribit may be unavailable. Saving empty results.")
        Path("market_comparison_results.json").write_text(
            json.dumps({"error": str(e)}), encoding="utf-8")
        return

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    print(f"  Retrieved {len(all_options)} option instruments")

    # Group calls by expiry
    expiry_calls: Dict[str, List] = {}
    for opt in all_options:
        parsed = parse_instrument_name(opt.get("instrument_name", ""))
        if not parsed or parsed["option_type"] != "C":
            continue
        # mid price in BTC terms
        bid = opt.get("bid_price") or 0.0
        ask = opt.get("ask_price") or 0.0
        if bid <= 0 or ask <= 0:
            continue
        mid = (bid + ask) / 2.0
        expiry_str = parsed["expiry_str"]
        if expiry_str not in expiry_calls:
            expiry_calls[expiry_str] = []
        expiry_calls[expiry_str].append({
            "strike": parsed["strike"],
            "mid_btc": mid,
            "mid_usd": mid * S0,
            "bid_usd": bid * S0,
            "ask_usd": ask * S0,
        })

    # Find nearest expiries (1-7 days out)
    results = {}
    for expiry_str, calls in sorted(expiry_calls.items()):
        try:
            expiry_dt = deribit_expiry_to_datetime(expiry_str)
        except (ValueError, KeyError):
            continue
        hours_to_expiry = (expiry_dt - now_utc).total_seconds() / 3600.0
        days_to_expiry = hours_to_expiry / 24.0

        if days_to_expiry < 0.5 or days_to_expiry > 7.0:
            continue
        if len(calls) < 5:
            continue

        print(f"\n  Expiry: {expiry_str} ({days_to_expiry:.1f}d, {hours_to_expiry:.0f}h)")

        strikes = np.array([c["strike"] for c in calls])
        call_mids_usd = np.array([c["mid_usd"] for c in calls])

        # Market-implied digital prices (call-spread approximation)
        digital_strikes, market_digital = digital_from_calls(strikes, call_mids_usd, delta_k=500.0)
        if len(digital_strikes) < 3:
            print(f"    Skipping: too few valid strikes for digital approximation")
            continue

        # Brulant v1.2 digital prices at the same strikes
        model_digital = price_v12_digital(S0, digital_strikes, hours_to_expiry,
                                          num_paths=200000, seed=42)

        # Comparison
        abs_err = np.abs(model_digital - market_digital)
        rmse = float(np.sqrt(np.mean(abs_err ** 2)))
        max_err = float(np.max(abs_err))
        mean_err = float(np.mean(abs_err))

        print(f"    {len(digital_strikes)} strikes compared")
        print(f"    RMSE: {rmse:.6f} | Max error: {max_err:.6f} | Mean error: {mean_err:.6f}")
        print(f"    {'Strike':>10s} {'Market':>10s} {'Model':>10s} {'Error':>10s}")
        print(f"    {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for j, K in enumerate(digital_strikes):
            print(f"    ${int(K):>8,} {market_digital[j]:>10.6f} {model_digital[j]:>10.6f} "
                  f"{abs_err[j]:>10.6f}")

        results[expiry_str] = {
            "days_to_expiry": days_to_expiry,
            "hours_to_expiry": hours_to_expiry,
            "strikes": digital_strikes.tolist(),
            "market_digital": market_digital.tolist(),
            "model_digital": model_digital.tolist(),
            "rmse": rmse,
            "max_error": max_err,
            "mean_error": mean_err,
        }

    # Summary
    if results:
        all_rmse = [r["rmse"] for r in results.values()]
        print(f"\n  Overall RMSE across {len(results)} expiries: "
              f"mean={np.mean(all_rmse):.6f} max={np.max(all_rmse):.6f}")

    # Save
    output = {
        "timestamp": now_utc.isoformat(),
        "spot": S0,
        "expiries": results,
    }
    out_path = Path("market_comparison_results.json")
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")

    # Plot
    if _HAS_PLT and results:
        n_exp = min(len(results), 4)
        fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 4), squeeze=False)
        for i, (exp_str, r) in enumerate(list(results.items())[:n_exp]):
            ax = axes[0, i]
            strikes = np.array(r["strikes"])
            ax.plot(strikes / 1000, r["market_digital"], "o-", label="Market (Deribit)",
                    color="#3498db", markersize=4)
            ax.plot(strikes / 1000, r["model_digital"], "s--", label="Brulant v1.2",
                    color="#e74c3c", markersize=4)
            ax.axvline(S0 / 1000, color="black", linestyle=":", alpha=0.5)
            ax.set_xlabel("Strike ($k)")
            ax.set_ylabel("Digital Price")
            ax.set_title(f"{exp_str} ({r['days_to_expiry']:.1f}d)\nRMSE={r['rmse']:.4f}",
                         fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Brulant v1.2 vs Deribit Market Digital Prices", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig("market_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print("Saved market_comparison.png")


if __name__ == "__main__":
    main()
