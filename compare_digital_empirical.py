"""
Compare model digital prices against empirical (historical) P(S_T >= K).
Uses rolling windows over 15-min BTC candles to compute realized digital prices.
"""
from __future__ import annotations
import json, sys
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# --- Load model benchmark results ---
bench = json.loads(Path("digital_benchmark.json").read_text(encoding="utf-8"))
MONEYNESS = np.array(bench["moneyness"])
model_prices = bench["digital_prices"]  # {expiry: {model: [prices]}}
models = list(next(iter(model_prices.values())).keys())

# --- Fetch historical data ---
print("Fetching 15-min BTC candles...")
try:
    import requests
    url = "https://api.binance.com/api/v3/klines"
    all_candles = []
    end_time = None
    for _ in range(10):  # 10 x 1000 = 10000 candles ~ 104 days
        params = {"symbol": "BTCUSDT", "interval": "15m", "limit": 1000}
        if end_time:
            params["endTime"] = end_time - 1
        r = requests.get(url, params=params, timeout=30)
        batch = r.json()
        if not batch:
            break
        all_candles = batch + all_candles
        end_time = batch[0][0]
    closes = np.array([float(c[4]) for c in all_candles])
    print(f"  Got {len(closes)} 15-min closes ({len(closes)/96:.0f} days)")
except Exception as e:
    print(f"  Error fetching data: {e}")
    sys.exit(1)

# --- Compute empirical digital prices ---
EXPIRIES_BARS = {"4h": 16, "1d": 96, "3d": 288, "7d": 672}

print("\n" + "=" * 75)
print("  EMPIRICAL vs MODEL DIGITAL PRICES")
print("=" * 75)

all_mae = {m: [] for m in models}
all_results = {}

for exp_name, n_bars in EXPIRIES_BARS.items():
    # Rolling windows: S0 = closes[i], S_T = closes[i + n_bars]
    n_windows = len(closes) - n_bars
    if n_windows < 100:
        print(f"\n  {exp_name}: insufficient data ({n_windows} windows), skipping")
        continue

    S0_arr = closes[:n_windows]
    ST_arr = closes[n_bars:n_bars + n_windows]
    ratios = ST_arr / S0_arr  # S_T / S_0

    # Empirical P(S_T >= K) = P(S_T/S0 >= moneyness)
    emp_prices = np.array([np.mean(ratios >= m) for m in MONEYNESS])

    print(f"\n{'=' * 75}")
    print(f"  EXPIRY: {exp_name}  ({n_windows:,} rolling windows)")
    print(f"{'=' * 75}")

    # Header
    header = f"  {'K/S':>6s} {'Empirical':>10s}"
    for name in models:
        header += f" {name[:10]:>10s}"
    print(header)
    print(f"  {'-' * 6} {'-' * 10}" + "".join(f" {'-' * 10}" for _ in models))

    # Print prices
    for j, m in enumerate(MONEYNESS):
        row = f"  {m:>6.2f} {emp_prices[j]:>10.4f}"
        for name in models:
            mp = model_prices[exp_name][name][j]
            row += f" {mp:>10.4f}"
        print(row)

    # Compute errors per model (exclude SABR from ranking — degenerate)
    print(f"\n  {'Model':<14s} {'MAE':>8s} {'RMSE':>8s} {'MaxErr':>8s}  Best strikes")
    print(f"  {'-' * 14} {'-' * 8} {'-' * 8} {'-' * 8}  {'-' * 30}")

    exp_results = {}
    for name in models:
        mp = np.array(model_prices[exp_name][name])
        errors = mp - emp_prices
        abs_errors = np.abs(errors)
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        max_err = np.max(abs_errors)
        # Find which strikes this model is closest on
        best_k = np.argsort(abs_errors)[:3]
        best_str = ", ".join(f"{MONEYNESS[k]:.2f}({abs_errors[k]:.4f})" for k in best_k)
        print(f"  {name:<14s} {mae:>8.4f} {rmse:>8.4f} {max_err:>8.4f}  {best_str}")
        all_mae[name].append(mae)
        exp_results[name] = {"mae": mae, "rmse": rmse, "max_err": max_err}

    all_results[exp_name] = exp_results

# --- Overall ranking ---
print(f"\n{'=' * 75}")
print("  OVERALL RANKING: MAE vs EMPIRICAL (mean across expiries)")
print(f"{'=' * 75}")
print(f"  {'Rank':>4s} {'Model':<14s}", end="")
for exp in EXPIRIES_BARS:
    if exp in all_results:
        print(f" {exp:>8s}", end="")
print(f" {'Average':>8s}")
print(f"  {'-' * 4} {'-' * 14}" + "".join(f" {'-' * 8}" for _ in all_results) + f" {'-' * 8}")

ranked = sorted(models, key=lambda m: np.mean(all_mae[m]))
for rank, name in enumerate(ranked, 1):
    row = f"  {rank:>4d} {name:<14s}"
    for exp in EXPIRIES_BARS:
        if exp in all_results:
            row += f" {all_results[exp][name]['mae']:>8.4f}"
    avg = np.mean(all_mae[name])
    row += f" {avg:>8.4f}"
    print(row)

# --- Key insight: at near-ATM vs wings ---
print(f"\n{'=' * 75}")
print("  NEAR-ATM (0.97-1.03) vs WINGS (<=0.95, >=1.05) MAE")
print(f"{'=' * 75}")
atm_mask = (MONEYNESS >= 0.97) & (MONEYNESS <= 1.03)
wing_mask = (MONEYNESS <= 0.95) | (MONEYNESS >= 1.05)

print(f"  {'Model':<14s} {'ATM MAE':>8s} {'Wing MAE':>8s} {'Ratio':>6s}")
print(f"  {'-' * 14} {'-' * 8} {'-' * 8} {'-' * 6}")

for name in ranked:
    atm_errs = []
    wing_errs = []
    for exp in EXPIRIES_BARS:
        if exp not in all_results:
            continue
        mp = np.array(model_prices[exp][name])
        # Recompute empirical for this expiry
        n_bars = EXPIRIES_BARS[exp]
        n_windows = len(closes) - n_bars
        S0_arr = closes[:n_windows]
        ST_arr = closes[n_bars:n_bars + n_windows]
        ratios = ST_arr / S0_arr
        emp = np.array([np.mean(ratios >= m) for m in MONEYNESS])
        abs_err = np.abs(mp - emp)
        atm_errs.extend(abs_err[atm_mask])
        wing_errs.extend(abs_err[wing_mask])
    atm_mae = np.mean(atm_errs)
    wing_mae = np.mean(wing_errs)
    ratio = atm_mae / wing_mae if wing_mae > 0 else float('inf')
    print(f"  {name:<14s} {atm_mae:>8.4f} {wing_mae:>8.4f} {ratio:>6.2f}")

print()
