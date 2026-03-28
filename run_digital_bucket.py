"""
Digital Option Pricing — Bucket Analysis
=========================================
Dense moneyness grid, then aggregate into ±2%, ±2-4%, ±4-6%, ±6-10% buckets.
Reports RMSE, MAE, and signed bias per bucket per model vs empirical.
"""
from __future__ import annotations
import json, time, sys
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from multi_scale_benchmark import simulate_v11_excitation
from experiment_v12 import simulate_v12
from benchmark_comparison import (
    simulate_gbm, simulate_heston, simulate_merton, simulate_sabr,
)
from fit_sandpile import _to_jsonable

NUM_PATHS = 200_000
N_SEEDS = 3

# Dense moneyness grid: 1% steps from 0.90 to 1.10
MONEYNESS = np.arange(0.90, 1.105, 0.01)
MONEYNESS = np.round(MONEYNESS, 2)

EXPIRIES = {"4h": 4, "1d": 24, "3d": 72, "7d": 168}

# Buckets defined as (label, lo_moneyness, hi_moneyness)
BUCKETS = [
    ("ITM 6-10%",  0.90, 0.94),
    ("ITM 4-6%",   0.94, 0.96),
    ("ITM 2-4%",   0.96, 0.98),
    ("ITM 0-2%",   0.98, 1.00),
    ("ATM",        0.995, 1.005),
    ("OTM 0-2%",   1.00, 1.02),
    ("OTM 2-4%",   1.02, 1.04),
    ("OTM 4-6%",   1.04, 1.06),
    ("OTM 6-10%",  1.06, 1.10),
]


def simulate_terminal(tag, params, S0, n_steps, dt, num_paths, seed):
    p = {k: v for k, v in params.items() if not k.startswith('_')}
    if tag == "v11_exc":
        _, S_T = simulate_v11_excitation(n_steps, dt, num_paths, seed=seed, S0=S0, **p)
    elif tag == "v12":
        _, S_T = simulate_v12(n_steps, dt, num_paths, seed=seed, S0=S0, **p)
    elif tag == "gbm":
        _, S_T = simulate_gbm(n_steps, dt, num_paths, S0, p["sigma"], seed=seed)
    elif tag == "heston":
        _, S_T = simulate_heston(n_steps, dt, num_paths, S0, **p, seed=seed)
    elif tag == "merton":
        _, S_T = simulate_merton(n_steps, dt, num_paths, S0, **p, seed=seed)
    elif tag == "sabr":
        _, S_T = simulate_sabr(n_steps, dt, num_paths, S0, **p, seed=seed)
    else:
        raise ValueError(f"Unknown tag: {tag}")
    return np.asarray(S_T).ravel()


def price_digitals(tag, params, S0, strikes, hours, num_paths, seed):
    T = hours / (24.0 * 365.0)
    n_steps = max(1, int(hours * 4))
    dt = T / n_steps
    S_T = simulate_terminal(tag, params, S0, n_steps, dt, num_paths, seed)
    prices = np.array([np.mean(S_T >= K) for K in strikes])
    return prices


# ============================================================================
#  FETCH EMPIRICAL DATA
# ============================================================================
print("Fetching 15-min BTC candles for empirical benchmark...")
try:
    import requests
    url = "https://api.binance.com/api/v3/klines"
    all_candles = []
    end_time = None
    for _ in range(10):
        params = {"symbol": "BTCUSDT", "interval": "15m", "limit": 1000}
        if end_time:
            params["endTime"] = end_time - 1
        r = requests.get(url, params=params, timeout=30)
        batch = r.json()
        if not batch:
            break
        all_candles = batch + all_candles
        end_time = batch[0][0]
    all_closes = np.array([float(c[4]) for c in all_candles])
    print(f"  Got {len(all_closes)} closes ({len(all_closes)/96:.0f} days)")
except Exception as e:
    print(f"  Error: {e}")
    sys.exit(1)

# OUT-OF-SAMPLE ONLY: models were calibrated on the first 2500 bars of the
# most-recent 5000-bar window.  Use only the second half (newest 5000 bars,
# ~52 days) so empirical prices are purely out-of-sample.
oos_start = len(all_closes) // 2
closes = all_closes[oos_start:]
print(f"  Out-of-sample window: last {len(closes)} bars ({len(closes)/96:.0f} days)")

# Compute empirical digital prices per expiry
EXPIRIES_BARS = {"4h": 16, "1d": 96, "3d": 288, "7d": 672}
empirical = {}
for exp_name, n_bars in EXPIRIES_BARS.items():
    n_win = len(closes) - n_bars
    ratios = closes[n_bars:n_bars + n_win] / closes[:n_win]
    empirical[exp_name] = np.array([np.mean(ratios >= m) for m in MONEYNESS])
    print(f"  {exp_name}: {n_win:,} windows")

# ============================================================================
#  LOAD MODELS & SIMULATE
# ============================================================================
print("\nLoading calibrated params from multi_scale_benchmark.json...")
prev = json.loads(Path("multi_scale_benchmark.json").read_text(encoding="utf-8"))
pp = prev["model_params"]

try:
    S0 = float(requests.get("https://api.binance.com/api/v3/ticker/price",
                            params={"symbol": "BTCUSDT"}, timeout=10).json()["price"])
except Exception:
    S0 = 67000.0

strikes = S0 * MONEYNESS
print(f"  Spot: ${S0:,.2f}  |  {len(MONEYNESS)} strikes ({MONEYNESS[0]:.0%}-{MONEYNESS[-1]:.0%})")

base_exc = dict(mu0=0, rho=1.3, nu=1.6, alpha=10, beta=0.0,
                phi=1.6, sigma_Y=0.20, kappa=15, theta_p=1.5,
                gamma=20, eta=1, eps=1e-3, jump_to_ret=False)

models = {
    "GBM":          ("gbm",     pp["GBM"]),
    "Heston":       ("heston",  pp["Heston"]),
    "Merton":       ("merton",  pp["Merton"]),
    "SABR":         ("sabr",    pp["SABR"]),
    "Brulant v1.2": ("v12",     pp["Brulant v1.2"]),
    "Exc-A":        ("v11_exc", {**base_exc, "sigma0": 0.42, "lambda0": 3.0,
                                  "exc_beta": 10.0, "exc_kappa": 100.0, "alpha_exc": 60.0}),
    "Exc (opt)":    ("v11_exc", {**base_exc, "sigma0": 0.40, "lambda0": 3.0,
                                  "exc_beta": 10.0, "exc_kappa": 130.0, "alpha_exc": 200.0}),
}

# Price all models across all expiries
all_model_prices = {}  # {exp: {model: array}}
n_total = len(models) * len(EXPIRIES)
done = 0

for exp_name, hours in EXPIRIES.items():
    all_model_prices[exp_name] = {}
    for name, (tag, params) in models.items():
        t0 = time.perf_counter()
        seed_prices = []
        for s in range(N_SEEDS):
            seed = 42 + s * 137
            p = price_digitals(tag, params, S0, strikes, hours, NUM_PATHS, seed)
            seed_prices.append(p)
        avg = np.mean(seed_prices, axis=0)
        all_model_prices[exp_name][name] = avg
        done += 1
        elapsed = time.perf_counter() - t0
        pct = done * 100 // n_total
        print(f"  [{pct:>3d}%] {exp_name} {name:<14s} ({elapsed:.0f}s)")

# ============================================================================
#  BUCKET ANALYSIS
# ============================================================================
print(f"\n{'='*90}")
print("  DIGITAL PRICING ERROR BY MONEYNESS BUCKET (vs Empirical)")
print(f"  {NUM_PATHS:,} paths x {N_SEEDS} seeds | Metric: RMSE across strikes in bucket")
print(f"{'='*90}")

# Skip SABR (degenerate)
model_names = [m for m in models if m != "SABR"]

for exp_name in EXPIRIES:
    print(f"\n  EXPIRY: {exp_name}")
    # Header
    header = f"  {'Bucket':<14s}"
    for name in model_names:
        header += f" {name[:10]:>10s}"
    print(header)
    print(f"  {'-'*14}" + "".join(f" {'-'*10}" for _ in model_names))

    emp = empirical[exp_name]

    for bkt_name, bkt_lo, bkt_hi in BUCKETS:
        mask = (MONEYNESS >= bkt_lo) & (MONEYNESS <= bkt_hi)
        if not np.any(mask):
            continue
        row = f"  {bkt_name:<14s}"
        for name in model_names:
            mp = all_model_prices[exp_name][name]
            errs = mp[mask] - emp[mask]
            rmse = np.sqrt(np.mean(errs**2))
            row += f" {rmse:>10.4f}"
        print(row)

# ============================================================================
#  SIGNED BIAS (model - empirical, averaged over strikes in bucket)
# ============================================================================
print(f"\n{'='*90}")
print("  SIGNED BIAS BY BUCKET (model - empirical, mean across strikes)")
print(f"  Positive = model overprices digital = assigns too much prob to S_T >= K")
print(f"{'='*90}")

for exp_name in EXPIRIES:
    print(f"\n  EXPIRY: {exp_name}")
    header = f"  {'Bucket':<14s}"
    for name in model_names:
        header += f" {name[:10]:>10s}"
    print(header)
    print(f"  {'-'*14}" + "".join(f" {'-'*10}" for _ in model_names))

    emp = empirical[exp_name]

    for bkt_name, bkt_lo, bkt_hi in BUCKETS:
        mask = (MONEYNESS >= bkt_lo) & (MONEYNESS <= bkt_hi)
        if not np.any(mask):
            continue
        row = f"  {bkt_name:<14s}"
        for name in model_names:
            mp = all_model_prices[exp_name][name]
            bias = np.mean(mp[mask] - emp[mask])
            row += f" {bias:>+10.4f}"
        print(row)

# ============================================================================
#  OVERALL RANKING BY BUCKET (RMSE averaged across expiries)
# ============================================================================
print(f"\n{'='*90}")
print("  OVERALL RANKING: RMSE BY BUCKET (averaged across all expiries)")
print(f"{'='*90}")

header = f"  {'Bucket':<14s}"
for name in model_names:
    header += f" {name[:10]:>10s}"
print(header)
print(f"  {'-'*14}" + "".join(f" {'-'*10}" for _ in model_names))

for bkt_name, bkt_lo, bkt_hi in BUCKETS:
    mask = (MONEYNESS >= bkt_lo) & (MONEYNESS <= bkt_hi)
    if not np.any(mask):
        continue
    row = f"  {bkt_name:<14s}"
    for name in model_names:
        rmses = []
        for exp_name in EXPIRIES:
            emp = empirical[exp_name]
            mp = all_model_prices[exp_name][name]
            errs = mp[mask] - emp[mask]
            rmses.append(np.sqrt(np.mean(errs**2)))
        row += f" {np.mean(rmses):>10.4f}"
    print(row)

# Average across all buckets
print(f"  {'-'*14}" + "".join(f" {'-'*10}" for _ in model_names))
row = f"  {'TOTAL':.<14s}"
for name in model_names:
    all_rmse = []
    for exp_name in EXPIRIES:
        emp = empirical[exp_name]
        mp = all_model_prices[exp_name][name]
        all_rmse.append(np.sqrt(np.mean((mp - emp)**2)))
    row += f" {np.mean(all_rmse):>10.4f}"
print(row)

# ============================================================================
#  BEST MODEL PER BUCKET
# ============================================================================
print(f"\n{'='*90}")
print("  BEST MODEL PER BUCKET (lowest avg RMSE across expiries)")
print(f"{'='*90}")

for bkt_name, bkt_lo, bkt_hi in BUCKETS:
    mask = (MONEYNESS >= bkt_lo) & (MONEYNESS <= bkt_hi)
    if not np.any(mask):
        continue
    best_name, best_rmse = None, 1e9
    for name in model_names:
        rmses = []
        for exp_name in EXPIRIES:
            emp = empirical[exp_name]
            mp = all_model_prices[exp_name][name]
            errs = mp[mask] - emp[mask]
            rmses.append(np.sqrt(np.mean(errs**2)))
        avg = np.mean(rmses)
        if avg < best_rmse:
            best_name, best_rmse = name, avg
    print(f"  {bkt_name:<14s}  {best_name:<14s} (RMSE={best_rmse:.4f})")

# Save results
result = {
    "spot": S0,
    "moneyness": MONEYNESS.tolist(),
    "expiries_hours": dict(EXPIRIES),
    "num_paths": NUM_PATHS,
    "n_seeds": N_SEEDS,
    "buckets": [(n, lo, hi) for n, lo, hi in BUCKETS],
    "model_prices": {exp: {name: all_model_prices[exp][name].tolist()
                           for name in models} for exp in EXPIRIES},
    "empirical_prices": {exp: empirical[exp].tolist() for exp in EXPIRIES},
}
Path("digital_bucket_benchmark.json").write_text(
    json.dumps(_to_jsonable(result), indent=2), encoding="utf-8")
print(f"\nSaved digital_bucket_benchmark.json")
