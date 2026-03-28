"""
Digital Option Pricing Benchmark
================================
Prices cash-or-nothing digital calls across models at multiple
moneyness levels and expiries.  Digital price = P(S_T > K), so it
directly tests the risk-neutral CDF shape and tail behaviour.

Methodology (Cont 2001, Schoutens et al. 2004):
  1. All models calibrated to same historical return moments
  2. MC digital prices at moneyness grid × expiry grid
  3. Compare against each other (model spread = model risk)
  4. Report RMSE/MAE by moneyness bucket
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
from backtest_buffer_model import simulate_buffer_paths
from fit_sandpile import _to_jsonable

NUM_PATHS = 200_000
N_SEEDS = 3

# Moneyness grid (K/S)
MONEYNESS = np.array([0.90, 0.95, 0.97, 0.99,
                       1.00, 1.01, 1.03, 1.05, 1.10])

# Expiry grid in hours
EXPIRIES = {"4h": 4, "1d": 24, "3d": 72, "7d": 168}


def simulate_terminal(tag, params, S0, n_steps, dt, num_paths, seed):
    """Simulate and return terminal prices S_T (array of shape num_paths)."""
    p = {k: v for k, v in params.items() if not k.startswith('_')}
    if tag == "v11_exc":
        _, S_T = simulate_v11_excitation(n_steps, dt, num_paths, seed=seed, S0=S0, **p)
    elif tag == "v12":
        _, S_T = simulate_v12(n_steps, dt, num_paths, seed=seed, S0=S0, **p)
    elif tag == "v11":
        _, S_T = simulate_buffer_paths(n_steps, dt, num_paths, seed=seed, S0=S0, **p)
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
    """Price digital calls P(S_T >= K) for a vector of strikes."""
    T = hours / (24.0 * 365.0)
    n_steps = max(1, int(hours * 4))  # 15-min resolution
    dt = T / n_steps
    S_T = simulate_terminal(tag, params, S0, n_steps, dt, num_paths, seed)
    S_T = np.asarray(S_T).ravel()
    prices = np.array([np.mean(S_T >= K) for K in strikes])
    stderrs = np.array([np.sqrt(p * (1 - p) / num_paths) for p in prices])
    return prices, stderrs


# ============================================================================
#  MAIN
# ============================================================================
print("=" * 70)
print("  DIGITAL OPTION PRICING BENCHMARK")
print(f"  Paths: {NUM_PATHS:,} | Seeds: {N_SEEDS}")
print("=" * 70)

# --- Load pre-calibrated params ---
print("\nLoading calibrated params from multi_scale_benchmark.json...")
prev = json.loads(Path("multi_scale_benchmark.json").read_text(encoding="utf-8"))
pp = prev["model_params"]

gbm_params = pp["GBM"]
heston_params = pp["Heston"]
merton_params = pp["Merton"]
sabr_params = pp["SABR"]
v12_params = pp["Brulant v1.2"]

try:
    import requests
    S0 = float(requests.get("https://api.binance.com/api/v3/ticker/price",
                            params={"symbol": "BTCUSDT"}, timeout=10).json()["price"])
except Exception:
    S0 = 67000.0

print(f"  Spot: ${S0:,.2f}")
strikes = S0 * MONEYNESS
print(f"  Strikes: ${strikes.min():,.0f} - ${strikes.max():,.0f} "
      f"({len(MONEYNESS)} points, {MONEYNESS.min():.0%}-{MONEYNESS.max():.0%} moneyness)")
print(f"  GBM sigma={gbm_params['sigma']:.4f}")
print(f"  Heston v0={heston_params['v0']:.4f} kappa={heston_params['kappa']:.2f}")
print(f"  Merton lam={merton_params['lam']:.4f} jsig={merton_params['jump_sigma']:.4f}")
print(f"  SABR alpha={sabr_params['alpha_s']:.4f} nu={sabr_params['nu_s']:.4f}")

# Excitation configs
base_exc = dict(mu0=0, rho=1.3, nu=1.6, alpha=10, beta=0.0,
                phi=1.6, sigma_Y=0.20, kappa=15, theta_p=1.5,
                gamma=20, eta=1, eps=1e-3, jump_to_ret=False)

exc_a = {**base_exc, "sigma0": 0.42, "lambda0": 3.0,
         "exc_beta": 10.0, "exc_kappa": 100.0, "alpha_exc": 60.0}
exc_opt = {**base_exc, "sigma0": 0.40, "lambda0": 3.0,
           "exc_beta": 10.0, "exc_kappa": 130.0, "alpha_exc": 200.0}

# All models
models = {
    "GBM":          ("gbm",     gbm_params),
    "Heston":       ("heston",  heston_params),
    "Merton":       ("merton",  merton_params),
    "SABR":         ("sabr",    sabr_params),
    "Brulant v1.2": ("v12",     v12_params),
    "Exc-A":        ("v11_exc", exc_a),
    "Exc (opt)":    ("v11_exc", exc_opt),
}

# --- Price digitals ---
all_results = {}
n_total = len(models) * len(EXPIRIES) * N_SEEDS
done = 0

for exp_name, hours in EXPIRIES.items():
    print(f"\n{'='*70}")
    print(f"  EXPIRY: {exp_name} ({hours}h)")
    print(f"{'='*70}")

    # Header
    header = f"  {'K/S':>6s}"
    for name in models:
        header += f" {name[:10]:>10s}"
    header += f" {'Spread':>8s}"
    print(header)
    print(f"  {'-'*6}" + "".join(f" {'-'*10}" for _ in models) + f" {'-'*8}")

    exp_prices = {}
    for name, (tag, params) in models.items():
        t0 = time.perf_counter()
        # Average over N_SEEDS for reduced MC noise
        seed_prices = []
        for s in range(N_SEEDS):
            seed = 42 + s * 137
            p, _ = price_digitals(tag, params, S0, strikes, hours,
                                  NUM_PATHS, seed)
            seed_prices.append(p)
        avg_prices = np.mean(seed_prices, axis=0)
        exp_prices[name] = avg_prices
        elapsed = time.perf_counter() - t0
        done += N_SEEDS
        pct = done * 100 // n_total
        print(f"  [{pct:>3d}%] {name} priced ({elapsed:.0f}s)")

    # Print price grid
    print()
    print(header)
    print(f"  {'-'*6}" + "".join(f" {'-'*10}" for _ in models) + f" {'-'*8}")

    for j, m in enumerate(MONEYNESS):
        row = f"  {m:>6.2f}"
        prices_at_strike = [exp_prices[name][j] for name in models]
        for p in prices_at_strike:
            row += f" {p:>10.4f}"
        spread = max(prices_at_strike) - min(prices_at_strike)
        row += f" {spread:>8.4f}"
        print(row)

    all_results[exp_name] = {
        name: exp_prices[name].tolist() for name in models
    }

# --- Summary: Model spread by moneyness bucket ---
print(f"\n{'='*70}")
print("  MODEL SPREAD (max - min across models) BY MONEYNESS & EXPIRY")
print(f"{'='*70}")

buckets = {
    "Deep ITM (0.85-0.93)": MONEYNESS <= 0.93,
    "Near ATM (0.97-1.03)": (MONEYNESS >= 0.97) & (MONEYNESS <= 1.03),
    "Deep OTM (1.07-1.15)": MONEYNESS >= 1.07,
}

header = f"  {'Bucket':<22s}"
for exp_name in EXPIRIES:
    header += f" {exp_name:>8s}"
print(header)
print(f"  {'-'*22}" + "".join(f" {'-'*8}" for _ in EXPIRIES))

for bucket_name, mask in buckets.items():
    row = f"  {bucket_name:<22s}"
    for exp_name in EXPIRIES:
        prices_matrix = np.array([all_results[exp_name][name] for name in models])
        spreads = prices_matrix[:, mask].max(axis=0) - prices_matrix[:, mask].min(axis=0)
        row += f" {np.mean(spreads):>8.4f}"
    print(row)

# --- Pairwise comparison: each model vs GBM (baseline) ---
print(f"\n{'='*70}")
print("  DIGITAL PRICE DEVIATION FROM GBM (mean across strikes)")
print(f"{'='*70}")

header = f"  {'Model':<14s}"
for exp_name in EXPIRIES:
    header += f" {exp_name:>8s}"
print(header)
print(f"  {'-'*14}" + "".join(f" {'-'*8}" for _ in EXPIRIES))

for name in models:
    if name == "GBM":
        continue
    row = f"  {name:<14s}"
    for exp_name in EXPIRIES:
        gbm_p = np.array(all_results[exp_name]["GBM"])
        mod_p = np.array(all_results[exp_name][name])
        mae = np.mean(np.abs(mod_p - gbm_p))
        row += f" {mae:>8.4f}"
    print(row)

# --- Save ---
result = {
    "spot": S0,
    "moneyness": MONEYNESS.tolist(),
    "strikes": strikes.tolist(),
    "expiries_hours": {k: v for k, v in EXPIRIES.items()},
    "num_paths": NUM_PATHS,
    "n_seeds": N_SEEDS,
    "digital_prices": all_results,
    "model_params": {name: {k: v for k, v in params.items() if not k.startswith('_')}
                     for name, (_, params) in models.items()},
}
Path("digital_benchmark.json").write_text(
    json.dumps(_to_jsonable(result), indent=2), encoding="utf-8")
print(f"\nSaved digital_benchmark.json")
