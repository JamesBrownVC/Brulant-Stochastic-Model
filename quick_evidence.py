"""
Quick Evidence: Fast subset of run_full_evidence.py for testing.
10 seeds, 2 windows. Should complete in ~5 minutes.
"""
from __future__ import annotations
import json, time, datetime, sys
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

def main():
    print("=" * 70)
    print("  QUICK EVIDENCE (10 seeds, 2 windows)")
    print("=" * 70)

    from experiment_v12 import simulate_v12
    from benchmark_comparison import (
        simulate_gbm, calibrate_gbm,
        simulate_heston, calibrate_heston,
        simulate_merton, calibrate_merton,
        simulate_sabr, calibrate_sabr,
    )
    from fit_sandpile import fetch_binance_log_returns, interval_to_dt_years, moment_vector
    from benchmark_v12 import BRULANT_V12, diebold_mariano_test, bootstrap_ci

    N_SEEDS = 10

    print("\nFetching 5000 1-min candles...")
    returns_raw = fetch_binance_log_returns("BTCUSDT", "1m", 5000)
    dt = interval_to_dt_years("1m")

    n_split = int(len(returns_raw) * 0.5)
    train_r_raw, test_r_raw = returns_raw[:n_split], returns_raw[n_split:]
    mu = np.median(train_r_raw)
    mad = np.percentile(np.abs(train_r_raw - mu), 75) * 1.4826
    train_r = np.clip(train_r_raw, mu - 5 * mad, mu + 5 * mad)
    test_r = np.clip(test_r_raw, mu - 5 * mad, mu + 5 * mad)

    emp = moment_vector(test_r, w=None, acf_recent_bars=300)
    emp_std = np.std(test_r)
    scales = np.maximum(np.abs(emp), np.array([1e-12, 1e-12, 0.5, 1.0, 0.05, 0.1]))
    scales = np.maximum(scales, 1e-9)

    try:
        import requests
        S0 = float(requests.get("https://api.binance.com/api/v3/ticker/price",
                                params={"symbol": "BTCUSDT"}, timeout=10).json()["price"])
    except Exception:
        S0 = 85000.0

    print(f"  Spot: ${S0:,.2f} | Train: {train_r.size} | Test: {test_r.size}")
    print(f"  Empirical: kurt={emp[3]:.2f} skew={emp[2]:.3f} std={emp_std:.8f}")

    # Calibrate
    print("\nCalibrating benchmarks...")
    t0 = time.perf_counter()
    gbm_p = calibrate_gbm(train_r, dt)
    heston_p = calibrate_heston(train_r, dt)
    merton_p = calibrate_merton(train_r, dt)
    sabr_p = calibrate_sabr(train_r, dt, S0=1.0)
    print(f"  Done in {time.perf_counter()-t0:.0f}s")

    model_configs = {
        "Brulant v1.2": ("v12", BRULANT_V12),
        "GBM": ("gbm", gbm_p),
        "Heston": ("heston", heston_p),
        "Merton": ("merton", merton_p),
        "SABR": ("sabr", sabr_p),
    }

    # Run seeds
    print(f"\nRunning {N_SEEDS} seeds per model...")
    model_losses = {}

    for name, (tag, params) in model_configs.items():
        t0 = time.perf_counter()
        losses = []
        for i in range(N_SEEDS):
            seed = 42 + i * 77
            if tag == "v12":
                lr, _ = simulate_v12(test_r.size, dt, 1000, seed=seed, S0=1.0, **params)
            elif tag == "gbm":
                lr, _ = simulate_gbm(test_r.size, dt, 1000, 1.0, params["sigma"], seed=seed)
            elif tag == "heston":
                lr, _ = simulate_heston(test_r.size, dt, 1000, 1.0, **params, seed=seed)
            elif tag == "merton":
                lr, _ = simulate_merton(test_r.size, dt, 1000, 1.0, **params, seed=seed)
            elif tag == "sabr":
                lr, _ = simulate_sabr(test_r.size, dt, 1000, 1.0, **params, seed=seed)
            sim = moment_vector(lr.ravel(), w=None, acf_recent_bars=300)
            losses.append(float(np.sum(((sim - emp) / scales) ** 2)))

        losses = np.array(losses)
        model_losses[name] = losses
        print(f"  {name}: median={np.median(losses):.2f} mean={np.mean(losses):.2f} ({time.perf_counter()-t0:.1f}s)")

    # Paired tests
    v12_losses = model_losses["Brulant v1.2"]
    print(f"\n  {'Model':<14s} {'MedDiff':>8s} {'Wilcoxon p':>11s}")
    for name in model_configs:
        if name == "Brulant v1.2":
            continue
        diffs = model_losses[name] - v12_losses
        try:
            _, w_pval = sp_stats.wilcoxon(diffs, alternative='two-sided')
        except ValueError:
            w_pval = float('nan')
        sig = "***" if w_pval < 0.001 else ("**" if w_pval < 0.01 else ("*" if w_pval < 0.05 else "ns"))
        print(f"  {name:<14s} {np.median(diffs):>+8.2f} {w_pval:>11.4g} {sig}")

    print(f"\nDone in {time.perf_counter()-t0:.0f}s total")


if __name__ == "__main__":
    main()
