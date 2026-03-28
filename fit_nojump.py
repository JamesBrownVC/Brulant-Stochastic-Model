"""
Proper calibration: no-jump stochastic vol + directional buffer.
50/50 train/test on 10000 1-min candles.
"""
import numpy as np
import time
from scipy.optimize import differential_evolution
from experiment_v12 import simulate_v12
from fit_sandpile import (
    fetch_binance_log_returns, interval_to_dt_years,
    moment_vector, recent_exponential_weights,
)
from backtest_buffer_model import MOMENT_NAMES


def main():
    # ===== FETCH LARGE DATASET =====
    print("Fetching 10000 candles...")
    returns_raw = fetch_binance_log_returns("BTCUSDT", "1m", 10000)
    dt = interval_to_dt_years("1m")

    # 50/50 split BEFORE winsorization to prevent test data leaking into thresholds
    n_half = len(returns_raw) // 2
    train_r_raw, test_r_raw = returns_raw[:n_half], returns_raw[n_half:]

    # Compute winsorization thresholds from TRAIN ONLY
    mu = np.median(train_r_raw)
    mad = np.percentile(np.abs(train_r_raw - mu), 75) * 1.4826
    train_r = np.clip(train_r_raw, mu - 5 * mad, mu + 5 * mad)
    test_r = np.clip(test_r_raw, mu - 5 * mad, mu + 5 * mad)
    n_clipped_train = int(np.sum(train_r_raw != train_r))
    n_clipped_test = int(np.sum(test_r_raw != test_r))
    print(f"Train: {train_r.size} | Test: {test_r.size} (50/50)")
    print(f"Winsorized (train-derived thresholds): {n_clipped_train} train, {n_clipped_test} test")

    emp_train = moment_vector(train_r, w=None, acf_recent_bars=300)
    emp_test = moment_vector(test_r, w=None, acf_recent_bars=300)
    print(f"Train: kurt={emp_train[3]:.2f} skew={emp_train[2]:.3f} std={emp_train[1]:.8f}")
    print(f"Test:  kurt={emp_test[3]:.2f} skew={emp_test[2]:.3f} std={emp_test[1]:.8f}")

    w = recent_exponential_weights(train_r.size, 400.0)
    target = moment_vector(train_r, w=w, acf_recent_bars=300)
    scales = np.maximum(np.abs(target), np.array([1e-12, 1e-12, 0.5, 1.0, 0.05, 0.1]))
    scales = np.maximum(scales, 1e-9)

    test_scales = np.maximum(np.abs(emp_test), np.array([1e-12, 1e-12, 0.5, 1.0, 0.05, 0.1]))
    test_scales = np.maximum(test_scales, 1e-9)

    # ===== FIT 1: Single buffer + stochastic vol target =====
    print("\n" + "=" * 60)
    print("FIT 1: Single buffer + stochastic vol mean-reversion")
    print("=" * 60)

    names1 = ["sigma0", "sigma0_bar", "alpha_s", "xi_s", "rho", "alpha", "kappa", "theta_p"]
    bounds1 = [
        (0.15, 0.80),  # sigma0
        (0.10, 0.70),  # sigma0_bar
        (0.1, 15.0),   # alpha_s
        (0.01, 3.0),   # xi_s
        (0.5, 4.0),    # rho
        (1.0, 20.0),   # alpha
        (1.0, 60.0),   # kappa
        (0.5, 3.0),    # theta_p
    ]
    fixed1 = dict(mu0=0.0, nu=1.0, beta=0.0, lambda0=0.0, gamma=20.0,
                  eta=1.0, phi=0.56, sigma_Y=0.01, eps=0.001,
                  stoch_vol_target=True)

    rng1 = np.random.default_rng(42)

    def obj1(theta):
        p = {k: float(v) for k, v in zip(names1, theta)}
        p.update(fixed1)
        s = int(rng1.integers(0, 2**31 - 1))
        sim_lr, _ = simulate_v12(train_r.size, dt, 800, seed=s, S0=1.0, **p)
        pooled = sim_lr.ravel()
        if pooled.size > 50000:
            pooled = np.random.default_rng(s).choice(pooled, 50000, replace=False)
        sim = moment_vector(pooled, w=None, acf_recent_bars=300)
        z = (sim - target) / scales
        penalty = 2.0 * max(0, p["rho"] - 3.0) ** 2
        penalty += 1.0 * max(0, p["xi_s"] - 2.0) ** 2
        return float(np.sum(z * z)) + penalty

    t0 = time.perf_counter()
    res1 = differential_evolution(obj1, bounds1, maxiter=14, seed=42, workers=1,
                                   polish=False, popsize=10, tol=1e-3, atol=1e-4)
    print(f"Done in {time.perf_counter() - t0:.1f}s, train loss={res1.fun:.4f}")

    fit1 = {k: float(v) for k, v in zip(names1, res1.x)}
    fit1.update(fixed1)
    print("\nParameters:")
    for k in names1:
        print(f"  {k:>12s} = {fit1[k]:.6f}")

    # Train moments
    sim_lr, _ = simulate_v12(train_r.size, dt, 5000, seed=999, S0=1.0, **fit1)
    sp = sim_lr.ravel()[:50000]
    sim_m = moment_vector(sp, w=None, acf_recent_bars=300)
    print("\nTrain moments (target | sim):")
    for n, a, b in zip(MOMENT_NAMES, target, sim_m):
        print(f"  {n:>8s}: {a:>12.6g} | {b:>12.6g}")

    # OOS
    losses1 = []
    for i in range(20):
        sim_lr, _ = simulate_v12(test_r.size, dt, 1000, seed=42 + i * 77, S0=1.0, **fit1)
        sim = moment_vector(sim_lr.ravel(), w=None, acf_recent_bars=300)
        loss = float(np.sum(((sim - emp_test) / test_scales) ** 2))
        losses1.append(loss)
    losses1 = np.array(losses1)

    sim_lr, _ = simulate_v12(test_r.size, dt, 5000, seed=42, S0=1.0, **fit1)
    sp = sim_lr.ravel()[:50000]
    sim_test1 = moment_vector(sp, w=None, acf_recent_bars=300)
    emp_std = np.std(test_r)

    print(f"\nOOS ({test_r.size} bars, 20 seeds):")
    print(f"  Loss: median={np.median(losses1):.4f} mean={np.mean(losses1):.4f} std={np.std(losses1):.4f}")
    print(f"  Std ratio: {np.std(sp) / emp_std:.4f}")
    print("  Moments (emp | sim):")
    for n, a, b in zip(MOMENT_NAMES, emp_test, sim_test1):
        print(f"    {n:>8s}: {a:>12.6g} | {b:>12.6g}")

    # ===== FIT 2: Multi-buffer (fast+slow) + stochastic vol =====
    print("\n" + "=" * 60)
    print("FIT 2: 2-layer buffer + stochastic vol mean-reversion")
    print("=" * 60)

    names2 = ["sigma0", "sigma0_bar", "alpha_s", "xi_s", "rho", "alpha",
              "kappa_fast", "kappa_slow", "theta_fast", "theta_slow", "w_slow"]
    bounds2 = [
        (0.15, 0.80), (0.10, 0.70), (0.1, 15.0), (0.01, 3.0),
        (0.5, 4.0), (1.0, 20.0),
        (20.0, 100.0),   # kappa_fast
        (0.5, 8.0),      # kappa_slow
        (0.5, 3.0),      # theta_fast
        (0.3, 2.0),      # theta_slow
        (0.1, 0.6),      # w_slow
    ]
    fixed2 = dict(mu0=0.0, nu=1.0, beta=0.0, lambda0=0.0, gamma=20.0,
                  eta=1.0, phi=0.56, sigma_Y=0.01, eps=0.001,
                  stoch_vol_target=True, multi_buffer=True,
                  kappa_mid=15.0, theta_mid=1.5, w_mid=0.0)

    rng2 = np.random.default_rng(99)

    def obj2(theta):
        p = {k: float(v) for k, v in zip(names2, theta)}
        p["w_fast"] = 1.0 - p["w_slow"]
        p.update(fixed2)
        s = int(rng2.integers(0, 2**31 - 1))
        sim_lr, _ = simulate_v12(train_r.size, dt, 800, seed=s, S0=1.0, **p)
        pooled = sim_lr.ravel()
        if pooled.size > 50000:
            pooled = np.random.default_rng(s).choice(pooled, 50000, replace=False)
        sim = moment_vector(pooled, w=None, acf_recent_bars=300)
        z = (sim - target) / scales
        penalty = 2.0 * max(0, p["rho"] - 3.0) ** 2
        penalty += 1.0 * max(0, p["xi_s"] - 2.0) ** 2
        return float(np.sum(z * z)) + penalty

    t0 = time.perf_counter()
    res2 = differential_evolution(obj2, bounds2, maxiter=14, seed=99, workers=1,
                                   polish=False, popsize=10, tol=1e-3, atol=1e-4)
    print(f"Done in {time.perf_counter() - t0:.1f}s, train loss={res2.fun:.4f}")

    fit2 = {k: float(v) for k, v in zip(names2, res2.x)}
    fit2["w_fast"] = 1.0 - fit2["w_slow"]
    fit2.update(fixed2)
    print("\nParameters:")
    for k in names2 + ["w_fast"]:
        print(f"  {k:>12s} = {fit2[k]:.6f}")

    # OOS
    losses2 = []
    for i in range(20):
        sim_lr, _ = simulate_v12(test_r.size, dt, 1000, seed=42 + i * 77, S0=1.0, **fit2)
        sim = moment_vector(sim_lr.ravel(), w=None, acf_recent_bars=300)
        loss = float(np.sum(((sim - emp_test) / test_scales) ** 2))
        losses2.append(loss)
    losses2 = np.array(losses2)

    sim_lr, _ = simulate_v12(test_r.size, dt, 5000, seed=42, S0=1.0, **fit2)
    sp2 = sim_lr.ravel()[:50000]
    sim_test2 = moment_vector(sp2, w=None, acf_recent_bars=300)

    print(f"\nOOS ({test_r.size} bars, 20 seeds):")
    print(f"  Loss: median={np.median(losses2):.4f} mean={np.mean(losses2):.4f} std={np.std(losses2):.4f}")
    print(f"  Std ratio: {np.std(sp2) / emp_std:.4f}")
    print("  Moments (emp | sim):")
    for n, a, b in zip(MOMENT_NAMES, emp_test, sim_test2):
        print(f"    {n:>8s}: {a:>12.6g} | {b:>12.6g}")

    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Single-buffer: train={res1.fun:.4f}  OOS median={np.median(losses1):.4f} mean={np.mean(losses1):.4f}")
    print(f"  Multi-buffer:  train={res2.fun:.4f}  OOS median={np.median(losses2):.4f} mean={np.mean(losses2):.4f}")
    print(f"  V1.1 ref:      train~2.0           OOS median~16")


if __name__ == "__main__":
    main()
