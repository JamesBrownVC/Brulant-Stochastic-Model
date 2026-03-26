"""
Brulant Model vs Benchmarks: SABR, Heston, Merton, GBM
=======================================================
Full head-to-head on:
  1. OOS moment matching (same 5000 1-min BTC returns)
  2. Digital option pricing grid (60k-80k, +0d..+4d at 17:00 CET)
  3. Distributional quality (std ratio, tail matching, kurtosis)
  4. Path realism (visual comparison)
"""

from __future__ import annotations
import json, time, datetime
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from scipy import stats
from scipy.optimize import minimize

from backtest_buffer_model import simulate_buffer_paths, MOMENT_NAMES
from fit_sandpile import (
    fetch_binance_log_returns, interval_to_dt_years,
    moment_vector, recent_exponential_weights, _to_jsonable,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False


# ============================================================================
#  BRULANT V1.1 (sigma_Y adjusted for long-standing vol regimes)
# ============================================================================
BRULANT_PARAMS = {
    "mu0": 0.0, "sigma0": 0.596377, "rho": 1.78402, "nu": 1.54849,
    "alpha": 9.90562, "beta": 0.128777, "lambda0": 1.18401,
    "gamma": 20.0, "eta": 1.0, "kappa": 15.0, "theta_p": 1.5,
    "phi": 0.560709, "sigma_Y": 0.045, "eps": 0.001,
}


# ============================================================================
#  BENCHMARK 1: Geometric Brownian Motion (Black-Scholes)
# ============================================================================
def simulate_gbm(n_steps, dt, num_paths, S0, sigma, mu=0.0, seed=42):
    """Standard GBM: dS = mu*S*dt + sigma*S*dW"""
    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)
    lr = np.zeros((num_paths, n_steps), dtype=np.float64)
    for t in range(n_steps):
        dW = sqrt_dt * rng.standard_normal(num_paths)
        lr[:, t] = (mu - 0.5 * sigma**2) * dt + sigma * dW
    S_T = S0 * np.exp(np.sum(lr, axis=1))
    return lr, S_T


def calibrate_gbm(train_r, dt):
    """Calibrate GBM: just match mean and std."""
    mu = np.mean(train_r) / dt
    sigma = np.std(train_r) / np.sqrt(dt)
    return {"mu": 0.0, "sigma": float(sigma)}


# ============================================================================
#  BENCHMARK 2: Heston Stochastic Volatility
# ============================================================================
def simulate_heston(n_steps, dt, num_paths, S0, v0, kappa, theta, xi, rho_h,
                    mu=0.0, seed=42):
    """
    Heston model:
      dS = mu*S*dt + sqrt(v)*S*dW1
      dv = kappa*(theta - v)*dt + xi*sqrt(v)*dW2
      corr(dW1, dW2) = rho_h
    """
    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)
    S = np.full(num_paths, S0, dtype=np.float64)
    v = np.full(num_paths, v0, dtype=np.float64)
    lr = np.zeros((num_paths, n_steps), dtype=np.float64)

    for t in range(n_steps):
        Z1 = rng.standard_normal(num_paths)
        Z2 = rng.standard_normal(num_paths)
        W1 = sqrt_dt * Z1
        W2 = sqrt_dt * (rho_h * Z1 + np.sqrt(1 - rho_h**2) * Z2)

        v_pos = np.maximum(v, 1e-8)
        sqrt_v = np.sqrt(v_pos)

        S_prev = S
        S = S * np.exp((mu - 0.5 * v_pos) * dt + sqrt_v * W1)
        S = np.maximum(S, 1e-12)
        lr[:, t] = np.log(S / S_prev)

        v = v + kappa * (theta - v_pos) * dt + xi * sqrt_v * W2
        v = np.maximum(v, 1e-8)

    return lr, S


def calibrate_heston(train_r, dt):
    """Calibrate Heston via simple moment matching."""
    emp_var = np.var(train_r)
    emp_kurt = float(stats.kurtosis(train_r))
    ann_var = emp_var / dt

    # Initial guess
    v0 = ann_var
    theta = ann_var
    kappa = 5.0
    xi = 0.5
    rho_h = -0.3

    def obj(x):
        k, th, x_, r_ = x
        k = max(0.1, k)
        th = max(0.01, th)
        x_ = max(0.01, min(x_, 3.0))
        r_ = max(-0.99, min(r_, 0.99))
        try:
            lr, _ = simulate_heston(train_r.size, dt, 500, 1.0, v0,
                                     k, th, x_, r_, mu=0.0, seed=42)
            sim_std = np.std(lr.ravel())
            sim_kurt = float(stats.kurtosis(lr.ravel()))
            emp_std = np.std(train_r)
            return (sim_std/emp_std - 1)**2 + 0.1*(sim_kurt - emp_kurt)**2
        except Exception:
            return 1e6

    from scipy.optimize import differential_evolution
    bounds = [(0.5, 30.0), (0.01, 2.0), (0.05, 3.0), (-0.95, 0.3)]
    res = differential_evolution(obj, bounds, maxiter=8, seed=42, popsize=6,
                                  polish=False, tol=0.01)
    k, th, x_, r_ = res.x
    return {"v0": float(v0), "kappa": max(0.1, float(k)),
            "theta": max(0.01, float(th)), "xi": max(0.01, float(x_)),
            "rho_h": max(-0.99, min(float(r_), 0.99))}


# ============================================================================
#  BENCHMARK 3: Merton Jump-Diffusion
# ============================================================================
def simulate_merton(n_steps, dt, num_paths, S0, sigma, lam, jump_mu, jump_sigma,
                    mu=0.0, seed=42):
    """
    Merton (1976):
      dS/S = (mu - lam*k)*dt + sigma*dW + J*dN
      J ~ LogNormal(jump_mu, jump_sigma), k = E[J-1]
    """
    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)
    k = np.exp(jump_mu + 0.5 * jump_sigma**2) - 1
    lr = np.zeros((num_paths, n_steps), dtype=np.float64)
    S = np.full(num_paths, S0, dtype=np.float64)

    for t in range(n_steps):
        dW = sqrt_dt * rng.standard_normal(num_paths)
        # Poisson jumps
        N_jump = rng.poisson(lam * dt, num_paths)
        J = np.zeros(num_paths)
        for i in range(num_paths):
            if N_jump[i] > 0:
                jumps = rng.normal(jump_mu, jump_sigma, N_jump[i])
                J[i] = np.sum(jumps)
        S_prev = S
        S = S * np.exp((mu - lam * k - 0.5 * sigma**2) * dt + sigma * dW + J)
        S = np.maximum(S, 1e-12)
        lr[:, t] = np.log(S / S_prev)

    return lr, S


def calibrate_merton(train_r, dt):
    """Calibrate Merton via moment matching."""
    emp_std = np.std(train_r)
    emp_kurt = float(stats.kurtosis(train_r))
    ann_sigma = emp_std / np.sqrt(dt) * 0.8  # diffusion carries ~80% of variance

    def obj(x):
        lam, jmu, jsig = x
        try:
            lr, _ = simulate_merton(train_r.size, dt, 500, 1.0,
                                     ann_sigma, lam, jmu, jsig, seed=42)
            sim_std = np.std(lr.ravel())
            sim_kurt = float(stats.kurtosis(lr.ravel()))
            return (sim_std/emp_std - 1)**2 + 0.1*(sim_kurt - emp_kurt)**2
        except Exception:
            return 1e6

    from scipy.optimize import differential_evolution
    bounds = [(0.1, 20.0), (-0.05, 0.05), (0.01, 0.3)]
    res = differential_evolution(obj, bounds, maxiter=8, seed=42, popsize=6,
                                  polish=False, tol=0.01)
    lam, jmu, jsig = res.x
    return {"sigma": float(ann_sigma), "lam": float(lam),
            "jump_mu": float(jmu), "jump_sigma": float(jsig)}


# ============================================================================
#  BENCHMARK 4: SABR
# ============================================================================
def simulate_sabr(n_steps, dt, num_paths, S0, alpha_s, beta_s, rho_s, nu_s,
                  seed=42):
    """
    SABR model:
      dF = alpha * F^beta * dW1
      dalpha = nu * alpha * dW2
      corr(dW1, dW2) = rho
    """
    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)
    F = np.full(num_paths, S0, dtype=np.float64)
    a = np.full(num_paths, alpha_s, dtype=np.float64)
    lr = np.zeros((num_paths, n_steps), dtype=np.float64)

    for t in range(n_steps):
        Z1 = rng.standard_normal(num_paths)
        Z2 = rng.standard_normal(num_paths)
        W1 = sqrt_dt * Z1
        W2 = sqrt_dt * (rho_s * Z1 + np.sqrt(1 - rho_s**2) * Z2)

        F_prev = F
        F_beta = np.power(np.maximum(F, 1e-8), beta_s)
        F = F + a * F_beta * W1
        F = np.maximum(F, 1e-12)
        lr[:, t] = np.log(F / F_prev)

        a = a * np.exp(-0.5 * nu_s**2 * dt + nu_s * W2)
        a = np.maximum(a, 1e-8)

    return lr, F


def calibrate_sabr(train_r, dt, S0=1.0):
    """Calibrate SABR via moment matching."""
    emp_std = np.std(train_r)
    emp_kurt = float(stats.kurtosis(train_r))

    def obj(x):
        al, be, rh, nu = x
        try:
            lr, _ = simulate_sabr(train_r.size, dt, 500, S0,
                                   al, be, rh, nu, seed=42)
            sim_std = np.std(lr.ravel())
            sim_kurt = float(stats.kurtosis(lr.ravel()))
            return (sim_std/emp_std - 1)**2 + 0.1*(sim_kurt - emp_kurt)**2
        except Exception:
            return 1e6

    from scipy.optimize import differential_evolution
    bounds = [(0.001, 5.0), (0.1, 1.0), (-0.95, 0.95), (0.01, 5.0)]
    res = differential_evolution(obj, bounds, maxiter=8, seed=42, popsize=6,
                                  polish=False, tol=0.01)
    al, be, rh, nu = res.x
    return {"alpha_s": float(al), "beta_s": float(be),
            "rho_s": max(-0.99, min(float(rh), 0.99)), "nu_s": float(nu)}


# ============================================================================
#  DIGITAL OPTION PRICING (generic)
# ============================================================================
def price_digital_generic(sim_func, S0, strikes, hours, params, num_paths=200000, seed=42):
    T = hours / (24.0 * 365.0)
    n_steps = max(1, int(hours * 60))
    dt = T / n_steps
    _, S_T = sim_func(n_steps, dt, num_paths, S0, **params, seed=seed)
    strikes = np.asarray(strikes, dtype=np.float64)
    prices = np.array([np.mean(S_T >= K) for K in strikes])
    stderrs = np.array([np.std((S_T >= K).astype(float)) / np.sqrt(num_paths) for K in strikes])
    return prices, stderrs


def price_brulant_digital(S0, strikes, hours, params, num_paths=200000, seed=42):
    T = hours / (24.0 * 365.0)
    n_steps = max(1, int(hours * 60))
    dt = T / n_steps
    _, S_T = simulate_buffer_paths(n_steps, dt, num_paths, seed=seed, S0=S0, **params)
    strikes = np.asarray(strikes, dtype=np.float64)
    prices = np.array([np.mean(S_T >= K) for K in strikes])
    stderrs = np.array([np.std((S_T >= K).astype(float)) / np.sqrt(num_paths) for K in strikes])
    return prices, stderrs


# ============================================================================
#  MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("  BRULANT v1.1 vs BENCHMARKS (GBM, Heston, Merton, SABR)")
    print("  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # Fetch data
    print("\nFetching 5000 1-min candles...")
    returns_raw = fetch_binance_log_returns("BTCUSDT", "1m", 5000)
    dt = interval_to_dt_years("1m")
    mu = np.median(returns_raw)
    mad = np.percentile(np.abs(returns_raw - mu), 75) * 1.4826
    returns = np.clip(returns_raw, mu - 5*mad, mu + 5*mad)
    train_r = returns[:int(len(returns)*0.75)]
    test_r = returns[int(len(returns)*0.75):]

    try:
        import requests
        S0 = float(requests.get("https://api.binance.com/api/v3/ticker/price",
                                params={"symbol": "BTCUSDT"}, timeout=10).json()["price"])
    except Exception:
        S0 = 70000.0

    emp = moment_vector(test_r, w=None, acf_recent_bars=300)
    emp_std = np.std(test_r)
    emp_3sig = np.mean(np.abs(test_r) > 3 * emp_std)
    scales = np.maximum(np.abs(emp), np.array([1e-12, 1e-12, 0.5, 1.0, 0.05, 0.1]))
    scales = np.maximum(scales, 1e-9)

    print(f"  Spot: ${S0:,.2f}")
    print(f"  Train: {train_r.size} | Test: {test_r.size}")
    print(f"  Empirical: std={emp_std:.8f} kurt={emp[3]:.2f} skew={emp[2]:.3f}")

    # ===== CALIBRATE BENCHMARKS =====
    print("\n--- Calibrating benchmarks ---")

    t0 = time.perf_counter()
    gbm_params = calibrate_gbm(train_r, dt)
    print(f"  GBM: sigma={gbm_params['sigma']:.4f} ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    heston_params = calibrate_heston(train_r, dt)
    print(f"  Heston: v0={heston_params['v0']:.4f} kappa={heston_params['kappa']:.2f} "
          f"theta={heston_params['theta']:.4f} xi={heston_params['xi']:.4f} "
          f"rho={heston_params['rho_h']:.3f} ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    merton_params = calibrate_merton(train_r, dt)
    print(f"  Merton: sigma={merton_params['sigma']:.4f} lam={merton_params['lam']:.4f} "
          f"jmu={merton_params['jump_mu']:.5f} jsig={merton_params['jump_sigma']:.4f} ({time.perf_counter()-t0:.1f}s)")

    t0 = time.perf_counter()
    sabr_params = calibrate_sabr(train_r, dt, S0=1.0)
    print(f"  SABR: alpha={sabr_params['alpha_s']:.4f} beta={sabr_params['beta_s']:.3f} "
          f"rho={sabr_params['rho_s']:.3f} nu={sabr_params['nu_s']:.4f} ({time.perf_counter()-t0:.1f}s)")

    # ===== OOS EVALUATION =====
    print("\n" + "=" * 70)
    print("  OOS MOMENT MATCHING (10 seeds)")
    print("=" * 70)

    models = {}

    # Brulant
    brulant_losses = []
    for i in range(10):
        sim_lr, _ = simulate_buffer_paths(test_r.size, dt, 1000, seed=42+i*77, S0=1.0, **BRULANT_PARAMS)
        sim = moment_vector(sim_lr.ravel(), w=None, acf_recent_bars=300)
        brulant_losses.append(float(np.sum(((sim - emp)/scales)**2)))
    sim_lr, _ = simulate_buffer_paths(test_r.size, dt, 5000, seed=42, S0=1.0, **BRULANT_PARAMS)
    sp = sim_lr.ravel()[:50000]
    models["Brulant v1.1"] = {
        "losses": brulant_losses,
        "std_ratio": float(np.std(sp) / emp_std),
        "tail_3sig": float(np.mean(np.abs(sp) > 3*np.std(sp)) / max(emp_3sig, 1e-12)),
        "sim_moments": moment_vector(sp, w=None, acf_recent_bars=300),
    }

    # GBM
    gbm_losses = []
    for i in range(10):
        lr, _ = simulate_gbm(test_r.size, dt, 1000, 1.0, gbm_params["sigma"], seed=42+i*77)
        sim = moment_vector(lr.ravel(), w=None, acf_recent_bars=300)
        gbm_losses.append(float(np.sum(((sim - emp)/scales)**2)))
    lr, _ = simulate_gbm(test_r.size, dt, 5000, 1.0, gbm_params["sigma"], seed=42)
    sp = lr.ravel()[:50000]
    models["GBM (Black-Scholes)"] = {
        "losses": gbm_losses,
        "std_ratio": float(np.std(sp) / emp_std),
        "tail_3sig": float(np.mean(np.abs(sp) > 3*np.std(sp)) / max(emp_3sig, 1e-12)),
        "sim_moments": moment_vector(sp, w=None, acf_recent_bars=300),
    }

    # Heston
    heston_losses = []
    for i in range(10):
        lr, _ = simulate_heston(test_r.size, dt, 1000, 1.0, **heston_params, seed=42+i*77)
        sim = moment_vector(lr.ravel(), w=None, acf_recent_bars=300)
        heston_losses.append(float(np.sum(((sim - emp)/scales)**2)))
    lr, _ = simulate_heston(test_r.size, dt, 5000, 1.0, **heston_params, seed=42)
    sp = lr.ravel()[:50000]
    models["Heston"] = {
        "losses": heston_losses,
        "std_ratio": float(np.std(sp) / emp_std),
        "tail_3sig": float(np.mean(np.abs(sp) > 3*np.std(sp)) / max(emp_3sig, 1e-12)),
        "sim_moments": moment_vector(sp, w=None, acf_recent_bars=300),
    }

    # Merton
    merton_losses = []
    for i in range(10):
        lr, _ = simulate_merton(test_r.size, dt, 1000, 1.0, **merton_params, seed=42+i*77)
        sim = moment_vector(lr.ravel(), w=None, acf_recent_bars=300)
        merton_losses.append(float(np.sum(((sim - emp)/scales)**2)))
    lr, _ = simulate_merton(test_r.size, dt, 5000, 1.0, **merton_params, seed=42)
    sp = lr.ravel()[:50000]
    models["Merton Jump-Diff"] = {
        "losses": merton_losses,
        "std_ratio": float(np.std(sp) / emp_std),
        "tail_3sig": float(np.mean(np.abs(sp) > 3*np.std(sp)) / max(emp_3sig, 1e-12)),
        "sim_moments": moment_vector(sp, w=None, acf_recent_bars=300),
    }

    # SABR
    sabr_losses = []
    for i in range(10):
        lr, _ = simulate_sabr(test_r.size, dt, 1000, 1.0, **sabr_params, seed=42+i*77)
        sim = moment_vector(lr.ravel(), w=None, acf_recent_bars=300)
        sabr_losses.append(float(np.sum(((sim - emp)/scales)**2)))
    lr, _ = simulate_sabr(test_r.size, dt, 5000, 1.0, **sabr_params, seed=42)
    sp = lr.ravel()[:50000]
    models["SABR"] = {
        "losses": sabr_losses,
        "std_ratio": float(np.std(sp) / emp_std),
        "tail_3sig": float(np.mean(np.abs(sp) > 3*np.std(sp)) / max(emp_3sig, 1e-12)),
        "sim_moments": moment_vector(sp, w=None, acf_recent_bars=300),
    }

    # Print comparison table
    print(f"\n  {'Model':<22s} {'Med Loss':>10s} {'Mean Loss':>10s} {'Std Ratio':>10s} {'3sig Ratio':>10s} {'Kurt':>8s}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for name, m in models.items():
        l = np.array(m["losses"])
        print(f"  {name:<22s} {np.median(l):>10.2f} {np.mean(l):>10.2f} "
              f"{m['std_ratio']:>10.3f} {m['tail_3sig']:>10.3f} {m['sim_moments'][3]:>8.1f}")
    print(f"  {'EMPIRICAL':<22s} {'---':>10s} {'---':>10s} {'1.000':>10s} {'1.000':>10s} {emp[3]:>8.1f}")

    # ===== DIGITAL OPTION PRICING =====
    print("\n" + "=" * 70)
    print("  DIGITAL OPTION PRICING COMPARISON")
    print("=" * 70)

    strikes = np.arange(60000, 82000, 2000, dtype=np.float64)
    cet = datetime.timezone(datetime.timedelta(hours=1))
    now_cet = datetime.datetime.now(datetime.timezone.utc).astimezone(cet)
    print(f"  Spot: ${S0:,.2f} | {now_cet.strftime('%Y-%m-%d %H:%M')} CET")

    all_pricing = {}
    for k_day in [1, 3]:  # +1d and +3d for comparison
        target_date = now_cet.date() + datetime.timedelta(days=k_day)
        target_dt = datetime.datetime(target_date.year, target_date.month, target_date.day,
                                      17, 0, 0, tzinfo=cet)
        hours = max((target_dt - now_cet).total_seconds() / 3600.0, 0.01)
        T = hours / (24.0 * 365.0)
        n_steps = max(1, int(hours * 60))
        step_dt = T / n_steps

        print(f"\n  +{k_day}d (17:00 CET {target_date}, {hours:.1f}h)")
        print(f"    {'Strike':>10s}", end="")
        for name in models:
            print(f" {name[:10]:>12s}", end="")
        print()
        print(f"    {'-'*10}", end="")
        for _ in models:
            print(f" {'-'*12}", end="")
        print()

        day_prices = {}
        for name in models:
            if name == "Brulant v1.1":
                prices, _ = price_brulant_digital(S0, strikes, hours, BRULANT_PARAMS,
                                                   num_paths=100000, seed=42+k_day*1000)
            elif name == "GBM (Black-Scholes)":
                prices, _ = price_digital_generic(
                    lambda ns, d, np_, s0, sigma, seed: simulate_gbm(ns, d, np_, s0, sigma, seed=seed),
                    S0, strikes, hours, {"sigma": gbm_params["sigma"]}, num_paths=100000, seed=42+k_day*1000)
            elif name == "Heston":
                prices, _ = price_digital_generic(
                    lambda ns, d, np_, s0, v0, kappa, theta, xi, rho_h, seed: simulate_heston(ns, d, np_, s0, v0, kappa, theta, xi, rho_h, seed=seed),
                    S0, strikes, hours, heston_params, num_paths=100000, seed=42+k_day*1000)
            elif name == "Merton Jump-Diff":
                prices, _ = price_digital_generic(
                    lambda ns, d, np_, s0, sigma, lam, jump_mu, jump_sigma, seed: simulate_merton(ns, d, np_, s0, sigma, lam, jump_mu, jump_sigma, seed=seed),
                    S0, strikes, hours, merton_params, num_paths=100000, seed=42+k_day*1000)
            elif name == "SABR":
                prices, _ = price_digital_generic(
                    lambda ns, d, np_, s0, alpha_s, beta_s, rho_s, nu_s, seed: simulate_sabr(ns, d, np_, s0, alpha_s, beta_s, rho_s, nu_s, seed=seed),
                    S0, strikes, hours, sabr_params, num_paths=100000, seed=42+k_day*1000)
            day_prices[name] = prices

        for j, K in enumerate(strikes):
            print(f"    ${int(K):>8,}", end="")
            for name in models:
                print(f" {day_prices[name][j]:>12.6f}", end="")
            print()

        all_pricing[f"+{k_day}d"] = {name: day_prices[name].tolist() for name in models}

    # ===== PLOT =====
    if _HAS_PLT:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. OOS loss comparison
        ax = axes[0, 0]
        names = list(models.keys())
        medians = [np.median(models[n]["losses"]) for n in names]
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
        bars = ax.bar(range(len(names)), medians, color=colors[:len(names)])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n[:12] for n in names], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Median OOS Loss")
        ax.set_title("OOS Moment-Match Loss (lower = better)", fontweight="bold")
        ax.set_yscale("log")
        for bar, val in zip(bars, medians):
            ax.text(bar.get_x() + bar.get_width()/2, val * 1.3, f"{val:.1f}",
                    ha="center", fontsize=8)

        # 2. Std ratio comparison
        ax = axes[0, 1]
        ratios = [models[n]["std_ratio"] for n in names]
        bars = ax.bar(range(len(names)), ratios, color=colors[:len(names)])
        ax.axhline(1.0, color="black", linestyle="--", linewidth=2, label="Target")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n[:12] for n in names], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Std Ratio (sim/emp)")
        ax.set_title("Volatility Match (target=1.0)", fontweight="bold")

        # 3. Digital pricing curves +1d
        ax = axes[1, 0]
        if "+1d" in all_pricing:
            for i, (name, prices) in enumerate(all_pricing["+1d"].items()):
                ax.plot(strikes/1000, prices, "o-", label=name[:12], color=colors[i], markersize=3)
            ax.axvline(S0/1000, color="black", linestyle="--", alpha=0.5)
            ax.set_xlabel("Strike ($k)")
            ax.set_ylabel("Digital Price")
            ax.set_title("+1d Digital Call Prices", fontweight="bold")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # 4. Digital pricing curves +3d
        ax = axes[1, 1]
        if "+3d" in all_pricing:
            for i, (name, prices) in enumerate(all_pricing["+3d"].items()):
                ax.plot(strikes/1000, prices, "o-", label=name[:12], color=colors[i], markersize=3)
            ax.axvline(S0/1000, color="black", linestyle="--", alpha=0.5)
            ax.set_xlabel("Strike ($k)")
            ax.set_ylabel("Digital Price")
            ax.set_title("+3d Digital Call Prices", fontweight="bold")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Brulant v1.1 vs Benchmarks — Spot ${S0:,.0f}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig("benchmark_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print("\nSaved benchmark_comparison.png")

    # Save results
    output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "spot": S0,
        "empirical_moments": emp.tolist(),
        "models": {},
    }
    for name, m in models.items():
        output["models"][name] = {
            "median_loss": float(np.median(m["losses"])),
            "mean_loss": float(np.mean(m["losses"])),
            "std_ratio": m["std_ratio"],
            "tail_3sig_ratio": m["tail_3sig"],
            "sim_moments": m["sim_moments"].tolist(),
        }
    output["pricing"] = _to_jsonable(all_pricing)
    Path("benchmark_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Saved benchmark_results.json")

    # Final ranking
    print("\n" + "=" * 70)
    print("  FINAL RANKING (by median OOS loss)")
    print("=" * 70)
    ranked = sorted(models.items(), key=lambda x: np.median(x[1]["losses"]))
    for rank, (name, m) in enumerate(ranked, 1):
        l = np.array(m["losses"])
        print(f"  #{rank} {name:<22s}  median={np.median(l):.2f}  std_ratio={m['std_ratio']:.3f}")


if __name__ == "__main__":
    main()
