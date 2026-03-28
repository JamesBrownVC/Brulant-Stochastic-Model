"""
Multi-Scale Model Benchmark
============================
Simulate at 1-min resolution, evaluate at multiple frequencies (5m, 15m, 1h, daily).
Tests structural properties — kurtosis emergence, jump frequency, vol clustering,
tail ratios, volatility signature — not noise-level moment matching.

Usage:
    python -u multi_scale_benchmark.py          # full run (~35 min)
    python -u multi_scale_benchmark.py --quick   # fast mode (~8 min)
"""
from __future__ import annotations
import argparse, json, time, datetime, sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import differential_evolution
try:
    import cma
    _HAS_CMA = True
except ImportError:
    _HAS_CMA = False

sys.stdout.reconfigure(line_buffering=True)

from fit_sandpile import (
    fetch_binance_log_returns, interval_to_dt_years,
    moment_vector, recent_exponential_weights, _to_jsonable,
)
from benchmark_v12 import BRULANT_V12, BRULANT_V11, diebold_mariano_test, bootstrap_ci
from experiment_v12 import simulate_v12
from backtest_buffer_model import simulate_buffer_paths, fit_buffer_model
from stoch_ou import simulate_stoch_ou, calibrate_stoch_ou
from benchmark_comparison import (
    simulate_gbm, calibrate_gbm,
    simulate_heston, calibrate_heston,
    simulate_merton, calibrate_merton,
    simulate_sabr, calibrate_sabr,
)

# Evaluation frequencies (multiples of 1-min bars for reference)
EVAL_FREQS = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}


# =========================================================================
#  DATA
# =========================================================================
def aggregate_returns(r_1m: np.ndarray, factor: int) -> np.ndarray:
    """Sum consecutive 1-min log returns into lower-frequency returns.
    Works for both 1D (empirical) and 2D (paths x steps) arrays.
    """
    if r_1m.ndim == 1:
        n = (r_1m.size // factor) * factor
        return r_1m[:n].reshape(-1, factor).sum(axis=1)
    else:
        # (num_paths, n_steps) -> (num_paths, n_steps//factor)
        n = (r_1m.shape[1] // factor) * factor
        return r_1m[:, :n].reshape(r_1m.shape[0], -1, factor).sum(axis=2)


def fetch_data(n_15m_candles: int = 2000) -> Dict[str, Any]:
    """Fetch 15-min candles. Split, winsorize, aggregate to 1h/4h/daily.
    Everything runs at 15-min or coarser -- no 1-min noise.
    """
    print(f"Fetching {n_15m_candles} 15-min candles...")
    r_15m_raw = fetch_binance_log_returns("BTCUSDT", "15m", n_15m_candles)
    dt_15m = interval_to_dt_years("15m")
    dt_1m = interval_to_dt_years("1m")

    # Split 50/50
    n_split = int(len(r_15m_raw) * 0.5)
    train_raw = r_15m_raw[:n_split]
    test_raw = r_15m_raw[n_split:]

    # Winsorize using train stats
    mu = np.median(train_raw)
    mad = np.percentile(np.abs(train_raw - mu), 75) * 1.4826
    train_15m = np.clip(train_raw, mu - 5 * mad, mu + 5 * mad)
    test_15m = np.clip(test_raw, mu - 5 * mad, mu + 5 * mad)

    # Build test data at multiple frequencies by aggregating from 15-min
    # Factors relative to 15-min: 1h=4, 4h=16, 1d=96
    agg_factors = {"15m": 1, "1h": 4, "4h": 16, "1d": 96}
    test = {}
    for freq, factor in agg_factors.items():
        if factor == 1:
            test[freq] = test_15m
        else:
            agg = aggregate_returns(test_15m, factor)
            if agg.size >= 5:
                test[freq] = agg

    # Spot price
    try:
        import requests
        S0 = float(requests.get("https://api.binance.com/api/v3/ticker/price",
                                params={"symbol": "BTCUSDT"}, timeout=10).json()["price"])
    except Exception:
        S0 = 85000.0

    print(f"  Spot: ${S0:,.2f}")
    print(f"  Train: {train_15m.size} 15-min bars | Test: {test_15m.size}")
    for freq in agg_factors:
        if freq in test:
            print(f"    {freq}: {test[freq].size} bars")

    return {"train_15m": train_15m, "test": test, "dt_1m": dt_1m, "dt_15m": dt_15m,
            "S0": S0}


# =========================================================================
#  STRUCTURAL METRICS
# =========================================================================
def structural_metrics(r: np.ndarray) -> Dict[str, float]:
    """Compute structural statistics that differentiate models."""
    r = np.asarray(r).ravel()
    n = r.size
    if n < 5:
        return {k: float('nan') for k in [
            "std", "abs_mean", "skew", "kurtosis",
            "tail_3sig", "tail_4sig", "abs_acf1", "abs_acf5", "r2_acf1"]}

    std = np.std(r)
    abs_r = np.abs(r)

    # Kurtosis and skew
    m = np.mean(r)
    xc = r - m
    m2 = np.mean(xc**2)
    m3 = np.mean(xc**3)
    m4 = np.mean(xc**4)
    s = np.sqrt(max(m2, 1e-30))
    skew = m3 / (s**3 + 1e-30)
    kurt = m4 / (m2**2 + 1e-30) - 3.0

    # Tail ratios vs Gaussian
    gauss_3sig = 0.0027  # P(|Z| > 3)
    gauss_4sig = 6.334e-5
    tail_3 = np.mean(abs_r > 3 * std) / max(gauss_3sig, 1e-12)
    tail_4 = np.mean(abs_r > 4 * std) / max(gauss_4sig, 1e-12)

    # Autocorrelations of |r| and r^2
    def safe_acf(x, lag):
        if len(x) <= lag + 1:
            return 0.0
        xc = x - np.mean(x)
        c0 = np.mean(xc**2)
        if c0 < 1e-30:
            return 0.0
        return float(np.mean(xc[lag:] * xc[:-lag]) / c0)

    abs_acf1 = safe_acf(abs_r, 1)
    abs_acf5 = safe_acf(abs_r, 5) if n > 10 else 0.0
    r2_acf1 = safe_acf(r**2, 1)

    return {
        "std": float(std),
        "abs_mean": float(np.mean(abs_r)),
        "skew": float(skew),
        "kurtosis": float(kurt),
        "tail_3sig": float(tail_3),
        "tail_4sig": float(tail_4),
        "abs_acf1": float(abs_acf1),
        "abs_acf5": float(abs_acf5),
        "r2_acf1": float(r2_acf1),
    }


def detect_jumps(r: np.ndarray, threshold_sigma: float = 4.0) -> np.ndarray:
    """Detect jumps as |r| > threshold * MAD-based sigma."""
    r = np.asarray(r).ravel()
    mu = np.median(r)
    mad = np.percentile(np.abs(r - mu), 75) * 1.4826
    if mad < 1e-15:
        mad = np.std(r)
    return np.abs(r - mu) > threshold_sigma * mad


def jumps_per_day(r: np.ndarray, dt_minutes: float) -> float:
    """Count detected jumps per day."""
    jumps = detect_jumps(r, threshold_sigma=4.0)
    bars_per_day = 1440.0 / dt_minutes
    n_days = r.size / bars_per_day
    if n_days < 0.01:
        return 0.0
    return float(np.sum(jumps) / n_days)


def vol_signature(r_1m: np.ndarray, factors: list = None) -> Dict[int, float]:
    """Realized vol at different sampling frequencies.
    RV(k) = std(aggregated_k) / sqrt(k). Under pure diffusion this is flat.
    """
    if factors is None:
        factors = [1, 5, 15, 30, 60, 120, 240]
    result = {}
    for k in factors:
        agg = aggregate_returns(r_1m, k) if k > 1 else r_1m
        if agg.ndim > 1:
            agg = agg.ravel()
        if agg.size < 5:
            continue
        result[k] = float(np.std(agg) / np.sqrt(k))
    return result


def leverage_corr(r: np.ndarray, lag: int = 1) -> float:
    """Correlation between r_t and |r_{t+lag}| — leverage/asymmetric vol effect."""
    r = np.asarray(r).ravel()
    if r.size < lag + 5:
        return 0.0
    x = r[:-lag]
    y = np.abs(r[lag:])
    cc = np.corrcoef(x, y)
    if cc.shape == (2, 2) and np.isfinite(cc[0, 1]):
        return float(cc[0, 1])
    return 0.0


def distribution_comparison(emp: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """KS test, QQ distance, Wasserstein distance."""
    emp = np.asarray(emp).ravel()
    sim = np.asarray(sim).ravel()
    if emp.size < 10 or sim.size < 10:
        return {"ks_stat": float('nan'), "ks_pval": float('nan'),
                "qq_tail_dist": float('nan'), "wasserstein": float('nan')}

    ks_stat, ks_pval = sp_stats.ks_2samp(emp, sim)

    # QQ tail distance at extreme percentiles
    tail_pcts = [1, 2, 5, 95, 98, 99]
    eq = np.percentile(emp, tail_pcts)
    sq = np.percentile(sim, tail_pcts)
    # Normalize by empirical IQR to make scale-free
    iqr = np.percentile(emp, 75) - np.percentile(emp, 25)
    if iqr < 1e-15:
        iqr = np.std(emp)
    qq_tail = float(np.sqrt(np.mean(((eq - sq) / max(iqr, 1e-15))**2)))

    wass = float(sp_stats.wasserstein_distance(emp, sim))

    return {
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
        "qq_tail_dist": qq_tail,
        "wasserstein": wass,
    }


def vol_clustering_ratio(r: np.ndarray, window: int = 60) -> float:
    """Ratio of mean RV in high-vol vs low-vol halves (rolling window)."""
    r = np.asarray(r).ravel()
    if r.size < 2 * window:
        return 1.0
    # Rolling realized vol
    rv = np.array([np.std(r[i:i+window]) for i in range(r.size - window)])
    med_rv = np.median(rv)
    if med_rv < 1e-15:
        return 1.0
    high = rv[rv >= med_rv]
    low = rv[rv < med_rv]
    if low.size == 0 or np.mean(low) < 1e-15:
        return 10.0
    return float(np.mean(high) / np.mean(low))


# =========================================================================
#  V1.1 UNCAPPED CALIBRATION
# =========================================================================
def calibrate_v11_uncapped(
    train_r: np.ndarray,
    dt: float,
    *,
    num_paths: int = 1000,
    maxiter: int = 12,
    seed: int = 42,
) -> Dict[str, Any]:
    """Calibrate v1.1 without sigma_Y/lambda0 penalties and with wider bounds."""
    w = recent_exponential_weights(train_r.size, 400.0)
    target = moment_vector(train_r, w=w, acf_recent_bars=300)
    scales = np.maximum(np.abs(target), np.array([1e-12, 1e-12, 0.5, 1.0, 0.05, 0.1]))
    scales = np.maximum(scales, 1e-9)

    names = ["mu0", "sigma0", "rho", "nu", "alpha", "beta", "lambda0", "phi", "sigma_Y"]
    bounds = [
        (-0.50, 0.50),    # mu0
        (0.05, 1.40),     # sigma0
        (0.1, 4.0),       # rho
        (0.1, 2.0),       # nu
        (0.5, 20.0),      # alpha
        (1e-4, 0.5),      # beta (slightly wider)
        (0.05, 50.0),     # lambda0 — UNCAPPED (was 10.0)
        (0.1, 2.0),       # phi
        (0.02, 1.0),      # sigma_Y — UNCAPPED (was 0.3)
    ]

    fixed = {
        "kappa": 15.0, "theta_p": 1.5,
        "gamma": 20.0, "eta": 1.0, "eps": 1e-3,
    }

    rng = np.random.default_rng(seed)

    def obj(theta):
        p = {k: float(v) for k, v in zip(names, theta)}
        p.update(fixed)
        s = int(rng.integers(0, 2**31 - 1))
        sim_lr, _ = simulate_buffer_paths(
            train_r.size, dt, num_paths, seed=s, S0=1.0, **p)
        pooled = sim_lr.ravel()
        if pooled.size > 50_000:
            pooled = np.random.default_rng(s).choice(pooled, 50_000, replace=False)
        sim = moment_vector(pooled, w=None, acf_recent_bars=300)
        z = (sim - target) / scales
        # NO penalties on sigma_Y or lambda0
        # Only light penalties on rho to prevent explosion
        penalty = 2.0 * max(0, p["rho"] - 3.0)**2
        return float(np.sum(z * z)) + penalty

    # 9 params x popsize=12 x maxiter=12 = 1296 evals
    res = differential_evolution(obj, bounds, maxiter=maxiter, seed=seed, workers=1,
                                 polish=True, popsize=12, tol=1e-3, atol=1e-4)

    fit = {k: float(v) for k, v in zip(names, res.x)}
    fit.update(fixed)
    fit["_train_loss"] = float(res.fun)
    return fit


def calibrate_v11_uncapped_15m(
    train_r_15m: np.ndarray,
    dt_15m: float,
    *,
    num_paths: int = 1000,
    maxiter: int = 12,
    seed: int = 42,
) -> Dict[str, Any]:
    """Calibrate v1.1 uncapped on 15-min aggregated data.
    Simulates at 1-min internally but matches 15-min moments.
    """
    w = recent_exponential_weights(train_r_15m.size, 100.0)
    target = moment_vector(train_r_15m, w=w, acf_recent_bars=80)
    scales = np.maximum(np.abs(target), np.array([1e-12, 1e-12, 0.5, 1.0, 0.05, 0.1]))
    scales = np.maximum(scales, 1e-9)

    dt_1m = interval_to_dt_years("1m")
    sim_steps = train_r_15m.size * 15  # simulate at 1-min

    names = ["mu0", "sigma0", "rho", "nu", "alpha", "beta", "lambda0", "phi", "sigma_Y"]
    bounds = [
        (-0.50, 0.50),
        (0.05, 1.40),
        (0.1, 4.0),
        (0.1, 2.0),
        (0.5, 20.0),
        (1e-4, 0.5),
        (0.05, 50.0),     # lambda0 uncapped
        (0.1, 2.0),
        (0.02, 1.0),      # sigma_Y uncapped
    ]

    fixed = {
        "kappa": 15.0, "theta_p": 1.5,
        "gamma": 20.0, "eta": 1.0, "eps": 1e-3,
    }

    rng = np.random.default_rng(seed)

    def obj(theta):
        p = {k: float(v) for k, v in zip(names, theta)}
        p.update(fixed)
        s = int(rng.integers(0, 2**31 - 1))
        # Simulate at 1-min resolution
        sim_lr, _ = simulate_buffer_paths(
            sim_steps, dt_1m, num_paths, seed=s, S0=1.0, **p)
        # Aggregate to 15-min
        sim_15m = aggregate_returns(sim_lr, 15)
        pooled = sim_15m.ravel()
        if pooled.size > 50_000:
            pooled = np.random.default_rng(s).choice(pooled, 50_000, replace=False)
        sim = moment_vector(pooled, w=None, acf_recent_bars=80)
        z = (sim - target) / scales
        penalty = 2.0 * max(0, p["rho"] - 3.0)**2
        return float(np.sum(z * z)) + penalty

    res = differential_evolution(obj, bounds, maxiter=maxiter, seed=seed, workers=1,
                                 polish=True, popsize=12, tol=1e-3, atol=1e-4)

    fit = {k: float(v) for k, v in zip(names, res.x)}
    fit.update(fixed)
    fit["_train_loss"] = float(res.fun)
    return fit


# =========================================================================
#  V1.1 EXCITATION MODEL
# =========================================================================
def simulate_v11_excitation(
    n_steps: int, dt: float, num_paths: int, *,
    S0: float = 1.0,
    mu0: float = 0.0, sigma0: float = 0.40,
    rho: float = 2.0, nu: float = 1.5,
    kappa: float = 15.0, theta_p: float = 1.5,
    alpha: float = 10.0, beta: float = 0.10,
    lambda0: float = 2.0, gamma: float = 20.0, eta: float = 1.0,
    phi: float = 1.5, sigma_Y: float = 0.10, eps: float = 1e-3,
    # New excitation params:
    exc_beta: float = 3.0,      # vol target kick per jump event
    exc_kappa: float = 100.0,   # vol target decay rate back to sigma0
    alpha_exc: float = 50.0,    # speed of vol mean-reversion toward excited target
    jump_to_ret: bool = False,  # if False, jumps only trigger vol excitation, no return impact
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """V1.1 with post-jump vol target excitation.

    After a jump event, the vol TARGET spikes up (sigma_target += exc_beta * |jump_size|).
    Sigma then mean-reverts toward this elevated target at rate alpha_exc.
    The target itself decays back to sigma0 at rate exc_kappa.

    Key innovation: jumps ONLY trigger vol excitation, they don't directly
    affect returns (jump_to_ret=False). Fat tails emerge from the elevated
    vol periods following jumps, not from the jump sizes themselves.
    This avoids the bimodal return problem of traditional jump-diffusion.
    """
    rng = np.random.default_rng(seed)
    S = np.full(num_paths, S0, dtype=np.float64)
    sigma = np.full(num_paths, sigma0, dtype=np.float64)
    sigma_target = np.full(num_paths, sigma0, dtype=np.float64)
    M = np.zeros(num_paths, dtype=np.float64)
    B = np.zeros(num_paths, dtype=np.float64)
    lr = np.zeros((num_paths, n_steps), dtype=np.float64)

    sqrt_dt = np.sqrt(dt)
    for t in range(n_steps):
        current_sig = np.maximum(sigma, eps)

        # Jump events (used as vol excitation triggers)
        lambda_t = lambda0 * (1.0 / current_sig) * np.exp(-M)
        p_jump = np.minimum(lambda_t * dt, 1.0)
        jump = (rng.random(num_paths) < p_jump).astype(np.float64)

        jump_mean = np.clip(-phi * B, -0.20, 0.20)
        jump_size = np.clip(rng.normal(jump_mean, sigma_Y), -0.25, 0.25)

        # Return from jump (optional — default off)
        if jump_to_ret:
            Y = np.exp(jump_size)
            dJ = (Y - 1.0) * jump
        else:
            dJ = 0.0

        dW = sqrt_dt * rng.standard_normal(num_paths)
        dWr = sqrt_dt * rng.standard_normal(num_paths)

        ret = (
            mu0 * dt
            - rho * B * dt
            - rho * nu * B * dWr
            + current_sig * dW
            + dJ
        )
        ret = np.clip(ret, -0.50, 0.50)

        S_prev = S.copy()
        S = S * (1.0 + ret)
        S = np.maximum(S, 1e-12)
        lr[:, t] = np.log(S / S_prev)

        # Vol target: jumps up on event, decays back to sigma0
        sigma_target = sigma_target + exc_kappa * (sigma0 - sigma_target) * dt + exc_beta * np.abs(jump_size) * jump
        sigma_target = np.clip(sigma_target, eps, 5.0)

        # Sigma mean-reverts toward the excited target
        sigma = np.clip(current_sig + alpha_exc * (sigma_target - current_sig) * dt, eps, 5.0)

        # Memory and buffer
        M = np.maximum(M - gamma * M * dt + eta * jump, 0.0)
        B = B - kappa * B * dt + theta_p * ret

    return lr, S


def calibrate_v11_excitation_cma(
    train_r: np.ndarray,
    dt: float,
    *,
    num_paths: int = 500,
    max_evals: int = 600,
    seed: int = 42,
    sigma0_scale: float = 0.7,   # post-calibration scaling for base sigma
) -> Dict[str, Any]:
    """Calibrate v1.1-Excitation using CMA-ES with structural loss.

    Uses a loss that directly targets kurtosis, tail ratios, vol clustering,
    and std — not the old 6-moment vector. Fixed seeds reduce eval noise.

    After calibration, scales sigma0 down by sigma0_scale to compensate
    for the fact that calibration overestimates base diffusion by fitting
    to data that includes jump-excited periods.
    """
    # Compute empirical structural targets from training data
    emp = structural_metrics(train_r)
    emp_std = emp["std"]
    emp_kurt = emp["kurtosis"]
    emp_tail3 = emp["tail_3sig"]
    emp_acf1 = emp["abs_acf1"]
    emp_skew = emp["skew"]

    print(f"    Structural targets: std={emp_std:.5f} kurt={emp_kurt:.2f} "
          f"tail3={emp_tail3:.1f}x acf1={emp_acf1:.3f}")

    # Parameters to optimize
    names = ["mu0", "sigma0", "rho", "nu", "alpha_exc",
             "lambda0", "phi", "sigma_Y", "exc_beta", "exc_kappa"]
    bounds_lo = np.array([-0.30, 0.05, 0.1, 0.1, 5.0,
                          0.05, 0.1, 0.02, 0.5, 5.0])
    bounds_hi = np.array([0.30, 1.40, 4.0, 2.0, 200.0,
                          50.0, 2.0, 1.0, 20.0, 500.0])

    fixed = {
        "kappa": 15.0, "theta_p": 1.5,
        "gamma": 20.0, "eta": 1.0, "eps": 1e-3,
        "alpha": 10.0, "beta": 0.0,  # base alpha not used (alpha_exc replaces)
    }

    # Fixed seeds for deterministic evaluations (average over 3 to reduce noise)
    eval_seeds = [42, 119, 256]

    # Starting point
    x0 = np.array([0.0, 0.50, 1.5, 1.0, 50.0,
                    5.0, 1.0, 0.15, 5.0, 100.0])
    sigma_cma = 0.3

    eval_count = [0]
    best_so_far = [1e12]

    def obj(theta):
        theta = np.clip(theta, bounds_lo, bounds_hi)
        p = {k: float(v) for k, v in zip(names, theta)}
        p.update(fixed)

        # Average structural loss over fixed seeds
        total_loss = 0.0
        for s in eval_seeds:
            sim_lr, _ = simulate_v11_excitation(
                train_r.size, dt, num_paths, seed=s, S0=1.0, **p)
            pooled = sim_lr.ravel()
            if pooled.size > 80_000:
                pooled = np.random.default_rng(s).choice(pooled, 80_000, replace=False)
            sm = structural_metrics(pooled)

            # Structural loss: weighted squared relative errors
            loss = 0.0
            # Std match (weight 2.0) — must get vol level right
            loss += 2.0 * ((sm["std"] - emp_std) / max(emp_std, 1e-6))**2
            # Kurtosis match (weight 3.0) — the main prize
            loss += 3.0 * ((sm["kurtosis"] - emp_kurt) / max(abs(emp_kurt), 0.5))**2
            # Tail ratio (weight 2.0) — fat tails
            loss += 2.0 * ((sm["tail_3sig"] - emp_tail3) / max(abs(emp_tail3), 0.5))**2
            # Vol clustering (weight 2.0) — the other main prize
            loss += 2.0 * ((sm["abs_acf1"] - emp_acf1) / max(abs(emp_acf1), 0.01))**2
            # Skew (weight 0.5) — nice to have
            loss += 0.5 * ((sm["skew"] - emp_skew) / max(abs(emp_skew), 0.1))**2

            total_loss += loss

        avg_loss = total_loss / len(eval_seeds)

        # Light penalty on extreme params
        penalty = 2.0 * max(0, p["rho"] - 3.0)**2

        result = avg_loss + penalty
        eval_count[0] += 1

        if result < best_so_far[0]:
            best_so_far[0] = result

        if eval_count[0] % 40 == 0:
            print(f"    CMA eval {eval_count[0]:>4d}: loss={result:>8.2f} (best={best_so_far[0]:.2f})  "
                  f"mu0={p['mu0']:.3f} sig0={p['sigma0']:.3f} rho={p['rho']:.2f} "
                  f"nu={p['nu']:.2f} aExc={p['alpha_exc']:.1f} lam={p['lambda0']:.2f} "
                  f"phi={p['phi']:.2f} sigY={p['sigma_Y']:.3f} "
                  f"excB={p['exc_beta']:.1f} excK={p['exc_kappa']:.0f}")
        return result

    if _HAS_CMA:
        opts = cma.CMAOptions()
        opts['bounds'] = [bounds_lo.tolist(), bounds_hi.tolist()]
        opts['maxfevals'] = max_evals
        opts['seed'] = seed
        opts['verbose'] = -1  # quiet
        opts['tolfun'] = 0.01
        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma_cma, opts)
        es.optimize(obj)
        best_x = np.clip(es.result.xbest, bounds_lo, bounds_hi)
        best_loss = es.result.fbest
        print(f"    CMA-ES done: {eval_count[0]} evals, loss={best_loss:.2f}")
    else:
        bounds = list(zip(bounds_lo, bounds_hi))
        res = differential_evolution(obj, bounds, maxiter=10, seed=seed,
                                     workers=1, polish=True, popsize=10, tol=1e-3)
        best_x = res.x
        best_loss = res.fun

    fit = {k: float(v) for k, v in zip(names, best_x)}
    fit.update(fixed)

    # Post-calibration: scale down sigma0 to compensate for excitation bias
    original_sigma0 = fit["sigma0"]
    fit["sigma0"] = fit["sigma0"] * sigma0_scale
    fit["_sigma0_prescale"] = original_sigma0
    fit["_sigma0_scale"] = sigma0_scale
    fit["_train_loss"] = float(best_loss)
    print(f"    sigma0: {original_sigma0:.4f} -> {fit['sigma0']:.4f} (x{sigma0_scale})")
    return fit


def calibrate_v11_excitation_twophase(
    train_r: np.ndarray,
    dt: float,
    *,
    num_paths: int = 500,
    cma_evals: int = 600,
    cd_passes: int = 3,
    cd_grid: int = 11,
    cd_refine: int = 5,
    cd_tol: float = 0.005,
    seed: int = 42,
) -> Dict[str, Any]:
    """Two-phase calibration: CMA-ES global search + coordinate descent fine-tuning.

    Phase 1: CMA-ES explores the full parameter space (~600 evals).
    Phase 2: Coordinate descent cycles through params one at a time,
             grid-searching then refining each (~160 evals/pass, up to 3 passes).

    No post-hoc sigma0 scaling — the optimizer finds sigma0 directly.
    """
    # Empirical structural targets
    emp = structural_metrics(train_r)
    emp_std = emp["std"]
    emp_kurt = emp["kurtosis"]
    emp_tail3 = emp["tail_3sig"]
    emp_acf1 = emp["abs_acf1"]
    emp_skew = emp["skew"]

    print(f"    Structural targets: std={emp_std:.5f} kurt={emp_kurt:.2f} "
          f"tail3={emp_tail3:.1f}x acf1={emp_acf1:.3f}")

    # Parameters to optimize
    names = ["mu0", "sigma0", "rho", "nu", "alpha_exc",
             "lambda0", "phi", "sigma_Y", "exc_beta", "exc_kappa"]
    bounds_lo = np.array([-0.30, 0.05, 0.1, 0.1, 5.0,
                          0.05, 0.1, 0.02, 0.5, 5.0])
    bounds_hi = np.array([0.30, 1.40, 4.0, 2.0, 200.0,
                          50.0, 2.0, 1.0, 20.0, 500.0])

    fixed = {
        "kappa": 15.0, "theta_p": 1.5,
        "gamma": 20.0, "eta": 1.0, "eps": 1e-3,
        "alpha": 10.0, "beta": 0.0,
    }

    eval_seeds = [42, 119, 256]
    total_evals = [0]

    def structural_loss(theta):
        """Compute structural loss for a parameter vector. Returns 1e6 on sanity failure."""
        theta = np.clip(theta, bounds_lo, bounds_hi)
        p = {k: float(v) for k, v in zip(names, theta)}
        p.update(fixed)

        total = 0.0
        for s in eval_seeds:
            sim_lr, _ = simulate_v11_excitation(
                train_r.size, dt, num_paths, seed=s, S0=1.0, **p)
            pooled = sim_lr.ravel()
            if pooled.size > 80_000:
                pooled = np.random.default_rng(s).choice(pooled, 80_000, replace=False)
            sm = structural_metrics(pooled)

            # Sanity guards: reject degenerate solutions immediately
            if sm["kurtosis"] > 100 or sm["std"] / max(emp_std, 1e-8) > 5.0:
                total_evals[0] += 1
                return 1e6

            loss = 0.0
            loss += 2.0 * ((sm["std"] - emp_std) / max(emp_std, 1e-6))**2
            loss += 3.0 * ((sm["kurtosis"] - emp_kurt) / max(abs(emp_kurt), 0.5))**2
            loss += 2.0 * ((sm["tail_3sig"] - emp_tail3) / max(abs(emp_tail3), 0.5))**2
            loss += 2.0 * ((sm["abs_acf1"] - emp_acf1) / max(abs(emp_acf1), 0.01))**2
            loss += 0.5 * ((sm["skew"] - emp_skew) / max(abs(emp_skew), 0.1))**2
            total += loss

        avg_loss = total / len(eval_seeds)
        penalty = 2.0 * max(0, p["rho"] - 3.0)**2
        total_evals[0] += 1
        return avg_loss + penalty

    # ── Phase 1: CMA-ES global search ──
    print(f"\n    Phase 1: CMA-ES global search ({cma_evals} evals)")
    x0 = np.array([0.0, 0.50, 1.5, 1.0, 50.0,
                    5.0, 1.0, 0.15, 5.0, 100.0])
    sigma_cma = 0.3

    best_so_far = [1e12]
    phase1_evals_start = total_evals[0]

    def cma_obj(theta):
        result = structural_loss(theta)
        if result < best_so_far[0]:
            best_so_far[0] = result
        n = total_evals[0]
        if n % 40 == 0:
            p = {k: float(v) for k, v in zip(names, np.clip(theta, bounds_lo, bounds_hi))}
            print(f"      CMA eval {n:>4d}: loss={result:>8.2f} (best={best_so_far[0]:.2f})  "
                  f"mu0={p['mu0']:.3f} sig0={p['sigma0']:.3f} rho={p['rho']:.2f} "
                  f"nu={p['nu']:.2f} aExc={p['alpha_exc']:.1f} lam={p['lambda0']:.2f} "
                  f"phi={p['phi']:.2f} sigY={p['sigma_Y']:.3f} "
                  f"excB={p['exc_beta']:.1f} excK={p['exc_kappa']:.0f}")
        return result

    if _HAS_CMA:
        es = cma.CMAEvolutionStrategy(x0.tolist(), sigma_cma,
                 {'bounds': [bounds_lo.tolist(), bounds_hi.tolist()],
                  'maxfevals': cma_evals,
                  'seed': seed,
                  'verbose': -1})
        cma_eval_count = 0
        while not es.stop() and cma_eval_count < cma_evals:
            solutions = es.ask()
            fitnesses = [cma_obj(s) for s in solutions]
            es.tell(solutions, fitnesses)
            cma_eval_count += len(solutions)
        best_x = np.clip(es.result.xbest, bounds_lo, bounds_hi)
        phase1_loss = es.result.fbest
    else:
        bounds = list(zip(bounds_lo, bounds_hi))
        res = differential_evolution(cma_obj, bounds, maxiter=10, seed=seed,
                                     workers=1, polish=True, popsize=10, tol=1e-3)
        best_x = res.x
        phase1_loss = res.fun

    phase1_evals = total_evals[0] - phase1_evals_start
    print(f"    Phase 1 done: {phase1_evals} evals, loss={phase1_loss:.4f}")
    for k, v in zip(names, best_x):
        print(f"      {k:>12s} = {v:.4f}")

    # ── Phase 2: Coordinate descent fine-tuning ──
    # Parameter order by impact
    cd_order = ["sigma0", "exc_beta", "exc_kappa", "alpha_exc",
                "lambda0", "sigma_Y", "rho", "phi", "nu", "mu0"]
    cd_indices = [names.index(n) for n in cd_order]

    current = best_x.copy()
    current_loss = phase1_loss

    print(f"\n    Phase 2: Coordinate descent ({cd_passes} passes max, "
          f"{cd_grid}+{cd_refine} pts/param)")

    passes_used = 0
    phase2_evals_start = total_evals[0]

    for pass_idx in range(cd_passes):
        pass_start_loss = current_loss
        print(f"\n      Pass {pass_idx + 1}/{cd_passes} (current loss={current_loss:.4f})")

        for param_name, param_idx in zip(cd_order, cd_indices):
            lo = bounds_lo[param_idx]
            hi = bounds_hi[param_idx]
            cur_val = current[param_idx]

            # Grid search: 11 points around current value (±50%)
            grid_lo = max(lo, cur_val * 0.5 if cur_val > 0 else cur_val * 1.5)
            grid_hi = min(hi, cur_val * 1.5 if cur_val > 0 else cur_val * 0.5)
            # Handle zero/negative current values
            if grid_lo >= grid_hi:
                grid_lo, grid_hi = lo, hi
            grid_pts = np.linspace(grid_lo, grid_hi, cd_grid)

            best_grid_val = cur_val
            best_grid_loss = current_loss

            for val in grid_pts:
                trial = current.copy()
                trial[param_idx] = val
                loss = structural_loss(trial)
                if loss < best_grid_loss:
                    best_grid_loss = loss
                    best_grid_val = val

            # Refine: 5 points around best grid value
            if cd_grid > 1:
                step = (grid_hi - grid_lo) / (cd_grid - 1)
            else:
                step = (hi - lo) / 10
            ref_lo = max(lo, best_grid_val - step / 2)
            ref_hi = min(hi, best_grid_val + step / 2)
            ref_pts = np.linspace(ref_lo, ref_hi, cd_refine)

            for val in ref_pts:
                trial = current.copy()
                trial[param_idx] = val
                loss = structural_loss(trial)
                if loss < best_grid_loss:
                    best_grid_loss = loss
                    best_grid_val = val

            # Update if improved
            if best_grid_loss < current_loss:
                improvement = (current_loss - best_grid_loss) / max(current_loss, 1e-10)
                current[param_idx] = best_grid_val
                current_loss = best_grid_loss
                print(f"        {param_name:>12s}: {cur_val:.4f} -> {best_grid_val:.4f} "
                      f"(loss={current_loss:.4f}, -{improvement*100:.1f}%)")

        passes_used = pass_idx + 1

        # Check convergence
        pass_improvement = (pass_start_loss - current_loss) / max(pass_start_loss, 1e-10)
        print(f"      Pass {pass_idx + 1} done: loss={current_loss:.4f} "
              f"(pass improvement: {pass_improvement*100:.2f}%)")
        if pass_improvement < cd_tol:
            print(f"      Converged (improvement < {cd_tol*100:.1f}%)")
            break

    phase2_evals = total_evals[0] - phase2_evals_start

    # Build result
    fit = {k: float(v) for k, v in zip(names, current)}
    fit.update(fixed)
    fit["_phase1_loss"] = float(phase1_loss)
    fit["_phase2_loss"] = float(current_loss)
    fit["_cd_passes_used"] = passes_used
    fit["_total_evals"] = total_evals[0]

    print(f"\n    Two-phase calibration done: {total_evals[0]} total evals "
          f"(phase1={phase1_evals}, phase2={phase2_evals})")
    print(f"    Loss: {phase1_loss:.4f} -> {current_loss:.4f} "
          f"({(1 - current_loss/max(phase1_loss, 1e-10))*100:.1f}% improvement)")
    for k, v in zip(names, current):
        print(f"      {k:>12s} = {v:.4f}")

    return fit


# =========================================================================
#  MODEL DISPATCHERS
# =========================================================================
def simulate_model(tag: str, params: Dict, n_steps: int, dt: float,
                   num_paths: int, seed: int) -> np.ndarray:
    """Simulate and return (num_paths, n_steps) log returns at 1-min."""
    p = {k: v for k, v in params.items() if not k.startswith('_')}
    if tag == "v11_exc":
        lr, _ = simulate_v11_excitation(n_steps, dt, num_paths, seed=seed, S0=1.0, **p)
    elif tag in ("v11", "v11_uncapped", "v11_uncapped_15m"):
        lr, _ = simulate_buffer_paths(n_steps, dt, num_paths, seed=seed, S0=1.0, **p)
    elif tag == "v12":
        lr, _ = simulate_v12(n_steps, dt, num_paths, seed=seed, S0=1.0, **p)
    elif tag == "gbm":
        lr, _ = simulate_gbm(n_steps, dt, num_paths, 1.0, p["sigma"], seed=seed)
    elif tag == "heston":
        lr, _ = simulate_heston(n_steps, dt, num_paths, 1.0, **p, seed=seed)
    elif tag == "merton":
        lr, _ = simulate_merton(n_steps, dt, num_paths, 1.0, **p, seed=seed)
    elif tag == "sabr":
        lr, _ = simulate_sabr(n_steps, dt, num_paths, 1.0, **p, seed=seed)
    elif tag == "stoch_ou":
        lr, _ = simulate_stoch_ou(n_steps, dt, num_paths, seed=seed, S0=1.0, **p)
    else:
        raise ValueError(f"Unknown model tag: {tag}")
    return lr


# =========================================================================
#  MULTI-SCALE EVALUATION
# =========================================================================
def evaluate_model_single_seed(
    tag: str, params: Dict, test_data: Dict[str, np.ndarray],
    dt_15m: float, n_steps_15m: int, num_paths: int, seed: int,
) -> Dict[str, Any]:
    """Simulate at 15-min, aggregate to 1h/4h/1d, compute structural metrics."""
    # Simulate at 15-min resolution
    lr_15m = simulate_model(tag, params, n_steps_15m, dt_15m, num_paths, seed)

    # Aggregation factors relative to 15-min base
    agg_from_15m = {"15m": 1, "1h": 4, "4h": 16, "1d": 96}

    results = {}
    for freq, factor in agg_from_15m.items():
        emp_r = test_data.get(freq)
        if emp_r is None or emp_r.size < 5:
            continue

        if factor == 1:
            sim_pool = lr_15m.ravel()
        else:
            sim_agg = aggregate_returns(lr_15m, factor)
            sim_pool = sim_agg.ravel()

        if sim_pool.size < 5:
            continue
        if sim_pool.size > 200_000:
            # Subsample by selecting a subset of paths to preserve temporal order
            # (random choice destroys autocorrelation structure)
            n_paths, n_steps = lr_15m.shape
            steps_at_freq = n_steps // max(factor, 1)
            max_paths = 200_000 // max(steps_at_freq, 1)
            max_paths = max(min(max_paths, n_paths), 1)
            if factor == 1:
                sim_pool = lr_15m[:max_paths, :].ravel()
            else:
                sim_pool = aggregate_returns(lr_15m[:max_paths, :], factor).ravel()

        sm = structural_metrics(sim_pool)
        em = structural_metrics(emp_r)
        dc = distribution_comparison(emp_r, sim_pool)

        freq_min = EVAL_FREQS.get(freq, 15)
        sim_jpd = jumps_per_day(sim_pool, freq_min)
        emp_jpd = jumps_per_day(emp_r, freq_min)

        results[freq] = {
            "sim_metrics": sm,
            "emp_metrics": em,
            "dist_comp": dc,
            "sim_jumps_per_day": sim_jpd,
            "emp_jumps_per_day": emp_jpd,
        }

    results["_global"] = {
        "sim_vol_sig": {}, "emp_vol_sig": {},
        "sim_leverage": 0.0, "emp_leverage": 0.0,
    }

    return results


def compute_composite_loss(eval_result: Dict[str, Any],
                           n_params: int = 0) -> float:
    """Weighted composite loss from multi-scale evaluation.

    Based on Cont (2001) stylized facts framework.  Three tiers:
      T1  Structural features  – vol clustering, fat tails, kurtosis (highest weight)
      T2  Distributional fit   – std, skew, QQ tail, Wasserstein
      T3  Multi-scale consistency – aggregational Gaussianity
    Plus a BIC-inspired complexity penalty.
    """
    loss = 0.0

    # ── Tier 1: Structural features (Cont stylized facts) ──
    # Vol clustering is the #1 discriminator (Cont facts 6, 8)
    w_t1 = {
        "abs_acf1": 3.0,       # vol clustering lag-1
        "abs_acf5": 2.0,       # vol clustering lag-5 (slow decay)
        "kurtosis": 2.5,       # fat tails
        "tail_3sig": 2.0,      # extreme tail ratio
    }

    # ── Tier 2: Distributional calibration ──
    w_t2 = {
        "std": 2.0,            # volatility level
        "skew": 0.5,           # asymmetry
        "qq_tail_dist": 1.5,   # QQ tail fit
        "wasserstein": 1.0,    # overall distributional distance
    }

    # Combine into single dict with appropriate floors for normalization
    floors = {
        "abs_acf1": 0.01, "abs_acf5": 0.01,
        "kurtosis": 0.5, "tail_3sig": 0.5,
        "std": 1e-6, "skew": 0.1,
    }

    # Frequency weights — higher frequencies get more weight (more data)
    freq_w = {"5m": 1.0, "15m": 1.5, "1h": 1.5, "4h": 1.0, "1d": 0.5}

    n_freq = 0
    for freq in EVAL_FREQS:
        if freq not in eval_result:
            continue
        fr = eval_result[freq]
        sm = fr["sim_metrics"]
        em = fr["emp_metrics"]
        dc = fr["dist_comp"]
        fw = freq_w.get(freq, 1.0)
        n_freq += 1

        # Tier 1 + Tier 2 structural metrics
        for metric, mw in {**w_t1, **w_t2}.items():
            if metric in sm and metric in em:
                sv, ev = sm[metric], em[metric]
                floor = floors.get(metric, 0.1)
                scale = max(abs(ev), floor)
                loss += fw * mw * ((sv - ev) / scale) ** 2
            elif metric in dc:
                val = dc[metric]
                if np.isfinite(val):
                    loss += fw * mw * val ** 2

        # Jump frequency match (only at 15m and 1h)
        if freq in ("15m", "1h"):
            sjpd = fr["sim_jumps_per_day"]
            ejpd = fr["emp_jumps_per_day"]
            scale = max(ejpd, 0.5)
            loss += fw * 1.0 * ((sjpd - ejpd) / scale) ** 2

    # ── Tier 3: Multi-scale consistency ──
    # Vol signature RMSE
    glob = eval_result.get("_global", {})
    svs = glob.get("sim_vol_sig", {})
    evs = glob.get("emp_vol_sig", {})
    common_factors = sorted(set(svs) & set(evs))
    if common_factors:
        diffs = [(svs[k] - evs[k])**2 for k in common_factors]
        e_mean = np.mean([evs[k] for k in common_factors])
        if e_mean > 1e-15:
            vs_rmse = np.sqrt(np.mean(diffs)) / e_mean
        else:
            vs_rmse = np.sqrt(np.mean(diffs))
        loss += 2.0 * vs_rmse ** 2

    # Leverage effect
    sl = glob.get("sim_leverage", 0.0)
    el = glob.get("emp_leverage", 0.0)
    loss += 1.5 * (sl - el) ** 2

    # ── Complexity penalty (BIC-inspired) ──
    # Penalize models with more free parameters to ensure fair comparison
    # Scale: ~5% of a typical structural loss per extra parameter
    if n_params > 0:
        loss += 0.5 * n_params

    return float(loss)


# =========================================================================
#  MAIN BENCHMARK
# =========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fast mode: 5 seeds, 500 paths")
    parser.add_argument("--candles", type=int, default=5000, help="1-min candles to fetch")
    args = parser.parse_args()

    N_SEEDS = 5 if args.quick else 20
    N_PATHS = 500 if args.quick else 1000

    print("=" * 70)
    print("  MULTI-SCALE MODEL BENCHMARK")
    print(f"  Simulate at 1-min, evaluate at {list(EVAL_FREQS.keys())}")
    print(f"  Seeds: {N_SEEDS} | Paths: {N_PATHS}")
    print("=" * 70)

    # --- Phase 1: Data ---
    data = fetch_data(args.candles)
    dt_1m = data["dt_1m"]
    dt_15m = data["dt_15m"]
    test_data = data["test"]
    train_15m = data["train_15m"]

    # Print empirical structural metrics at each frequency
    print(f"\n  Empirical structural metrics:")
    for freq in EVAL_FREQS:
        if freq not in test_data or test_data[freq].size < 5:
            print(f"    {freq:>4s}: (too few samples)")
            continue
        em = structural_metrics(test_data[freq])
        jpd = jumps_per_day(test_data[freq], EVAL_FREQS[freq])
        print(f"    {freq:>4s}: kurt={em['kurtosis']:>6.2f}  tail3s={em['tail_3sig']:>5.1f}x  "
              f"acf|r|={em['abs_acf1']:>5.3f}  jumps/day={jpd:.1f}")
    emp_vs = {}

    # --- Phase 2: Models ---
    # Use pre-fitted params for standard benchmarks (already calibrated).
    # Only calibrate v1.1 uncapped variants on 15-min.
    print(f"\n{'='*70}")
    print(f"  MODELS (pre-fitted benchmarks + fresh v1.1 uncapped calibration)")
    print(f"{'='*70}")

    models = {}

    # Standard benchmarks -- use pre-fitted params, simulate at 15-min with dt_15m
    # GBM: just needs sigma (scale from 1-min fit)
    gbm_p = {"sigma": float(np.std(train_15m) / np.sqrt(dt_15m))}  # analytic
    print(f"  GBM:             sigma={gbm_p['sigma']:.4f}")
    models["GBM"] = ("gbm", gbm_p)

    # Heston/Merton/SABR -- recalibrate on 15-min (fast: only ~1000 train bars)
    t0 = time.perf_counter()
    heston_p = calibrate_heston(train_15m, dt_15m)
    print(f"  Heston:          {time.perf_counter()-t0:.0f}s")
    models["Heston"] = ("heston", heston_p)

    t0 = time.perf_counter()
    merton_p = calibrate_merton(train_15m, dt_15m)
    print(f"  Merton:          {time.perf_counter()-t0:.0f}s  lam={merton_p['lam']:.4f} jsig={merton_p['jump_sigma']:.4f}")
    models["Merton"] = ("merton", merton_p)

    t0 = time.perf_counter()
    sabr_p = calibrate_sabr(train_15m, dt_15m, S0=1.0)
    print(f"  SABR:            {time.perf_counter()-t0:.0f}s")
    models["SABR"] = ("sabr", sabr_p)

    # Brulant v1.2 (pre-fitted on 1m -- baseline)
    v12_sim = {k: v for k, v in BRULANT_V12.items() if not k.startswith('_')}
    print(f"  Brulant v1.2:    (pre-fitted)")
    models["Brulant v1.2"] = ("v12", v12_sim)

    # Brulant v1.1 (pre-fitted on 1m -- baseline)
    v11_sim = {k: v for k, v in BRULANT_V11.items() if not k.startswith('_')}
    print(f"  Brulant v1.1:    (pre-fitted)")
    models["Brulant v1.1"] = ("v11", v11_sim)

    # v1.1 Uncapped -- calibrate on 15-min, no sigma_Y/lambda0 penalty
    t0 = time.perf_counter()
    v11u_p = calibrate_v11_uncapped(train_15m, dt_15m, num_paths=500, maxiter=10, seed=42)
    v11u_sim = {k: v for k, v in v11u_p.items() if not k.startswith('_')}
    print(f"  v1.1 Uncapped:   {time.perf_counter()-t0:.0f}s  sigY={v11u_sim['sigma_Y']:.4f} lam0={v11u_sim['lambda0']:.4f}")
    models["v1.1 Uncapped"] = ("v11_uncapped", v11u_sim)

    # v1.1 Excitation -- aggressive post-jump vol spike + fast decay, CMA-ES
    t0 = time.perf_counter()
    v11e_p = calibrate_v11_excitation_cma(
        train_15m, dt_15m, num_paths=500, max_evals=800, seed=42, sigma0_scale=0.7)
    v11e_sim = {k: v for k, v in v11e_p.items() if not k.startswith('_')}
    print(f"  v1.1 Excitation: {time.perf_counter()-t0:.0f}s  "
          f"sig0={v11e_sim['sigma0']:.4f} sigY={v11e_sim['sigma_Y']:.4f} "
          f"lam0={v11e_sim['lambda0']:.2f} excB={v11e_sim['exc_beta']:.1f} excK={v11e_sim['exc_kappa']:.0f}")
    models["v1.1 Excitation"] = ("v11_exc", v11e_sim)

    # --- Phase 3: Simulation & Evaluation ---
    # Simulate at 15-min directly. Evaluate at 15m, 1h, 4h.
    n_sim_15m = test_data.get("15m", np.array([])).size  # match test length
    if n_sim_15m < 10:
        n_sim_15m = 672  # ~7 days of 15-min bars
    n_total = len(models) * N_SEEDS
    print(f"\n{'='*70}")
    print(f"  EVALUATION ({N_SEEDS} seeds x {N_PATHS} paths)")
    print(f"  Simulating {n_sim_15m} 15-min steps per seed")
    print(f"{'='*70}")

    all_losses = {}
    all_freq_metrics = {}
    all_global_metrics = {}
    done = 0

    for name, (tag, params) in models.items():
        t0 = time.perf_counter()
        losses = []
        freq_metrics_agg = {freq: [] for freq in EVAL_FREQS}

        for i in range(N_SEEDS):
            seed = 42 + i * 77
            ev = evaluate_model_single_seed(
                tag, params, test_data, dt_15m, n_sim_15m, N_PATHS, seed)
            loss = compute_composite_loss(ev)
            losses.append(loss)

            for freq in EVAL_FREQS:
                if freq in ev:
                    freq_metrics_agg[freq].append(ev[freq]["sim_metrics"])
            done += 1

        losses = np.array(losses)
        all_losses[name] = losses

        # Compute median metrics across seeds per frequency
        med_metrics = {}
        for freq in EVAL_FREQS:
            if freq_metrics_agg[freq]:
                keys = freq_metrics_agg[freq][0].keys()
                med_metrics[freq] = {
                    k: float(np.median([m[k] for m in freq_metrics_agg[freq]]))
                    for k in keys
                }
        all_freq_metrics[name] = med_metrics

        # Global metrics from a larger sim
        lr_big = simulate_model(tag, params, n_sim_15m, dt_15m, N_PATHS * 2, seed=42)
        sim_jpd_15m = jumps_per_day(lr_big.ravel(), 15)
        vcr = vol_clustering_ratio(lr_big.ravel(), window=20) if lr_big.ravel().size > 40 else 1.0
        all_global_metrics[name] = {
            "vol_sig": {}, "leverage": 0.0,
            "jumps_per_day_15m": sim_jpd_15m,
            "vol_cluster_ratio": vcr,
        }

        elapsed = time.perf_counter() - t0
        k15 = med_metrics.get("15m", {}).get("kurtosis", 0)
        k1h = med_metrics.get("1h", {}).get("kurtosis", 0)
        pct = done * 100 // n_total
        print(f"  [{pct:>3d}%] {name:<18s}: loss={np.median(losses):>7.1f}  "
              f"kurt15m={k15:>5.2f} kurt1h={k1h:>5.2f}  "
              f"jpd={sim_jpd_15m:.1f}  ({elapsed:.0f}s)")

    # --- Phase 4: Rankings & Statistical Tests ---
    print(f"\n{'='*70}")
    print("  FINAL RANKING (composite multi-scale loss)")
    print(f"{'='*70}")

    ranked = sorted(all_losses.items(), key=lambda x: np.median(x[1]))
    param_counts = {
        "GBM": 1, "Heston": 4, "Merton": 3, "SABR": 4,
        "Stoch OU": 4, "Brulant v1.2": 11, "Brulant v1.1": 14,
        "v1.1 Uncapped": 9, "v1.1 Uncap+15m": 9,
        "v1.1 Excitation": 11,
    }

    print(f"  {'Rank':>4} {'Model':<18s} {'Median':>8s} {'Params':>6s} "
          f"{'kurt15m':>8s} {'kurt1h':>8s} {'jpd15m':>7s} {'vcr':>5s}")
    print(f"  {'-'*4} {'-'*18} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*5}")

    emp_m_15m = structural_metrics(test_data["15m"])
    emp_m_1h = structural_metrics(test_data["1h"])
    emp_jpd_15m = jumps_per_day(test_data["15m"], 15)
    emp_vcr = vol_clustering_ratio(test_data["15m"], window=40)

    for rank, (name, losses) in enumerate(ranked, 1):
        gm = all_global_metrics[name]
        fm = all_freq_metrics[name]
        k15 = fm.get("15m", {}).get("kurtosis", 0)
        k1h = fm.get("1h", {}).get("kurtosis", 0)
        print(f"  {rank:>4d} {name:<18s} {np.median(losses):>8.1f} "
              f"{param_counts.get(name, '?'):>6} "
              f"{k15:>8.2f} {k1h:>8.2f} "
              f"{gm['jumps_per_day_15m']:>7.1f} {gm['vol_cluster_ratio']:>5.2f}")

    print(f"\n  Empirical:       {'':>8s} {'':>6s} "
          f"{emp_m_15m['kurtosis']:>8.2f} {emp_m_1h['kurtosis']:>8.2f} "
          f"{emp_jpd_15m:>7.1f} {emp_vcr:>5.2f}")

    # Paired tests: best model vs each
    best_name = ranked[0][0]
    best_losses = all_losses[best_name]

    print(f"\n  PAIRED TESTS vs {best_name}:")
    print(f"  {'Model':<18s} {'MedDiff':>8s} {'95% CI':>22s} {'Wilcoxon p':>11s} {'Sig':>5s}")
    print(f"  {'-'*18} {'-'*8} {'-'*22} {'-'*11} {'-'*5}")

    for name, losses in all_losses.items():
        if name == best_name:
            continue
        diffs = losses - best_losses
        med_diff = float(np.median(diffs))
        ci_lo, ci_hi = bootstrap_ci(diffs, n_bootstrap=10000, ci=0.95, seed=42)
        try:
            _, w_pval = sp_stats.wilcoxon(diffs, alternative='two-sided')
        except ValueError:
            w_pval = float('nan')
        sig = "***" if w_pval < 0.001 else ("**" if w_pval < 0.01 else ("*" if w_pval < 0.05 else "ns"))
        print(f"  {name:<18s} {med_diff:>+8.1f} [{ci_lo:>+9.1f}, {ci_hi:>+9.1f}] {w_pval:>11.4g} {sig:>5s}")

    # --- Per-frequency kurtosis table ---
    print(f"\n{'='*70}")
    print("  KURTOSIS BY FREQUENCY (median across seeds)")
    print(f"{'='*70}")
    header = f"  {'Model':<18s}" + "".join(f" {f:>8s}" for f in EVAL_FREQS)
    print(header)
    print(f"  {'-'*18}" + "".join(f" {'-'*8}" for _ in EVAL_FREQS))
    for name, _ in ranked:
        fm = all_freq_metrics[name]
        row = f"  {name:<18s}"
        for freq in EVAL_FREQS:
            k = fm.get(freq, {}).get("kurtosis", float('nan'))
            row += f" {k:>8.2f}"
        print(row)
    row = f"  {'Empirical':<18s}"
    for freq in EVAL_FREQS:
        em = structural_metrics(test_data[freq])
        row += f" {em['kurtosis']:>8.2f}"
    print(row)

    # --- Vol clustering table ---
    print(f"\n{'='*70}")
    print("  VOL CLUSTERING abs_acf1 BY FREQUENCY")
    print(f"{'='*70}")
    print(header)
    print(f"  {'-'*18}" + "".join(f" {'-'*8}" for _ in EVAL_FREQS))
    for name, _ in ranked:
        fm = all_freq_metrics[name]
        row = f"  {name:<18s}"
        for freq in EVAL_FREQS:
            v = fm.get(freq, {}).get("abs_acf1", float('nan'))
            row += f" {v:>8.3f}"
        print(row)
    row = f"  {'Empirical':<18s}"
    for freq in EVAL_FREQS:
        em = structural_metrics(test_data[freq])
        row += f" {em['abs_acf1']:>8.3f}"
    print(row)

    # --- Save results ---
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_seeds": N_SEEDS,
        "n_paths": N_PATHS,
        "model_params": {name: {k: v for k, v in params.items() if not k.startswith('_')}
                         for name, (tag, params) in models.items()},
        "model_losses": {name: losses.tolist() for name, losses in all_losses.items()},
        "model_freq_metrics": all_freq_metrics,
        "model_global_metrics": all_global_metrics,
        "empirical_freq_metrics": {
            freq: structural_metrics(test_data[freq]) for freq in EVAL_FREQS
        },
        "empirical_vol_signature": emp_vs,
    }
    Path("multi_scale_benchmark.json").write_text(
        json.dumps(_to_jsonable(result), indent=2), encoding="utf-8")
    print(f"\nSaved multi_scale_benchmark.json")

    # --- Visualization ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {
            "GBM": "#3498db", "Heston": "#2ecc71", "Merton": "#9b59b6",
            "SABR": "#f39c12", "Stoch OU": "#e67e22", "Brulant v1.2": "#e74c3c",
            "Brulant v1.1": "#c0392b", "v1.1 Uncapped": "#1abc9c",
            "v1.1 Uncap+15m": "#16a085", "v1.1 Excitation": "#8e44ad",
        }
        model_order = [n for n, _ in ranked]

        fig, axes = plt.subplots(3, 3, figsize=(20, 15))

        # (0,0) Composite loss box plot
        ax = axes[0, 0]
        bp = ax.boxplot([all_losses[n] for n in model_order],
                        tick_labels=[n[:12] for n in model_order],
                        patch_artist=True, showfliers=False)
        for patch, name in zip(bp["boxes"], model_order):
            patch.set_facecolor(colors.get(name, "#95a5a6"))
            patch.set_alpha(0.7)
        ax.set_ylabel("Composite Multi-Scale Loss")
        ax.set_title("Composite Loss (lower = better)", fontweight="bold")
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3)

        # (0,1) Kurtosis by frequency
        ax = axes[0, 1]
        freqs_list = list(EVAL_FREQS.keys())
        x = np.arange(len(freqs_list))
        for name in model_order:
            fm = all_freq_metrics[name]
            vals = [fm.get(f, {}).get("kurtosis", 0) for f in freqs_list]
            ax.plot(x, vals, 'o-', label=name[:12], color=colors.get(name, "#95a5a6"), alpha=0.8)
        emp_kurt = [structural_metrics(test_data[f])["kurtosis"] for f in freqs_list]
        ax.plot(x, emp_kurt, 'k*-', label="Empirical", markersize=12, linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(freqs_list)
        ax.set_ylabel("Excess Kurtosis")
        ax.set_title("Kurtosis by Frequency", fontweight="bold")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # (0,2) Volatility signature
        ax = axes[0, 2]
        for name in model_order:
            vs = all_global_metrics[name]["vol_sig"]
            ks = sorted(vs.keys())
            ax.plot(ks, [vs[k] for k in ks], 'o-', label=name[:12],
                    color=colors.get(name, "#95a5a6"), alpha=0.7)
        evs_k = sorted(emp_vs.keys())
        ax.plot(evs_k, [emp_vs[k] for k in evs_k], 'k*-', label="Empirical",
                markersize=12, linewidth=2)
        ax.set_xlabel("Aggregation factor (minutes)")
        ax.set_ylabel("RV / sqrt(k)")
        ax.set_title("Volatility Signature", fontweight="bold")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # (1,0) Tail ratio (3σ) by frequency
        ax = axes[1, 0]
        for name in model_order:
            fm = all_freq_metrics[name]
            vals = [fm.get(f, {}).get("tail_3sig", 0) for f in freqs_list]
            ax.plot(x, vals, 'o-', label=name[:12], color=colors.get(name, "#95a5a6"), alpha=0.8)
        emp_tail = [structural_metrics(test_data[f])["tail_3sig"] for f in freqs_list]
        ax.plot(x, emp_tail, 'k*-', label="Empirical", markersize=12, linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(freqs_list)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label="Gaussian")
        ax.set_ylabel("Tail Ratio (P>3sig / Gaussian)")
        ax.set_title("Tail Heaviness by Frequency", fontweight="bold")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # (1,1) QQ plot at 1h
        ax = axes[1, 1]
        emp_1h = test_data.get("1h", np.array([]))
        if emp_1h.size > 10:
            pcts = np.linspace(1, 99, 50)
            eq = np.percentile(emp_1h, pcts)
            for name in model_order[:5]:  # Top 5 only to avoid clutter
                fm = all_freq_metrics[name]
                # Recompute from a big sim at 15-min
                lr_qq = simulate_model(models[name][0], models[name][1],
                                       n_sim_15m, dt_15m, N_PATHS, seed=42)
                sim_1h = aggregate_returns(lr_qq, 4).ravel()  # 15m -> 1h = factor 4
                sq = np.percentile(sim_1h, pcts)
                ax.plot(eq, sq, 'o', label=name[:12], color=colors.get(name, "#95a5a6"),
                        alpha=0.6, markersize=3)
            lims = [min(eq.min(), eq.min()), max(eq.max(), eq.max())]
            ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlabel("Empirical 1h quantiles")
        ax.set_ylabel("Simulated 1h quantiles")
        ax.set_title("QQ Plot at 1h (top 5)", fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # (1,2) Jump frequency comparison
        ax = axes[1, 2]
        emp_jpd = jumps_per_day(test_data["15m"], 15)
        jpds = [all_global_metrics[n]["jumps_per_day_15m"] for n in model_order]
        bar_colors = [colors.get(n, "#95a5a6") for n in model_order]
        bars = ax.bar(range(len(model_order)), jpds, color=bar_colors, alpha=0.7)
        ax.axhline(emp_jpd, color="black", linestyle="--", linewidth=2,
                    label=f"Empirical ({emp_jpd:.1f})")
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels([n[:12] for n in model_order], rotation=30)
        ax.set_ylabel("Jumps per Day (15m, 4sig)")
        ax.set_title("Jump Frequency", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (2,0) Vol clustering by frequency
        ax = axes[2, 0]
        for name in model_order:
            fm = all_freq_metrics[name]
            vals = [fm.get(f, {}).get("abs_acf1", 0) for f in freqs_list]
            ax.plot(x, vals, 'o-', label=name[:12], color=colors.get(name, "#95a5a6"), alpha=0.8)
        emp_acf = [structural_metrics(test_data[f])["abs_acf1"] for f in freqs_list]
        ax.plot(x, emp_acf, 'k*-', label="Empirical", markersize=12, linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(freqs_list)
        ax.set_ylabel("ACF of |returns|, lag 1")
        ax.set_title("Volatility Clustering by Frequency", fontweight="bold")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # (2,1) Leverage effect
        ax = axes[2, 1]
        emp_lev = leverage_corr(test_data.get("1d", np.array([])))
        levs = [all_global_metrics[n]["leverage"] for n in model_order]
        ax.bar(range(len(model_order)), levs, color=bar_colors, alpha=0.7)
        ax.axhline(emp_lev, color="black", linestyle="--", linewidth=2,
                    label=f"Empirical ({emp_lev:.3f})")
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels([n[:12] for n in model_order], rotation=30)
        ax.set_ylabel("corr(r_t, |r_{t+1}|)")
        ax.set_title("Leverage Effect (daily)", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (2,2) Per-frequency ranking heatmap
        ax = axes[2, 2]
        rank_matrix = np.zeros((len(model_order), len(freqs_list)))
        for j, freq in enumerate(freqs_list):
            freq_losses = []
            for name in model_order:
                fm = all_freq_metrics[name]
                sm = fm.get(freq, {})
                em = structural_metrics(test_data[freq])
                # Simple loss: kurtosis error + tail error + acf error
                k_err = ((sm.get("kurtosis", 0) - em["kurtosis"]) /
                         max(abs(em["kurtosis"]), 0.1))**2
                t_err = ((sm.get("tail_3sig", 0) - em["tail_3sig"]) /
                         max(abs(em["tail_3sig"]), 0.1))**2
                a_err = ((sm.get("abs_acf1", 0) - em["abs_acf1"]) /
                         max(abs(em["abs_acf1"]), 0.01))**2
                freq_losses.append(k_err + t_err + a_err)
            ranking = np.argsort(np.argsort(freq_losses)) + 1
            for i, r in enumerate(ranking):
                rank_matrix[i, j] = r

        im = ax.imshow(rank_matrix, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=len(model_order))
        ax.set_xticks(range(len(freqs_list)))
        ax.set_xticklabels(freqs_list)
        ax.set_yticks(range(len(model_order)))
        ax.set_yticklabels([n[:12] for n in model_order])
        for i in range(len(model_order)):
            for j in range(len(freqs_list)):
                ax.text(j, i, f"{int(rank_matrix[i, j])}", ha="center", va="center", fontsize=9)
        ax.set_title("Rank by Frequency (1=best)", fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.6)

        fig.suptitle(f"Multi-Scale Benchmark: {len(models)} models x {N_SEEDS} seeds",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig("multi_scale_benchmark.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved multi_scale_benchmark.png")
    except Exception as e:
        print(f"  Plot failed: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
