"""
Brulant Model v2 Retraining
============================
Changes from v1:
  1. xi (vol-of-vol): dσ += ξ·σ·dW^σ — generates fat tails from diffusion
  2. delta (structural jump bias): jm = -φ·B + δ — crypto short-gamma bias
  3. mu0=0 fixed (no drift in 1-min returns)
  4. nu=1.0 fixed (not separately identifiable from rho)
  5. Tighter sigma0 bounds [0.15, 0.55] — previous was too wide
  6. lambda0 floor raised [0.5, 8.0] — more jump activity
  7. Winsorized training data at 5-sigma (MAD-based)
"""

from __future__ import annotations

import json
import time
import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import differential_evolution

from backtest_buffer_model import simulate_buffer_paths, MOMENT_NAMES
from fit_sandpile import (
    fetch_binance_log_returns,
    interval_to_dt_years,
    moment_vector,
    recent_exponential_weights,
    _to_jsonable,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False


# Known-good v1 params as baseline comparison
V1_PARAMS = {
    "mu0": 0.0, "sigma0": 0.596377, "rho": 1.78402, "nu": 1.54849,
    "alpha": 9.90562, "beta": 0.128777, "lambda0": 1.18401,
    "gamma": 20.0, "eta": 1.0, "kappa": 15.0, "theta_p": 1.5,
    "phi": 0.560709, "sigma_Y": 0.0568364, "eps": 0.001,
    "xi": 0.0, "delta": 0.0,
}


def winsorize(r: np.ndarray, n_sigma: float = 5.0) -> np.ndarray:
    mu = np.median(r)
    mad = np.percentile(np.abs(r - mu), 75) * 1.4826
    return np.clip(r, mu - n_sigma * mad, mu + n_sigma * mad)


def split_train_test(r: np.ndarray, frac: float = 0.75):
    n = r.size
    n_train = max(200, min(int(n * frac), n - 80))
    return r[:n_train], r[n_train:]


def evaluate_oos(params: Dict[str, float], test_r: np.ndarray, dt: float,
                 num_paths: int, seed: int, acf_recent: int) -> Dict[str, Any]:
    sim_lr, _ = simulate_buffer_paths(test_r.size, dt, num_paths, seed=seed, S0=1.0, **params)
    emp = moment_vector(test_r, w=None, acf_recent_bars=acf_recent)
    sim = moment_vector(sim_lr.ravel(), w=None, acf_recent_bars=acf_recent)
    scales = np.maximum(np.abs(emp), np.array([1e-12, 1e-12, 0.5, 1.0, 0.05, 0.1]))
    scales = np.maximum(scales, 1e-9)
    loss = float(np.sum(((sim - emp) / scales) ** 2))
    return {"test_emp": emp, "test_sim": sim, "test_loss": loss}


def fit_v2(
    train_r: np.ndarray,
    dt: float,
    *,
    half_life: float = 250.0,
    acf_recent: int = 300,
    num_paths: int = 800,
    maxiter: int = 14,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    V2 calibration: 8 free params + xi + delta.
    Fixed: mu0=0, nu=1.0, kappa=15, theta_p=1.5, gamma=20, eta=1, eps=0.001
    """
    w = recent_exponential_weights(train_r.size, half_life)
    target = moment_vector(train_r, w=w, acf_recent_bars=acf_recent)

    # Free parameters: sigma0, rho, alpha, beta, lambda0, phi, sigma_Y, xi, delta
    names = ["sigma0", "rho", "alpha", "beta", "lambda0", "phi", "sigma_Y", "xi", "delta"]
    bounds = [
        (0.15, 0.55),    # sigma0: tighter — v1 was 0.596 which overshot diffusion
        (0.5, 3.5),      # rho: buffer coupling
        (2.0, 18.0),     # alpha: vol mean-reversion speed
        (1e-4, 0.15),    # beta: jump vol injection (tighter ceiling)
        (0.5, 8.0),      # lambda0: higher floor — more jump activity
        (0.1, 1.5),      # phi: trapdoor coupling
        (0.02, 0.20),    # sigma_Y: jump size dispersion
        (0.0, 2.5),      # xi: vol-of-vol (NEW)
        (-0.03, 0.01),   # delta: structural jump bias (NEW, slightly negative = short-gamma)
    ]

    fixed = {
        "mu0": 0.0, "nu": 1.0, "kappa": 15.0, "theta_p": 1.5,
        "gamma": 20.0, "eta": 1.0, "eps": 1e-3,
    }

    def unpack(theta):
        p = {k: float(v) for k, v in zip(names, theta)}
        p.update(fixed)
        return p

    scales = np.maximum(np.abs(target), np.array([1e-12, 1e-12, 0.5, 1.0, 0.05, 0.1]))
    scales = np.maximum(scales, 1e-9)
    rng = np.random.default_rng(seed)

    def obj(theta):
        p = unpack(theta)
        s = int(rng.integers(0, 2**31 - 1))
        sim_lr, _ = simulate_buffer_paths(
            train_r.size, dt, num_paths, seed=s, S0=1.0, **p,
        )
        pooled = sim_lr.ravel()
        if pooled.size > 50_000:
            pooled = np.random.default_rng(s).choice(pooled, 50_000, replace=False)
        sim = moment_vector(pooled, w=None, acf_recent_bars=acf_recent)
        z = (sim - target) / scales

        # L2 penalties — economically motivated
        penalty = 0.0
        penalty += 8.0 * (max(0.0, p["phi"] - 1.0)**2)       # trapdoor too deterministic
        penalty += 8.0 * (max(0.0, p["sigma_Y"] - 0.12)**2)   # jumps too violent
        penalty += 1.0 * (max(0.0, p["lambda0"] - 5.0)**2)    # too many jumps
        penalty += 8.0 * (max(0.0, p["beta"] - 0.08)**2)      # vol explosion risk
        penalty += 2.0 * (max(0.0, p["rho"] - 2.5)**2)        # buffer over-dominant
        penalty += 3.0 * (max(0.0, p["xi"] - 1.5)**2)         # vol-of-vol explosion
        penalty += 5.0 * (min(0.0, p["delta"] + 0.02)**2)     # bias too negative

        return float(np.sum(z * z)) + penalty

    t0 = time.perf_counter()
    res = differential_evolution(
        obj, bounds, maxiter=maxiter, seed=seed, workers=1,
        polish=False, popsize=10, tol=1e-3, atol=1e-4,
    )
    elapsed = time.perf_counter() - t0

    p = unpack(res.x)
    sim_lr, _ = simulate_buffer_paths(train_r.size, dt, num_paths, seed=seed + 111, S0=1.0, **p)
    sim_m = moment_vector(sim_lr.ravel(), w=None, acf_recent_bars=acf_recent)

    return {
        **p,
        "loss": float(res.fun),
        "target_moments": target,
        "train_sim_moments": sim_m,
        "elapsed_s": elapsed,
    }


def price_digital(S0, strikes, hours, params, num_paths=200000, seed=42):
    T = hours / (24.0 * 365.0)
    n_steps = max(1, int(hours * 60))
    dt = T / n_steps if n_steps > 0 else T
    _, S_T = simulate_buffer_paths(n_steps, dt, num_paths, seed=seed, S0=S0, **params)
    strikes = np.asarray(strikes, dtype=np.float64)
    prices = np.array([np.mean(S_T >= K) for K in strikes])
    stderrs = np.array([np.std((S_T >= K).astype(float)) / np.sqrt(num_paths) for K in strikes])
    return prices, stderrs


def main():
    print("=" * 70)
    print("  BRULANT MODEL v2: RETRAINING WITH xi AND delta")
    print("  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # Fetch data
    print("\nFetching 5000 1-min candles...")
    t0 = time.perf_counter()
    returns_raw = fetch_binance_log_returns("BTCUSDT", "1m", 5000)
    dt = interval_to_dt_years("1m")
    print(f"  Got {returns_raw.size} returns in {time.perf_counter()-t0:.1f}s")

    # Spot
    try:
        import requests
        S0 = float(requests.get("https://api.binance.com/api/v3/ticker/price",
                                params={"symbol": "BTCUSDT"}, timeout=10).json()["price"])
    except Exception:
        S0 = 70000.0
    print(f"  Spot: ${S0:,.2f}")

    # Winsorize and split
    returns = winsorize(returns_raw, 5.0)
    n_clipped = int(np.sum(returns_raw != returns))
    train_r, test_r = split_train_test(returns, 0.75)
    print(f"  Winsorized {n_clipped} outliers")
    print(f"  Train: {train_r.size} | Test: {test_r.size}")

    raw_kurt = moment_vector(returns_raw, w=None, acf_recent_bars=300)[3]
    clean_kurt = moment_vector(returns, w=None, acf_recent_bars=300)[3]
    print(f"  Raw kurtosis: {raw_kurt:.1f} -> Winsorized: {clean_kurt:.1f}")

    # ========== V1 BASELINE ==========
    print("\n" + "=" * 70)
    print("  V1 BASELINE (known-good params, xi=0, delta=0)")
    print("=" * 70)
    ev_v1 = evaluate_oos(V1_PARAMS, test_r, dt, 1000, 42, 300)
    print(f"  OOS loss: {ev_v1['test_loss']:.4f}")
    print("  Moments (emp | sim):")
    for n, a, b in zip(MOMENT_NAMES, ev_v1["test_emp"], ev_v1["test_sim"]):
        print(f"    {n:>8s}: {a:>12.6g} | {b:>12.6g}")

    # Distributional check
    sim_lr_v1, _ = simulate_buffer_paths(test_r.size, dt, 5000, seed=42, S0=1.0, **V1_PARAMS)
    sim_v1 = sim_lr_v1.ravel()
    emp_std = np.std(test_r)
    sim_std_v1 = np.std(sim_v1[:50000])
    emp_3sig = np.mean(np.abs(test_r) > 3 * emp_std)
    sim_3sig_v1 = np.mean(np.abs(sim_v1[:50000]) > 3 * sim_std_v1)
    print(f"\n  Std ratio: {sim_std_v1/emp_std:.3f}")
    print(f"  3-sigma tail: emp={emp_3sig:.6f} sim={sim_3sig_v1:.6f} (ratio={sim_3sig_v1/max(emp_3sig,1e-12):.2f})")

    # ========== V2 CALIBRATION ==========
    print("\n" + "=" * 70)
    print("  V2 CALIBRATION (xi, delta, tighter bounds)")
    print("=" * 70)

    fit = fit_v2(train_r, dt, half_life=250.0, acf_recent=300,
                 num_paths=800, maxiter=14, seed=42)

    v2_params = {k: fit[k] for k in [
        "mu0", "sigma0", "rho", "nu", "kappa", "theta_p",
        "alpha", "beta", "lambda0", "gamma", "eta",
        "phi", "sigma_Y", "eps", "xi", "delta"
    ]}

    print(f"\n  Calibration done in {fit['elapsed_s']:.1f}s")
    print(f"  Train loss: {fit['loss']:.4f}")
    print("\n  Fitted parameters:")
    for k in ["sigma0", "rho", "alpha", "beta", "lambda0", "phi", "sigma_Y", "xi", "delta"]:
        old = V1_PARAMS.get(k, 0.0)
        new = v2_params[k]
        arrow = "  NEW" if k in ("xi", "delta") else f"  (was {old:.4f})"
        print(f"    {k:>10s} = {new:.6f}{arrow}")

    print("\n  Train moments (target | sim):")
    for n, a, b in zip(MOMENT_NAMES, fit["target_moments"], fit["train_sim_moments"]):
        print(f"    {n:>8s}: {a:>12.6g} | {b:>12.6g}")

    # ========== V2 OOS TEST ==========
    print("\n" + "=" * 70)
    print("  V2 OUT-OF-SAMPLE TEST")
    print("=" * 70)

    ev_v2 = evaluate_oos(v2_params, test_r, dt, 1000, 42, 300)
    print(f"  OOS loss: {ev_v2['test_loss']:.4f}")
    print("  Moments (emp | sim):")
    for n, a, b in zip(MOMENT_NAMES, ev_v2["test_emp"], ev_v2["test_sim"]):
        print(f"    {n:>8s}: {a:>12.6g} | {b:>12.6g}")

    # Distributional check
    sim_lr_v2, _ = simulate_buffer_paths(test_r.size, dt, 5000, seed=42, S0=1.0, **v2_params)
    sim_v2 = sim_lr_v2.ravel()
    sim_std_v2 = np.std(sim_v2[:50000])
    sim_3sig_v2 = np.mean(np.abs(sim_v2[:50000]) > 3 * sim_std_v2)
    print(f"\n  Std ratio: {sim_std_v2/emp_std:.3f}")
    print(f"  3-sigma tail: emp={emp_3sig:.6f} sim={sim_3sig_v2:.6f} (ratio={sim_3sig_v2/max(emp_3sig,1e-12):.2f})")

    # Multi-seed robustness
    print("\n  Multi-seed robustness (5 seeds):")
    losses_v2 = []
    for i in range(5):
        ev_i = evaluate_oos(v2_params, test_r, dt, 1000, 2000 + i * 77, 300)
        losses_v2.append(ev_i["test_loss"])
        print(f"    Seed {i}: {ev_i['test_loss']:.4f}")
    losses_v2 = np.array(losses_v2)
    print(f"    Mean: {np.mean(losses_v2):.4f} +/- {np.std(losses_v2):.4f}")

    # ========== HEAD-TO-HEAD ==========
    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD: V1 vs V2")
    print("=" * 70)
    print(f"  {'Metric':<30s} {'V1':>12s} {'V2':>12s} {'Winner':>8s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*8}")

    metrics = [
        ("OOS loss", ev_v1["test_loss"], ev_v2["test_loss"], "lower"),
        ("Std ratio (target=1.0)", sim_std_v1/emp_std, sim_std_v2/emp_std, "closer1"),
        ("3-sig tail ratio (target=1.0)", sim_3sig_v1/max(emp_3sig,1e-12), sim_3sig_v2/max(emp_3sig,1e-12), "closer1"),
        ("Multi-seed loss mean", float('nan'), float(np.mean(losses_v2)), "lower"),
    ]
    for name, v1, v2, mode in metrics:
        if mode == "lower":
            winner = "V1" if v1 < v2 else "V2"
        elif mode == "closer1":
            winner = "V1" if abs(v1-1) < abs(v2-1) else "V2"
        else:
            winner = "?"
        if np.isnan(v1):
            print(f"  {name:<30s} {'N/A':>12s} {v2:>12.4f} {'V2':>8s}")
        else:
            print(f"  {name:<30s} {v1:>12.4f} {v2:>12.4f} {winner:>8s}")

    # ========== DIGITAL OPTION PRICING WITH BEST MODEL ==========
    best_params = v2_params if ev_v2["test_loss"] < ev_v1["test_loss"] else V1_PARAMS
    best_label = "V2" if ev_v2["test_loss"] < ev_v1["test_loss"] else "V1"
    print(f"\n  Using {best_label} params for pricing")

    print("\n" + "=" * 70)
    print(f"  DIGITAL OPTION PRICING ({best_label} MODEL)")
    print("=" * 70)

    strikes = np.arange(60000, 82000, 2000, dtype=np.float64)
    cet = datetime.timezone(datetime.timedelta(hours=1))
    now_cet = datetime.datetime.now(datetime.timezone.utc).astimezone(cet)

    print(f"  Spot: ${S0:,.2f}")
    print(f"  Time (CET): {now_cet.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Paths: 200,000")

    pricing_results = {}
    for k in range(5):
        target_date = now_cet.date() + datetime.timedelta(days=k)
        target_dt = datetime.datetime(target_date.year, target_date.month, target_date.day,
                                      17, 0, 0, tzinfo=cet)
        hours = max((target_dt - now_cet).total_seconds() / 3600.0, 0.01)

        print(f"\n  +{k}d (17:00 CET {target_date}, {hours:.1f}h)")
        prices, stderrs = price_digital(S0, strikes, hours, best_params,
                                         num_paths=200000, seed=42 + k * 1000)

        print(f"    {'Strike':>10s} {'Price':>10s} {'95% CI':>24s} {'SE':>10s}")
        print(f"    {'-'*10} {'-'*10} {'-'*24} {'-'*10}")
        for j, K in enumerate(strikes):
            ci_lo = max(0, prices[j] - 1.96 * stderrs[j])
            ci_hi = min(1, prices[j] + 1.96 * stderrs[j])
            tag = "ITM" if K < S0 - 1000 else ("ATM" if abs(K - S0) < 1000 else "OTM")
            print(f"    ${int(K):>8,} {prices[j]:>10.6f} [{ci_lo:.6f}, {ci_hi:.6f}] {stderrs[j]:>10.6f}  {tag}")

        pricing_results[f"+{k}d"] = {
            "hours": hours, "strikes": strikes.tolist(),
            "prices": prices.tolist(), "stderrs": stderrs.tolist(),
        }

    # Save everything
    output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "spot": S0,
        "v1_oos_loss": ev_v1["test_loss"],
        "v2_oos_loss": ev_v2["test_loss"],
        "v2_params": {k: float(v) for k, v in v2_params.items()},
        "v1_params": {k: float(v) for k, v in V1_PARAMS.items()},
        "best_model": best_label,
        "pricing": _to_jsonable(pricing_results),
        "v2_multi_seed": {"mean": float(np.mean(losses_v2)), "std": float(np.std(losses_v2))},
    }
    Path("retrain_v2_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to retrain_v2_results.json")

    # Plot comparison
    if _HAS_PLT:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Return distributions
        ax = axes[0]
        ax.hist(test_r, bins=80, density=True, alpha=0.5, label="Empirical", color="#2c3e50")
        rng = np.random.default_rng(42)
        ax.hist(rng.choice(sim_v1, min(50000, sim_v1.size), replace=False),
                bins=80, density=True, alpha=0.4, label="V1 sim", color="#3498db")
        ax.hist(rng.choice(sim_v2, min(50000, sim_v2.size), replace=False),
                bins=80, density=True, alpha=0.4, label="V2 sim", color="#e74c3c")
        ax.set_title("Return Distribution: Empirical vs V1 vs V2", fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xlabel("Log return")

        # Moment comparison
        ax = axes[1]
        x = np.arange(len(MOMENT_NAMES))
        w = 0.25
        ax.bar(x - w, ev_v1["test_emp"], width=w, label="Empirical", color="#2c3e50")
        ax.bar(x, ev_v1["test_sim"], width=w, label="V1 sim", color="#3498db")
        ax.bar(x + w, ev_v2["test_sim"], width=w, label="V2 sim", color="#e74c3c")
        ax.set_xticks(x)
        ax.set_xticklabels(MOMENT_NAMES, rotation=30, ha="right", fontsize=8)
        ax.set_title("OOS Moment Match", fontweight="bold")
        ax.legend(fontsize=8)

        # Digital option curves
        ax = axes[2]
        for label, data in pricing_results.items():
            ax.plot(np.array(data["strikes"])/1000, data["prices"], "o-", label=label, markersize=3)
        ax.axvline(S0/1000, color="black", linestyle="--", alpha=0.5, label="Spot")
        ax.set_xlabel("Strike ($k)")
        ax.set_ylabel("Digital Price")
        ax.set_title(f"Digital Options ({best_label} Model)", fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"Brulant v2 Retraining — V1 loss={ev_v1['test_loss']:.2f}, V2 loss={ev_v2['test_loss']:.2f}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig("retrain_v2_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print("Saved retrain_v2_comparison.png")


if __name__ == "__main__":
    main()
