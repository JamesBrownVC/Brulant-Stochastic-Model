"""
Stochastic Mean-Reversion OU Model
====================================
Take the simplest mean-reverting vol model (OU-driven GBM) and replace
the fixed vol target with a stochastic OU target — the key innovation
from Brulant v1.2, isolated from buffers and jumps.

SDE System:
  dS/S = sigma_t * dW1                          (price)
  dsigma = alpha * (sigma_bar_t - sigma) * dt   (vol mean-reverts to wandering target)
  dsigma_bar = alpha_s * (sigma0 - sigma_bar) * dt + xi_s * dW2  (target is OU)

4 free parameters: sigma0, alpha, alpha_s, xi_s
(plus sigma_bar0 = sigma0 as initial condition)
"""
from __future__ import annotations
import json, time, datetime, sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import differential_evolution

sys.stdout.reconfigure(line_buffering=True)

from fit_sandpile import (
    fetch_binance_log_returns, interval_to_dt_years,
    moment_vector, recent_exponential_weights, _to_jsonable,
)
from benchmark_v12 import BRULANT_V12, diebold_mariano_test, bootstrap_ci


# =========================================================================
#  SIMULATION
# =========================================================================
def simulate_stoch_ou(
    n_steps: int,
    dt: float,
    num_paths: int,
    *,
    S0: float = 1.0,
    sigma0: float = 0.50,       # initial vol & long-run target
    alpha: float = 10.0,        # vol mean-reversion speed
    alpha_s: float = 5.0,       # target mean-reversion speed
    xi_s: float = 0.5,          # target noise
    eps: float = 1e-3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate GBM with stochastic mean-reverting volatility."""
    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)

    S = np.full(num_paths, S0, dtype=np.float64)
    sig = np.full(num_paths, sigma0, dtype=np.float64)
    sig_target = np.full(num_paths, sigma0, dtype=np.float64)
    lr = np.zeros((num_paths, n_steps), dtype=np.float64)

    for t in range(n_steps):
        cs = np.maximum(sig, eps)

        # Price dynamics (pure diffusion, no drift, no buffer)
        dW1 = sqrt_dt * rng.standard_normal(num_paths)
        ret = cs * dW1
        ret = np.clip(ret, -0.50, 0.50)

        S_prev = S
        S = S * (1.0 + ret)
        S = np.maximum(S, 1e-12)
        lr[:, t] = np.log(S / S_prev)

        # Vol dynamics: mean-revert to stochastic target
        sig_new = cs + alpha * (sig_target - cs) * dt
        sig = np.clip(sig_new, eps, 5.0)

        # Target dynamics: OU process
        dW2 = sqrt_dt * rng.standard_normal(num_paths)
        sig_target = sig_target + alpha_s * (sigma0 - sig_target) * dt + xi_s * dW2
        sig_target = np.maximum(sig_target, eps)

    return lr, S


# =========================================================================
#  CALIBRATION
# =========================================================================
def calibrate_stoch_ou(
    train_r: np.ndarray,
    dt: float,
    *,
    num_paths: int = 1000,
    maxiter: int = 14,
    seed: int = 42,
) -> Dict[str, float]:
    """Calibrate stochastic OU via DE moment matching."""
    w = recent_exponential_weights(train_r.size, 400.0)
    target = moment_vector(train_r, w=w, acf_recent_bars=300)
    scales = np.maximum(np.abs(target), np.array([1e-12, 1e-12, 0.5, 1.0, 0.05, 0.1]))
    scales = np.maximum(scales, 1e-9)

    names = ["sigma0", "alpha", "alpha_s", "xi_s"]
    bounds = [
        (0.10, 1.00),   # sigma0
        (1.0, 30.0),    # alpha (vol MR speed)
        (0.1, 20.0),    # alpha_s (target MR speed)
        (0.01, 3.0),    # xi_s (target noise)
    ]

    rng = np.random.default_rng(seed)

    def obj(theta):
        p = {k: float(v) for k, v in zip(names, theta)}
        s = int(rng.integers(0, 2**31 - 1))
        sim_lr, _ = simulate_stoch_ou(train_r.size, dt, num_paths, seed=s, S0=1.0, **p)
        pooled = sim_lr.ravel()
        if pooled.size > 50000:
            pooled = np.random.default_rng(s).choice(pooled, 50000, replace=False)
        sim = moment_vector(pooled, w=None, acf_recent_bars=300)
        z = (sim - target) / scales
        return float(np.sum(z * z))

    # 4 params x popsize=15 x maxiter=14 = 840 evals (generous for 4 params)
    res = differential_evolution(obj, bounds, maxiter=maxiter, seed=seed, workers=1,
                                 polish=True, popsize=15, tol=1e-4, atol=1e-5)

    fit = {k: float(v) for k, v in zip(names, res.x)}
    fit["_train_loss"] = float(res.fun)
    return fit


# =========================================================================
#  BENCHMARK
# =========================================================================
def main():
    from experiment_v12 import simulate_v12
    from benchmark_comparison import (
        simulate_gbm, calibrate_gbm,
        simulate_heston, calibrate_heston,
        simulate_merton, calibrate_merton,
        simulate_sabr, calibrate_sabr,
    )

    N_SEEDS = 200

    print("=" * 70)
    print("  STOCHASTIC OU vs ALL (200 seeds)")
    print("  Isolating the stochastic mean-reversion idea from Brulant")
    print("=" * 70)

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

    # Calibrate stoch OU
    print("\nCalibrating Stochastic OU (4 params)...")
    t0 = time.perf_counter()
    sou_params = calibrate_stoch_ou(train_r, dt, num_paths=1000, maxiter=14, seed=42)
    sou_time = time.perf_counter() - t0
    print(f"  StochOU calibrated in {sou_time:.0f}s (loss={sou_params['_train_loss']:.3f})")
    print(f"  Params: sigma0={sou_params['sigma0']:.4f} alpha={sou_params['alpha']:.4f} "
          f"alpha_s={sou_params['alpha_s']:.4f} xi_s={sou_params['xi_s']:.4f}")

    # Calibrate benchmarks
    print("\nCalibrating benchmarks...")
    t0 = time.perf_counter()
    gbm_p = calibrate_gbm(train_r, dt)
    heston_p = calibrate_heston(train_r, dt)
    merton_p = calibrate_merton(train_r, dt)
    sabr_p = calibrate_sabr(train_r, dt, S0=1.0)
    print(f"  Benchmarks calibrated in {time.perf_counter()-t0:.0f}s")

    # Clean params for simulation
    sou_sim = {k: v for k, v in sou_params.items() if not k.startswith('_')}

    model_configs = {
        "Stoch OU": ("sou", sou_sim),
        "Brulant v1.2": ("v12", BRULANT_V12),
        "GBM": ("gbm", gbm_p),
        "Heston": ("heston", heston_p),
        "Merton": ("merton", merton_p),
        "SABR": ("sabr", sabr_p),
    }

    print(f"\nRunning {N_SEEDS} seeds per model...")
    model_losses = {}
    model_stats = {}

    for name, (tag, params) in model_configs.items():
        t0 = time.perf_counter()
        losses = []
        for i in range(N_SEEDS):
            seed = 42 + i * 77
            if tag == "sou":
                lr, _ = simulate_stoch_ou(test_r.size, dt, 1000, seed=seed, S0=1.0, **params)
            elif tag == "v12":
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

        rng_bs = np.random.default_rng(42)
        boot_medians = np.array([np.median(rng_bs.choice(losses, size=len(losses), replace=True))
                                 for _ in range(10000)])

        # Distributional stats
        if tag == "sou":
            lr, _ = simulate_stoch_ou(test_r.size, dt, 5000, seed=42, S0=1.0, **params)
        elif tag == "v12":
            lr, _ = simulate_v12(test_r.size, dt, 5000, seed=42, S0=1.0, **params)
        elif tag == "gbm":
            lr, _ = simulate_gbm(test_r.size, dt, 5000, 1.0, params["sigma"], seed=42)
        elif tag == "heston":
            lr, _ = simulate_heston(test_r.size, dt, 5000, 1.0, **params, seed=42)
        elif tag == "merton":
            lr, _ = simulate_merton(test_r.size, dt, 5000, 1.0, **params, seed=seed)
        elif tag == "sabr":
            lr, _ = simulate_sabr(test_r.size, dt, 5000, 1.0, **params, seed=42)
        sp = lr.ravel()[:50000]
        sm = moment_vector(sp, w=None, acf_recent_bars=300)

        model_stats[name] = {
            "median_loss": float(np.median(losses)),
            "mean_loss": float(np.mean(losses)),
            "std_loss": float(np.std(losses)),
            "se_median": float(np.std(boot_medians)),
            "ci_median_lo": float(np.percentile(boot_medians, 2.5)),
            "ci_median_hi": float(np.percentile(boot_medians, 97.5)),
            "std_ratio": float(np.std(sp) / emp_std),
            "kurt": float(sm[3]),
            "skew": float(sm[2]),
            "acf_r2": float(sm[5]),
        }
        elapsed = time.perf_counter() - t0
        print(f"  {name:<14s}: median={np.median(losses):.2f} "
              f"CI=[{model_stats[name]['ci_median_lo']:.2f}, {model_stats[name]['ci_median_hi']:.2f}] "
              f"kurt={sm[3]:.2f} acf={sm[5]:.3f} ({elapsed:.0f}s)")

    # Paired tests: StochOU vs each
    print(f"\n{'='*70}")
    print(f"  PAIRED TESTS: Stochastic OU vs each")
    print(f"{'='*70}")
    sou_losses = model_losses["Stoch OU"]
    print(f"  {'Model':<14s} {'MedDiff':>8s} {'95% CI':>22s} {'Wilcoxon p':>11s} {'Sig':>5s}")
    print(f"  {'-'*14} {'-'*8} {'-'*22} {'-'*11} {'-'*5}")

    for name in model_configs:
        if name == "Stoch OU":
            continue
        diffs = model_losses[name] - sou_losses
        med_diff = float(np.median(diffs))
        ci_lo, ci_hi = bootstrap_ci(diffs, n_bootstrap=10000, ci=0.95, seed=42)
        try:
            _, w_pval = sp_stats.wilcoxon(diffs, alternative='two-sided')
        except ValueError:
            w_pval = float('nan')
        sig = "***" if w_pval < 0.001 else ("**" if w_pval < 0.01 else ("*" if w_pval < 0.05 else "ns"))
        print(f"  {name:<14s} {med_diff:>+8.2f} [{ci_lo:>+9.2f}, {ci_hi:>+9.2f}] {w_pval:>11.4g} {sig:>5s}")

    # Final ranking
    print(f"\n{'='*70}")
    print(f"  FINAL RANKING")
    print(f"{'='*70}")
    ranked = sorted(model_stats.items(), key=lambda x: x[1]["median_loss"])
    param_counts = {"Stoch OU": 4, "Brulant v1.2": 11, "GBM": 1,
                    "Heston": 4, "Merton": 3, "SABR": 4}
    print(f"  {'Rank':>4s} {'Model':<14s} {'Median':>8s} {'Params':>6s} {'Kurt':>6s} {'ACF':>6s}")
    print(f"  {'-'*4} {'-'*14} {'-'*8} {'-'*6} {'-'*6} {'-'*6}")
    for rank, (name, s) in enumerate(ranked, 1):
        marker = " <--" if "Stoch" in name else ""
        print(f"  {rank:>4d} {name:<14s} {s['median_loss']:>8.2f} "
              f"{param_counts.get(name, '?'):>6} {s['kurt']:>6.2f} {s['acf_r2']:>6.3f}{marker}")

    print(f"\n  Empirical: kurt={emp[3]:.2f} acf_r2={emp[5]:.3f}")

    # Save
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "spot": S0,
        "n_seeds": N_SEEDS,
        "empirical_moments": emp.tolist(),
        "stoch_ou_params": sou_sim,
        "stoch_ou_train_loss": sou_params.get("_train_loss"),
        "model_stats": model_stats,
    }
    Path("evidence_stoch_ou.json").write_text(json.dumps(_to_jsonable(result), indent=2), encoding="utf-8")
    print(f"\nSaved evidence_stoch_ou.json")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = {"Stoch OU": "#e67e22", "Brulant v1.2": "#e74c3c", "GBM": "#3498db",
                  "Heston": "#2ecc71", "Merton": "#9b59b6", "SABR": "#f39c12"}
        names_sorted = [n for n, _ in ranked]

        # 1. Loss distributions
        ax = axes[0]
        bp = ax.boxplot([model_losses[n] for n in names_sorted],
                        tick_labels=[n[:8] for n in names_sorted],
                        patch_artist=True, showfliers=False)
        for patch, name in zip(bp["boxes"], names_sorted):
            patch.set_facecolor(colors.get(name, "#95a5a6"))
            patch.set_alpha(0.7)
        ax.set_ylabel("OOS Moment-Matching Loss")
        ax.set_title(f"Loss Distribution ({N_SEEDS} seeds)", fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 2. Paired differences (StochOU vs each)
        ax = axes[1]
        others = [n for n in names_sorted if n != "Stoch OU"]
        diffs_list = [model_losses[n] - sou_losses for n in others]
        bp2 = ax.boxplot(diffs_list,
                         tick_labels=[n[:8] for n in others],
                         patch_artist=True, showfliers=False)
        for patch, name in zip(bp2["boxes"], others):
            patch.set_facecolor(colors.get(name, "#95a5a6"))
            patch.set_alpha(0.7)
        ax.axhline(0, color="black", linestyle="--", linewidth=2, alpha=0.5)
        ax.set_ylabel("Loss(Model) - Loss(StochOU)")
        ax.set_title("StochOU vs All (>0 = StochOU wins)", fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 3. Kurtosis
        ax = axes[2]
        kurt_vals = [model_stats[n]["kurt"] for n in names_sorted]
        bar_colors = [colors.get(n, "#95a5a6") for n in names_sorted]
        ax.bar(range(len(names_sorted)), kurt_vals, color=bar_colors, alpha=0.7)
        ax.axhline(emp[3], color="red", linestyle="--", linewidth=2, label=f"Empirical ({emp[3]:.1f})")
        ax.set_xticks(range(len(names_sorted)))
        ax.set_xticklabels([n[:8] for n in names_sorted], rotation=15)
        ax.set_ylabel("Excess Kurtosis")
        ax.set_title("Kurtosis Match", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"Stochastic OU: Isolated Stochastic Mean-Reversion ({N_SEEDS} seeds)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig("evidence_stoch_ou.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("Saved evidence_stoch_ou.png")
    except Exception as e:
        print(f"  Plot failed: {e}")


if __name__ == "__main__":
    main()
