"""
Benchmark v1.1 (with L2 penalties) against v1.2 and standard models.
200 seeds, paired tests, same methodology as evidence_benchmark.
"""
from __future__ import annotations
import json, time, datetime, sys
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

sys.stdout.reconfigure(line_buffering=True)

from experiment_v12 import simulate_v12
from backtest_buffer_model import simulate_buffer_paths, fit_buffer_model
from benchmark_comparison import (
    simulate_gbm, calibrate_gbm,
    simulate_heston, calibrate_heston,
    simulate_merton, calibrate_merton,
    simulate_sabr, calibrate_sabr,
)
from fit_sandpile import fetch_binance_log_returns, interval_to_dt_years, moment_vector
from benchmark_v12 import BRULANT_V12, BRULANT_V11, diebold_mariano_test, bootstrap_ci


N_SEEDS = 200


def main():
    print("=" * 70)
    print("  BENCHMARK v1.1 (L2 penalties) vs v1.2 vs ALL (200 seeds)")
    print("=" * 70)

    print("\nFetching 5000 1-min candles...")
    returns_raw = fetch_binance_log_returns("BTCUSDT", "1m", 5000)
    dt = interval_to_dt_years("1m")

    # Train-only winsorization
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

    # Calibrate v1.1 fresh with L2 penalties
    print("\nCalibrating v1.1 (buffer model with L2 penalties)...")
    t0 = time.perf_counter()
    v11_fit = fit_buffer_model(
        train_r, dt,
        half_life_bars=300.0,
        acf_recent_bars=300,
        num_paths=1000,
        maxiter=14,   # ~1500 evals (9 params x popsize=8 x 14 iters = 1008, + init)
        seed=42,
    )
    v11_time = time.perf_counter() - t0
    print(f"  v1.1 calibrated in {v11_time:.0f}s (loss={v11_fit['loss']:.3f})")

    # Extract sim-compatible params for v1.1
    v11_sim_keys = ["mu0", "sigma0", "rho", "nu", "alpha", "beta", "lambda0",
                    "gamma", "eta", "kappa", "theta_p", "phi", "sigma_Y", "eps"]
    v11_params = {k: v11_fit[k] for k in v11_sim_keys}
    print(f"  v1.1 params: sigma0={v11_params['sigma0']:.4f} rho={v11_params['rho']:.4f} "
          f"alpha={v11_params['alpha']:.4f} lambda0={v11_params['lambda0']:.4f} "
          f"phi={v11_params['phi']:.4f} beta={v11_params['beta']:.4f}")

    # Calibrate benchmarks
    print("\nCalibrating benchmarks...")
    t0 = time.perf_counter()
    gbm_p = calibrate_gbm(train_r, dt)
    heston_p = calibrate_heston(train_r, dt)
    merton_p = calibrate_merton(train_r, dt)
    sabr_p = calibrate_sabr(train_r, dt, S0=1.0)
    print(f"  Benchmarks calibrated in {time.perf_counter()-t0:.0f}s")

    model_configs = {
        "Brulant v1.1": ("v11", v11_params),
        "Brulant v1.2": ("v12", BRULANT_V12),
        "GBM": ("gbm", gbm_p),
        "Heston": ("heston", heston_p),
        "Merton": ("merton", merton_p),
        "SABR": ("sabr", sabr_p),
    }

    # Run N_SEEDS per model
    print(f"\nRunning {N_SEEDS} seeds per model...")
    model_losses = {}
    model_stats = {}

    for name, (tag, params) in model_configs.items():
        t0 = time.perf_counter()
        losses = []
        for i in range(N_SEEDS):
            seed = 42 + i * 77
            if tag == "v12":
                lr, _ = simulate_v12(test_r.size, dt, 1000, seed=seed, S0=1.0, **params)
            elif tag == "v11":
                lr, _ = simulate_buffer_paths(test_r.size, dt, 1000, seed=seed, S0=1.0, **params)
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

        # Bootstrap SE of median
        rng_bs = np.random.default_rng(42)
        boot_medians = np.array([np.median(rng_bs.choice(losses, size=len(losses), replace=True))
                                 for _ in range(10000)])

        # Single large sim for distributional stats
        if tag == "v12":
            lr, _ = simulate_v12(test_r.size, dt, 5000, seed=42, S0=1.0, **params)
        elif tag == "v11":
            lr, _ = simulate_buffer_paths(test_r.size, dt, 5000, seed=42, S0=1.0, **params)
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
        }
        elapsed = time.perf_counter() - t0
        print(f"  {name}: median={np.median(losses):.2f} "
              f"95%CI=[{model_stats[name]['ci_median_lo']:.2f}, {model_stats[name]['ci_median_hi']:.2f}] "
              f"kurt={sm[3]:.2f} ({elapsed:.0f}s)")

    # Paired significance tests (v1.1 vs each, v1.2 vs each)
    print(f"\n{'='*70}")
    print(f"  PAIRED SIGNIFICANCE TESTS")
    print(f"{'='*70}")

    for ref_name in ["Brulant v1.1", "Brulant v1.2"]:
        ref_losses = model_losses[ref_name]
        print(f"\n  {ref_name} vs each:")
        print(f"  {'Model':<14s} {'MedDiff':>8s} {'95% CI':>22s} {'Wilcoxon p':>11s} {'DM p':>8s}")
        print(f"  {'-'*14} {'-'*8} {'-'*22} {'-'*11} {'-'*8}")

        for name in model_configs:
            if name == ref_name:
                continue
            diffs = model_losses[name] - ref_losses
            med_diff = float(np.median(diffs))
            ci_lo, ci_hi = bootstrap_ci(diffs, n_bootstrap=10000, ci=0.95, seed=42)
            try:
                _, w_pval = sp_stats.wilcoxon(diffs, alternative='two-sided')
            except ValueError:
                w_pval = float('nan')
            _, dm_pval = diebold_mariano_test(model_losses[name].tolist(), ref_losses.tolist())
            sig = " ***" if w_pval < 0.001 else (" **" if w_pval < 0.01 else (" *" if w_pval < 0.05 else ""))
            print(f"  {name:<14s} {med_diff:>+8.2f} [{ci_lo:>+9.2f}, {ci_hi:>+9.2f}] "
                  f"{w_pval:>11.4g} {dm_pval:>8.4g}{sig}")

    # v1.1 vs v1.2 head-to-head
    print(f"\n{'='*70}")
    print(f"  HEAD-TO-HEAD: v1.1 vs v1.2")
    print(f"{'='*70}")
    diffs_11v12 = model_losses["Brulant v1.2"] - model_losses["Brulant v1.1"]
    med = float(np.median(diffs_11v12))
    ci_lo, ci_hi = bootstrap_ci(diffs_11v12, n_bootstrap=10000, ci=0.95, seed=42)
    try:
        _, wp = sp_stats.wilcoxon(diffs_11v12, alternative='two-sided')
    except ValueError:
        wp = float('nan')
    _, dmp = diebold_mariano_test(model_losses["Brulant v1.2"].tolist(),
                                   model_losses["Brulant v1.1"].tolist())
    print(f"  v1.2 - v1.1 median diff: {med:+.3f}")
    print(f"  95% CI: [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    print(f"  Wilcoxon p: {wp:.4g}")
    print(f"  DM p: {dmp:.4g}")
    if med > 0:
        print(f"  >> v1.1 WINS (v1.2 has higher loss)")
    else:
        print(f"  >> v1.2 WINS (v1.2 has lower loss)")

    # Save
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "spot": S0,
        "n_seeds": N_SEEDS,
        "empirical_moments": emp.tolist(),
        "v11_calibration_time": v11_time,
        "v11_train_loss": v11_fit["loss"],
        "v11_params": v11_params,
        "model_stats": model_stats,
    }
    Path("evidence_v11_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved evidence_v11_benchmark.json")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  FINAL RANKING (sorted by median loss)")
    print(f"{'='*70}")
    ranked = sorted(model_stats.items(), key=lambda x: x[1]["median_loss"])
    print(f"  {'Rank':>4s} {'Model':<14s} {'Median':>8s} {'95% CI':>22s} {'Params':>6s} {'Kurt':>6s}")
    print(f"  {'-'*4} {'-'*14} {'-'*8} {'-'*22} {'-'*6} {'-'*6}")
    param_counts = {"Brulant v1.1": 9, "Brulant v1.2": 11, "GBM": 1,
                    "Heston": 4, "Merton": 3, "SABR": 4}
    for rank, (name, s) in enumerate(ranked, 1):
        marker = " <--" if "v1.1" in name else ""
        print(f"  {rank:>4d} {name:<14s} {s['median_loss']:>8.2f} "
              f"[{s['ci_median_lo']:>8.2f}, {s['ci_median_hi']:>8.2f}] "
              f"{param_counts.get(name, '?'):>6} {s['kurt']:>6.2f}{marker}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = {"Brulant v1.1": "#ff6b35", "Brulant v1.2": "#e74c3c", "GBM": "#3498db",
                  "Heston": "#2ecc71", "Merton": "#9b59b6", "SABR": "#f39c12"}
        names_sorted = [n for n, _ in ranked]

        # 1. Loss distributions
        ax = axes[0]
        bp = ax.boxplot([model_losses[n] for n in names_sorted],
                        tick_labels=[n.replace("Brulant ", "B") for n in names_sorted],
                        patch_artist=True, showfliers=False)
        for patch, name in zip(bp["boxes"], names_sorted):
            patch.set_facecolor(colors.get(name, "#95a5a6"))
            patch.set_alpha(0.7)
        ax.set_ylabel("OOS Moment-Matching Loss")
        ax.set_title(f"Loss Distribution ({N_SEEDS} seeds)", fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 2. v1.1 vs all paired differences
        ax = axes[1]
        v11_losses = model_losses["Brulant v1.1"]
        others = [n for n in names_sorted if n != "Brulant v1.1"]
        diffs_list = [model_losses[n] - v11_losses for n in others]
        bp2 = ax.boxplot(diffs_list,
                         tick_labels=[n.replace("Brulant ", "B")[:8] for n in others],
                         patch_artist=True, showfliers=False)
        for patch, name in zip(bp2["boxes"], others):
            patch.set_facecolor(colors.get(name, "#95a5a6"))
            patch.set_alpha(0.7)
        ax.axhline(0, color="black", linestyle="--", linewidth=2, alpha=0.5)
        ax.set_ylabel("Loss(Model) - Loss(v1.1)")
        ax.set_title("v1.1 vs All (>0 = v1.1 wins)", fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 3. Kurtosis comparison
        ax = axes[2]
        kurt_vals = [model_stats[n]["kurt"] for n in names_sorted]
        bar_colors = [colors.get(n, "#95a5a6") for n in names_sorted]
        bars = ax.bar(range(len(names_sorted)), kurt_vals, color=bar_colors, alpha=0.7)
        ax.axhline(emp[3], color="red", linestyle="--", linewidth=2, label=f"Empirical ({emp[3]:.1f})")
        ax.set_xticks(range(len(names_sorted)))
        ax.set_xticklabels([n.replace("Brulant ", "B") for n in names_sorted], rotation=15)
        ax.set_ylabel("Excess Kurtosis")
        ax.set_title("Kurtosis Match", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"Brulant v1.1 vs v1.2 vs Benchmarks (Spot ${S0:,.0f}, {N_SEEDS} seeds)",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig("evidence_v11_benchmark.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("Saved evidence_v11_benchmark.png")
    except Exception as e:
        print(f"  Plot failed: {e}")


if __name__ == "__main__":
    main()
