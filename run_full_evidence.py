"""
Master Evidence Runner: Generates all results for the Brulant v1.2 presentation.
==================================================================================
Runs in sequence:
  1. Quick benchmark (200 seeds, significance tests) -- ~15 min
  2. Temporal validation (12 weeks, 30 seeds per model) -- ~45 min
  3. Market comparison (Deribit quotes) -- ~5 min
  4. Cross-asset check (ETH, quick 4-window test) -- ~15 min

Total estimated: ~80 min on 22-core CPU without Numba.
All results saved to JSON + figures for the presentation.
"""
from __future__ import annotations
import json, time, datetime, sys
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

# ===========================================================================
#  PART 1: BENCHMARK V1.2 (200 seeds, paired tests)
# ===========================================================================
def run_benchmark():
    print("\n" + "=" * 70)
    print("  PART 1: BENCHMARK v1.2 vs ALL MODELS (200 seeds)")
    print("=" * 70)

    from experiment_v12 import simulate_v12
    from benchmark_comparison import (
        simulate_gbm, calibrate_gbm,
        simulate_heston, calibrate_heston,
        simulate_merton, calibrate_merton,
        simulate_sabr, calibrate_sabr,
    )
    from fit_sandpile import fetch_binance_log_returns, interval_to_dt_years, moment_vector
    from backtest_buffer_model import simulate_buffer_paths, MOMENT_NAMES
    from benchmark_v12 import BRULANT_V12, BRULANT_V11, diebold_mariano_test, bootstrap_ci

    N_SEEDS = 200

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

    # Calibrate benchmarks
    print("\nCalibrating benchmarks (equalized DE budgets)...")
    t0 = time.perf_counter()
    gbm_p = calibrate_gbm(train_r, dt)
    heston_p = calibrate_heston(train_r, dt)
    merton_p = calibrate_merton(train_r, dt)
    sabr_p = calibrate_sabr(train_r, dt, S0=1.0)
    print(f"  Calibration done in {time.perf_counter()-t0:.0f}s")

    model_configs = {
        "Brulant v1.2": ("v12", BRULANT_V12),
        "GBM": ("gbm", gbm_p),
        "Heston": ("heston", heston_p),
        "Merton": ("merton", merton_p),
        "SABR": ("sabr", sabr_p),
    }

    # Run 200 seeds per model
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

        # Distributional stats
        if tag == "v12":
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
        }
        elapsed = time.perf_counter() - t0
        print(f"  {name}: median={np.median(losses):.2f} "
              f"95%CI=[{model_stats[name]['ci_median_lo']:.2f}, {model_stats[name]['ci_median_hi']:.2f}] "
              f"({elapsed:.0f}s)")

    # Paired significance tests
    print("\n--- PAIRED SIGNIFICANCE TESTS (v1.2 vs each) ---")
    v12_losses = model_losses["Brulant v1.2"]
    sig_results = {}
    print(f"  {'Model':<14s} {'MedDiff':>8s} {'95% CI':>22s} {'Wilcoxon p':>11s} {'DM p':>8s}")
    print(f"  {'-'*14} {'-'*8} {'-'*22} {'-'*11} {'-'*8}")

    for name in model_configs:
        if name == "Brulant v1.2":
            continue
        diffs = model_losses[name] - v12_losses
        med_diff = float(np.median(diffs))
        ci_lo, ci_hi = bootstrap_ci(diffs, n_bootstrap=10000, ci=0.95, seed=42)
        try:
            _, w_pval = sp_stats.wilcoxon(diffs, alternative='two-sided')
        except ValueError:
            w_pval = float('nan')
        _, dm_pval = diebold_mariano_test(model_losses[name], v12_losses)
        sig = " ***" if w_pval < 0.001 else (" **" if w_pval < 0.01 else (" *" if w_pval < 0.05 else ""))
        print(f"  {name:<14s} {med_diff:>+8.2f} [{ci_lo:>+9.2f}, {ci_hi:>+9.2f}] {w_pval:>11.4g} {dm_pval:>8.4g}{sig}")
        sig_results[name] = {
            "median_diff": med_diff,
            "ci_95": [ci_lo, ci_hi],
            "wilcoxon_p": float(w_pval),
            "dm_p": float(dm_pval),
        }

    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "spot": S0,
        "n_seeds": N_SEEDS,
        "empirical_moments": emp.tolist(),
        "model_stats": model_stats,
        "significance_tests": sig_results,
    }
    Path("evidence_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nSaved evidence_benchmark.json")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1. Loss distributions
        ax = axes[0]
        names = list(model_losses.keys())
        colors = {"Brulant v1.2": "#e74c3c", "GBM": "#3498db", "Heston": "#2ecc71",
                  "Merton": "#9b59b6", "SABR": "#f39c12"}
        bp = ax.boxplot([model_losses[n] for n in names],
                        labels=[n.replace("Brulant ", "") for n in names],
                        patch_artist=True, showfliers=False)
        for patch, name in zip(bp["boxes"], names):
            patch.set_facecolor(colors.get(name, "#95a5a6"))
            patch.set_alpha(0.7)
        ax.set_ylabel("OOS Moment-Matching Loss")
        ax.set_title(f"Loss Distribution ({N_SEEDS} seeds)", fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 2. Paired differences (v1.2 vs each)
        ax = axes[1]
        bench_names = [n for n in names if n != "Brulant v1.2"]
        diffs_list = [model_losses[n] - v12_losses for n in bench_names]
        bp2 = ax.boxplot(diffs_list,
                         labels=[n[:8] for n in bench_names],
                         patch_artist=True, showfliers=False)
        for patch, name in zip(bp2["boxes"], bench_names):
            patch.set_facecolor(colors.get(name, "#95a5a6"))
            patch.set_alpha(0.7)
        ax.axhline(0, color="black", linestyle="--", linewidth=2, alpha=0.5)
        ax.set_ylabel("Loss(Benchmark) - Loss(v1.2)")
        ax.set_title("Paired Differences (>0 = v1.2 wins)", fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 3. Significance summary
        ax = axes[2]
        ax.axis('off')
        sig_text = f"Paired Wilcoxon Signed-Rank Tests\n(v1.2 vs each, {N_SEEDS} seeds)\n\n"
        for name, sr in sig_results.items():
            p = sr['wilcoxon_p']
            stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            sig_text += f"{name:<12s}  p={p:.4g}  {stars}\n"
            sig_text += f"  median diff: {sr['median_diff']:+.2f}\n"
            sig_text += f"  95% CI: [{sr['ci_95'][0]:+.2f}, {sr['ci_95'][1]:+.2f}]\n\n"
        ax.text(0.1, 0.95, sig_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle(f"Brulant v1.2: Statistical Evidence (Spot ${S0:,.0f})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig("evidence_benchmark.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("Saved evidence_benchmark.png")
    except Exception as e:
        print(f"  Plot failed: {e}")

    return result


# ===========================================================================
#  PART 2: TEMPORAL VALIDATION (12 weeks, compact)
# ===========================================================================
def run_temporal():
    print("\n" + "=" * 70)
    print("  PART 2: TEMPORAL VALIDATION (12 weeks, 5 models)")
    print("=" * 70)

    from experiment_v12 import simulate_v12
    from benchmark_comparison import (
        simulate_gbm, calibrate_gbm,
        simulate_heston, calibrate_heston,
        simulate_merton, calibrate_merton,
        simulate_sabr, calibrate_sabr,
    )
    from fit_sandpile import (
        interval_to_dt_years, moment_vector, recent_exponential_weights,
    )
    from temporal_validation import fetch_extended_history, calibrate_v12, compute_oos_loss
    from fit_sandpile import _to_jsonable

    WINDOW_SIZE = 10_080  # 7 days
    N_SEEDS = 30
    TARGET_CANDLES = 90_000  # ~9 weeks

    print(f"\nFetching ~{TARGET_CANDLES:,} 1-min candles (~9 weeks)...")
    t0 = time.perf_counter()
    all_returns = fetch_extended_history("BTCUSDT", "1m", TARGET_CANDLES)
    dt = interval_to_dt_years("1m")
    print(f"  Fetched {all_returns.size:,} returns in {time.perf_counter()-t0:.0f}s")

    n_windows = all_returns.size // WINDOW_SIZE
    print(f"  Windows: {n_windows} x {WINDOW_SIZE} bars")

    model_tags = ["v12", "gbm", "heston", "merton", "sabr"]
    model_names = {"v12": "Brulant v1.2", "gbm": "GBM", "heston": "Heston",
                   "merton": "Merton", "sabr": "SABR"}

    all_median_losses = {tag: [] for tag in model_tags}
    window_results = []

    for w_idx in range(n_windows):
        w_start = w_idx * WINDOW_SIZE
        w_end = w_start + WINDOW_SIZE
        window_r = all_returns[w_start:w_end]

        n_half = WINDOW_SIZE // 2
        train_r_raw = window_r[:n_half]
        test_r_raw = window_r[n_half:]

        mu = np.median(train_r_raw)
        mad = np.percentile(np.abs(train_r_raw - mu), 75) * 1.4826
        train_r = np.clip(train_r_raw, mu - 5 * mad, mu + 5 * mad)
        test_r = np.clip(test_r_raw, mu - 5 * mad, mu + 5 * mad)

        window_return = float(np.sum(window_r))
        window_vol = float(np.std(window_r))

        t0 = time.perf_counter()
        v12_params = calibrate_v12(train_r, dt, seed=42 + w_idx)
        gbm_p = calibrate_gbm(train_r, dt)
        heston_p = calibrate_heston(train_r, dt)
        merton_p = calibrate_merton(train_r, dt)
        sabr_p = calibrate_sabr(train_r, dt, S0=1.0)
        cal_time = time.perf_counter() - t0

        all_params = {"v12": v12_params, "gbm": gbm_p, "heston": heston_p,
                      "merton": merton_p, "sabr": sabr_p}

        medians = {}
        for tag in model_tags:
            losses = compute_oos_loss(tag, all_params[tag], test_r, dt,
                                      n_seeds=N_SEEDS, base_seed=42 + w_idx * 1000)
            medians[tag] = float(np.median(losses))
            all_median_losses[tag].append(medians[tag])

        winner = min(medians, key=medians.get)
        loss_str = "  ".join(f"{model_names[t][:5]}={medians[t]:.1f}" for t in model_tags)
        print(f"  W{w_idx+1}/{n_windows}: {loss_str} | {model_names[winner]} | {cal_time:.0f}s")

        # Classify regime
        if window_return > 0.05:
            regime = "bull"
        elif window_return < -0.05:
            regime = "bear"
        else:
            regime = "sideways"

        window_results.append({
            "window": w_idx, "return": window_return, "vol": window_vol,
            "regime": regime, "medians": medians, "winner": winner,
        })

    # Aggregate
    wins = {tag: sum(1 for w in window_results if w["winner"] == tag) for tag in model_tags}
    ranks = {tag: [] for tag in model_tags}
    for w in window_results:
        sorted_tags = sorted(model_tags, key=lambda t: w["medians"][t])
        for rank, tag in enumerate(sorted_tags, 1):
            ranks[tag].append(rank)

    print(f"\n--- AGGREGATE ({n_windows} windows) ---")
    print(f"  {'Model':<14s} {'Wins':>5s} {'Win%':>6s} {'AvgLoss':>8s} {'AvgRank':>8s}")
    for tag in model_tags:
        print(f"  {model_names[tag]:<14s} {wins[tag]:>5d} {100*wins[tag]/n_windows:>5.1f}% "
              f"{np.mean(all_median_losses[tag]):>8.2f} {np.mean(ranks[tag]):>8.2f}")

    # Pooled Wilcoxon
    v12_arr = np.array(all_median_losses["v12"])
    print(f"\n--- POOLED SIGNIFICANCE ---")
    pooled_sig = {}
    for tag in model_tags:
        if tag == "v12":
            continue
        diffs = np.array(all_median_losses[tag]) - v12_arr
        try:
            _, p = sp_stats.wilcoxon(diffs, alternative='two-sided')
        except ValueError:
            p = float('nan')
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  vs {model_names[tag]:<10s}: mean_diff={np.mean(diffs):+.2f} p={p:.4g} {stars}")
        pooled_sig[tag] = {"mean_diff": float(np.mean(diffs)), "p": float(p)}

    # Regime breakdown
    print(f"\n--- REGIME BREAKDOWN ---")
    regime_results = {}
    for regime in ["bull", "bear", "sideways"]:
        rw = [w for w in window_results if w["regime"] == regime]
        if not rw:
            continue
        regime_wins = {tag: sum(1 for w in rw if w["winner"] == tag) for tag in model_tags}
        regime_avgs = {tag: np.mean([w["medians"][tag] for w in rw]) for tag in model_tags}
        print(f"  {regime.upper()} ({len(rw)} windows):")
        for tag in model_tags:
            print(f"    {model_names[tag]:<14s} wins={regime_wins[tag]} avg={regime_avgs[tag]:.2f}")
        regime_results[regime] = {"n": len(rw), "wins": regime_wins,
                                   "avg_loss": {model_names[t]: regime_avgs[t] for t in model_tags}}

    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_windows": n_windows, "n_seeds": N_SEEDS,
        "wins": {model_names[t]: wins[t] for t in model_tags},
        "avg_rank": {model_names[t]: float(np.mean(ranks[t])) for t in model_tags},
        "avg_loss": {model_names[t]: float(np.mean(all_median_losses[t])) for t in model_tags},
        "pooled_significance": {model_names[t]: pooled_sig[t] for t in pooled_sig},
        "regime_results": regime_results,
        "windows": window_results,
    }
    Path("evidence_temporal.json").write_text(json.dumps(_to_jsonable(result), indent=2), encoding="utf-8")
    print("\nSaved evidence_temporal.json")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

        # 1. Box plot
        ax = axes[0, 0]
        data = [all_median_losses[tag] for tag in model_tags]
        bp = ax.boxplot(data, labels=[model_names[t][:8] for t in model_tags], patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_ylabel("Median OOS Loss")
        ax.set_title("Loss Across Windows", fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 2. Win rate
        ax = axes[0, 1]
        pcts = [100 * wins[t] / n_windows for t in model_tags]
        bars = ax.bar(range(len(model_tags)), pcts, color=colors,
                      tick_label=[model_names[t][:8] for t in model_tags])
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Win Rate", fontweight="bold")
        for bar, p in zip(bars, pcts):
            ax.text(bar.get_x()+bar.get_width()/2, p+1, f"{p:.0f}%", ha="center", fontsize=9)

        # 3. Loss trajectory
        ax = axes[1, 0]
        for i, tag in enumerate(model_tags):
            ax.plot(all_median_losses[tag], "o-", label=model_names[tag][:8],
                    color=colors[i], markersize=3, alpha=0.8)
        ax.set_xlabel("Window")
        ax.set_ylabel("Median Loss")
        ax.set_title("Loss Over Time", fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # 4. Regime heatmap
        ax = axes[1, 1]
        ax.axis('off')
        regime_text = "REGIME-CONDITIONAL RESULTS\n\n"
        for regime, rd in regime_results.items():
            regime_text += f"{regime.upper()} ({rd['n']} windows):\n"
            for name, avg in sorted(rd['avg_loss'].items(), key=lambda x: x[1]):
                w = rd['wins'].get([t for t in model_tags if model_names[t]==name][0], 0)
                regime_text += f"  {name:<14s} avg={avg:.1f}  wins={w}\n"
            regime_text += "\n"
        ax.text(0.05, 0.95, regime_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        fig.suptitle(f"Temporal Validation: {n_windows} Windows x {N_SEEDS} Seeds",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig("evidence_temporal.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("Saved evidence_temporal.png")
    except Exception as e:
        print(f"  Plot failed: {e}")

    return result


# ===========================================================================
#  PART 3: DERIBIT MARKET COMPARISON
# ===========================================================================
def run_market_comparison():
    print("\n" + "=" * 70)
    print("  PART 3: DERIBIT MARKET COMPARISON")
    print("=" * 70)
    try:
        from market_comparison import main as mc_main
        mc_main()
    except Exception as e:
        print(f"  Market comparison failed: {e}")
        print("  (Deribit may be geo-restricted or unavailable)")


# ===========================================================================
#  MAIN
# ===========================================================================
if __name__ == "__main__":
    t_start = time.perf_counter()
    print("=" * 70)
    print("  BRULANT v1.2: FULL EVIDENCE GENERATION")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}

    results["benchmark"] = run_benchmark()
    results["temporal"] = run_temporal()
    run_market_comparison()

    total_time = time.perf_counter() - t_start
    print(f"\n{'='*70}")
    print(f"  ALL DONE in {total_time/60:.1f} minutes")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    for f in ["evidence_benchmark.json", "evidence_benchmark.png",
              "evidence_temporal.json", "evidence_temporal.png",
              "market_comparison_results.json", "market_comparison.png"]:
        if Path(f).exists():
            print(f"  {f} ({Path(f).stat().st_size/1024:.0f} KB)")
