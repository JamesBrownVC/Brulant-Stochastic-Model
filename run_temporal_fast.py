"""
Fast temporal validation: 4 windows, 10 seeds, reduced DE budget.
Should complete in ~40 min. Enough to confirm the pattern.
"""
from __future__ import annotations
import json, time, datetime, sys
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import differential_evolution

sys.stdout.reconfigure(line_buffering=True)

from experiment_v12 import simulate_v12
from benchmark_comparison import (
    simulate_gbm, calibrate_gbm,
    simulate_heston, calibrate_heston,
    simulate_merton, calibrate_merton,
    simulate_sabr, calibrate_sabr,
)
from fit_sandpile import (
    interval_to_dt_years, moment_vector, recent_exponential_weights,
    _to_jsonable,
)
from temporal_validation import fetch_extended_history


def calibrate_v12_fast(train_r, dt, seed=42):
    """Fast v1.2 calibration: popsize=8, maxiter=10, 500 paths."""
    w = recent_exponential_weights(train_r.size, 400.0)
    target = moment_vector(train_r, w=w, acf_recent_bars=300)
    scales = np.maximum(np.abs(target), np.array([1e-12, 1e-12, 0.5, 1.0, 0.05, 0.1]))
    scales = np.maximum(scales, 1e-9)

    names = ["sigma0", "sigma0_bar", "alpha_s", "xi_s", "rho", "alpha",
             "kappa_fast", "kappa_slow", "theta_fast", "theta_slow", "w_slow"]
    bounds = [
        (0.15, 0.80), (0.10, 0.70), (0.1, 15.0), (0.01, 3.0),
        (0.5, 4.0), (1.0, 20.0),
        (20.0, 100.0), (0.5, 8.0), (0.5, 3.0), (0.3, 2.0), (0.1, 0.6),
    ]
    fixed = dict(mu0=0.0, nu=1.0, beta=0.0, lambda0=0.0, gamma=20.0,
                 eta=1.0, phi=0.56, sigma_Y=0.01, eps=0.001,
                 stoch_vol_target=True, multi_buffer=True,
                 kappa_mid=15.0, theta_mid=1.5, w_mid=0.0)

    rng = np.random.default_rng(seed)

    def obj(theta):
        p = {k: float(v) for k, v in zip(names, theta)}
        p["w_fast"] = 1.0 - p["w_slow"]
        p.update(fixed)
        s = int(rng.integers(0, 2**31 - 1))
        sim_lr, _ = simulate_v12(train_r.size, dt, 500, seed=s, S0=1.0, **p)
        pooled = sim_lr.ravel()
        if pooled.size > 30000:
            pooled = np.random.default_rng(s).choice(pooled, 30000, replace=False)
        sim = moment_vector(pooled, w=None, acf_recent_bars=300)
        z = (sim - target) / scales
        penalty = 2.0 * max(0, p["rho"] - 3.0) ** 2
        return float(np.sum(z * z)) + penalty

    res = differential_evolution(obj, bounds, maxiter=10, seed=seed, workers=1,
                                 polish=False, popsize=8, tol=1e-3, atol=1e-3)

    fit = {k: float(v) for k, v in zip(names, res.x)}
    fit["w_fast"] = 1.0 - fit["w_slow"]
    fit.update(fixed)
    return fit


def compute_loss_fast(tag, params, test_r, dt, n_seeds=10, base_seed=42):
    """Quick OOS loss with fewer seeds."""
    emp = moment_vector(test_r, w=None, acf_recent_bars=300)
    scales = np.maximum(np.abs(emp), np.array([1e-12, 1e-12, 0.5, 1.0, 0.05, 0.1]))
    scales = np.maximum(scales, 1e-9)
    clean = {k: v for k, v in params.items() if not k.startswith('_')}

    losses = []
    for i in range(n_seeds):
        seed = base_seed + i * 77
        if tag == "v12":
            lr, _ = simulate_v12(test_r.size, dt, 800, seed=seed, S0=1.0, **clean)
        elif tag == "gbm":
            lr, _ = simulate_gbm(test_r.size, dt, 800, 1.0, params["sigma"], seed=seed)
        elif tag == "heston":
            lr, _ = simulate_heston(test_r.size, dt, 800, 1.0, **clean, seed=seed)
        elif tag == "merton":
            lr, _ = simulate_merton(test_r.size, dt, 800, 1.0, **clean, seed=seed)
        elif tag == "sabr":
            lr, _ = simulate_sabr(test_r.size, dt, 800, 1.0, **clean, seed=seed)
        sim = moment_vector(lr.ravel(), w=None, acf_recent_bars=300)
        loss = float(np.sum(((sim - emp) / scales) ** 2))
        # Cap at 1000 to avoid Merton-type explosions
        losses.append(min(loss, 1000.0))
    return np.array(losses)


def main():
    print("=" * 70)
    print("  FAST TEMPORAL VALIDATION (4 windows, 10 seeds)")
    print("=" * 70)

    WINDOW_SIZE = 10_080  # 7 days
    N_SEEDS = 10
    TARGET_CANDLES = 50_000  # ~5 weeks

    print(f"\nFetching ~{TARGET_CANDLES:,} 1-min candles...")
    t0 = time.perf_counter()
    all_returns = fetch_extended_history("BTCUSDT", "1m", TARGET_CANDLES)
    dt = interval_to_dt_years("1m")
    print(f"  Fetched {all_returns.size:,} returns in {time.perf_counter()-t0:.0f}s")

    n_windows = min(all_returns.size // WINDOW_SIZE, 4)
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
        print(f"\n  W{w_idx+1}/{n_windows}: Calibrating...", end=" ", flush=True)
        v12_params = calibrate_v12_fast(train_r, dt, seed=42 + w_idx)
        print("v1.2", end=" ", flush=True)
        gbm_p = calibrate_gbm(train_r, dt)
        print("GBM", end=" ", flush=True)
        heston_p = calibrate_heston(train_r, dt)
        print("Heston", end=" ", flush=True)
        merton_p = calibrate_merton(train_r, dt)
        print("Merton", end=" ", flush=True)
        sabr_p = calibrate_sabr(train_r, dt, S0=1.0)
        print("SABR", end=" ", flush=True)
        cal_time = time.perf_counter() - t0

        all_params = {"v12": v12_params, "gbm": gbm_p, "heston": heston_p,
                      "merton": merton_p, "sabr": sabr_p}

        medians = {}
        for tag in model_tags:
            losses = compute_loss_fast(tag, all_params[tag], test_r, dt,
                                       n_seeds=N_SEEDS, base_seed=42 + w_idx * 1000)
            medians[tag] = float(np.median(losses))
            all_median_losses[tag].append(medians[tag])

        winner = min(medians, key=medians.get)
        loss_str = "  ".join(f"{model_names[t][:5]}={medians[t]:.1f}" for t in model_tags)
        print(f"\n    {loss_str} | Winner: {model_names[winner]} | {cal_time:.0f}s")

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

    print(f"\n{'='*70}")
    print(f"  AGGREGATE ({n_windows} windows)")
    print(f"{'='*70}")
    print(f"  {'Model':<14s} {'Wins':>5s} {'Win%':>6s} {'AvgLoss':>8s} {'AvgRank':>8s}")
    for tag in model_tags:
        print(f"  {model_names[tag]:<14s} {wins[tag]:>5d} {100*wins[tag]/n_windows:>5.1f}% "
              f"{np.mean(all_median_losses[tag]):>8.2f} {np.mean(ranks[tag]):>8.2f}")

    # Pooled significance
    v12_arr = np.array(all_median_losses["v12"])
    print(f"\n  POOLED SIGNIFICANCE (v1.2 vs each)")
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
        print(f"    vs {model_names[tag]:<10s}: mean_diff={np.mean(diffs):+.2f} p={p:.4g} {stars}")
        pooled_sig[tag] = {"mean_diff": float(np.mean(diffs)), "p": float(p)}

    # Regime breakdown
    regime_results = {}
    for regime in ["bull", "bear", "sideways"]:
        rw = [w for w in window_results if w["regime"] == regime]
        if not rw:
            continue
        regime_wins = {tag: sum(1 for w in rw if w["winner"] == tag) for tag in model_tags}
        regime_avgs = {tag: np.mean([w["medians"][tag] for w in rw]) for tag in model_tags}
        print(f"\n  {regime.upper()} ({len(rw)} windows):")
        for tag in model_tags:
            print(f"    {model_names[tag]:<14s} wins={regime_wins[tag]} avg={regime_avgs[tag]:.2f}")
        regime_results[regime] = {"n": len(rw), "wins": regime_wins,
                                   "avg_loss": {model_names[t]: regime_avgs[t] for t in model_tags}}

    # Save as evidence_temporal.json (same format as full runner)
    result = {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_windows": n_windows, "n_seeds": N_SEEDS,
        "note": "Fast run (4 windows, 10 seeds, reduced DE). Confirms pattern.",
        "wins": {model_names[t]: wins[t] for t in model_tags},
        "avg_rank": {model_names[t]: float(np.mean(ranks[t])) for t in model_tags},
        "avg_loss": {model_names[t]: float(np.mean(all_median_losses[t])) for t in model_tags},
        "pooled_significance": {model_names[t]: pooled_sig[t] for t in pooled_sig},
        "regime_results": regime_results,
        "windows": window_results,
    }
    Path("evidence_temporal.json").write_text(json.dumps(_to_jsonable(result), indent=2), encoding="utf-8")
    print(f"\nSaved evidence_temporal.json")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

        # 1. Box plot
        ax = axes[0]
        data = [all_median_losses[tag] for tag in model_tags]
        bp = ax.boxplot(data, tick_labels=[model_names[t][:8] for t in model_tags], patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_ylabel("Median OOS Loss")
        ax.set_title(f"Loss Across {n_windows} Windows", fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 2. Win rate
        ax = axes[1]
        pcts = [100 * wins[t] / n_windows for t in model_tags]
        bars = ax.bar(range(len(model_tags)), pcts, color=colors,
                      tick_label=[model_names[t][:8] for t in model_tags])
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Win Rate", fontweight="bold")
        for bar, p in zip(bars, pcts):
            ax.text(bar.get_x()+bar.get_width()/2, p+1, f"{p:.0f}%", ha="center", fontsize=9)

        # 3. Loss trajectory
        ax = axes[2]
        for i, tag in enumerate(model_tags):
            ax.plot(all_median_losses[tag], "o-", label=model_names[tag][:8],
                    color=colors[i], markersize=5, alpha=0.8)
        ax.set_xlabel("Window")
        ax.set_ylabel("Median Loss")
        ax.set_title("Loss Over Time", fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f"Temporal Validation: {n_windows} Windows x {N_SEEDS} Seeds",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig("evidence_temporal.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("Saved evidence_temporal.png")
    except Exception as e:
        print(f"  Plot failed: {e}")

    # Also run Deribit market comparison
    print(f"\n{'='*70}")
    print(f"  PART 3: DERIBIT MARKET COMPARISON")
    print(f"{'='*70}")
    try:
        from market_comparison import main as mc_main
        mc_main()
    except Exception as e:
        print(f"  Market comparison failed: {e}")

    total = time.perf_counter() - t0
    print(f"\nTotal time: {total/60:.1f} min")


if __name__ == "__main__":
    main()
