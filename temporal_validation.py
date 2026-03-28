"""
Temporal Validation: Brulant v1.2 vs benchmarks across non-overlapping windows.
================================================================================
Downloads extended BTC/USDT 1-min history, splits into ~36 non-overlapping
7-day windows, calibrates ALL models per window with equalized DE budgets,
and runs paired significance tests pooled across windows.

This is the core evidence for publication: single-window results are anecdotes;
multi-window results with significance tests are science.
"""
from __future__ import annotations

import json
import time
import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import differential_evolution

from experiment_v12 import simulate_v12
from benchmark_comparison import (
    simulate_gbm, simulate_heston, simulate_merton, simulate_sabr,
    calibrate_gbm, calibrate_heston, calibrate_merton, calibrate_sabr,
)
from fit_sandpile import (
    fetch_binance_log_returns, interval_to_dt_years,
    moment_vector, recent_exponential_weights, _to_jsonable,
)
from backtest_buffer_model import MOMENT_NAMES

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False


# ---------------------------------------------------------------------------
#  V1.2 calibration (inline, equalized budget)
# ---------------------------------------------------------------------------
def calibrate_v12(train_r: np.ndarray, dt: float, seed: int = 42) -> Dict[str, float]:
    """Calibrate Brulant v1.2 multi-buffer model via DE moment matching."""
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
        sim_lr, _ = simulate_v12(train_r.size, dt, 800, seed=s, S0=1.0, **p)
        pooled = sim_lr.ravel()
        if pooled.size > 50000:
            pooled = np.random.default_rng(s).choice(pooled, 50000, replace=False)
        sim = moment_vector(pooled, w=None, acf_recent_bars=300)
        z = (sim - target) / scales
        penalty = 2.0 * max(0, p["rho"] - 3.0) ** 2
        penalty += 1.0 * max(0, p["xi_s"] - 2.0) ** 2
        return float(np.sum(z * z)) + penalty

    # 11 params x popsize=10 x maxiter=14 = 1540 evals
    res = differential_evolution(obj, bounds, maxiter=14, seed=seed, workers=1,
                                 polish=False, popsize=10, tol=1e-3, atol=1e-4)

    fit = {k: float(v) for k, v in zip(names, res.x)}
    fit["w_fast"] = 1.0 - fit["w_slow"]
    fit.update(fixed)
    fit["_train_loss"] = float(res.fun)
    return fit


# ---------------------------------------------------------------------------
#  OOS loss computation
# ---------------------------------------------------------------------------
def compute_oos_loss(
    model_tag: str,
    params: Dict,
    test_r: np.ndarray,
    dt: float,
    n_seeds: int = 50,
    base_seed: int = 42,
) -> np.ndarray:
    """Compute OOS moment-matching loss for n_seeds, return loss array."""
    emp = moment_vector(test_r, w=None, acf_recent_bars=300)
    scales = np.maximum(np.abs(emp), np.array([1e-12, 1e-12, 0.5, 1.0, 0.05, 0.1]))
    scales = np.maximum(scales, 1e-9)

    # Strip any private keys (e.g., _train_loss) from params
    clean_params = {k: v for k, v in params.items() if not k.startswith('_')}

    losses = []
    for i in range(n_seeds):
        seed = base_seed + i * 77
        if model_tag == "v12":
            lr, _ = simulate_v12(test_r.size, dt, 1000, seed=seed, S0=1.0, **clean_params)
        elif model_tag == "gbm":
            lr, _ = simulate_gbm(test_r.size, dt, 1000, 1.0, params["sigma"], seed=seed)
        elif model_tag == "heston":
            lr, _ = simulate_heston(test_r.size, dt, 1000, 1.0, **clean_params, seed=seed)
        elif model_tag == "merton":
            lr, _ = simulate_merton(test_r.size, dt, 1000, 1.0, **clean_params, seed=seed)
        elif model_tag == "sabr":
            lr, _ = simulate_sabr(test_r.size, dt, 1000, 1.0, **clean_params, seed=seed)
        else:
            raise ValueError(f"Unknown model tag: {model_tag}")
        sim = moment_vector(lr.ravel(), w=None, acf_recent_bars=300)
        losses.append(float(np.sum(((sim - emp) / scales) ** 2)))
    return np.array(losses)


# ---------------------------------------------------------------------------
#  Data fetching (extended history)
# ---------------------------------------------------------------------------
def fetch_extended_history(symbol: str, interval: str, total_candles: int) -> np.ndarray:
    """Fetch extended history by paginating Binance API (max 1000 per call)."""
    import requests

    interval_ms = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}
    step_ms = interval_ms.get(interval, 60_000)

    all_closes = []
    end_time = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
    remaining = total_candles

    while remaining > 0:
        limit = min(remaining, 1000)
        start_time = end_time - limit * step_ms
        url = "https://api.binance.com/api/v3/klines"
        resp = requests.get(url, params={
            "symbol": symbol, "interval": interval,
            "startTime": start_time, "endTime": end_time - 1,
            "limit": limit,
        }, timeout=30)
        data = resp.json()
        if not data:
            break
        closes = [float(c[4]) for c in data]
        all_closes = closes + all_closes
        remaining -= len(data)
        end_time = start_time
        time.sleep(0.2)  # rate limiting

    closes = np.array(all_closes, dtype=np.float64)
    log_returns = np.diff(np.log(closes))
    return log_returns


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Temporal validation across non-overlapping windows")
    parser.add_argument("--symbol", default="BTCUSDT", help="Binance symbol (e.g. BTCUSDT, ETHUSDT, SOLUSDT)")
    parser.add_argument("--candles", type=int, default=260_000, help="Total candles to fetch (~6 months)")
    parser.add_argument("--window-size", type=int, default=10_080, help="Window size in bars (7 days = 10080)")
    parser.add_argument("--seeds", type=int, default=50, help="MC seeds per model per window")
    parser.add_argument("--csv", type=str, default=None, help="Load from CSV instead of fetching")
    args = parser.parse_args()

    SYMBOL = args.symbol
    TARGET_CANDLES = args.candles
    WINDOW_SIZE = args.window_size
    N_SEEDS = args.seeds

    print("=" * 70)
    print(f"  TEMPORAL VALIDATION: Brulant v1.2 vs Benchmarks ({SYMBOL})")
    print("  Non-overlapping 7-day windows across extended history")
    print("=" * 70)

    if args.csv:
        print(f"\nLoading from {args.csv}...")
        all_returns = np.loadtxt(args.csv, delimiter=",", skiprows=1)
        dt = interval_to_dt_years("1m")
    else:
        print(f"\nFetching ~{TARGET_CANDLES:,} 1-min {SYMBOL} candles...")
        t0 = time.perf_counter()
        all_returns = fetch_extended_history(SYMBOL, "1m", TARGET_CANDLES)
        dt = interval_to_dt_years("1m")
        print(f"  Fetched {all_returns.size:,} log returns in {time.perf_counter()-t0:.1f}s")

        # Save locally for reproducibility
        csv_path = Path(f"{SYMBOL.lower()}_1m_extended.csv")
        np.savetxt(csv_path, all_returns, delimiter=",", header="log_return", comments="")
        print(f"  Saved to {csv_path}")

    # Split into non-overlapping windows
    n_windows = all_returns.size // WINDOW_SIZE
    print(f"\n  Window size: {WINDOW_SIZE} bars (7 days)")
    print(f"  Total windows: {n_windows}")
    print(f"  Seeds per model per window: {N_SEEDS}")

    model_tags = ["v12", "gbm", "heston", "merton", "sabr"]
    model_names = {
        "v12": "Brulant v1.2", "gbm": "GBM", "heston": "Heston",
        "merton": "Merton", "sabr": "SABR",
    }

    # Results storage
    window_results: List[Dict[str, Any]] = []
    all_median_losses = {tag: [] for tag in model_tags}

    for w_idx in range(n_windows):
        w_start = w_idx * WINDOW_SIZE
        w_end = w_start + WINDOW_SIZE
        window_r = all_returns[w_start:w_end]

        # 50/50 train/test within window
        n_half = WINDOW_SIZE // 2
        train_r_raw = window_r[:n_half]
        test_r_raw = window_r[n_half:]

        # Train-only winsorization
        mu = np.median(train_r_raw)
        mad = np.percentile(np.abs(train_r_raw - mu), 75) * 1.4826
        lo, hi = mu - 5 * mad, mu + 5 * mad
        train_r = np.clip(train_r_raw, lo, hi)
        test_r = np.clip(test_r_raw, lo, hi)

        window_return = float(np.sum(window_r))  # total log return for regime classification
        window_vol = float(np.std(window_r))

        print(f"\n  Window {w_idx+1}/{n_windows} [bars {w_start}:{w_end}] "
              f"return={window_return:+.4f} vol={window_vol:.6f}")

        # Calibrate all models with equalized effort
        t0 = time.perf_counter()
        v12_params = calibrate_v12(train_r, dt, seed=42 + w_idx)
        gbm_p = calibrate_gbm(train_r, dt)
        heston_p = calibrate_heston(train_r, dt)
        merton_p = calibrate_merton(train_r, dt)
        sabr_p = calibrate_sabr(train_r, dt, S0=1.0)
        cal_time = time.perf_counter() - t0

        all_params = {
            "v12": v12_params, "gbm": gbm_p, "heston": heston_p,
            "merton": merton_p, "sabr": sabr_p,
        }

        # OOS evaluation
        window_losses = {}
        for tag in model_tags:
            losses = compute_oos_loss(tag, all_params[tag], test_r, dt,
                                      n_seeds=N_SEEDS, base_seed=42 + w_idx * 1000)
            window_losses[tag] = losses
            all_median_losses[tag].append(float(np.median(losses)))

        # Window winner
        medians = {tag: np.median(window_losses[tag]) for tag in model_tags}
        winner = min(medians, key=medians.get)

        loss_str = "  ".join(f"{model_names[t][:6]}={medians[t]:.1f}" for t in model_tags)
        print(f"    Calibrated in {cal_time:.0f}s | {loss_str} | Winner: {model_names[winner]}")

        window_results.append({
            "window_idx": w_idx,
            "bars": [w_start, w_end],
            "total_return": window_return,
            "volatility": window_vol,
            "median_losses": {tag: float(medians[tag]) for tag in model_tags},
            "winner": winner,
            "calibration_time_s": cal_time,
        })

    # ===== AGGREGATE RESULTS =====
    print("\n" + "=" * 70)
    print("  AGGREGATE RESULTS ACROSS ALL WINDOWS")
    print("=" * 70)

    wins = {tag: sum(1 for w in window_results if w["winner"] == tag) for tag in model_tags}
    avg_med = {tag: np.mean(all_median_losses[tag]) for tag in model_tags}
    std_med = {tag: np.std(all_median_losses[tag]) for tag in model_tags}

    print(f"\n  {'Model':<18s} {'Wins':>5s} {'Win%':>6s} {'AvgMedLoss':>11s} {'StdMedLoss':>11s} {'AvgRank':>8s}")
    print(f"  {'-'*18} {'-'*5} {'-'*6} {'-'*11} {'-'*11} {'-'*8}")

    # Compute average rank per model
    ranks = {tag: [] for tag in model_tags}
    for w in window_results:
        sorted_tags = sorted(model_tags, key=lambda t: w["median_losses"][t])
        for rank, tag in enumerate(sorted_tags, 1):
            ranks[tag].append(rank)
    avg_ranks = {tag: np.mean(ranks[tag]) for tag in model_tags}

    for tag in model_tags:
        print(f"  {model_names[tag]:<18s} {wins[tag]:>5d} {100*wins[tag]/n_windows:>5.1f}% "
              f"{avg_med[tag]:>11.2f} {std_med[tag]:>11.2f} {avg_ranks[tag]:>8.2f}")

    # ===== POOLED PAIRED TESTS (v1.2 vs each benchmark) =====
    print("\n" + "=" * 70)
    print("  POOLED PAIRED SIGNIFICANCE TESTS (v1.2 vs each)")
    print("=" * 70)

    v12_medians = np.array(all_median_losses["v12"])
    print(f"\n  {'Benchmark':<18s} {'MeanDiff':>9s} {'95% CI':>22s} {'Wilcoxon p':>11s} {'Effect r':>9s}")
    print(f"  {'-'*18} {'-'*9} {'-'*22} {'-'*11} {'-'*9}")

    for tag in model_tags:
        if tag == "v12":
            continue
        bench_medians = np.array(all_median_losses[tag])
        diffs = bench_medians - v12_medians  # positive = v1.2 wins

        mean_diff = float(np.mean(diffs))
        # Bootstrap CI on mean difference
        rng_bs = np.random.default_rng(42)
        boot_means = np.array([np.mean(rng_bs.choice(diffs, size=len(diffs), replace=True))
                               for _ in range(10000)])
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))

        # Wilcoxon signed-rank test
        try:
            w_stat, w_pval = sp_stats.wilcoxon(diffs, alternative='two-sided')
            # Effect size r = Z / sqrt(N)
            z_approx = sp_stats.norm.ppf(1 - w_pval / 2)
            effect_r = z_approx / np.sqrt(len(diffs))
        except ValueError:
            w_pval = float('nan')
            effect_r = float('nan')

        sig = " ***" if w_pval < 0.001 else (" **" if w_pval < 0.01 else (" *" if w_pval < 0.05 else ""))
        print(f"  {model_names[tag]:<18s} {mean_diff:>+9.2f} [{ci_lo:>+9.2f}, {ci_hi:>+9.2f}] "
              f"{w_pval:>11.4g} {effect_r:>9.3f}{sig}")

    # ===== REGIME ANALYSIS =====
    print("\n" + "=" * 70)
    print("  REGIME-CONDITIONAL RESULTS")
    print("=" * 70)

    # Classify windows
    returns_arr = np.array([w["total_return"] for w in window_results])
    vol_arr = np.array([w["volatility"] for w in window_results])
    vol_80 = np.percentile(vol_arr, 80)
    vol_20 = np.percentile(vol_arr, 20)

    for w in window_results:
        if w["total_return"] > 0.05:
            w["regime"] = "bull"
        elif w["total_return"] < -0.05:
            w["regime"] = "bear"
        else:
            w["regime"] = "sideways"
        if w["volatility"] > vol_80:
            w["vol_regime"] = "high_vol"
        elif w["volatility"] < vol_20:
            w["vol_regime"] = "low_vol"
        else:
            w["vol_regime"] = "normal_vol"

    for regime_key in ["regime", "vol_regime"]:
        regimes = sorted(set(w[regime_key] for w in window_results))
        for regime in regimes:
            rw = [w for w in window_results if w[regime_key] == regime]
            if len(rw) < 3:
                continue
            regime_wins = {tag: sum(1 for w in rw if w["winner"] == tag) for tag in model_tags}
            print(f"\n  {regime.upper()} ({len(rw)} windows):")
            for tag in model_tags:
                avg = np.mean([w["median_losses"][tag] for w in rw])
                print(f"    {model_names[tag]:<18s} wins={regime_wins[tag]:>2d} avg_med_loss={avg:.2f}")

    # ===== SAVE =====
    output = {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_windows": n_windows,
        "window_size": WINDOW_SIZE,
        "n_seeds": N_SEEDS,
        "total_returns": all_returns.size,
        "windows": window_results,
        "aggregate": {
            "wins": wins,
            "avg_median_loss": {tag: float(avg_med[tag]) for tag in model_tags},
            "avg_rank": {tag: float(avg_ranks[tag]) for tag in model_tags},
        },
    }
    out_path = Path(f"temporal_validation_{SYMBOL.lower()}.json")
    out_path.write_text(json.dumps(_to_jsonable(output), indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")

    # ===== PLOT =====
    if _HAS_PLT and n_windows > 1:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Box plot of median losses per model
        ax = axes[0, 0]
        data = [all_median_losses[tag] for tag in model_tags]
        bp = ax.boxplot(data, labels=[model_names[t][:8] for t in model_tags], patch_artist=True)
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Median OOS Loss")
        ax.set_title("Loss Distribution Across Windows", fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 2. Win rate bar chart
        ax = axes[0, 1]
        win_pcts = [100 * wins[tag] / n_windows for tag in model_tags]
        bars = ax.bar(range(len(model_tags)), win_pcts,
                      color=colors, tick_label=[model_names[t][:8] for t in model_tags])
        ax.set_ylabel("Win Rate (%)")
        ax.set_title("Win Rate Across Windows", fontweight="bold")
        for bar, pct in zip(bars, win_pcts):
            ax.text(bar.get_x() + bar.get_width()/2, pct + 1, f"{pct:.0f}%",
                    ha="center", fontsize=9)

        # 3. Median loss trajectory across windows
        ax = axes[1, 0]
        for i, tag in enumerate(model_tags):
            ax.plot(all_median_losses[tag], "o-", label=model_names[tag][:8],
                    color=colors[i], markersize=3, alpha=0.8)
        ax.set_xlabel("Window Index")
        ax.set_ylabel("Median OOS Loss")
        ax.set_title("Loss Trajectory Over Time", fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # 4. Average rank bar chart
        ax = axes[1, 1]
        avg_r = [avg_ranks[tag] for tag in model_tags]
        bars = ax.bar(range(len(model_tags)), avg_r,
                      color=colors, tick_label=[model_names[t][:8] for t in model_tags])
        ax.set_ylabel("Average Rank (1=best)")
        ax.set_title("Average Rank Across Windows", fontweight="bold")
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.3)
        for bar, r in zip(bars, avg_r):
            ax.text(bar.get_x() + bar.get_width()/2, r + 0.05, f"{r:.2f}",
                    ha="center", fontsize=9)

        fig.suptitle(f"Temporal Validation: Brulant v1.2 vs Benchmarks ({SYMBOL})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig_path = f"temporal_validation_{SYMBOL.lower()}.png"
        fig.savefig(fig_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
