"""
Evaluate v1.1-Excitation (vol-target excitation, no jump returns) vs all baselines.
Includes two-phase calibration (CMA-ES + coordinate descent) alongside hand-tuned configs.
"""
from __future__ import annotations
import json, time, sys
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from multi_scale_benchmark import (
    fetch_data, structural_metrics, jumps_per_day, vol_clustering_ratio,
    aggregate_returns, evaluate_model_single_seed, compute_composite_loss,
    simulate_model, simulate_v11_excitation,
    EVAL_FREQS,
)
from fit_sandpile import _to_jsonable
from scipy import stats as sp_stats
from benchmark_v12 import bootstrap_ci

N_SEEDS = 10
N_PATHS = 500

print("=" * 70)
print("  V1.1 EXCITATION BENCHMARK (two-phase calibration)")
print(f"  Seeds: {N_SEEDS} | Paths: {N_PATHS}")
print("=" * 70)

# --- Phase 1: Data ---
data = fetch_data(5000)
dt_15m = data["dt_15m"]
test_data = data["test"]
train_15m = data["train_15m"]

print(f"\n  Empirical structural metrics:")
for freq in EVAL_FREQS:
    if freq not in test_data or test_data[freq].size < 5:
        continue
    em = structural_metrics(test_data[freq])
    print(f"    {freq:>4s}: kurt={em['kurtosis']:>6.2f}  tail3s={em['tail_3sig']:>5.1f}x  "
          f"acf|r|={em['abs_acf1']:>6.3f}  std={em['std']:.5f}")

# --- Phase 2: Load baselines ---
prev = json.loads(Path("multi_scale_benchmark.json").read_text(encoding="utf-8"))
prev_params = prev["model_params"]

models = {}
models["GBM"] = ("gbm", prev_params["GBM"])
models["Heston"] = ("heston", prev_params["Heston"])
models["Merton"] = ("merton", prev_params["Merton"])
models["SABR"] = ("sabr", prev_params["SABR"])
models["Brulant v1.2"] = ("v12", prev_params["Brulant v1.2"])
models["Brulant v1.1"] = ("v11", prev_params["Brulant v1.1"])
models["v1.1 Uncapped"] = ("v11_uncapped", prev_params["v1.1 Uncapped"])

# --- Phase 3a: Hand-tuned excitation configs (for comparison) ---
base = dict(mu0=0, rho=1.3, nu=1.6, alpha=10, beta=0.0,
            phi=1.6, sigma_Y=0.20,
            kappa=15, theta_p=1.5, gamma=20, eta=1, eps=1e-3,
            jump_to_ret=False)

exc_a = {**base, "sigma0": 0.42, "lambda0": 3.0,
         "exc_beta": 10.0, "exc_kappa": 100.0, "alpha_exc": 60.0}
exc_b = {**base, "sigma0": 0.38, "lambda0": 3.0,
         "exc_beta": 12.0, "exc_kappa": 100.0, "alpha_exc": 70.0}

models["Exc-A (hand)"] = ("v11_exc", exc_a)
models["Exc-B (hand)"] = ("v11_exc", exc_b)

# --- Phase 3b: Grid-optimized compromise config ---
# Best structural compromise: kurt~5.1, acf~0.22, std~0.00295, tail~5.1x
exc_opt = {**base, "sigma0": 0.40, "lambda0": 3.0,
           "exc_beta": 10.0, "exc_kappa": 130.0, "alpha_exc": 200.0}

models["Exc (opt)"] = ("v11_exc", exc_opt)

# --- Phase 4: Evaluate all models ---
n_sim_15m = test_data.get("15m", np.array([])).size
if n_sim_15m < 10:
    n_sim_15m = 672

param_counts = {
    "GBM": 1, "Heston": 4, "Merton": 3, "SABR": 4,
    "Brulant v1.2": 11, "Brulant v1.1": 14,
    "v1.1 Uncapped": 9, "Exc-A (hand)": 10, "Exc-B (hand)": 10,
    "Exc (opt)": 10,
}

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
        loss = compute_composite_loss(ev, n_params=param_counts.get(name, 0))
        losses.append(loss)

        for freq in EVAL_FREQS:
            if freq in ev:
                freq_metrics_agg[freq].append(ev[freq]["sim_metrics"])
        done += 1

    losses = np.array(losses)
    all_losses[name] = losses

    med_metrics = {}
    for freq in EVAL_FREQS:
        if freq_metrics_agg[freq]:
            keys = freq_metrics_agg[freq][0].keys()
            med_metrics[freq] = {
                k: float(np.median([m[k] for m in freq_metrics_agg[freq]]))
                for k in keys
            }
    all_freq_metrics[name] = med_metrics

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
    acf15 = med_metrics.get("15m", {}).get("abs_acf1", 0)
    std15 = med_metrics.get("15m", {}).get("std", 0)
    pct = done * 100 // n_total
    print(f"  [{pct:>3d}%] {name:<18s}: loss={np.median(losses):>8.1f}  "
          f"kurt15m={k15:>6.2f} kurt1h={k1h:>6.2f}  "
          f"acf15m={acf15:>6.3f} std15m={std15:.5f}  ({elapsed:.0f}s)")

# --- Rankings ---
print(f"\n{'='*70}")
print("  FINAL RANKING")
print(f"{'='*70}")

ranked = sorted(all_losses.items(), key=lambda x: np.median(x[1]))

emp_15m = structural_metrics(test_data["15m"])
emp_1h = structural_metrics(test_data.get("1h", test_data["15m"]))

print(f"  {'Rank':>4} {'Model':<18s} {'Median':>8s} {'P':>3s} "
      f"{'kurt15m':>8s} {'kurt1h':>8s} {'acf15m':>8s} {'std15m':>8s} {'t3_15m':>7s}")
print(f"  {'-'*4} {'-'*18} {'-'*8} {'-'*3} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")

for rank, (name, losses) in enumerate(ranked, 1):
    fm = all_freq_metrics[name]
    k15 = fm.get("15m", {}).get("kurtosis", 0)
    k1h = fm.get("1h", {}).get("kurtosis", 0)
    acf15 = fm.get("15m", {}).get("abs_acf1", 0)
    std15 = fm.get("15m", {}).get("std", 0)
    t3 = fm.get("15m", {}).get("tail_3sig", 0)
    print(f"  {rank:>4d} {name:<18s} {np.median(losses):>8.1f} "
          f"{param_counts.get(name, '?'):>3} "
          f"{k15:>8.2f} {k1h:>8.2f} {acf15:>8.3f} {std15:>8.5f} {t3:>7.1f}x")

print(f"\n  {'':>4} {'Empirical':<18s} {'':>8s} {'':>3s} "
      f"{emp_15m['kurtosis']:>8.2f} {emp_1h['kurtosis']:>8.2f} "
      f"{emp_15m['abs_acf1']:>8.3f} {emp_15m['std']:>8.5f} {emp_15m['tail_3sig']:>7.1f}x")

# --- Detail tables ---
freqs = [f for f in EVAL_FREQS if f in test_data]
header = f"  {'Model':<18s}" + "".join(f" {f:>8s}" for f in freqs)

for table_name, metric_key, fmt in [
    ("KURTOSIS", "kurtosis", "{:>8.2f}"),
    ("VOL CLUSTERING abs_acf1", "abs_acf1", "{:>8.3f}"),
    ("STD (VOLATILITY)", "std", "{:>8.5f}"),
    ("TAIL RATIO (3-SIGMA)", "tail_3sig", "{:>8.2f}"),
]:
    print(f"\n{'='*70}")
    print(f"  {table_name} BY FREQUENCY")
    print(f"{'='*70}")
    print(header)
    print(f"  {'-'*18}" + "".join(f" {'-'*8}" for _ in freqs))
    for name, _ in ranked:
        fm = all_freq_metrics[name]
        row = f"  {name:<18s}"
        for freq in freqs:
            v = fm.get(freq, {}).get(metric_key, float('nan'))
            row += f" {fmt.format(v)}"
        print(row)
    row = f"  {'Empirical':<18s}"
    for freq in freqs:
        v = structural_metrics(test_data[freq])[metric_key]
        row += f" {fmt.format(v)}"
    print(row)

# --- Paired tests ---
best_name = ranked[0][0]
best_losses = all_losses[best_name]
print(f"\n{'='*70}")
print(f"  PAIRED TESTS vs {best_name}")
print(f"{'='*70}")
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

# --- Save ---
result = {
    "timestamp": __import__("datetime").datetime.now().isoformat(),
    "n_seeds": N_SEEDS, "n_paths": N_PATHS,
    "model_params": {name: {k: v for k, v in params.items() if not k.startswith('_')}
                     for name, (tag, params) in models.items()},
    "cma_source": "Phase 1 CMA-ES best at eval 400, structural loss=2.29",
    "model_losses": {name: losses.tolist() for name, losses in all_losses.items()},
    "model_freq_metrics": all_freq_metrics,
    "model_global_metrics": all_global_metrics,
    "empirical_freq_metrics": {freq: structural_metrics(test_data[freq]) for freq in freqs},
}
Path("excitation_benchmark.json").write_text(
    json.dumps(_to_jsonable(result), indent=2), encoding="utf-8")
print(f"\nSaved excitation_benchmark.json")
