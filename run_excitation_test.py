"""
Focused test: v1.1-Excitation model vs all baselines.
Loads standard model params from previous benchmark, only calibrates the new model.
"""
from __future__ import annotations
import json, time, sys
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from multi_scale_benchmark import (
    fetch_data, structural_metrics, jumps_per_day, vol_clustering_ratio,
    aggregate_returns, evaluate_model_single_seed, compute_composite_loss,
    simulate_model, simulate_v11_excitation, calibrate_v11_excitation_cma,
    EVAL_FREQS,
)
from fit_sandpile import _to_jsonable, interval_to_dt_years
from scipy import stats as sp_stats
from benchmark_v12 import bootstrap_ci

N_SEEDS = 10
N_PATHS = 500

print("=" * 70)
print("  V1.1 EXCITATION BENCHMARK")
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
    jpd = jumps_per_day(test_data[freq], EVAL_FREQS[freq])
    print(f"    {freq:>4s}: kurt={em['kurtosis']:>6.2f}  tail3s={em['tail_3sig']:>5.1f}x  "
          f"acf|r|={em['abs_acf1']:>5.3f}  std={em['std']:.5f}  jumps/day={jpd:.1f}")

# --- Phase 2: Load baseline models from previous run ---
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

print(f"\n  Loaded 7 baseline models from previous benchmark")
for name in models:
    print(f"    {name}")

# --- Phase 3: Calibrate v1.1 Excitation with CMA-ES ---
print(f"\n{'='*70}")
print("  CALIBRATING v1.1 Excitation (CMA-ES)")
print(f"{'='*70}")

t0 = time.perf_counter()
v11e_p = calibrate_v11_excitation_cma(
    train_15m, dt_15m, num_paths=500, max_evals=800, seed=42, sigma0_scale=0.7)
v11e_sim = {k: v for k, v in v11e_p.items() if not k.startswith('_')}
elapsed = time.perf_counter() - t0
print(f"\n  v1.1 Excitation calibrated in {elapsed:.0f}s")
print(f"    sigma0={v11e_sim['sigma0']:.4f} (pre-scale: {v11e_p['_sigma0_prescale']:.4f})")
print(f"    sigma_Y={v11e_sim['sigma_Y']:.4f}")
print(f"    lambda0={v11e_sim['lambda0']:.2f}")
print(f"    exc_beta={v11e_sim['exc_beta']:.2f}")
print(f"    exc_kappa={v11e_sim['exc_kappa']:.1f} (half-life: {0.693/v11e_sim['exc_kappa']/dt_15m:.1f} 15m bars)")
models["v1.1 Excitation"] = ("v11_exc", v11e_sim)

# --- Phase 4: Evaluate all models ---
n_sim_15m = test_data.get("15m", np.array([])).size
if n_sim_15m < 10:
    n_sim_15m = 672

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
    acf15 = med_metrics.get("15m", {}).get("abs_acf1", 0)
    std15 = med_metrics.get("15m", {}).get("std", 0)
    pct = done * 100 // n_total
    print(f"  [{pct:>3d}%] {name:<18s}: loss={np.median(losses):>8.1f}  "
          f"kurt15m={k15:>6.2f} kurt1h={k1h:>6.2f}  "
          f"acf15m={acf15:>6.3f} std15m={std15:.5f}  ({elapsed:.0f}s)")

# --- Phase 5: Rankings ---
print(f"\n{'='*70}")
print("  FINAL RANKING")
print(f"{'='*70}")

ranked = sorted(all_losses.items(), key=lambda x: np.median(x[1]))
param_counts = {
    "GBM": 1, "Heston": 4, "Merton": 3, "SABR": 4,
    "Brulant v1.2": 11, "Brulant v1.1": 14,
    "v1.1 Uncapped": 9, "v1.1 Excitation": 11,
}

print(f"  {'Rank':>4} {'Model':<18s} {'Median':>8s} {'Params':>6s} "
      f"{'kurt15m':>8s} {'kurt1h':>8s} {'acf15m':>8s} {'std15m':>8s}")
print(f"  {'-'*4} {'-'*18} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

emp_m_15m = structural_metrics(test_data["15m"])
emp_m_1h = structural_metrics(test_data["1h"])

for rank, (name, losses) in enumerate(ranked, 1):
    fm = all_freq_metrics[name]
    k15 = fm.get("15m", {}).get("kurtosis", 0)
    k1h = fm.get("1h", {}).get("kurtosis", 0)
    acf15 = fm.get("15m", {}).get("abs_acf1", 0)
    std15 = fm.get("15m", {}).get("std", 0)
    print(f"  {rank:>4d} {name:<18s} {np.median(losses):>8.1f} "
          f"{param_counts.get(name, '?'):>6} "
          f"{k15:>8.2f} {k1h:>8.2f} {acf15:>8.3f} {std15:>8.5f}")

print(f"\n  {'':>4} {'Empirical':<18s} {'':>8s} {'':>6s} "
      f"{emp_m_15m['kurtosis']:>8.2f} {emp_m_1h['kurtosis']:>8.2f} "
      f"{emp_m_15m['abs_acf1']:>8.3f} {emp_m_15m['std']:>8.5f}")

# --- Kurtosis table ---
print(f"\n{'='*70}")
print("  KURTOSIS BY FREQUENCY")
print(f"{'='*70}")
freqs = [f for f in EVAL_FREQS if f in test_data]
header = f"  {'Model':<18s}" + "".join(f" {f:>8s}" for f in freqs)
print(header)
print(f"  {'-'*18}" + "".join(f" {'-'*8}" for _ in freqs))
for name, _ in ranked:
    fm = all_freq_metrics[name]
    row = f"  {name:<18s}"
    for freq in freqs:
        k = fm.get(freq, {}).get("kurtosis", float('nan'))
        row += f" {k:>8.2f}"
    print(row)
row = f"  {'Empirical':<18s}"
for freq in freqs:
    row += f" {structural_metrics(test_data[freq])['kurtosis']:>8.2f}"
print(row)

# --- Vol clustering table ---
print(f"\n{'='*70}")
print("  VOL CLUSTERING abs_acf1 BY FREQUENCY")
print(f"{'='*70}")
print(header)
print(f"  {'-'*18}" + "".join(f" {'-'*8}" for _ in freqs))
for name, _ in ranked:
    fm = all_freq_metrics[name]
    row = f"  {name:<18s}"
    for freq in freqs:
        v = fm.get(freq, {}).get("abs_acf1", float('nan'))
        row += f" {v:>8.4f}"
    print(row)
row = f"  {'Empirical':<18s}"
for freq in freqs:
    row += f" {structural_metrics(test_data[freq])['abs_acf1']:>8.4f}"
print(row)

# --- Std table ---
print(f"\n{'='*70}")
print("  STD (VOLATILITY) BY FREQUENCY")
print(f"{'='*70}")
print(header)
print(f"  {'-'*18}" + "".join(f" {'-'*8}" for _ in freqs))
for name, _ in ranked:
    fm = all_freq_metrics[name]
    row = f"  {name:<18s}"
    for freq in freqs:
        v = fm.get(freq, {}).get("std", float('nan'))
        row += f" {v:>8.5f}"
    print(row)
row = f"  {'Empirical':<18s}"
for freq in freqs:
    row += f" {structural_metrics(test_data[freq])['std']:>8.5f}"
print(row)

# --- Tail 3sig table ---
print(f"\n{'='*70}")
print("  TAIL RATIO (3-SIGMA) BY FREQUENCY")
print(f"{'='*70}")
print(header)
print(f"  {'-'*18}" + "".join(f" {'-'*8}" for _ in freqs))
for name, _ in ranked:
    fm = all_freq_metrics[name]
    row = f"  {name:<18s}"
    for freq in freqs:
        v = fm.get(freq, {}).get("tail_3sig", float('nan'))
        row += f" {v:>8.2f}"
    print(row)
row = f"  {'Empirical':<18s}"
for freq in freqs:
    row += f" {structural_metrics(test_data[freq])['tail_3sig']:>8.2f}"
print(row)

# --- Paired tests vs best ---
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
    "n_seeds": N_SEEDS,
    "n_paths": N_PATHS,
    "model_params": {name: {k: v for k, v in params.items() if not k.startswith('_')}
                     for name, (tag, params) in models.items()},
    "excitation_params_full": v11e_p,
    "model_losses": {name: losses.tolist() for name, losses in all_losses.items()},
    "model_freq_metrics": all_freq_metrics,
    "model_global_metrics": all_global_metrics,
    "empirical_freq_metrics": {
        freq: structural_metrics(test_data[freq]) for freq in freqs
    },
}
Path("excitation_benchmark.json").write_text(
    json.dumps(_to_jsonable(result), indent=2), encoding="utf-8")
print(f"\nSaved excitation_benchmark.json")
