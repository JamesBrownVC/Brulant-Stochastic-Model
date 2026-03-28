"""
Digital RMSE surface plot + bootstrap confidence bounds.
Uses pre-computed model & empirical prices from digital_bucket_benchmark.json.
No retraining — just analysis of existing results.
"""
from __future__ import annotations
import json, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# --- Load pre-computed results ---
bench = json.loads(Path("digital_bucket_benchmark.json").read_text(encoding="utf-8"))
MONEYNESS = np.array(bench["moneyness"])
EXPIRIES = bench["expiries_hours"]  # {"4h": 4, "1d": 24, "3d": 72, "7d": 168}
model_prices = bench["model_prices"]
emp_prices = bench["empirical_prices"]

exp_names = list(EXPIRIES.keys())
exp_hours = np.array([EXPIRIES[e] for e in exp_names])
models = [m for m in model_prices[exp_names[0]] if m != "SABR"]

# ============================================================================
#  1. BOOTSTRAP 95% CONFIDENCE BOUNDS ON RMSE (Brulant v1.2)
# ============================================================================
print("=" * 70)
print("  BOOTSTRAP 95% CI ON RMSE vs EMPIRICAL (10,000 resamples)")
print("  Using pre-computed prices from digital_bucket_benchmark.json")
print("=" * 70)

N_BOOT = 10_000
rng = np.random.default_rng(42)

# Build error matrix: (n_expiries, n_strikes) for each model
for name in models:
    errors_matrix = []
    for exp in exp_names:
        mp = np.array(model_prices[exp][name])
        ep = np.array(emp_prices[exp])
        errors_matrix.append(mp - ep)
    errors_matrix = np.array(errors_matrix)  # (4, 21)

    # Flatten all errors for overall RMSE bootstrap
    flat_errors = errors_matrix.ravel()
    n = len(flat_errors)

    boot_rmses = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boot_rmses.append(np.sqrt(np.mean(flat_errors[idx] ** 2)))
    boot_rmses = np.sort(boot_rmses)

    rmse_point = np.sqrt(np.mean(flat_errors ** 2))
    ci_lo = boot_rmses[int(0.025 * N_BOOT)]
    ci_hi = boot_rmses[int(0.975 * N_BOOT)]
    ci_95_worst = boot_rmses[int(0.95 * N_BOOT)]

    print(f"\n  {name:<14s}")
    print(f"    RMSE (point):    {rmse_point:.4f}")
    print(f"    95% CI:          [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"    95% worst case:  {ci_95_worst:.4f}")

# --- Per-bucket bootstrap for Brulant v1.2 ---
print(f"\n{'=' * 70}")
print("  BRULANT v1.2: PER-BUCKET 95% WORST-CASE RMSE")
print(f"{'=' * 70}")

BUCKETS = [
    ("ITM 6-10%",  0.90, 0.94),
    ("ITM 4-6%",   0.94, 0.96),
    ("ITM 2-4%",   0.96, 0.98),
    ("ITM 0-2%",   0.98, 1.00),
    ("ATM",        0.995, 1.005),
    ("OTM 0-2%",   1.00, 1.02),
    ("OTM 2-4%",   1.02, 1.04),
    ("OTM 4-6%",   1.04, 1.06),
    ("OTM 6-10%",  1.06, 1.10),
]

name = "Brulant v1.2"
print(f"\n  {'Bucket':<14s} {'RMSE':>8s} {'95% CI':>18s} {'95% worst':>10s}")
print(f"  {'-'*14} {'-'*8} {'-'*18} {'-'*10}")

for bkt_name, bkt_lo, bkt_hi in BUCKETS:
    mask = (MONEYNESS >= bkt_lo) & (MONEYNESS <= bkt_hi)
    if not np.any(mask):
        continue

    # Collect errors across all expiries for this bucket
    bucket_errors = []
    for exp in exp_names:
        mp = np.array(model_prices[exp][name])
        ep = np.array(emp_prices[exp])
        bucket_errors.extend((mp - ep)[mask])
    bucket_errors = np.array(bucket_errors)
    n = len(bucket_errors)

    boot_rmses = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boot_rmses.append(np.sqrt(np.mean(bucket_errors[idx] ** 2)))
    boot_rmses = np.sort(boot_rmses)

    rmse_pt = np.sqrt(np.mean(bucket_errors ** 2))
    ci_lo = boot_rmses[int(0.025 * N_BOOT)]
    ci_hi = boot_rmses[int(0.975 * N_BOOT)]
    worst_95 = boot_rmses[int(0.95 * N_BOOT)]

    print(f"  {bkt_name:<14s} {rmse_pt:>8.4f} [{ci_lo:>7.4f}, {ci_hi:>7.4f}] {worst_95:>10.4f}")


# ============================================================================
#  2. 3D RMSE SURFACE PLOT
# ============================================================================
print(f"\n{'=' * 70}")
print("  Generating 3D RMSE surface plots...")
print(f"{'=' * 70}")

# Build RMSE grid: (n_expiries, n_strikes) per model
# Each cell = |model_price - empirical_price| at that (expiry, strike)
rmse_grids = {}
for name in models:
    grid = np.zeros((len(exp_names), len(MONEYNESS)))
    for i, exp in enumerate(exp_names):
        mp = np.array(model_prices[exp][name])
        ep = np.array(emp_prices[exp])
        grid[i, :] = np.abs(mp - ep)  # absolute error at each point
    rmse_grids[name] = grid

# Interpolate to smooth surface
from scipy.interpolate import RectBivariateSpline

# Fine grid for interpolation
hours_fine = np.linspace(exp_hours.min(), exp_hours.max(), 80)
money_fine = np.linspace(MONEYNESS.min(), MONEYNESS.max(), 80)
H_fine, M_fine = np.meshgrid(hours_fine, money_fine, indexing='ij')

# --- Plot: All models side by side ---
fig = plt.figure(figsize=(18, 10))
fig.suptitle("Digital Option Pricing Error vs Empirical (Out-of-Sample)\n"
             "|Model Price − Empirical P(S_T≥K)|  •  Interpolated Surface",
             fontsize=13, fontweight='bold')

plot_models = ["GBM", "Heston", "Merton", "Brulant v1.2", "Exc-A", "Exc (opt)"]
vmax = 0.20  # shared color scale

for idx, name in enumerate(plot_models):
    ax = fig.add_subplot(2, 3, idx + 1, projection='3d')

    grid = rmse_grids[name]
    # Interpolate
    spline = RectBivariateSpline(exp_hours, MONEYNESS, grid, kx=3, ky=3)
    Z_fine = spline(hours_fine, money_fine)
    Z_fine = np.clip(Z_fine, 0, None)

    surf = ax.plot_surface(H_fine, M_fine, Z_fine,
                           cmap=cm.RdYlGn_r, alpha=0.85,
                           vmin=0, vmax=vmax,
                           edgecolor='none')

    ax.set_xlabel('Hours', fontsize=8)
    ax.set_ylabel('K/S', fontsize=8)
    ax.set_zlabel('|Error|', fontsize=8)
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_zlim(0, vmax)
    ax.view_init(elev=25, azim=-60)
    ax.tick_params(labelsize=7)

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
fig.colorbar(surf, ax=fig.axes, shrink=0.4, aspect=20, label='|Price Error|',
             pad=0.1)
plt.savefig("digital_rmse_surface_all.png", dpi=150, bbox_inches='tight')
print("  Saved digital_rmse_surface_all.png")

# --- Focused Brulant v1.2 plot with contours ---
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Brulant v1.2 — Digital Pricing Error vs Empirical (OOS)",
              fontsize=13, fontweight='bold')

grid_b = rmse_grids["Brulant v1.2"]
spline_b = RectBivariateSpline(exp_hours, MONEYNESS, grid_b, kx=3, ky=3)
Z_b = np.clip(spline_b(hours_fine, money_fine), 0, None)

# Left: 3D surface
ax3d = fig2.add_subplot(1, 2, 1, projection='3d')
surf_b = ax3d.plot_surface(H_fine, M_fine, Z_b,
                           cmap=cm.RdYlGn_r, alpha=0.85,
                           vmin=0, vmax=0.12,
                           edgecolor='none')
ax3d.set_xlabel('Hours to Expiry')
ax3d.set_ylabel('K/S (Moneyness)')
ax3d.set_zlabel('|Price Error|')
ax3d.set_title('3D Surface')
ax3d.set_zlim(0, 0.12)
ax3d.view_init(elev=25, azim=-55)
fig2.colorbar(surf_b, ax=ax3d, shrink=0.6, label='|Error|')

# Right: Contour heatmap (top-down view)
ax2d = axes[1]
cf = ax2d.contourf(M_fine, H_fine, Z_b,
                   levels=np.linspace(0, 0.10, 21),
                   cmap=cm.RdYlGn_r)
ax2d.contour(M_fine, H_fine, Z_b,
             levels=[0.01, 0.02, 0.03, 0.05, 0.08],
             colors='black', linewidths=0.5, linestyles='--')
ax2d.set_xlabel('K/S (Moneyness)')
ax2d.set_ylabel('Hours to Expiry')
ax2d.set_title('Contour Map (iso-error lines)')
fig2.colorbar(cf, ax=ax2d, label='|Price Error|')

# Mark actual data points
for h in exp_hours:
    for m in MONEYNESS:
        ax2d.plot(m, h, 'k.', markersize=1.5)

plt.tight_layout()
plt.savefig("digital_rmse_surface_brulant.png", dpi=150, bbox_inches='tight')
print("  Saved digital_rmse_surface_brulant.png")

plt.close('all')
print("\nDone.")
