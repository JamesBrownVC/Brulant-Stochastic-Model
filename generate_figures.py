"""
Generate all figures for the Brulant Model GitHub repo.
Produces: assets/benchmark_oos.png, assets/pricing_surface.png,
          assets/paths_comparison.png, assets/vol_dynamics.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime

from experiment_v12 import simulate_v12
from backtest_buffer_model import simulate_buffer_paths
from benchmark_comparison import simulate_gbm, simulate_heston, simulate_merton
from fit_sandpile import fetch_binance_log_returns, interval_to_dt_years, moment_vector
from backtest_buffer_model import MOMENT_NAMES

# Style
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "font.size": 10,
})

V12_PARAMS = dict(
    mu0=0.0, sigma0=0.500183, sigma0_bar=0.236662, alpha_s=14.393487,
    xi_s=1.576421, rho=2.917580, nu=1.0, alpha=10.291423,
    beta=0.0, lambda0=0.0, gamma=20.0, eta=1.0,
    phi=0.56, sigma_Y=0.01, eps=0.001,
    stoch_vol_target=True, multi_buffer=True,
    kappa_fast=95.670338, kappa_mid=15.0, kappa_slow=4.522279,
    theta_fast=1.786391, theta_mid=1.5, theta_slow=1.021288,
    w_fast=0.435199, w_mid=0.0, w_slow=0.564801,
)

V11_PARAMS = dict(
    mu0=0.0, sigma0=0.596377, rho=1.78402, nu=1.54849,
    alpha=9.90562, beta=0.128777, lambda0=1.18401,
    gamma=20.0, eta=1.0, kappa=15.0, theta_p=1.5,
    phi=0.560709, sigma_Y=0.045, eps=0.001,
)

COLORS = {
    "v12": "#58a6ff",
    "v11": "#f78166",
    "gbm": "#8b949e",
    "heston": "#7ee787",
    "merton": "#d2a8ff",
    "empirical": "#ffffff",
    "spot": "#ffd700",
}


def fig1_benchmark_oos():
    """Bar chart: OOS moment-match loss comparison."""
    print("  Figure 1: Benchmark OOS comparison...")

    models = ["Brulant\nv1.2", "GBM", "Heston", "Merton", "SABR", "Brulant\nv1.1"]
    medians = [4.08, 4.82, 4.89, 4.85, 5.47, 88.95]
    stds = [0.91, 1.25, 1.03, 5.54, 1.13, 212.88]
    colors = [COLORS["v12"], COLORS["gbm"], COLORS["heston"],
              COLORS["merton"], "#d2a8ff", COLORS["v11"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Median loss
    bars = ax1.bar(range(len(models)), medians, color=colors, edgecolor="#30363d", linewidth=0.5)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylabel("Median OOS Loss")
    ax1.set_title("Out-of-Sample Moment-Match Loss", fontweight="bold", fontsize=13)
    for bar, val in zip(bars, medians):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.1f}",
                ha="center", fontsize=9, color="#c9d1d9")
    ax1.set_ylim(0, max(medians[:5]) * 1.4)
    ax1.grid(True, alpha=0.3, axis="y")

    # Loss stability (std)
    bars2 = ax2.bar(range(len(models)), stds, color=colors, edgecolor="#30363d", linewidth=0.5)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, fontsize=9)
    ax2.set_ylabel("Loss Std (across seeds)")
    ax2.set_title("Calibration Stability (lower = more reliable)", fontweight="bold", fontsize=13)
    for bar, val in zip(bars2, stds):
        if val < 10:
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.2, f"{val:.2f}",
                    ha="center", fontsize=9, color="#c9d1d9")
    ax2.set_ylim(0, 8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Brulant v1.2 vs Industry Benchmarks", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("assets/benchmark_oos.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig2_pricing_surface(S0):
    """Digital option pricing surface: strikes x maturities."""
    print("  Figure 2: Pricing surface...")

    strikes = np.arange(60000, 82000, 2000, dtype=np.float64)
    hours_list = [6, 12, 24, 48, 72, 96]
    labels = ["6h", "12h", "1d", "2d", "3d", "4d"]

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.cool

    for i, (h, label) in enumerate(zip(hours_list, labels)):
        T = h / (24.0 * 365.0)
        n_steps = max(1, int(h * 60))
        dt = T / n_steps
        _, S_T = simulate_v12(n_steps, dt, 100000, seed=42 + i * 1000, S0=S0, **V12_PARAMS)
        prices = np.array([np.mean(S_T >= K) for K in strikes])
        color = cmap(i / (len(hours_list) - 1))
        ax.plot(strikes / 1000, prices, "o-", color=color, label=f"T={label}",
                markersize=4, linewidth=2)

    ax.axvline(S0 / 1000, color=COLORS["spot"], linestyle="--", linewidth=2,
               alpha=0.7, label=f"Spot ${S0/1000:.1f}k")
    ax.set_xlabel("Strike ($k)", fontsize=12)
    ax.set_ylabel("Digital Call Price  P(S_T ≥ K)", fontsize=12)
    ax.set_title("Brulant v1.2: Digital Option Term Structure", fontweight="bold", fontsize=14)
    ax.legend(fontsize=9, ncol=4)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig("assets/pricing_surface.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig3_paths_comparison(S0):
    """Side-by-side simulated paths: v1.2 vs v1.1."""
    print("  Figure 3: Path comparison...")

    n_paths = 50
    n_steps = 1440 * 7
    dt = 1.0 / (365.0 * 24.0 * 60.0)
    times = np.arange(n_steps + 1) / 1440.0

    lr12, _ = simulate_v12(n_steps, dt, n_paths, seed=42, S0=S0, **V12_PARAMS)
    paths12 = S0 * np.exp(np.cumsum(np.hstack((np.zeros((n_paths, 1)), lr12)), axis=1))

    lr11, _ = simulate_buffer_paths(n_steps, dt, n_paths, seed=42, S0=S0, **V11_PARAMS)
    paths11 = S0 * np.exp(np.cumsum(np.hstack((np.zeros((n_paths, 1)), lr11)), axis=1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for i in range(n_paths):
        ax1.plot(times, paths12[i, :], lw=0.8, alpha=0.5, color=COLORS["v12"])
    ax1.axhline(S0, color=COLORS["spot"], linestyle="--", linewidth=2, alpha=0.8)
    ax1.set_title("Brulant v1.2\n(Stochastic Vol + Multi-Buffer)", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Days")
    ax1.set_ylabel("BTC/USDT")
    ax1.grid(True, alpha=0.3)

    for i in range(n_paths):
        ax2.plot(times, paths11[i, :], lw=0.8, alpha=0.5, color=COLORS["v11"])
    ax2.axhline(S0, color=COLORS["spot"], linestyle="--", linewidth=2, alpha=0.8)
    ax2.set_title("Brulant v1.1\n(Jump-Diffusion + Sandpile)", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Days")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"50 Simulated 7-Day Paths (Spot = ${S0:,.0f})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("assets/paths_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig4_pricing_vs_benchmarks(S0):
    """Digital pricing curves: v1.2 vs benchmarks at +1d and +3d."""
    print("  Figure 4: Pricing vs benchmarks...")

    strikes = np.arange(60000, 82000, 2000, dtype=np.float64)

    # Calibrate benchmarks quickly
    returns_raw = fetch_binance_log_returns("BTCUSDT", "1m", 3000)
    dt = interval_to_dt_years("1m")
    mu = np.median(returns_raw)
    mad = np.percentile(np.abs(returns_raw - mu), 75) * 1.4826
    returns = np.clip(returns_raw, mu - 5 * mad, mu + 5 * mad)
    train_r = returns[:len(returns)//2]
    gbm_sigma = float(np.std(train_r) / np.sqrt(dt))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, hours, title in [(ax1, 24, "+1d (24h)"), (ax2, 72, "+3d (72h)")]:
        T = hours / (24.0 * 365.0)
        n_steps = int(hours * 60)
        step_dt = T / n_steps

        # v1.2
        _, S12 = simulate_v12(n_steps, step_dt, 100000, seed=42, S0=S0, **V12_PARAMS)
        p12 = [np.mean(S12 >= K) for K in strikes]

        # v1.1
        _, S11 = simulate_buffer_paths(n_steps, step_dt, 100000, seed=42, S0=S0, **V11_PARAMS)
        p11 = [np.mean(S11 >= K) for K in strikes]

        # GBM
        _, S_gbm = simulate_gbm(n_steps, step_dt, 100000, S0, gbm_sigma, seed=42)
        p_gbm = [np.mean(S_gbm >= K) for K in strikes]

        ax.plot(strikes/1000, p12, "o-", color=COLORS["v12"], label="Brulant v1.2", linewidth=2, markersize=4)
        ax.plot(strikes/1000, p11, "s--", color=COLORS["v11"], label="Brulant v1.1", linewidth=1.5, markersize=3, alpha=0.8)
        ax.plot(strikes/1000, p_gbm, "^:", color=COLORS["gbm"], label="GBM (BS)", linewidth=1.5, markersize=3, alpha=0.8)
        ax.axvline(S0/1000, color=COLORS["spot"], linestyle="--", linewidth=2, alpha=0.5, label="Spot")
        ax.set_xlabel("Strike ($k)", fontsize=11)
        ax.set_ylabel("Digital Price", fontsize=11)
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.02)

    fig.suptitle(f"Digital Call Pricing Comparison (Spot = ${S0:,.0f})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("assets/pricing_vs_benchmarks.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Generating figures for GitHub repo...")

    try:
        import requests
        S0 = float(requests.get("https://api.binance.com/api/v3/ticker/price",
                                params={"symbol": "BTCUSDT"}, timeout=10).json()["price"])
    except Exception:
        S0 = 70000.0
    print(f"  Spot: ${S0:,.2f}")

    import os
    os.makedirs("assets", exist_ok=True)

    fig1_benchmark_oos()
    fig2_pricing_surface(S0)
    fig3_paths_comparison(S0)
    fig4_pricing_vs_benchmarks(S0)

    print("\nAll figures saved to assets/")


if __name__ == "__main__":
    main()
