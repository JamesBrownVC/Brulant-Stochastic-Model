"""
Brulant v1.2 (no-jump stoch-vol + 2-layer buffer) vs benchmarks.
Digital option pricing comparison using MC.
Includes paired significance tests (Wilcoxon, Diebold-Mariano, bootstrap CI).
"""
import numpy as np
import time
import datetime
from scipy import stats as sp_stats
from experiment_v12 import simulate_v12
from benchmark_comparison import (
    simulate_gbm, calibrate_gbm,
    simulate_heston, calibrate_heston,
    simulate_merton, calibrate_merton,
    simulate_sabr, calibrate_sabr,
)
from fit_sandpile import fetch_binance_log_returns, interval_to_dt_years, moment_vector
from backtest_buffer_model import simulate_buffer_paths, MOMENT_NAMES


def diebold_mariano_test(losses_1, losses_2):
    """Diebold-Mariano test for equal predictive accuracy.
    H0: E[d_t] = 0 where d_t = L1_t - L2_t.
    Returns (DM statistic, two-sided p-value).
    """
    d = np.array(losses_1) - np.array(losses_2)
    n = len(d)
    d_bar = np.mean(d)
    # HAC variance (Newey-West with 0 lags for independent seeds)
    var_d = np.var(d, ddof=1) / n
    if var_d < 1e-20:
        return 0.0, 1.0
    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2.0 * (1.0 - sp_stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def bootstrap_ci(data, n_bootstrap=10000, ci=0.95, seed=42):
    """Bootstrap percentile confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means[i] = np.mean(sample)
    alpha = (1 - ci) / 2
    return float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))

# Fitted params from the DE calibration
BRULANT_V12 = dict(
    mu0=0.0, sigma0=0.500183, sigma0_bar=0.236662, alpha_s=14.393487,
    xi_s=1.576421, rho=2.917580, nu=1.0, alpha=10.291423,
    beta=0.0, lambda0=0.0, gamma=20.0, eta=1.0,
    phi=0.56, sigma_Y=0.01, eps=0.001,
    stoch_vol_target=True, multi_buffer=True,
    kappa_fast=95.670338, kappa_mid=15.0, kappa_slow=4.522279,
    theta_fast=1.786391, theta_mid=1.5, theta_slow=1.021288,
    w_fast=0.435199, w_mid=0.0, w_slow=0.564801,
)

BRULANT_V11 = dict(
    mu0=0.0, sigma0=0.596377, rho=1.78402, nu=1.54849,
    alpha=9.90562, beta=0.128777, lambda0=1.18401,
    gamma=20.0, eta=1.0, kappa=15.0, theta_p=1.5,
    phi=0.560709, sigma_Y=0.045, eps=0.001,
)


def price_v12(S0, strikes, hours, num_paths=100000, seed=42):
    T = hours / (24.0 * 365.0)
    n_steps = max(1, int(hours * 60))
    dt = T / n_steps
    _, S_T = simulate_v12(n_steps, dt, num_paths, seed=seed, S0=S0, **BRULANT_V12)
    strikes = np.asarray(strikes, dtype=np.float64)
    prices = np.array([np.mean(S_T >= K) for K in strikes])
    se = np.array([np.std((S_T >= K).astype(float)) / np.sqrt(num_paths) for K in strikes])
    return prices, se


def price_v11(S0, strikes, hours, num_paths=100000, seed=42):
    T = hours / (24.0 * 365.0)
    n_steps = max(1, int(hours * 60))
    dt = T / n_steps
    _, S_T = simulate_buffer_paths(n_steps, dt, num_paths, seed=seed, S0=S0, **BRULANT_V11)
    strikes = np.asarray(strikes, dtype=np.float64)
    prices = np.array([np.mean(S_T >= K) for K in strikes])
    se = np.array([np.std((S_T >= K).astype(float)) / np.sqrt(num_paths) for K in strikes])
    return prices, se


def price_model(sim_func, S0, strikes, hours, params, num_paths=100000, seed=42):
    T = hours / (24.0 * 365.0)
    n_steps = max(1, int(hours * 60))
    dt = T / n_steps
    _, S_T = sim_func(n_steps, dt, num_paths, S0, **params, seed=seed)
    strikes = np.asarray(strikes, dtype=np.float64)
    prices = np.array([np.mean(S_T >= K) for K in strikes])
    se = np.array([np.std((S_T >= K).astype(float)) / np.sqrt(num_paths) for K in strikes])
    return prices, se


def main():
    print("=" * 70)
    print("  DIGITAL OPTION PRICING: Brulant v1.2 vs v1.1 vs Benchmarks")
    print("=" * 70)

    # Fetch data for benchmark calibration
    print("\nFetching data for benchmark calibration...")
    returns_raw = fetch_binance_log_returns("BTCUSDT", "1m", 5000)
    dt = interval_to_dt_years("1m")

    # Split BEFORE winsorization; thresholds from train only (no test leakage)
    n_split = int(len(returns_raw) * 0.5)
    train_r_raw = returns_raw[:n_split]
    test_r_raw = returns_raw[n_split:]
    mu = np.median(train_r_raw)
    mad = np.percentile(np.abs(train_r_raw - mu), 75) * 1.4826
    train_r = np.clip(train_r_raw, mu - 5 * mad, mu + 5 * mad)
    test_r = np.clip(test_r_raw, mu - 5 * mad, mu + 5 * mad)
    print(f"  Winsorization: train-derived thresholds [{mu - 5*mad:.8f}, {mu + 5*mad:.8f}]")

    try:
        import requests
        S0 = float(requests.get("https://api.binance.com/api/v3/ticker/price",
                                params={"symbol": "BTCUSDT"}, timeout=10).json()["price"])
    except Exception:
        S0 = 70000.0

    emp = moment_vector(test_r, w=None, acf_recent_bars=300)
    emp_std = np.std(test_r)
    emp_3sig = np.mean(np.abs(test_r) > 3 * emp_std)
    scales = np.maximum(np.abs(emp), np.array([1e-12, 1e-12, 0.5, 1.0, 0.05, 0.1]))
    scales = np.maximum(scales, 1e-9)

    print(f"  Spot: ${S0:,.2f}")
    print(f"  Empirical: kurt={emp[3]:.2f} skew={emp[2]:.3f} std={emp_std:.8f}")

    # Calibrate benchmarks
    print("\nCalibrating benchmarks...")
    gbm_p = calibrate_gbm(train_r, dt)
    print(f"  GBM: sigma={gbm_p['sigma']:.4f}")

    heston_p = calibrate_heston(train_r, dt)
    print(f"  Heston: kappa={heston_p['kappa']:.2f} theta={heston_p['theta']:.4f} xi={heston_p['xi']:.4f}")

    merton_p = calibrate_merton(train_r, dt)
    print(f"  Merton: lam={merton_p['lam']:.4f} jsig={merton_p['jump_sigma']:.4f}")

    sabr_p = calibrate_sabr(train_r, dt, S0=1.0)
    print(f"  SABR: alpha={sabr_p['alpha_s']:.4f} nu={sabr_p['nu_s']:.4f}")

    # OOS moment match
    N_SEEDS = 200
    print("\n" + "=" * 70)
    print(f"  OOS MOMENT MATCH ({N_SEEDS} seeds, frozen params)")
    print("=" * 70)

    model_configs = {
        "Brulant v1.2": ("v12", None),
        "Brulant v1.1": ("v11", None),
        "GBM": ("gbm", gbm_p),
        "Heston": ("heston", heston_p),
        "Merton": ("merton", merton_p),
        "SABR": ("sabr", sabr_p),
    }

    model_results = {}
    model_loss_arrays = {}  # for paired tests
    for name, (tag, params) in model_configs.items():
        t0 = time.perf_counter()
        losses = []
        for i in range(N_SEEDS):
            seed = 42 + i * 77
            if tag == "v12":
                lr, _ = simulate_v12(test_r.size, dt, 1000, seed=seed, S0=1.0, **BRULANT_V12)
            elif tag == "v11":
                lr, _ = simulate_buffer_paths(test_r.size, dt, 1000, seed=seed, S0=1.0, **BRULANT_V11)
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

        # Distributional stats from one big run
        if tag == "v12":
            lr, _ = simulate_v12(test_r.size, dt, 5000, seed=42, S0=1.0, **BRULANT_V12)
        elif tag == "v11":
            lr, _ = simulate_buffer_paths(test_r.size, dt, 5000, seed=42, S0=1.0, **BRULANT_V11)
        elif tag == "gbm":
            lr, _ = simulate_gbm(test_r.size, dt, 5000, 1.0, params["sigma"], seed=42)
        elif tag == "heston":
            lr, _ = simulate_heston(test_r.size, dt, 5000, 1.0, **params, seed=42)
        elif tag == "merton":
            lr, _ = simulate_merton(test_r.size, dt, 5000, 1.0, **params, seed=42)
        elif tag == "sabr":
            lr, _ = simulate_sabr(test_r.size, dt, 5000, 1.0, **params, seed=42)

        sp = lr.ravel()[:50000]
        sm = moment_vector(sp, w=None, acf_recent_bars=300)
        losses = np.array(losses)
        model_loss_arrays[name] = losses

        # Bootstrap SE of median
        rng_bs = np.random.default_rng(42)
        boot_medians = np.array([np.median(rng_bs.choice(losses, size=len(losses), replace=True))
                                 for _ in range(10000)])
        se_median = float(np.std(boot_medians))

        model_results[name] = {
            "med": float(np.median(losses)),
            "mean": float(np.mean(losses)),
            "std": float(np.std(losses)),
            "se_median": se_median,
            "std_ratio": float(np.std(sp) / emp_std),
            "kurt": float(sm[3]),
            "skew": float(sm[2]),
            "tail3": float(np.mean(np.abs(sp) > 3 * np.std(sp)) / max(emp_3sig, 1e-12)),
        }
        elapsed = time.perf_counter() - t0
        print(f"  {name}: median={np.median(losses):.2f} SE(med)={se_median:.3f} ({elapsed:.1f}s)")

    print(f"\n  {'Model':<18s} {'MedLoss':>8s} {'SE(med)':>8s} {'MeanLoss':>9s} {'LossStd':>8s} {'StdR':>6s} {'Kurt':>7s} {'Skew':>7s} {'3sig':>6s}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*6} {'-'*7} {'-'*7} {'-'*6}")
    for name, r in model_results.items():
        print(f"  {name:<18s} {r['med']:>8.2f} {r['se_median']:>8.3f} {r['mean']:>9.2f} {r['std']:>8.2f} "
              f"{r['std_ratio']:>6.3f} {r['kurt']:>7.1f} {r['skew']:>7.3f} {r['tail3']:>6.3f}")
    print(f"  {'EMPIRICAL':<18s} {'---':>8s} {'---':>8s} {'---':>9s} {'---':>8s} {'1.000':>6s} {emp[3]:>7.1f} {emp[2]:>7.3f} {'1.000':>6s}")

    # ===== PAIRED SIGNIFICANCE TESTS (v1.2 vs each benchmark) =====
    print("\n" + "=" * 70)
    print("  PAIRED SIGNIFICANCE TESTS (Brulant v1.2 vs each model)")
    print("=" * 70)
    v12_losses = model_loss_arrays["Brulant v1.2"]
    print(f"\n  {'Benchmark':<18s} {'MedDiff':>8s} {'95% CI':>22s} {'Wilcoxon p':>11s} {'DM stat':>8s} {'DM p':>8s}")
    print(f"  {'-'*18} {'-'*8} {'-'*22} {'-'*11} {'-'*8} {'-'*8}")
    for name in model_configs:
        if name == "Brulant v1.2":
            continue
        bench_losses = model_loss_arrays[name]
        # Paired differences: positive means benchmark is worse (v1.2 wins)
        diffs = bench_losses - v12_losses
        med_diff = float(np.median(diffs))
        ci_lo, ci_hi = bootstrap_ci(diffs, n_bootstrap=10000, ci=0.95, seed=42)
        # Wilcoxon signed-rank test (H0: median difference = 0)
        try:
            w_stat, w_pval = sp_stats.wilcoxon(diffs, alternative='two-sided')
        except ValueError:
            w_stat, w_pval = float('nan'), float('nan')
        dm_stat, dm_pval = diebold_mariano_test(bench_losses, v12_losses)
        sig = " ***" if w_pval < 0.001 else (" **" if w_pval < 0.01 else (" *" if w_pval < 0.05 else ""))
        print(f"  {name:<18s} {med_diff:>+8.2f} [{ci_lo:>+9.2f}, {ci_hi:>+9.2f}] {w_pval:>11.4g} {dm_stat:>+8.2f} {dm_pval:>8.4g}{sig}")

    # ===== DIGITAL OPTION PRICING =====
    print("\n" + "=" * 70)
    print("  DIGITAL OPTION PRICING GRID")
    print("=" * 70)

    strikes = np.arange(60000, 82000, 2000, dtype=np.float64)
    cet = datetime.timezone(datetime.timedelta(hours=1))
    now_cet = datetime.datetime.now(datetime.timezone.utc).astimezone(cet)
    print(f"  Spot: ${S0:,.2f} | {now_cet.strftime('%Y-%m-%d %H:%M')} CET | 100k paths")

    for k_day in [1, 3]:
        target_date = now_cet.date() + datetime.timedelta(days=k_day)
        target_dt = datetime.datetime(target_date.year, target_date.month, target_date.day,
                                      17, 0, 0, tzinfo=cet)
        hours = max((target_dt - now_cet).total_seconds() / 3600.0, 0.01)
        T = hours / (24.0 * 365.0)
        n_steps = max(1, int(hours * 60))
        step_dt = T / n_steps

        print(f"\n  +{k_day}d (17:00 CET {target_date}, {hours:.1f}h)")

        all_prices = {}

        # Brulant v1.2
        t0 = time.perf_counter()
        p12, _ = price_v12(S0, strikes, hours, 100000, seed=42 + k_day * 1000)
        all_prices["v1.2"] = p12
        print(f"    v1.2 done ({time.perf_counter()-t0:.1f}s)")

        # Brulant v1.1
        t0 = time.perf_counter()
        p11, _ = price_v11(S0, strikes, hours, 100000, seed=42 + k_day * 1000)
        all_prices["v1.1"] = p11
        print(f"    v1.1 done ({time.perf_counter()-t0:.1f}s)")

        # GBM
        t0 = time.perf_counter()
        p_gbm, _ = price_model(
            lambda ns, d, np_, s0, sigma, seed: simulate_gbm(ns, d, np_, s0, sigma, seed=seed),
            S0, strikes, hours, {"sigma": gbm_p["sigma"]}, 100000, seed=42 + k_day * 1000)
        all_prices["GBM"] = p_gbm
        print(f"    GBM done ({time.perf_counter()-t0:.1f}s)")

        # Heston
        t0 = time.perf_counter()
        p_hes, _ = price_model(
            lambda ns, d, np_, s0, v0, kappa, theta, xi, rho_h, seed: simulate_heston(ns, d, np_, s0, v0, kappa, theta, xi, rho_h, seed=seed),
            S0, strikes, hours, heston_p, 100000, seed=42 + k_day * 1000)
        all_prices["Heston"] = p_hes
        print(f"    Heston done ({time.perf_counter()-t0:.1f}s)")

        # Merton
        t0 = time.perf_counter()
        p_mer, _ = price_model(
            lambda ns, d, np_, s0, sigma, lam, jump_mu, jump_sigma, seed: simulate_merton(ns, d, np_, s0, sigma, lam, jump_mu, jump_sigma, seed=seed),
            S0, strikes, hours, merton_p, 100000, seed=42 + k_day * 1000)
        all_prices["Merton"] = p_mer
        print(f"    Merton done ({time.perf_counter()-t0:.1f}s)")

        # Print table
        print(f"\n    {'Strike':>10s} {'v1.2':>10s} {'v1.1':>10s} {'GBM':>10s} {'Heston':>10s} {'Merton':>10s}")
        print(f"    {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for j, K in enumerate(strikes):
            tag = "ITM" if K < S0 - 1000 else ("ATM" if abs(K - S0) < 1000 else "OTM")
            print(f"    ${int(K):>8,} {all_prices['v1.2'][j]:>10.6f} {all_prices['v1.1'][j]:>10.6f} "
                  f"{all_prices['GBM'][j]:>10.6f} {all_prices['Heston'][j]:>10.6f} "
                  f"{all_prices['Merton'][j]:>10.6f}  {tag}")

        # Wing analysis
        print(f"\n    Wing analysis (deep OTM, >{S0+8000:.0f}):")
        for j, K in enumerate(strikes):
            if K > S0 + 8000:
                print(f"      ${int(K):>8,}: v1.2={all_prices['v1.2'][j]:.6f}  v1.1={all_prices['v1.1'][j]:.6f}  "
                      f"GBM={all_prices['GBM'][j]:.6f}  Heston={all_prices['Heston'][j]:.6f}  Merton={all_prices['Merton'][j]:.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
