"""
Brulant v1.2 Experimental Engine
================================
New mechanics (all backward-compatible, off by default):

1. STOCHASTIC VOL MEAN-REVERSION: sigma0 itself follows an OU process
     d(sigma0_t) = alpha_s * (sigma0_bar - sigma0_t) * dt + xi_s * dW
   So the "target" vol drifts — captures regime shifts.

2. AGGRESSIVE HEIGHT-SCALED VOL DECAY: after jumps, vol decays proportional
   to how far above baseline it is:
     decay_rate = alpha + alpha_fast * max(0, sigma - sigma0)
   Higher sigma = faster snap-back. Captures post-liquidation vol crush.

3. WARM BUFFER INITIALIZATION: B_0 = theta_p * weighted_sum(recent_returns)
   instead of B_0 = 0. The buffer starts "loaded" with recent history.

4. MULTI-LAYER BUFFERS: B_fast (minutes), B_mid (hours), B_slow (days).
   Each has different kappa (decay) and theta_p (sensitivity).
   Jump direction uses weighted combination: jm = -phi * (w1*B_fast + w2*B_mid + w3*B_slow)
   Drift uses the same: mu = mu0 - rho * (w1*B_fast + w2*B_mid + w3*B_slow)

5. JUMP CLUSTERING (high diffusion burst then fast decay):
   On jump: sigma += beta_burst (large spike)
   Post-jump: alpha_fast channel drains the excess quickly
   Net effect: sharp vol spike that decays in minutes, not days.
"""

from __future__ import annotations
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple, List
from fit_sandpile import (
    fetch_binance_log_returns, interval_to_dt_years,
    moment_vector, recent_exponential_weights, _to_jsonable,
)
from backtest_buffer_model import MOMENT_NAMES


def simulate_v12(
    n_steps: int,
    dt: float,
    num_paths: int,
    *,
    S0: float = 1.0,
    # --- V1 base params ---
    mu0: float = 0.0,
    sigma0: float = 0.596,
    rho: float = 1.784,
    nu: float = 1.549,
    alpha: float = 9.906,
    beta: float = 0.129,
    lambda0: float = 1.184,
    gamma: float = 20.0,
    eta: float = 1.0,
    kappa: float = 15.0,
    theta_p: float = 1.5,
    phi: float = 0.561,
    sigma_Y: float = 0.045,
    eps: float = 1e-3,
    # --- NEW: Stochastic vol target ---
    stoch_vol_target: bool = False,
    sigma0_bar: float = 0.50,     # long-run vol target
    alpha_s: float = 2.0,         # mean-reversion speed of vol target
    xi_s: float = 0.3,            # noise on vol target
    # --- NEW: Height-scaled fast decay ---
    alpha_fast: float = 0.0,      # extra decay rate per unit above sigma0 (0=off)
    # --- NEW: Warm buffer ---
    warm_buffer: bool = False,
    warm_returns: Optional[np.ndarray] = None,  # recent returns to warm B with
    warm_halflife: float = 200.0,  # halflife for warm weighting
    # --- NEW: Multi-layer buffers ---
    multi_buffer: bool = False,
    kappa_fast: float = 50.0,     # fast buffer decay (~30 min halflife)
    kappa_mid: float = 15.0,      # mid buffer decay (~hours)
    kappa_slow: float = 3.0,      # slow buffer decay (~days)
    theta_fast: float = 2.0,      # fast buffer sensitivity
    theta_mid: float = 1.5,       # mid buffer sensitivity
    theta_slow: float = 0.8,      # slow buffer sensitivity
    w_fast: float = 0.5,          # weight on fast buffer
    w_mid: float = 0.3,           # weight on mid buffer
    w_slow: float = 0.2,          # weight on slow buffer
    # ---
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)

    S = np.full(num_paths, S0, dtype=np.float64)
    sig = np.full(num_paths, sigma0, dtype=np.float64)
    sig_target = np.full(num_paths, sigma0, dtype=np.float64)  # stochastic target
    M = np.zeros(num_paths, dtype=np.float64)
    lr = np.zeros((num_paths, n_steps), dtype=np.float64)

    if multi_buffer:
        B_f = np.zeros(num_paths, dtype=np.float64)
        B_m = np.zeros(num_paths, dtype=np.float64)
        B_s = np.zeros(num_paths, dtype=np.float64)

        # Warm initialization for each layer
        if warm_buffer and warm_returns is not None and warm_returns.size > 0:
            wr = warm_returns.astype(np.float64)
            n_w = wr.size
            # Initialize each buffer with its own exponential weighting
            for i in range(n_w):
                age = n_w - 1 - i
                B_f += theta_fast * wr[i] * np.exp(-kappa_fast * age * dt)
                B_m += theta_mid * wr[i] * np.exp(-kappa_mid * age * dt)
                B_s += theta_slow * wr[i] * np.exp(-kappa_slow * age * dt)
            # Broadcast to all paths
            b_f_init, b_m_init, b_s_init = B_f[0], B_m[0], B_s[0]
            B_f[:] = b_f_init
            B_m[:] = b_m_init
            B_s[:] = b_s_init
    else:
        B = np.zeros(num_paths, dtype=np.float64)
        if warm_buffer and warm_returns is not None and warm_returns.size > 0:
            wr = warm_returns.astype(np.float64)
            n_w = wr.size
            b_init = 0.0
            for i in range(n_w):
                age = n_w - 1 - i
                b_init += theta_p * wr[i] * np.exp(-kappa * age * dt)
            B[:] = b_init

    for t in range(n_steps):
        cs = np.maximum(sig, eps)

        # Jump intensity
        lambda_t = lambda0 * (1.0 / cs) * np.exp(-M)
        p_jump = np.minimum(lambda_t * dt, 1.0)
        jump = (rng.random(num_paths) < p_jump).astype(np.float64)

        # Jump direction from buffer(s)
        if multi_buffer:
            B_eff = w_fast * B_f + w_mid * B_m + w_slow * B_s
        else:
            B_eff = B

        jump_mean = np.clip(-phi * B_eff, -0.20, 0.20)
        jump_size = np.clip(rng.normal(jump_mean, sigma_Y), -0.25, 0.25)
        Y = np.exp(jump_size)
        dJ = (Y - 1.0) * jump

        # Price dynamics
        dW = sqrt_dt * rng.standard_normal(num_paths)
        dWr = sqrt_dt * rng.standard_normal(num_paths)

        ret = (
            mu0 * dt
            - rho * B_eff * dt
            - rho * nu * B_eff * dWr
            + cs * dW
            + dJ
        )
        ret = np.clip(ret, -0.50, 0.50)

        S_prev = S
        S = S * (1.0 + ret)
        S = np.maximum(S, 1e-12)
        lr[:, t] = np.log(S / S_prev)

        # --- Vol dynamics ---
        if stoch_vol_target:
            # Stochastic vol target: sigma0 drifts
            dW_s = sqrt_dt * rng.standard_normal(num_paths)
            sig_target = sig_target + alpha_s * (sigma0_bar - sig_target) * dt + xi_s * dW_s
            sig_target = np.maximum(sig_target, eps)
            mean_rev_target = sig_target
        else:
            mean_rev_target = sigma0

        # Base mean-reversion + height-scaled fast decay
        excess = np.maximum(cs - mean_rev_target, 0.0)
        effective_alpha = alpha + alpha_fast * excess
        sig_new = cs + effective_alpha * (mean_rev_target - cs) * dt + beta * jump
        sig = np.clip(sig_new, eps, 5.0)

        # Memory
        M = np.maximum(M - gamma * M * dt + eta * jump, 0.0)

        # Buffer(s)
        if multi_buffer:
            B_f = B_f - kappa_fast * B_f * dt + theta_fast * ret
            B_m = B_m - kappa_mid * B_m * dt + theta_mid * ret
            B_s = B_s - kappa_slow * B_s * dt + theta_slow * ret
        else:
            B = B - kappa * B * dt + theta_p * ret

    return lr, S


def evaluate(params, test_r, dt, acf_recent=300, n_seeds=10, paths=1000):
    emp = moment_vector(test_r, w=None, acf_recent_bars=acf_recent)
    emp_std = np.std(test_r)
    emp_3sig = np.mean(np.abs(test_r) > 3 * emp_std)
    scales = np.maximum(np.abs(emp), np.array([1e-12, 1e-12, 0.5, 1.0, 0.05, 0.1]))
    scales = np.maximum(scales, 1e-9)

    losses = []
    for i in range(n_seeds):
        sim_lr, _ = simulate_v12(test_r.size, dt, paths, seed=42+i*77, **params)
        sim = moment_vector(sim_lr.ravel(), w=None, acf_recent_bars=acf_recent)
        losses.append(float(np.sum(((sim - emp)/scales)**2)))

    sim_lr, _ = simulate_v12(test_r.size, dt, 5000, seed=42, **params)
    sp = sim_lr.ravel()[:50000]
    sm = moment_vector(sp, w=None, acf_recent_bars=acf_recent)
    sim_std = np.std(sp)

    losses = np.array(losses)
    return {
        "med_loss": float(np.median(losses)),
        "mean_loss": float(np.mean(losses)),
        "max_loss": float(np.max(losses)),
        "kurt": float(sm[3]),
        "skew": float(sm[2]),
        "std_ratio": float(sim_std / emp_std),
        "tail_3sig": float(np.mean(np.abs(sp) > 3*sim_std) / max(emp_3sig, 1e-12)),
        "acf_r2": float(sm[5]),
    }


def main():
    print("=" * 70)
    print("  BRULANT v1.2 EXPERIMENTS")
    print("=" * 70)

    # Fetch data
    returns_raw = fetch_binance_log_returns("BTCUSDT", "1m", 5000)
    dt = interval_to_dt_years("1m")
    mu = np.median(returns_raw)
    mad = np.percentile(np.abs(returns_raw - mu), 75) * 1.4826
    returns = np.clip(returns_raw, mu - 5*mad, mu + 5*mad)
    n_train = int(len(returns) * 0.75)
    train_r, test_r = returns[:n_train], returns[n_train:]

    emp = moment_vector(test_r, w=None, acf_recent_bars=300)
    print(f"  Empirical: std={np.std(test_r):.8f} kurt={emp[3]:.2f} skew={emp[2]:.3f}")
    print(f"  Train: {train_r.size} | Test: {test_r.size}")

    # V1 base (for warm buffer initialization)
    base = dict(
        mu0=0.0, sigma0=0.596377, rho=1.78402, nu=1.54849,
        alpha=9.90562, beta=0.128777, lambda0=1.18401,
        gamma=20.0, eta=1.0, kappa=15.0, theta_p=1.5,
        phi=0.560709, sigma_Y=0.045, eps=0.001,
    )

    # Use the last 500 training returns for warm buffer
    warm_ret = train_r[-500:]

    configs = {
        # === BASELINE ===
        "V1.1 baseline": {**base},

        # === 1. Stochastic vol target only ===
        "1a: stoch vol target (xi_s=0.3)": {
            **base, "stoch_vol_target": True, "sigma0_bar": 0.50,
            "alpha_s": 2.0, "xi_s": 0.3,
        },
        "1b: stoch vol target (xi_s=0.5)": {
            **base, "stoch_vol_target": True, "sigma0_bar": 0.50,
            "alpha_s": 2.0, "xi_s": 0.5,
        },

        # === 2. Height-scaled fast decay only ===
        "2a: fast decay (alpha_fast=50)": {**base, "alpha_fast": 50.0},
        "2b: fast decay (alpha_fast=100)": {**base, "alpha_fast": 100.0},
        "2c: fast decay (alpha_fast=200)": {**base, "alpha_fast": 200.0},

        # === 3. Warm buffer only ===
        "3a: warm buffer": {
            **base, "warm_buffer": True, "warm_returns": warm_ret,
        },

        # === 4. Multi-layer buffer only ===
        "4a: multi-buffer (3 layers)": {
            **base, "multi_buffer": True,
            "kappa_fast": 50.0, "kappa_mid": 15.0, "kappa_slow": 3.0,
            "theta_fast": 2.0, "theta_mid": 1.5, "theta_slow": 0.8,
            "w_fast": 0.5, "w_mid": 0.3, "w_slow": 0.2,
        },
        "4b: multi-buffer (heavier slow)": {
            **base, "multi_buffer": True,
            "kappa_fast": 60.0, "kappa_mid": 12.0, "kappa_slow": 2.0,
            "theta_fast": 1.5, "theta_mid": 1.5, "theta_slow": 1.2,
            "w_fast": 0.3, "w_mid": 0.3, "w_slow": 0.4,
        },

        # === 5. Fast decay + warm buffer ===
        "5a: fast_decay=100 + warm": {
            **base, "alpha_fast": 100.0,
            "warm_buffer": True, "warm_returns": warm_ret,
        },

        # === 6. Multi-buffer + warm ===
        "6a: multi-buffer + warm": {
            **base, "multi_buffer": True,
            "kappa_fast": 50.0, "kappa_mid": 15.0, "kappa_slow": 3.0,
            "theta_fast": 2.0, "theta_mid": 1.5, "theta_slow": 0.8,
            "w_fast": 0.5, "w_mid": 0.3, "w_slow": 0.2,
            "warm_buffer": True, "warm_returns": warm_ret,
        },

        # === 7. Stoch vol + fast decay ===
        "7a: stoch_vol + fast_decay=100": {
            **base, "stoch_vol_target": True, "sigma0_bar": 0.50,
            "alpha_s": 2.0, "xi_s": 0.3, "alpha_fast": 100.0,
        },

        # === 8. FULL COMBO: stoch vol + fast decay + multi-buffer + warm ===
        "8a: FULL (stoch+fast+multi+warm)": {
            **base, "stoch_vol_target": True, "sigma0_bar": 0.50,
            "alpha_s": 2.0, "xi_s": 0.3, "alpha_fast": 100.0,
            "multi_buffer": True,
            "kappa_fast": 50.0, "kappa_mid": 15.0, "kappa_slow": 3.0,
            "theta_fast": 2.0, "theta_mid": 1.5, "theta_slow": 0.8,
            "w_fast": 0.5, "w_mid": 0.3, "w_slow": 0.2,
            "warm_buffer": True, "warm_returns": warm_ret,
        },
        "8b: FULL (aggressive decay)": {
            **base, "stoch_vol_target": True, "sigma0_bar": 0.50,
            "alpha_s": 2.0, "xi_s": 0.3, "alpha_fast": 200.0,
            "multi_buffer": True,
            "kappa_fast": 60.0, "kappa_mid": 12.0, "kappa_slow": 2.0,
            "theta_fast": 1.5, "theta_mid": 1.5, "theta_slow": 1.2,
            "w_fast": 0.3, "w_mid": 0.3, "w_slow": 0.4,
            "warm_buffer": True, "warm_returns": warm_ret,
        },
    }

    print(f"\n  Testing {len(configs)} configurations (10 seeds each)...\n")

    header = f"  {'Config':<38s} {'Med':>7s} {'Mean':>7s} {'Kurt':>7s} {'Skew':>7s} {'StdR':>6s} {'3sig':>6s} {'ACF':>6s}"
    print(header)
    print("  " + "-" * len(header.strip()))

    results = {}
    for name, params in configs.items():
        t0 = time.perf_counter()
        r = evaluate(params, test_r, dt)
        elapsed = time.perf_counter() - t0
        results[name] = r
        print(f"  {name:<38s} {r['med_loss']:>7.2f} {r['mean_loss']:>7.1f} "
              f"{r['kurt']:>7.1f} {r['skew']:>7.3f} {r['std_ratio']:>6.3f} "
              f"{r['tail_3sig']:>6.3f} {r['acf_r2']:>6.3f}  ({elapsed:.0f}s)")

    # Rank by median loss
    print(f"\n  Empirical targets: kurt={emp[3]:.1f} skew={emp[2]:.3f} std_ratio=1.000 3sig=1.000")
    print("\n  === RANKING (by median OOS loss) ===")
    ranked = sorted(results.items(), key=lambda x: x[1]["med_loss"])
    for rank, (name, r) in enumerate(ranked, 1):
        flag = " *** BEST ***" if rank == 1 else ""
        print(f"  #{rank:>2d} {name:<38s} med={r['med_loss']:.2f} kurt={r['kurt']:.1f} std_r={r['std_ratio']:.3f}{flag}")

    # Save
    import json
    from pathlib import Path
    out = {"empirical": emp.tolist(), "results": {}}
    for name, r in results.items():
        out["results"][name] = {k: v for k, v in r.items()}
    Path("experiment_v12_results.json").write_text(json.dumps(out, indent=2))
    print("\nSaved experiment_v12_results.json")


if __name__ == "__main__":
    main()
