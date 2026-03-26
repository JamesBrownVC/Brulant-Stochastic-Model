from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution


from fit_sandpile import (
    fetch_binance_log_returns,
    interval_to_dt_years,
    moment_vector,
    recent_exponential_weights,
    _to_jsonable,
)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


MOMENT_NAMES = ["mean", "std", "skew", "ex_kurt", "tail", "acf_r2"]

try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _buffer_core(n_steps, dt, num_paths, S0, mu0, sigma0, rho, nu, kappa,
                     theta_p, alpha, beta, lambda0, gamma, eta, phi, sigma_Y, eps, seed,
                     xi, delta):
        sqrt_dt = np.sqrt(dt)
        np.random.seed(seed % (2**31))
        S = np.full(num_paths, S0)
        sig = np.full(num_paths, sigma0)
        M = np.zeros(num_paths)
        B = np.zeros(num_paths)
        lr = np.empty((num_paths, n_steps))
        for t in range(n_steps):
            for i in range(num_paths):
                cs = max(sig[i], eps)
                lam = lambda0 / cs * np.exp(-M[i])
                pj = min(lam * dt, 1.0)
                jmp = 1.0 if np.random.random() < pj else 0.0
                jm = -phi * B[i] + delta
                jm = min(max(jm, -0.20), 0.20)
                Y_val = jm + sigma_Y * np.random.randn()
                Y_val = min(max(Y_val, -0.25), 0.25)
                Y = np.exp(Y_val)
                dJ = (Y - 1.0) * jmp
                dW = sqrt_dt * np.random.randn()
                dWr = sqrt_dt * np.random.randn()
                dW_sig = sqrt_dt * np.random.randn()
                ret = mu0 * dt - rho * B[i] * dt - rho * nu * B[i] * dWr + cs * dW + dJ
                ret = min(max(ret, -0.50), 0.50)
                sp = S[i]
                S[i] = max(sp * (1.0 + ret), 1e-12)
                lr[i, t] = np.log(S[i] / sp)
                sig[i] = min(max(cs + alpha * (sigma0 - cs) * dt + xi * cs * dW_sig + beta * jmp, eps), 5.0)
                M[i] = max(M[i] - gamma * M[i] * dt + eta * jmp, 0.0)
                B[i] = B[i] - kappa * B[i] * dt + theta_p * ret
        return lr, S

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def simulate_buffer_paths(
    n_steps: int,
    dt: float,
    num_paths: int,
    *,
    S0: float = 1.0,
    mu0: float = 0.05,
    sigma0: float = 0.40,
    rho: float = 2.0,
    nu: float = 1.5,
    kappa: float = 15.0,
    theta_p: float = 1.5,
    alpha: float = 10.0,
    beta: float = 0.10,
    lambda0: float = 2.0,
    gamma: float = 20.0,
    eta: float = 1.0,
    phi: float = 1.5,
    sigma_Y: float = 0.10,
    eps: float = 1e-3,
    seed: Optional[int] = 42,
    xi: float = 0.0,
    delta: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if _HAS_NUMBA:
        s = int(seed) % (2**31) if seed is not None else 0
        return _buffer_core(n_steps, dt, num_paths, S0, mu0, sigma0, rho, nu,
                            kappa, theta_p, alpha, beta, lambda0, gamma, eta,
                            phi, sigma_Y, eps, s, xi, delta)
    rng = np.random.default_rng(seed)
    S = np.full(num_paths, S0, dtype=np.float64)
    sigma = np.full(num_paths, sigma0, dtype=np.float64)
    M = np.zeros(num_paths, dtype=np.float64)
    B = np.zeros(num_paths, dtype=np.float64)
    lr = np.zeros((num_paths, n_steps), dtype=np.float64)

    for t in range(n_steps):
        current_sig = np.maximum(sigma, eps)
        lambda_t = lambda0 * (1.0 / current_sig) * np.exp(-M)
        p_jump = np.minimum(lambda_t * dt, 1.0)
        jump = (rng.random(num_paths) < p_jump).astype(np.float64)

        jump_mean = np.clip(-phi * B + delta, -0.20, 0.20)
        jump_size = np.clip(rng.normal(jump_mean, sigma_Y), -0.25, 0.25)
        Y = np.exp(jump_size)
        dJ = (Y - 1.0) * jump

        dW = np.sqrt(dt) * rng.standard_normal(num_paths)
        dWr = np.sqrt(dt) * rng.standard_normal(num_paths)
        dW_sig = np.sqrt(dt) * rng.standard_normal(num_paths)

        ret = (
            mu0 * dt
            - rho * B * dt
            - rho * nu * B * dWr
            + current_sig * dW
            + dJ
        )
        ret = np.clip(ret, -0.50, 0.50)

        S_prev = S
        S = S * (1.0 + ret)
        S = np.maximum(S, 1e-12)
        lr[:, t] = np.log(S / S_prev)

        sigma = np.clip(
            current_sig + alpha * (sigma0 - current_sig) * dt
            + xi * current_sig * dW_sig
            + beta * jump,
            eps, 5.0
        )
        M = np.maximum(M - gamma * M * dt + eta * jump, 0.0)
        B = B - kappa * B * dt + theta_p * ret

    return lr, S


def split_train_test(r: np.ndarray, train_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    n = r.size
    n_train = max(200, min(int(n * train_frac), n - 80))
    return r[:n_train], r[n_train:]


def fit_buffer_model(
    train_r: np.ndarray,
    dt: float,
    *,
    half_life_bars: float = 300.0,
    acf_recent_bars: Optional[int] = 400,
    num_paths: int = 1000,
    maxiter: int = 10,
    seed: int = 42,
    fixed: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    fixed = dict(fixed or {})
    w = recent_exponential_weights(train_r.size, half_life_bars)
    target = moment_vector(train_r, w=w, acf_recent_bars=acf_recent_bars)

    names = ["mu0", "sigma0", "rho", "nu", "alpha", "beta", "lambda0", "phi", "sigma_Y"]
    bounds = [
        (-0.50, 0.50),
        (0.05, 1.40),
        (0.1, 4.0),
        (0.1, 2.0),
        (0.5, 20.0),
        (1e-4, 0.2),
        (0.05, 10.0),
        (0.1, 2.0),
        (0.02, 0.3),
    ]

    def unpack(theta: np.ndarray) -> Dict[str, float]:
        p = {k: float(v) for k, v in zip(names, theta)}
        p.update(
            {
                "kappa": float(fixed.get("kappa", 15.0)),
                "theta_p": float(fixed.get("theta_p", 1.5)),
                "gamma": float(fixed.get("gamma", 20.0)),
                "eta": float(fixed.get("eta", 1.0)),
                "eps": float(fixed.get("eps", 1e-3)),
            }
        )
        return p

    scales = np.maximum(np.abs(target), np.array([1e-12, 1e-12, 0.5, 1.0, 0.05, 0.1]))
    scales = np.maximum(scales, 1e-9)
    rng = np.random.default_rng(seed)

    def obj(theta: np.ndarray) -> float:
        p = unpack(theta)
        s = int(rng.integers(0, 2**31 - 1))
        sim_lr, _ = simulate_buffer_paths(
            train_r.size,
            dt,
            num_paths,
            seed=s,
            S0=1.0,
            **p,
        )
        pooled = sim_lr.ravel()
        if pooled.size > 50_000:
            pooled = np.random.default_rng(s).choice(pooled, 50_000, replace=False)
        sim = moment_vector(pooled, w=None, acf_recent_bars=acf_recent_bars)
        z = (sim - target) / scales
        
        penalty = 0.0
        penalty += 10.0 * (max(0.0, p["phi"] - 1.0)**2)
        penalty += 10.0 * (max(0.0, p["sigma_Y"] - 0.15)**2)
        penalty += 1.0 * (max(0.0, p["lambda0"] - 5.0)**2)
        penalty += 10.0 * (max(0.0, p["beta"] - 0.05)**2)
        penalty += 2.0 * (max(0.0, p["rho"] - 2.0)**2)

        return float(np.sum(z * z)) + penalty

    res = differential_evolution(
        obj, bounds, maxiter=maxiter, seed=seed, workers=1,
        polish=False, popsize=8, tol=1e-3, atol=1e-4,
    )
    p = unpack(res.x)

    sim_lr, _ = simulate_buffer_paths(train_r.size, dt, num_paths, seed=seed + 111, S0=1.0, **p)
    sim_m = moment_vector(sim_lr.ravel(), w=None, acf_recent_bars=acf_recent_bars)

    return {
        **p,
        "loss": float(res.fun),
        "target_moments": target,
        "train_sim_moments": sim_m,
        "train_n": int(train_r.size),
        "half_life_bars": float(half_life_bars),
        "acf_recent_bars": acf_recent_bars,
    }


def evaluate_test(params: Dict[str, float], test_r: np.ndarray, dt: float, num_paths: int, seed: int, acf_recent_bars: Optional[int]) -> Dict[str, Any]:
    sim_lr, _ = simulate_buffer_paths(test_r.size, dt, num_paths, seed=seed, S0=1.0, **params)
    emp = moment_vector(test_r, w=None, acf_recent_bars=acf_recent_bars)
    sim = moment_vector(sim_lr.ravel(), w=None, acf_recent_bars=acf_recent_bars)
    z = np.maximum(np.abs(emp), np.array([1e-12, 1e-12, 0.5, 1.0, 0.05, 0.1]))
    z = np.maximum(z, 1e-9)
    loss = float(np.sum(((sim - emp) / z) ** 2))
    return {"test_emp": emp, "test_sim": sim, "test_loss": loss}


def plot_result(train_r: np.ndarray, test_r: np.ndarray, fit: Dict[str, Any], ev: Dict[str, Any], out_path: str) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = 80
    ax[0].hist(train_r, bins=bins, density=True, alpha=0.6, label="train")
    ax[0].set_title("Train returns")
    ax[1].hist(test_r, bins=bins, density=True, alpha=0.6, label="test", color="C2")
    ax[1].set_title("Test returns")
    for a in ax:
        a.grid(True, alpha=0.2)
    fig.suptitle(f"Buffer-model fit: train loss={fit['loss']:.3g}, test loss={ev['test_loss']:.3g}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest new 4-factor buffer model on historical BTC returns.")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1m")
    p.add_argument("--n-candles", type=int, default=3000, help="Use fewer candles first for speed")
    p.add_argument("--train-frac", type=float, default=0.75)
    p.add_argument("--half-life", type=float, default=250.0)
    p.add_argument("--acf-recent", type=int, default=300)
    p.add_argument("--paths", type=int, default=500)
    p.add_argument("--maxiter", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--plot-out", type=str, default="buffer_btc_backtest.png")
    p.add_argument("--output-json", type=str, default="buffer_btc_backtest.json")
    args = p.parse_args()

    if args.csv:
        r = np.loadtxt(args.csv, delimiter=",", usecols=0)
    else:
        r = fetch_binance_log_returns(args.symbol, args.interval, args.n_candles)

    dt = interval_to_dt_years(args.interval)
    train_r, test_r = split_train_test(r, args.train_frac)

    fit = fit_buffer_model(
        train_r,
        dt,
        half_life_bars=args.half_life,
        acf_recent_bars=args.acf_recent,
        num_paths=args.paths,
        maxiter=args.maxiter,
        seed=args.seed,
    )

    params = {
        "mu0": fit["mu0"],
        "sigma0": fit["sigma0"],
        "rho": fit["rho"],
        "nu": fit["nu"],
        "kappa": fit["kappa"],
        "theta_p": fit["theta_p"],
        "alpha": fit["alpha"],
        "beta": fit["beta"],
        "lambda0": fit["lambda0"],
        "gamma": fit["gamma"],
        "eta": fit["eta"],
        "phi": fit["phi"],
        "sigma_Y": fit["sigma_Y"],
        "eps": fit["eps"],
    }

    ev = evaluate_test(params, test_r, dt, max(600, args.paths), args.seed + 7, args.acf_recent)

    print("New buffer model fitted on train, frozen on test")
    print("-" * 50)
    for k in ["mu0", "sigma0", "rho", "nu", "alpha", "beta", "lambda0", "gamma", "eta", "kappa", "theta_p", "phi", "sigma_Y"]:
        print(f"  {k}: {fit[k]:.6g}")
    print("-" * 50)
    print(f"  n_train={train_r.size} n_test={test_r.size}")
    print(f"  train loss={fit['loss']:.6g}  test loss={ev['test_loss']:.6g}")
    print("\nTest moments (emp | sim):")
    for n, a, b in zip(MOMENT_NAMES, ev["test_emp"], ev["test_sim"]):
        print(f"  {n}: {a:.6g} | {b:.6g}")

    plot_result(train_r, test_r, fit, ev, args.plot_out)
    print(f"Saved plot: {args.plot_out}")

    out = {
        "fit": _to_jsonable({k: v for k, v in fit.items() if k not in ("target_moments", "train_sim_moments")}),
        "target_moments": fit["target_moments"].tolist(),
        "train_sim_moments": fit["train_sim_moments"].tolist(),
        "test_empirical_moments": ev["test_emp"].tolist(),
        "test_simulated_moments": ev["test_sim"].tolist(),
        "test_loss": ev["test_loss"],
    }
    Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
