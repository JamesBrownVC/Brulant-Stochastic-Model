"""
Calibrate sandpile SDE parameters to historical log returns using simulated
method of moments. Recent bars receive exponentially higher weight.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from digital_option import simulate_sandpile_paths

try:
    import requests
except ImportError:
    requests = None


def interval_to_dt_years(interval: str) -> float:
    """Binance-style interval string -> year fraction per bar."""
    m = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "1d": 1440,
    }.get(interval, 1)
    return m / (365.0 * 24.0 * 60.0)


def recent_exponential_weights(n: int, half_life_bars: float) -> np.ndarray:
    """
    Weights for chronological series [oldest, ..., newest]. Index n-1 is most recent.
    Weight doubles every `half_life_bars` toward the present.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    i = np.arange(n, dtype=np.float64)
    lam = np.log(2.0) / float(half_life_bars)
    w = np.exp(lam * i)
    w /= w.sum()
    return w


def moment_vector(
    r: np.ndarray,
    w: Optional[np.ndarray] = None,
    acf_recent_bars: Optional[int] = None,
) -> np.ndarray:
    """
    Summary statistics for MSM: mean, std, skew, excess kurtosis, tail mass,
    lag-1 ACF of squared (demeaned) returns.

    Mean / std / skew / kurtosis / tail use the same weights `w` when given
    (typically exponential toward the present).

    Tail: weighted fraction with |r - mu_w| > 3 * sigma_w (sigma_w from weighted variance).

    ACF of r^2: if `acf_recent_bars` is set, correlation is computed only on the
    last `acf_recent_bars` observations (extra emphasis on very recent volatility clustering).
    """
    r = np.asarray(r, dtype=np.float64).ravel()
    n = r.size
    if n < 4:
        raise ValueError("Need at least 4 returns")

    if w is None:
        w = np.ones(n) / n
    else:
        w = np.asarray(w, dtype=np.float64).ravel()
        w = w / w.sum()

    mu = np.sum(w * r)
    xc = r - mu
    m2 = np.sum(w * xc**2)
    m3 = np.sum(w * xc**3)
    m4 = np.sum(w * xc**4)
    std = np.sqrt(max(m2, 1e-30))
    skew = m3 / (std**3 + 1e-30)
    exkurt = m4 / (m2**2 + 1e-30) - 3.0
    tail = np.sum(w * (np.abs(xc) > 3.0 * std))

    k = n if acf_recent_bars is None else int(min(max(4, acf_recent_bars), n))
    r_tail = r[-k:]
    r2 = r_tail * r_tail
    r2c = r2 - np.mean(r2)
    if r2c.size > 2:
        acf1 = np.corrcoef(r2c[1:], r2c[:-1])[0, 1]
        if not np.isfinite(acf1):
            acf1 = 0.0
    else:
        acf1 = 0.0

    return np.array([mu, std, skew, exkurt, tail, acf1], dtype=np.float64)


def weight_diagnostics(w: np.ndarray) -> dict:
    """Effective sample size and mass on the most recent 10% of bars."""
    w = np.asarray(w, dtype=np.float64).ravel()
    w = w / w.sum()
    n = w.size
    ess = 1.0 / np.sum(w**2)
    k10 = max(1, int(np.ceil(0.1 * n)))
    mass_last_10pct = float(np.sum(w[-k10:]))
    mass_last_1pct = float(np.sum(w[-max(1, int(np.ceil(0.01 * n))) :]))
    return {
        "effective_sample_size": float(ess),
        "mass_last_10pct": mass_last_10pct,
        "mass_last_1pct": mass_last_1pct,
    }


def fetch_binance_log_returns(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    n_candles: int = 5000,
) -> np.ndarray:
    if requests is None:
        raise ImportError("Install requests to download Binance data: pip install requests")
    url = "https://api.binance.com/api/v3/klines"
    max_limit = 1000
    all_data = []
    end_time = None
    while len(all_data) < n_candles:
        remaining = n_candles - len(all_data)
        limit = min(max_limit, remaining)
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time is not None:
            params["endTime"] = end_time
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_data = data + all_data
        end_time = int(data[0][0]) - 1
        time.sleep(0.05)

    closes = np.array([float(x[4]) for x in all_data], dtype=np.float64)
    lr = np.diff(np.log(closes))
    return lr


def _pooled_simulated_moments(
    n_steps: int,
    dt: float,
    num_paths: int,
    mu: float,
    theta: np.ndarray,
    eps: float,
    seed: int,
    acf_recent_bars: Optional[int],
    use_gpu: Optional[bool] = False,
) -> np.ndarray:
    """theta = [sigma0, alpha, beta, lambda0, gamma, eta, jump_mu, jump_sigma]"""
    sigma0, alpha, beta, lambda0, gamma, eta, jump_mu, jump_sigma = theta
    lr, _ = simulate_sandpile_paths(
        n_steps=n_steps,
        dt=dt,
        num_paths=num_paths,
        mu=mu,
        sigma0=sigma0,
        alpha=alpha,
        beta=beta,
        lambda0=lambda0,
        gamma=gamma,
        eta=eta,
        eps=eps,
        jump_mu=jump_mu,
        jump_sigma=jump_sigma,
        seed=seed,
        S0=1.0,
        use_gpu=use_gpu,
    )
    pooled = lr.ravel()
    # Subsample large pooled arrays for speed (50k is enough for stable moments)
    max_pool = 50_000
    if pooled.size > max_pool:
        rng_sub = np.random.default_rng(seed)
        pooled = rng_sub.choice(pooled, max_pool, replace=False)
    return moment_vector(pooled, w=None, acf_recent_bars=acf_recent_bars)


def smm_objective(
    theta: np.ndarray,
    target: np.ndarray,
    n_steps: int,
    dt: float,
    num_paths: int,
    mu: float,
    eps: float,
    seed: int,
    scales: np.ndarray,
    acf_recent_bars: Optional[int],
    use_gpu: Optional[bool] = False,
) -> float:
    sim = _pooled_simulated_moments(
        n_steps, dt, num_paths, mu, theta, eps, seed, acf_recent_bars, use_gpu
    )
    z = (sim - target) / scales

    sigma0, alpha, beta, lambda0, gamma, eta, jump_mu, jump_sigma = theta
    penalty = 0.0
    penalty += 10.0 * (jump_mu**2)
    penalty += 10.0 * (max(0.0, jump_sigma - 0.2)**2)
    penalty += 1.0 * (max(0.0, lambda0 - 5.0)**2)
    penalty += 10.0 * (max(0.0, beta - 0.1)**2)

    return float(np.sum(z * z)) + penalty


def fit_to_returns(
    log_returns: np.ndarray,
    dt_years: float,
    *,
    half_life_bars: float = 500.0,
    use_window: Optional[int] = None,
    acf_recent_bars: Optional[int] = 800,
    fix_mu_from_data: bool = True,
    mu_fixed: Optional[float] = None,
    eps: float = 1e-3,
    num_paths: int = 2500,
    maxiter: int = 22,
    seed: int = 42,
    workers: int = 1,
    use_gpu: Optional[bool] = False,
) -> Tuple[dict, float]:
    """
    Fit [sigma0, alpha, beta, lambda0, gamma, eta, jump_mu, jump_sigma] by
    minimizing weighted squared error between target moments (recent-emphasized)
    and pooled simulated moments.

    Parameters
    ----------
    log_returns
        Chronological log returns (oldest first).
    dt_years
        Length of one bar in years (e.g. 1m -> 1/(365*24*60)).
    half_life_bars
        For exponential weights: weight on the newest bar is twice the weight
        at `half_life_bars` before the end (smaller = more emphasis on last few bars).
    use_window
        If set, keep only the last `use_window` returns before weighting.
    acf_recent_bars
        For the ACF(r^2) moment only: use the trailing `acf_recent_bars` of the
        (possibly windowed) series. None means use the full series for ACF.
    fix_mu_from_data
        If True, set mu = weighted_mean(r) / dt_years (annualized drift).
    mu_fixed
        If not None, overrides drift and ignores fix_mu_from_data.
    use_gpu
        If True, run inner Monte Carlo on CUDA (PyTorch). Use ``workers=1`` to
        avoid multiple processes sharing one GPU.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(log_returns, dtype=np.float64).ravel()
    if use_window is not None and use_window < len(r):
        r = r[-int(use_window) :]
    n = r.size
    w = recent_exponential_weights(n, half_life_bars)
    target = moment_vector(r, w=w, acf_recent_bars=acf_recent_bars)
    diag = weight_diagnostics(w)

    if mu_fixed is not None:
        mu = float(mu_fixed)
    elif fix_mu_from_data:
        mu = float(np.sum(w * r) / dt_years)
    else:
        mu = 0.0

    scales = np.maximum(np.abs(target), np.array([1e-12, 1e-12, 0.5, 1.0, 0.05, 0.1]))
    scales = np.maximum(scales, 1e-9)

    bounds = [
        (0.05, 2.5),
        (0.5, 50.0),
        (1e-4, 0.3),
        (0.05, 10.0),
        (0.5, 50.0),
        (0.05, 10.0),
        (-0.1, 0.1),
        (0.01, 0.4),
    ]

    eval_counter = [0]

    def fun(theta: np.ndarray) -> float:
        eval_counter[0] += 1
        s = int(rng.integers(0, 2**31 - 1))
        return smm_objective(
            theta,
            target,
            n,
            dt_years,
            num_paths,
            mu,
            eps,
            s,
            scales,
            acf_recent_bars,
            use_gpu,
        )

    t0 = time.perf_counter()
    res = differential_evolution(
        fun,
        bounds,
        maxiter=maxiter,
        seed=seed,
        workers=workers,
        polish=False,
        popsize=8,
        atol=1e-4,
        tol=1e-3,
    )
    elapsed = time.perf_counter() - t0

    sigma0, alpha, beta, lambda0, gamma, eta, jump_mu, jump_sigma = res.x
    out = {
        "mu": mu,
        "sigma0": sigma0,
        "alpha": alpha,
        "beta": beta,
        "lambda0": lambda0,
        "gamma": gamma,
        "eta": eta,
        "eps": eps,
        "jump_mu": jump_mu,
        "jump_sigma": jump_sigma,
        "n_bars": n,
        "half_life_bars": half_life_bars,
        "acf_recent_bars": acf_recent_bars,
        "dt_years": dt_years,
        "weight_diagnostics": diag,
        "target_moments": target,
        "sim_moments_at_opt": _pooled_simulated_moments(
            n, dt_years, num_paths, mu, res.x, eps, seed + 999, acf_recent_bars, use_gpu
        ),
        "use_gpu": bool(use_gpu),
        "loss": res.fun,
        "n_evals": eval_counter[0],
        "elapsed_s": elapsed,
        "success": res.success,
        "message": res.message,
    }
    return out, res.fun


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def chronological_split(
    log_returns: np.ndarray,
    train_frac: float = 0.75,
    min_train: int = 400,
    min_test: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Time-ordered split: train = earliest segment, test = future holdout.
    No shuffling — preserves causality for backtesting.
    """
    r = np.asarray(log_returns, dtype=np.float64).ravel()
    n = r.size
    if n < min_train + min_test:
        raise ValueError(
            f"Need at least min_train+min_test={min_train + min_test} returns, got {n}"
        )
    n_train = int(np.floor(n * float(train_frac)))
    n_train = max(min_train, min(n_train, n - min_test))
    return r[:n_train].copy(), r[n_train:].copy()


def simulate_pooled_log_returns(
    n_steps: int,
    dt: float,
    num_paths: int,
    mu: float,
    theta: np.ndarray,
    eps: float,
    seed: int,
    use_gpu: Optional[bool] = False,
) -> np.ndarray:
    """Pooled one-step log returns from many paths (same length as train/test for MSM)."""
    sigma0, alpha, beta, lambda0, gamma, eta, jump_mu, jump_sigma = theta
    lr, _ = simulate_sandpile_paths(
        n_steps=n_steps,
        dt=dt,
        num_paths=num_paths,
        mu=mu,
        sigma0=sigma0,
        alpha=alpha,
        beta=beta,
        lambda0=lambda0,
        gamma=gamma,
        eta=eta,
        eps=eps,
        jump_mu=jump_mu,
        jump_sigma=jump_sigma,
        seed=seed,
        S0=1.0,
        use_gpu=use_gpu,
    )
    return lr.ravel()


def moments_from_fit_dict(
    fit: dict,
    n_steps: int,
    dt: float,
    num_paths: int,
    seed: int,
    acf_recent_bars: Optional[int],
    use_gpu: Optional[bool] = False,
) -> np.ndarray:
    """Simulated MSM vector using stored theta from a train fit."""
    theta = np.array(
        [
            fit["sigma0"],
            fit["alpha"],
            fit["beta"],
            fit["lambda0"],
            fit["gamma"],
            fit["eta"],
            fit["jump_mu"],
            fit["jump_sigma"],
        ],
        dtype=np.float64,
    )
    pooled = simulate_pooled_log_returns(
        n_steps, dt, num_paths, fit["mu"], theta, fit["eps"], seed, use_gpu
    )
    return moment_vector(pooled, w=None, acf_recent_bars=acf_recent_bars)


def fitted_params_for_digital_option(fit: dict) -> Dict[str, Any]:
    """Keyword args for `digital_option.price_digital_option` from a fit dict."""
    return {
        "mu": fit["mu"],
        "sigma0": fit["sigma0"],
        "alpha": fit["alpha"],
        "beta": fit["beta"],
        "lambda0": fit["lambda0"],
        "gamma": fit["gamma"],
        "eta": fit["eta"],
        "eps": fit["eps"],
        "jump_mu": fit["jump_mu"],
        "jump_sigma": fit["jump_sigma"],
    }


def _print_result(fit: dict) -> None:
    wd = fit.get("weight_diagnostics") or {}
    print("Fitted parameters (recent-weighted MSM)")
    print("-" * 50)
    for k in (
        "mu",
        "sigma0",
        "alpha",
        "beta",
        "lambda0",
        "gamma",
        "eta",
        "jump_mu",
        "jump_sigma",
        "eps",
    ):
        print(f"  {k}: {fit[k]:.6g}")
    print("-" * 50)
    print(f"  bars used: {fit['n_bars']}, half_life_bars: {fit['half_life_bars']}")
    if wd:
        print(
            f"  weight ESS: {wd['effective_sample_size']:.0f}  "
            f"mass last 10%: {wd['mass_last_10pct']:.1%}  last 1%: {wd['mass_last_1pct']:.1%}"
        )
    if fit.get("acf_recent_bars") is not None:
        print(f"  ACF(r^2) moment: last {fit['acf_recent_bars']} bars only")
    print(f"  loss: {fit['loss']:.6g}  evals: {fit['n_evals']}  time: {fit['elapsed_s']:.1f}s")
    names = ["mean", "std", "skew", "ex_kurt", "tail", "acf_r2"]
    print("\nMoments (target vs sim at optimum):")
    for name, a, b in zip(names, fit["target_moments"], fit["sim_moments_at_opt"]):
        print(f"  {name}: {a:.6g}  |  {b:.6g}")


def main():
    p = argparse.ArgumentParser(description="Fit sandpile SDE to historical returns (recent-weighted).")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1m")
    p.add_argument("--n-candles", type=int, default=8000, help="History length to fetch (closes -> n-1 returns)")
    p.add_argument("--window", type=int, default=6000, help="Use only last N returns (recent emphasis)")
    p.add_argument("--half-life", type=float, default=400.0, dest="half_life", help="Exp. weight half-life in bars (smaller -> more weight on very last bars)")
    p.add_argument(
        "--acf-recent",
        type=int,
        default=800,
        dest="acf_recent",
        help="ACF(r^2) moment uses only the last N bars (strong recent vol clustering)",
    )
    p.add_argument("--acf-full", action="store_true", help="Use full window for ACF(r^2) instead of --acf-recent")
    p.add_argument("--paths", type=int, default=1200, help="MC paths per objective eval (lower = faster)")
    p.add_argument("--maxiter", type=int, default=14, help="DE iterations (lower = faster, rougher fit)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--gpu",
        action="store_true",
        help="Run path simulation on CUDA (PyTorch). Forces workers=1.",
    )
    p.add_argument("--mu", type=float, default=None, help="Fix annual drift; default: from weighted returns/dt")
    p.add_argument("--csv", type=str, default=None, help="Optional path to one column of log returns (skip fetch)")
    p.add_argument("--output-json", type=str, default=None, help="Write fitted parameters and metadata to this path")
    args = p.parse_args()

    if args.csv:
        r = np.loadtxt(args.csv, delimiter=",", usecols=0)
    else:
        r = fetch_binance_log_returns(
            symbol=args.symbol, interval=args.interval, n_candles=args.n_candles
        )

    dt = interval_to_dt_years(args.interval)
    acf_recent = None if args.acf_full else args.acf_recent
    workers = 1 if args.gpu else args.workers
    if args.gpu and args.workers != 1:
        print("Note: --gpu set: using workers=1 (single process on one GPU).")
    fit, loss = fit_to_returns(
        r,
        dt,
        half_life_bars=args.half_life,
        use_window=args.window,
        acf_recent_bars=acf_recent,
        mu_fixed=args.mu,
        fix_mu_from_data=args.mu is None,
        num_paths=args.paths,
        maxiter=args.maxiter,
        seed=args.seed,
        workers=workers,
        use_gpu=args.gpu,
    )
    _print_result(fit)

    if args.output_json:
        fit_meta = {
            k: v
            for k, v in fit.items()
            if k not in ("target_moments", "sim_moments_at_opt")
        }
        out = {
            "digital_option_kwargs": _to_jsonable(fitted_params_for_digital_option(fit)),
            "fit": _to_jsonable(fit_meta),
            "target_moments": fit["target_moments"].tolist(),
            "sim_moments_at_opt": fit["sim_moments_at_opt"].tolist(),
        }
        Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
