from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from backtest_buffer_model import fit_buffer_model, evaluate_test
from fit_sandpile import fetch_binance_log_returns, interval_to_dt_years, _to_jsonable

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def run_walk_forward(
    returns: np.ndarray,
    dt: float,
    *,
    train_size: int,
    test_size: int,
    step_size: int,
    half_life: float,
    acf_recent: int,
    paths: int,
    maxiter: int,
    seed: int,
) -> Dict[str, Any]:
    r = np.asarray(returns, dtype=np.float64).ravel()
    folds: List[Dict[str, Any]] = []
    start = 0
    fold_id = 0

    while start + train_size + test_size <= r.size:
        train_r = r[start : start + train_size]
        test_r = r[start + train_size : start + train_size + test_size]

        fit = fit_buffer_model(
            train_r,
            dt,
            half_life_bars=half_life,
            acf_recent_bars=acf_recent,
            num_paths=paths,
            maxiter=maxiter,
            seed=seed + fold_id,
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
        ev = evaluate_test(
            params,
            test_r,
            dt,
            num_paths=max(500, paths),
            seed=seed + 1000 + fold_id,
            acf_recent_bars=acf_recent,
        )

        folds.append(
            {
                "fold": fold_id,
                "train_start": int(start),
                "train_end": int(start + train_size),
                "test_end": int(start + train_size + test_size),
                "train_loss": float(fit["loss"]),
                "test_loss": float(ev["test_loss"]),
                "fit_params": {
                    k: float(fit[k])
                    for k in (
                        "mu0",
                        "sigma0",
                        "rho",
                        "nu",
                        "alpha",
                        "beta",
                        "lambda0",
                        "gamma",
                        "eta",
                        "kappa",
                        "theta_p",
                        "phi",
                        "sigma_Y",
                        "eps",
                    )
                },
            }
        )
        fold_id += 1
        start += step_size

    if not folds:
        raise ValueError("Not enough data for requested train/test/step sizes.")

    test_losses = np.array([f["test_loss"] for f in folds], dtype=np.float64)
    train_losses = np.array([f["train_loss"] for f in folds], dtype=np.float64)
    return {
        "folds": folds,
        "summary": {
            "n_folds": int(len(folds)),
            "train_loss_mean": float(np.mean(train_losses)),
            "train_loss_median": float(np.median(train_losses)),
            "test_loss_mean": float(np.mean(test_losses)),
            "test_loss_median": float(np.median(test_losses)),
            "test_loss_std": float(np.std(test_losses)),
        },
    }


def maybe_plot_losses(folds: List[Dict[str, Any]], out_path: str) -> None:
    if plt is None:
        return
    x = np.arange(len(folds))
    train = [f["train_loss"] for f in folds]
    test = [f["test_loss"] for f in folds]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, train, marker="o", label="Train loss")
    ax.plot(x, test, marker="o", label="Forward test loss")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Loss")
    ax.set_title("Walk-forward losses (buffer model)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward out-of-sample test for buffer model.")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1m")
    p.add_argument("--n-candles", type=int, default=5000)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument("--train-size", type=int, default=2000)
    p.add_argument("--test-size", type=int, default=400)
    p.add_argument("--step-size", type=int, default=400)
    p.add_argument("--half-life", type=float, default=250.0)
    p.add_argument("--acf-recent", type=int, default=300)
    p.add_argument("--paths", type=int, default=700)
    p.add_argument("--maxiter", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot-out", default="buffer_forward_losses.png")
    p.add_argument("--output-json", default="buffer_forward_test.json")
    args = p.parse_args()

    if args.csv:
        r = np.loadtxt(args.csv, delimiter=",", usecols=0)
    else:
        r = fetch_binance_log_returns(args.symbol, args.interval, args.n_candles)
    dt = interval_to_dt_years(args.interval)

    result = run_walk_forward(
        r,
        dt,
        train_size=args.train_size,
        test_size=args.test_size,
        step_size=args.step_size,
        half_life=args.half_life,
        acf_recent=args.acf_recent,
        paths=args.paths,
        maxiter=args.maxiter,
        seed=args.seed,
    )
    folds = result["folds"]
    summary = result["summary"]

    print("Walk-forward out-of-sample summary")
    print("-" * 50)
    print(f"  folds: {summary['n_folds']}")
    print(f"  train loss mean/median: {summary['train_loss_mean']:.4g} / {summary['train_loss_median']:.4g}")
    print(f"  test loss  mean/median/std: {summary['test_loss_mean']:.4g} / {summary['test_loss_median']:.4g} / {summary['test_loss_std']:.4g}")
    print("-" * 50)
    for f in folds:
        print(
            f"  fold {f['fold']}: train[{f['train_start']}:{f['train_end']}] "
            f"test_end={f['test_end']}  train={f['train_loss']:.4g}  test={f['test_loss']:.4g}"
        )

    maybe_plot_losses(folds, args.plot_out)
    print(f"Saved plot: {args.plot_out}")

    Path(args.output_json).write_text(json.dumps(_to_jsonable(result), indent=2), encoding="utf-8")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
