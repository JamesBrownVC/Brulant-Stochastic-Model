"""
Compression & release (sandpile) model — coupled SDEs
-----------------------------------------------------

Spot, stochastic volatility, and jump memory::

    dS_t   = μ S_t dt + σ_t S_t dW_t + S_t (Y - 1) dN_t
    dσ_t   = α (σ_0 - σ_t) dt + β dN_t
    dM_t   = -γ M_t dt + η dN_t

Jump intensity (volatility compression × exhaustion)::

    λ_t = λ_0 · f(σ_t) · g(M_t),
    f(σ) = 1 / (σ + ε),   g(M) = exp(-M).

Here ε > 0 avoids singularity at σ → 0. Conditional on a jump, Y > 0 is drawn
(lognormal by default) so the price jumps by factor Y. N_t is an inhomogeneous
Poisson process with rate λ_t; over [t, t+dt) we use P(ΔN=1) ≈ min(λ_t dt, 1).

Monte Carlo uses an explicit Euler step consistent with the formal SDEs above.
"""

import argparse
import os
from typing import Optional, Tuple

import numpy as np

try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _sandpile_core(n_steps, dt, num_paths, mu, sigma0, alpha, beta,
                       lambda0, gamma, eta, eps, jump_mu, jump_sigma, seed, S0):
        sqrt_dt = np.sqrt(dt)
        mu_dt = mu * dt
        np.random.seed(seed % (2**31))
        S = np.full(num_paths, S0)
        sig = np.full(num_paths, sigma0)
        M = np.zeros(num_paths)
        lr = np.empty((num_paths, n_steps))
        for t in range(n_steps):
            for i in range(num_paths):
                lam = lambda0 / (sig[i] + eps) * np.exp(-M[i])
                pj = min(lam * dt, 1.0)
                jmp = 1.0 if np.random.random() < pj else 0.0
                Y_val = jump_mu + jump_sigma * np.random.randn()
                Y_val = min(max(Y_val, -0.20), 0.20)
                Y = np.exp(Y_val)
                dW = sqrt_dt * np.random.randn()
                sp = S[i]
                S[i] = sp * (1.0 + mu_dt + sig[i] * dW) + sp * (Y - 1.0) * jmp
                lr[i, t] = np.log(S[i] / sp)
                sig[i] = min(max(sig[i] + alpha * (sigma0 - sig[i]) * dt + beta * jmp, eps), 5.0)
                M[i] = max(M[i] - gamma * M[i] * dt + eta * jmp, 0.0)
        return lr, S

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

try:
    import torch
except ImportError:
    torch = None


def _pick_torch_device(use_gpu: Optional[bool]) -> Optional["torch.device"]:
    """
    use_gpu True: require CUDA or raise.
    use_gpu False: CPU numpy path.
    use_gpu None: use env SANDPILE_USE_GPU=1 to try CUDA, else numpy.
    """
    if torch is None:
        if use_gpu is True:
            raise ImportError("GPU path requires PyTorch: pip install torch")
        return None
    if use_gpu is False:
        return None
    if use_gpu is True:
        if not torch.cuda.is_available():
            raise RuntimeError("use_gpu=True but torch.cuda.is_available() is False")
        return torch.device("cuda")
    env = os.environ.get("SANDPILE_USE_GPU", "").strip().lower()
    if env in ("1", "true", "yes") and torch.cuda.is_available():
        return torch.device("cuda")
    return None


def _simulate_sandpile_paths_torch(
    n_steps: int,
    dt: float,
    num_paths: int,
    mu: float,
    sigma0: float,
    alpha: float,
    beta: float,
    lambda0: float,
    gamma: float,
    eta: float,
    eps: float,
    jump_mu: float,
    jump_sigma: float,
    seed: Optional[int],
    S0: float,
    device: "torch.device",
) -> Tuple[np.ndarray, np.ndarray]:
    """Same dynamics as the NumPy path; float32 on GPU for throughput."""
    assert torch is not None
    dtype = torch.float32 if device.type == "cuda" else torch.float64
    if seed is not None:
        s = int(seed) % (2**31)
        torch.manual_seed(s)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(s)

    S = torch.full((num_paths,), S0, device=device, dtype=dtype)
    sigma = torch.full((num_paths,), sigma0, device=device, dtype=dtype)
    M = torch.zeros(num_paths, device=device, dtype=dtype)
    log_returns = torch.zeros((num_paths, n_steps), device=device, dtype=dtype)

    mu_dt = mu * dt
    sqrt_dt = float(np.sqrt(dt))
    sigma0_t = torch.tensor(sigma0, device=device, dtype=dtype)
    alpha_t = torch.tensor(alpha, device=device, dtype=dtype)
    beta_t = torch.tensor(beta, device=device, dtype=dtype)
    lambda0_t = torch.tensor(lambda0, device=device, dtype=dtype)
    gamma_t = torch.tensor(gamma, device=device, dtype=dtype)
    eta_t = torch.tensor(eta, device=device, dtype=dtype)
    eps_t = torch.tensor(eps, device=device, dtype=dtype)
    jump_mu_t = torch.tensor(jump_mu, device=device, dtype=dtype)
    jump_sigma_t = torch.tensor(jump_sigma, device=device, dtype=dtype)

    for _t in range(n_steps):
        lambda_t = lambda0_t * (1.0 / (sigma + eps_t)) * torch.exp(-M)
        p_jump = torch.clamp(lambda_t * dt, max=1.0)
        u = torch.rand(num_paths, device=device)
        jump = (u < p_jump).to(dtype)
        z_y = torch.randn(num_paths, device=device)
        jump_size = torch.clamp(jump_mu_t + jump_sigma_t * z_y, min=-0.20, max=0.20)
        Y = torch.exp(jump_size)
        dW = sqrt_dt * torch.randn(num_paths, device=device)
        S_prev = S
        S = S * (1.0 + mu_dt + sigma * dW) + S * (Y - 1.0) * jump
        log_returns[:, _t] = torch.log(S / S_prev)
        sigma = torch.clamp(sigma + alpha_t * (sigma0_t - sigma) * dt + beta_t * jump, min=eps_t.item(), max=5.0)
        M = torch.clamp(M - gamma_t * M * dt + eta_t * jump, min=0.0)

    return log_returns.detach().cpu().numpy(), S.detach().cpu().numpy()


def simulate_sandpile_paths(
    n_steps,
    dt,
    num_paths,
    mu,
    sigma0,
    alpha,
    beta,
    lambda0,
    gamma,
    eta,
    eps=1e-3,
    jump_mu=0.0,
    jump_sigma=0.25,
    seed=None,
    S0=1.0,
    use_gpu: Optional[bool] = False,
):
    """
    Euler simulation of the coupled SDEs. Returns per-step log returns and
    terminal spot (same units as S0).

    use_gpu
        If True, run on CUDA via PyTorch (requires ``pip install torch``).
        If False, NumPy on CPU. If None, honor env ``SANDPILE_USE_GPU=1``.

    Returns
    -------
    log_returns : ndarray, shape (num_paths, n_steps)
    S_T : ndarray, shape (num_paths,)
    """
    dev = _pick_torch_device(use_gpu)
    if dev is not None:
        return _simulate_sandpile_paths_torch(
            n_steps,
            dt,
            num_paths,
            mu,
            sigma0,
            alpha,
            beta,
            lambda0,
            gamma,
            eta,
            eps,
            jump_mu,
            jump_sigma,
            seed,
            S0,
            dev,
        )

    # Numba JIT path (scalar double loop, but compiled to machine code)
    if _HAS_NUMBA:
        s = int(seed) % (2**31) if seed is not None else 0
        return _sandpile_core(n_steps, dt, num_paths, mu, sigma0, alpha, beta,
                              lambda0, gamma, eta, eps, jump_mu, jump_sigma, s, S0)

    # Fallback: pure numpy vectorized over paths
    rng = np.random.default_rng(seed)
    S = np.full(num_paths, S0, dtype=np.float64)
    sigma = np.full(num_paths, sigma0, dtype=np.float64)
    M = np.zeros(num_paths, dtype=np.float64)
    log_returns = np.zeros((num_paths, n_steps), dtype=np.float64)

    for t in range(n_steps):
        lambda_t = lambda0 * (1.0 / (sigma + eps)) * np.exp(-M)
        p_jump = np.minimum(lambda_t * dt, 1.0)
        jump = (rng.random(num_paths) < p_jump).astype(np.float64)
        jump_size = np.clip(rng.normal(jump_mu, jump_sigma, num_paths), -0.20, 0.20)
        Y = np.exp(jump_size)
        dW = np.sqrt(dt) * rng.standard_normal(num_paths)
        S_prev = S
        S = S * (1.0 + mu * dt + sigma * dW) + S * (Y - 1.0) * jump
        log_returns[:, t] = np.log(S / S_prev)
        sigma = np.clip(sigma + alpha * (sigma0 - sigma) * dt + beta * jump, eps, 5.0)
        M = np.maximum(M - gamma * M * dt + eta * jump, 0.0)

    return log_returns, S


def price_digital_option(
    S0=65000.0,
    K=66000.0,
    hours=16.0,
    num_paths=50000,
    steps=200,
    mu=0.0,
    sigma0=0.20,
    alpha=4.0,
    beta=0.08,
    lambda0=1.5,
    gamma=15.0,
    eta=1.0,
    eps=1e-3,
    jump_mu=0.0,
    jump_sigma=0.25,
    r=0.0,
    seed=42,
    use_gpu: Optional[bool] = False,
):
    """
    Risk-neutral style digital call: discounted E[ 1{S_T > K} ] under the
    model in the module docstring (drift μ is a parameter; set μ = r for a
    common risk-neutral choice if desired).

    Parameters
    ----------
    S0, K : float
        Spot and strike.
    hours : float
        Time to expiry in hours (converted to years internally).
    num_paths, steps : int
        Monte Carlo paths and Euler time steps.
    mu : float
        Drift in dS (per year, as multiplier on S_t in the SDE).
    sigma0 : float
        Long-run / baseline σ in dσ_t = α(σ_0 - σ_t)dt + …
    alpha, beta : float
        Mean-reversion speed of σ and post-jump injection β dN_t.
    lambda0, gamma, eta : float
        Intensity prefactor, memory decay −γ M dt, jump mark η dN on M.
    eps : float
        Regularization in f(σ) = 1/(σ + ε).
    jump_mu, jump_sigma : float
        Log-normal jump multiplier Y ~ LogNormal(jump_mu, jump_sigma) (numpy
        parameterization: mean and sigma of log(Y)).
    r : float
        Discount exp(−r T) on the digital payoff.
    seed : int or None
        RNG seed; None for non-reproducible runs.
    """
    T = hours / (24.0 * 365.0)
    dt = T / steps

    _, S = simulate_sandpile_paths(
        n_steps=steps,
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
        S0=S0,
        use_gpu=use_gpu,
    )

    payoffs = (S > K).astype(np.float64)
    discount = np.exp(-r * T)
    digital_price = discount * np.mean(payoffs)

    return digital_price, S


def _build_parser():
    p = argparse.ArgumentParser(description="Digital call option MC pricer (custom SDE).")
    p.add_argument("--S0", type=float, default=65000.0)
    p.add_argument("--K", type=float, default=66000.0)
    p.add_argument("--hours", type=float, default=16.0)
    p.add_argument("--num-paths", type=int, default=50000)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--sigma0", type=float, default=0.20)
    p.add_argument("--alpha", type=float, default=4.0)
    p.add_argument("--beta", type=float, default=0.08)
    p.add_argument("--lambda0", type=float, default=1.5)
    p.add_argument("--gamma", type=float, default=15.0)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--jump-mu", type=float, default=0.0, dest="jump_mu")
    p.add_argument("--jump-sigma", type=float, default=0.25, dest="jump_sigma")
    p.add_argument("--r", type=float, default=0.0, help="Annual risk-free rate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--gpu",
        action="store_true",
        help="Run Monte Carlo on CUDA (requires PyTorch with GPU)",
    )
    return p


def main():
    args = _build_parser().parse_args()
    T_years = args.hours / (24.0 * 365.0)
    price, _ = price_digital_option(
        S0=args.S0,
        K=args.K,
        hours=args.hours,
        num_paths=args.num_paths,
        steps=args.steps,
        mu=args.mu,
        sigma0=args.sigma0,
        alpha=args.alpha,
        beta=args.beta,
        lambda0=args.lambda0,
        gamma=args.gamma,
        eta=args.eta,
        eps=args.eps,
        jump_mu=args.jump_mu,
        jump_sigma=args.jump_sigma,
        r=args.r,
        seed=args.seed,
        use_gpu=args.gpu,
    )
    rn_prob = price * np.exp(args.r * T_years)
    print(f"Digital Option Price (payout $1): ${price:.4f}")
    print(f"Risk-neutral P(S_T > K): {rn_prob * 100:.2f}%")


if __name__ == "__main__":
    main()
