"""Quick grid test of vol-target excitation configs."""
import numpy as np
from multi_scale_benchmark import simulate_v11_excitation, structural_metrics, aggregate_returns
from fit_sandpile import interval_to_dt_years

dt_15m = interval_to_dt_years('15m')

# Key params: exc_beta (target kick), exc_kappa (target decay), alpha_exc (sigma->target speed)
# After a jump of |size|=0.2:
#   sigma_target jumps by exc_beta * 0.2
#   sigma chases sigma_target at rate alpha_exc
#   sigma_target decays back to sigma0 at rate exc_kappa
# Half-life of target: ln(2)/exc_kappa / dt = 0.693/(exc_kappa * dt_15m) bars

configs = [
    # (name, exc_beta, exc_kappa, alpha_exc, sigma_Y, lambda0, sigma0)
    # Low excitation, fast chase
    ('eb=2 ek=50 ae=50',    2.0,  50, 50.0, 0.20, 3.0, 0.50),
    ('eb=3 ek=50 ae=50',    3.0,  50, 50.0, 0.20, 3.0, 0.45),
    ('eb=5 ek=50 ae=50',    5.0,  50, 50.0, 0.20, 3.0, 0.40),
    # Medium excitation, slower chase (more gradual ramp)
    ('eb=3 ek=50 ae=20',    3.0,  50, 20.0, 0.20, 3.0, 0.45),
    ('eb=5 ek=50 ae=20',    5.0,  50, 20.0, 0.20, 3.0, 0.40),
    ('eb=5 ek=50 ae=100',   5.0,  50, 100.0, 0.20, 3.0, 0.40),
    # Faster target decay (shorter excitation window)
    ('eb=5 ek=100 ae=50',   5.0, 100, 50.0, 0.20, 3.0, 0.40),
    ('eb=5 ek=200 ae=50',   5.0, 200, 50.0, 0.20, 3.0, 0.40),
    # Bigger jumps, less frequent
    ('eb=5 ek=50 ae=50 sY25', 5.0, 50, 50.0, 0.25, 2.0, 0.40),
    # More aggressive
    ('eb=10 ek=50 ae=50',  10.0,  50, 50.0, 0.20, 3.0, 0.35),
    ('eb=10 ek=100 ae=50', 10.0, 100, 50.0, 0.20, 3.0, 0.35),
    # Very gentle
    ('eb=1 ek=50 ae=50',    1.0,  50, 50.0, 0.20, 5.0, 0.55),
]

print(f"{'Config':<28s} {'k15m':>7s} {'k1h':>7s} {'acf15':>7s} {'std15':>8s} {'t3_15':>6s}")
print("-" * 68)

for name, eb, ek, ae, sy, lam, s0 in configs:
    p = dict(mu0=0, sigma0=s0, rho=1.3, nu=1.6, alpha=10, beta=0.0,
             lambda0=lam, phi=1.6, sigma_Y=sy, exc_beta=eb, exc_kappa=ek,
             alpha_exc=ae, kappa=15, theta_p=1.5, gamma=20, eta=1, eps=1e-3)
    kurts, acfs, stds, tails = [], [], [], []
    for s in [42, 119, 256]:
        lr, _ = simulate_v11_excitation(2500, dt_15m, 500, seed=s, S0=1.0, **p)
        sm = structural_metrics(lr.ravel())
        kurts.append(sm["kurtosis"])
        acfs.append(sm["abs_acf1"])
        stds.append(sm["std"])
        tails.append(sm["tail_3sig"])

    k = np.median(kurts)
    a = np.median(acfs)
    st = np.median(stds)
    t = np.median(tails)

    lr, _ = simulate_v11_excitation(2500, dt_15m, 500, seed=42, S0=1.0, **p)
    sm1h = structural_metrics(aggregate_returns(lr, 4).ravel())
    print(f"{name:<28s} {k:>7.2f} {sm1h['kurtosis']:>7.2f} {a:>7.3f} {st:>8.5f} {t:>6.1f}x")

hl = 0.693 / (50 * dt_15m)
print(f"\n  Target half-life at ek=50: {hl:.0f} 15m bars = {hl*0.25:.0f} hours")
hl = 0.693 / (100 * dt_15m)
print(f"  Target half-life at ek=100: {hl:.0f} 15m bars = {hl*0.25:.0f} hours")
hl = 0.693 / (200 * dt_15m)
print(f"  Target half-life at ek=200: {hl:.0f} 15m bars = {hl*0.25:.0f} hours")
print(f"\n  Empirical targets: kurt15=6.5, kurt1h=4.0, acf15=0.155, std15=0.00286, tail3=5.0x")
