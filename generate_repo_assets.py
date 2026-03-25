import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import warnings
warnings.filterwarnings("ignore")
from backtest_buffer_model import simulate_buffer_paths
from digital_option import simulate_sandpile_paths

# Use a clean, professional plotting style
style.use('ggplot')

# Create assets directory if not present
os.makedirs('assets', exist_ok=True)

def generate_benchmark_graphs():
    S0 = 71169.0
    n_steps = 1440 * 7 # 7 Days
    dt = 1.0 / (365.0 * 24.0 * 60.0)
    num_paths = 50

    # Execute 3-Factor Baseline
    sp_params = {"mu": 0.0, "sigma0": 0.235523, "alpha": 4.32765, "beta": 0.112472, "lambda0": 0.141961, "gamma": 18.5498, "eta": 0.717369, "jump_mu": -0.00381911, "jump_sigma": 0.345908, "eps": 0.001}
    lr_sp, _ = simulate_sandpile_paths(n_steps=n_steps, dt=dt, num_paths=num_paths, S0=S0, seed=105, **sp_params)
    paths_sp = S0 * np.exp(np.cumsum(np.hstack((np.zeros((num_paths, 1)), lr_sp)), axis=1))

    # Execute 4-Factor Brulant Model
    buf_params = {"mu0": 0.0, "sigma0": 0.596377, "rho": 1.78402, "nu": 1.54849, "alpha": 9.90562, "beta": 0.128777, "lambda0": 1.18401, "gamma": 20.0, "eta": 1.0, "kappa": 15.0, "theta_p": 1.5, "phi": 0.560709, "sigma_Y": 0.0568364, "eps": 0.001}
    lr_buf, _ = simulate_buffer_paths(n_steps=n_steps, dt=dt, num_paths=num_paths, S0=S0, seed=205, **buf_params)
    paths_buf = S0 * np.exp(np.cumsum(np.hstack((np.zeros((num_paths, 1)), lr_buf)), axis=1))

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    times = np.arange(n_steps + 1) / 1440.0 # x-axis in Days

    for i in range(num_paths):
        ax[0].plot(times, paths_sp[i, :], lw=1.2, alpha=0.5, color='#3498db') # Blue
        ax[1].plot(times, paths_buf[i, :], lw=1.2, alpha=0.5, color='#e74c3c') # Red

    ax[0].set_title("Baseline: 3-Factor Sandpile Model", fontsize=14, fontweight='bold', pad=15)
    ax[0].set_xlabel("Days")
    ax[0].set_ylabel("BTC/USDT Price")
    ax[0].axhline(S0, color='black', linestyle='--', linewidth=2, alpha=0.8, label="Starting Spot")
    ax[0].legend()

    ax[1].set_title("The Brulant Model (4-Factor Coupled SDE)", fontsize=14, fontweight='bold', pad=15)
    ax[1].set_xlabel("Days")
    ax[1].set_ylabel("BTC/USDT Price")
    ax[1].axhline(S0, color='black', linestyle='--', linewidth=2, alpha=0.8, label="Starting Spot")
    ax[1].legend()

    plt.tight_layout()
    fig.savefig('assets/benchmark_paths.png', dpi=200, bbox_inches='tight')
    print("Graph generated and saved to assets/benchmark_paths.png")

if __name__ == '__main__':
    generate_benchmark_graphs()
