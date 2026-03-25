import numpy as np
import requests
import warnings
warnings.filterwarnings("ignore")
from backtest_buffer_model import simulate_buffer_paths
from digital_option import simulate_sandpile_paths

def get_spot():
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=5)
        return float(r.json()["price"])
    except:
        return 71000.0

def main():
    S0 = get_spot()
    print(f"--- 7-Day Simulation Distribution Analysis ---")
    print(f"Starting Spot: ${S0:,.2f}")
    
    n_steps = 1440 * 7 # 7 days
    dt = 1.0 / (365.0 * 24.0 * 60.0)
    num_paths = 5000

    sp_params = {"mu": 0.0, "sigma0": 0.235523, "alpha": 4.32765, "beta": 0.112472, "lambda0": 0.141961, "gamma": 18.5498, "eta": 0.717369, "jump_mu": -0.00381911, "jump_sigma": 0.345908, "eps": 0.001}
    lr_sp, _ = simulate_sandpile_paths(n_steps=n_steps, dt=dt, num_paths=num_paths, S0=S0, seed=105, **sp_params)
    sp_terminal = S0 * np.exp(np.sum(lr_sp, axis=1))

    buf_params = {"mu0": 0.0, "sigma0": 0.596377, "rho": 1.78402, "nu": 1.54849, "alpha": 9.90562, "beta": 0.128777, "lambda0": 1.18401, "gamma": 20.0, "eta": 1.0, "kappa": 15.0, "theta_p": 1.5, "phi": 0.560709, "sigma_Y": 0.0568364, "eps": 0.001}
    lr_buf, _ = simulate_buffer_paths(n_steps=n_steps, dt=dt, num_paths=num_paths, S0=S0, seed=205, **buf_params)
    buf_terminal = S0 * np.exp(np.sum(lr_buf, axis=1))

    def print_stats(name, data):
        print(f"\n{name} Model (5000 Paths):")
        print(f"  Mean:   ${np.mean(data):,.2f}")
        print(f"  Median: ${np.median(data):,.2f}")
        print(f"  Max:    ${np.max(data):,.2f}")
        print(f"  Min:    ${np.min(data):,.2f}")
        print(f"  5th %:  ${np.percentile(data, 5):,.2f}")
        print(f"  95th %: ${np.percentile(data, 95):,.2f}")
        print(f"  Volatility (Std of log return): {np.std(np.log(data/S0)) * 100:.2f}%")

    print_stats("Sandpile", sp_terminal)
    print_stats("Buffer", buf_terminal)

if __name__ == '__main__':
    main()
