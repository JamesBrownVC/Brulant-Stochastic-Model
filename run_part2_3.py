"""Run just Part 2 (temporal) and Part 3 (market comparison)."""
import time, sys
sys.stdout.reconfigure(line_buffering=True)  # Force line buffering

t_start = time.perf_counter()

from run_full_evidence import run_temporal, run_market_comparison

print("Starting Part 2: Temporal Validation...")
try:
    run_temporal()
except Exception as e:
    print(f"Part 2 FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\nStarting Part 3: Market Comparison...")
try:
    run_market_comparison()
except Exception as e:
    print(f"Part 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

total = time.perf_counter() - t_start
print(f"\nDone in {total/60:.1f} minutes")
