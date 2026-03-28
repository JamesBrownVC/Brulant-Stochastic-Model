# Brulant v1.2: Executive Summary for Quant Committee
## March 27, 2026

---

## One-Line Summary

Brulant v1.2 introduces a structurally novel multi-factor SDE with multi-timescale
directional buffers and stochastic volatility targeting, but **does not outperform
standard benchmark models** (GBM, Heston, Merton, SABR) on BTC/USDT 1-minute
moment-matching loss after rigorous statistical testing.

---

## The Model (What It Does)

Brulant v1.2 is a 5-factor coupled SDE system for cryptocurrency microstructure:

1. **Spot price** with buffer-modulated drift: dS/S = [mu - rho*B_eff]dt + sigma*dW
2. **Stochastic volatility** mean-reverting to a stochastic target: dsigma = alpha*(sigma_target - sigma)dt
3. **Stochastic vol target** (OU process): dsigma_target = alpha_s*(sigma_bar - sigma_target)dt + xi_s*dW
4. **Fast buffer** (half-life ~10 min): dB_fast = -kappa_f*B_fast*dt + theta_f*d(log S)
5. **Slow buffer** (half-life ~56 days): dB_slow = -kappa_s*B_slow*dt + theta_s*d(log S)

**11 free parameters** (vs 1 for GBM, 4 for Heston/SABR, 3 for Merton).

---

## Testing Methodology (How We Tested It)

- **Calibration:** SMM via Differential Evolution, equalized compute budgets (~1500 evals/model)
- **Data:** BTC/USDT 1-min candles from Binance, 50/50 chronological train/test split
- **Winsorization:** Train-only thresholds (no data leakage)
- **Seeds:** 200 MC seeds per model per window
- **Statistics:** Paired Wilcoxon signed-rank test, Diebold-Mariano test, bootstrap 95% CI
- **Multi-window:** Non-overlapping 7-day windows across ~9 weeks of data
- **Market validation:** Deribit digital option price comparison

---

## Results (What We Found)

### Single-Window Benchmark (200 seeds)

| Rank | Model | Median Loss | Params |
|------|-------|-------------|--------|
| 1 | SABR | 3.79 | 4 |
| 2 | Merton | 3.79 | 3 |
| 3 | Heston | 3.97 | 4 |
| 4 | GBM | 4.04 | 1 |
| **5** | **Brulant v1.2** | **4.10** | **11** |

**v1.2 is last place.** Heston, Merton, and SABR significantly outperform (p < 0.001).

### Temporal Validation (4 windows, 10 seeds)

| Model | Win Rate | Avg Rank | Avg Loss |
|-------|----------|----------|----------|
| **SABR** | **100%** (4/4) | **1.00** | **2.22** |
| Heston | 0% | 2.50 | 2.86 |
| Brulant v1.2 | 0% | 3.25 | 3.16 |
| GBM | 0% | 3.25 | 3.16 |
| Merton | 0% | 5.00 | 506.72* |

*Merton suffers catastrophic calibration failures on 2/4 windows (loss capped at 1000).

SABR wins **all 4 windows**. v1.2 is middle-of-pack (tied with GBM). No pooled tests significant (p > 0.05, but only 4 windows — limited power).

### Market Comparison (Deribit Digital Options)

| Expiry | Days | RMSE | Mean Error |
|--------|------|------|------------|
| 28MAR26 | 1.1 | 0.1113 | 0.0565 |
| 29MAR26 | 2.1 | 0.0504 | 0.0283 |
| 30MAR26 | 3.1 | 0.0694 | 0.0403 |

**Bright spot:** v1.2 prices digital options reasonably well (mean RMSE 0.077).
The model tracks the market S-curve shape, with larger errors near ATM strikes.
This suggests the model may have value for pricing even if it can't match moments.

### Root Cause

- Empirical excess kurtosis: **8.27** (heavy-tailed, as expected for crypto)
- v1.2 simulated kurtosis: **-0.03** (near-Gaussian)
- The buffer mechanism smooths returns by creating counter-drift after directional moves,
  which **dampens the tail events** that produce heavy tails
- The model's core innovation is also its statistical weakness

---

## What's Still Valuable

Despite the negative performance result:

1. **Multi-timescale directional buffers** are a genuinely novel SDE mechanism with no
   precedent in the crypto derivatives literature
2. **Stochastic vol targets** extend Heston's framework in a natural way
3. **The jump frequency discovery** (jumps fire ~0.005 times per path at 1-min) is a
   quantified insight useful for anyone modeling crypto at high frequency
4. **The methodology** (equalized calibration, paired tests, multi-window) sets a standard
   for fair model comparison

---

## Honest Assessment

**The model is not ready for production or publication as a "better" model.** However:

- The negative result is scientifically valuable and publishable as a methods paper
- The buffer mechanism may work for applications beyond moment matching (hedging, pricing)
- Testing at lower frequencies (5-min, 15-min) where buffers have more time to accumulate
  may yield different results
- The testing framework itself is publishable

**Recommended framing:** "We introduce novel SDE mechanisms for crypto microstructure,
rigorously test them against benchmarks with equalized calibration, and find that while
structurally interesting, they do not improve distributional fit. We analyze why and
propose directions for future work."

---

## Files Generated

| File | Description |
|------|-------------|
| `presentation.html` | Interactive slide deck (open in browser) |
| `speaker_notes.md` | Detailed speaking notes with Q&A |
| `evidence_benchmark.json` | 200-seed benchmark data |
| `evidence_benchmark.png` | Benchmark visualization |
| `evidence_temporal.json` | Multi-window temporal validation |
| `evidence_temporal.png` | Temporal validation visualization |
| `market_comparison_results.json` | Deribit comparison data |
| `analyze_results.py` | Results analysis script |

All results are reproducible via: `python run_full_evidence.py`
