"""
Results Analyzer: Reads all evidence JSONs and produces a clean summary.
Run after run_full_evidence.py completes.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys


def load_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def analyze_benchmark(data: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("  BENCHMARK RESULTS (200 seeds, single window)")
    lines.append("=" * 70)

    stats = data.get("model_stats", {})
    lines.append(f"\n  Spot: ${data.get('spot', 0):,.2f}")
    lines.append(f"  Seeds: {data.get('n_seeds', '?')}")
    lines.append(f"  Empirical moments: {data.get('empirical_moments', [])}")

    # Sort by median loss
    sorted_models = sorted(stats.items(), key=lambda x: x[1]["median_loss"])
    lines.append(f"\n  {'Model':<14s} {'Median':>8s} {'95% CI':>22s} {'SE':>8s} {'Std/Emp':>8s} {'Kurt':>6s}")
    lines.append(f"  {'-'*14} {'-'*8} {'-'*22} {'-'*8} {'-'*8} {'-'*6}")
    for name, s in sorted_models:
        marker = " <--" if "v1.2" in name else ""
        lines.append(
            f"  {name:<14s} {s['median_loss']:>8.2f} "
            f"[{s['ci_median_lo']:>8.2f}, {s['ci_median_hi']:>8.2f}] "
            f"{s['se_median']:>8.3f} {s['std_ratio']:>8.4f} {s.get('kurt', 0):>6.1f}{marker}"
        )

    # Significance tests
    sig = data.get("significance_tests", {})
    if sig:
        lines.append(f"\n  PAIRED SIGNIFICANCE TESTS (Brulant v1.2 vs each)")
        lines.append(f"  {'Model':<14s} {'MedDiff':>8s} {'95% CI':>22s} {'Wilcoxon':>10s} {'DM':>10s} {'Sig':>5s}")
        lines.append(f"  {'-'*14} {'-'*8} {'-'*22} {'-'*10} {'-'*10} {'-'*5}")
        for name, s in sig.items():
            ci = s.get("ci_95", [0, 0])
            wp = s["wilcoxon_p"]
            stars = "***" if wp < 0.001 else ("**" if wp < 0.01 else ("*" if wp < 0.05 else "ns"))
            lines.append(
                f"  {name:<14s} {s['median_diff']:>+8.2f} "
                f"[{ci[0]:>+9.2f}, {ci[1]:>+9.2f}] "
                f"{wp:>10.4g} {s['dm_p']:>10.4g} {stars:>5s}"
            )

    return "\n".join(lines)


def analyze_temporal(data: dict) -> str:
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  TEMPORAL VALIDATION")
    lines.append("=" * 70)

    n_windows = data.get("n_windows", 0)
    n_seeds = data.get("n_seeds", 0)
    lines.append(f"\n  Windows: {n_windows} | Seeds per model: {n_seeds}")

    # Win rates and ranks
    wins = data.get("wins", {})
    avg_rank = data.get("avg_rank", {})
    avg_loss = data.get("avg_loss", {})
    pooled_sig = data.get("pooled_significance", {})

    models = ['Brulant v1.2', 'GBM', 'Heston', 'Merton', 'SABR']
    lines.append(f"\n  {'Model':<14s} {'Wins':>5s} {'Win%':>6s} {'AvgRank':>8s} {'AvgLoss':>8s} {'Pooled p':>10s}")
    lines.append(f"  {'-'*14} {'-'*5} {'-'*6} {'-'*8} {'-'*8} {'-'*10}")
    for m in models:
        w = wins.get(m, 0)
        wp = 100 * w / max(n_windows, 1)
        ar = avg_rank.get(m, 0)
        al = avg_loss.get(m, 0)
        ps = pooled_sig.get(m, {})
        p_str = f"{ps['p']:.4g}" if ps else "-"
        marker = " <--" if "v1.2" in m else ""
        lines.append(f"  {m:<14s} {w:>5d} {wp:>5.1f}% {ar:>8.2f} {al:>8.2f} {p_str:>10s}{marker}")

    # Regime breakdown
    regime = data.get("regime_results", {})
    if regime:
        lines.append(f"\n  REGIME BREAKDOWN")
        for rname, rd in regime.items():
            lines.append(f"\n  {rname.upper()} ({rd.get('n', '?')} windows)")
            al = rd.get("avg_loss", {})
            for name, avg in sorted(al.items(), key=lambda x: x[1]):
                lines.append(f"    {name:<14s} avg_loss={avg:.1f}")

    # Overall assessment
    lines.append(f"\n  ASSESSMENT")
    v12_wins = wins.get("Brulant v1.2", 0)
    v12_win_pct = 100 * v12_wins / max(n_windows, 1)
    v12_rank = avg_rank.get("Brulant v1.2", 5)

    any_sig = any(s.get("p", 1) < 0.05 for s in pooled_sig.values())
    all_sig = all(s.get("p", 1) < 0.05 for s in pooled_sig.values()) if pooled_sig else False
    all_pos = all(s.get("mean_diff", 0) > 0 for s in pooled_sig.values()) if pooled_sig else False

    if all_sig and all_pos and v12_rank <= 1.5:
        lines.append(f"  >> STRONG: v1.2 wins {v12_win_pct:.0f}% of windows, rank {v12_rank:.2f}, all tests significant")
    elif any_sig and v12_win_pct > 40:
        lines.append(f"  >> MODERATE: v1.2 wins {v12_win_pct:.0f}% of windows, rank {v12_rank:.2f}, partial significance")
    elif v12_win_pct > 30:
        lines.append(f"  >> WEAK: v1.2 wins {v12_win_pct:.0f}% of windows, rank {v12_rank:.2f}, mostly not significant")
    else:
        lines.append(f"  >> NEGATIVE: v1.2 wins only {v12_win_pct:.0f}% of windows, rank {v12_rank:.2f}")

    return "\n".join(lines)


def analyze_market(data: dict) -> str:
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  MARKET COMPARISON (Deribit)")
    lines.append("=" * 70)

    if data.get("error"):
        lines.append(f"  ERROR: {data['error']}")
        return "\n".join(lines)

    lines.append(f"\n  Spot: ${data.get('spot', 0):,.2f}")
    expiries = data.get("expiries", {})
    if not expiries:
        lines.append("  No expiries compared (API may be unavailable)")
        return "\n".join(lines)

    lines.append(f"  Expiries compared: {len(expiries)}")
    lines.append(f"\n  {'Expiry':<12s} {'Days':>6s} {'Strikes':>8s} {'RMSE':>8s} {'MaxErr':>8s} {'MeanErr':>8s}")
    lines.append(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for exp, r in expiries.items():
        lines.append(
            f"  {exp:<12s} {r['days_to_expiry']:>6.1f} {len(r['strikes']):>8d} "
            f"{r['rmse']:>8.4f} {r['max_error']:>8.4f} {r['mean_error']:>8.4f}"
        )

    all_rmse = [r["rmse"] for r in expiries.values()]
    import numpy as np
    lines.append(f"\n  Overall RMSE: mean={np.mean(all_rmse):.4f} max={np.max(all_rmse):.4f}")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("  BRULANT v1.2: RESULTS ANALYSIS")
    print("=" * 70)

    benchmark = load_json("evidence_benchmark.json")
    temporal = load_json("evidence_temporal.json")
    market = load_json("market_comparison_results.json")

    found = False
    if benchmark:
        found = True
        print(analyze_benchmark(benchmark))
    else:
        print("\n  evidence_benchmark.json not found")

    if temporal:
        found = True
        print(analyze_temporal(temporal))
    else:
        print("\n  evidence_temporal.json not found")

    if market:
        found = True
        print(analyze_market(market))
    else:
        print("\n  market_comparison_results.json not found")

    if not found:
        print("\n  No results found. Run: python run_full_evidence.py")
        return

    # Final verdict
    print("\n" + "=" * 70)
    print("  FINAL HONEST VERDICT")
    print("=" * 70)

    if temporal:
        wins = temporal.get("wins", {})
        n_windows = temporal.get("n_windows", 1)
        v12_wins = wins.get("Brulant v1.2", 0)
        v12_pct = 100 * v12_wins / max(n_windows, 1)
        v12_rank = temporal.get("avg_rank", {}).get("Brulant v1.2", 5)
        pooled = temporal.get("pooled_significance", {})

        sig_models = [m for m, s in pooled.items() if s.get("p", 1) < 0.05 and s.get("mean_diff", 0) > 0]

        print(f"\n  Win rate: {v12_pct:.0f}% ({v12_wins}/{n_windows} windows)")
        print(f"  Average rank: {v12_rank:.2f} out of 5")
        print(f"  Significant vs: {', '.join(sig_models) if sig_models else 'none'}")

        if v12_pct >= 50 and len(sig_models) >= 3:
            print(f"\n  CLAIM: 'Brulant v1.2 achieves statistically significant lower OOS")
            print(f"  moment-matching loss vs {len(sig_models)}/4 benchmarks across {n_windows} windows.'")
            print(f"  (Honest caveat: 11 params vs 1-4. AIC penalty not applied.)")
        elif v12_pct >= 30 and sig_models:
            print(f"\n  CLAIM: 'Brulant v1.2 shows partial advantage, significantly")
            print(f"  outperforming {', '.join(sig_models)} but not all benchmarks.'")
            print(f"  (Honest caveat: higher parameter count may explain some advantage.)")
        else:
            print(f"\n  CLAIM: 'Brulant v1.2 does not reliably outperform simpler models")
            print(f"  across temporal windows, despite lower loss on individual windows.'")
            print(f"  (The buffer mechanism is structurally interesting but needs more work.)")

    print()


if __name__ == "__main__":
    main()
