# Brulant v1.2: Speaker Notes for Quant Committee Presentation
## March 2026

---

### Slide 1: Title
- "Good morning. I'm presenting the Brulant Model v1.2 — a multi-factor SDE system
  designed for cryptocurrency microstructure. I want to be upfront: this is an honest
  presentation. I'm going to show you what we built, how we tested it, and what the
  results actually say — including where the model fails."

---

### Slide 2: The Problem
- Key point: crypto markets have structural features (liquidation cascades, 125x leverage,
  24/7 trading) that traditional models don't capture
- Existing models lack directional memory — they don't know "which way the market has been
  going" and don't model the feedback between leveraged positions and price
- The novel idea: what if we add a "tension accumulator" that builds up during directional
  moves and creates counter-pressure?

---

### Slide 3: The SDE System
- Walk through the 4 key innovations:
  1. Stochastic vol target (OU on the mean-reversion level itself)
  2. Multi-layer buffers at different timescales
  3. Buffer-modulated drift
  4. No jumps (they don't fire at 1-min resolution)
- Emphasize: this is a 5-factor coupled SDE system (S, sigma, sigma_target, B_fast, B_slow)

---

### Slide 4: Calibration Methodology
- Stress the fairness measures taken:
  - Train-only winsorization (no test data leakage)
  - Equalized DE budgets (~1500 function evals per model)
  - No parameter fallbacks (if calibration fails, it fails)
  - 200 MC seeds per model
  - Paired statistical tests
- "We deliberately gave every model the same computational budget. If anything,
  Brulant's 11-parameter space is harder to optimize, not easier."

---

### Slide 5: Parameter Count
- Be very honest here: "11 parameters vs 1-4. This is the first thing any
  reviewer will flag, and rightly so."
- Explain: more parameters = more fitting flexibility, but also more overfitting risk
- "Our test is: does the extra complexity buy real improvement, or is it just noise?"
- Foreshadow the answer: the temporal validation will tell us

---

### Slide 6: Benchmark Results
- DELIVER THE PUNCHLINE HONESTLY:
  - "On a single calibration window with 200 MC seeds: v1.2 comes in last place."
  - SABR and Merton at 3.79, Heston at 3.97, GBM at 4.04, v1.2 at 4.10
  - "The differences vs Heston, Merton, and SABR are highly significant (p < 0.001)"
  - "11 parameters and we can't beat GBM's 1 parameter."

---

### Slide 6b: Diagnosis
- "Why does v1.2 fail? The answer is in the kurtosis."
  - Empirical kurtosis: 8.27 (heavy tails, as expected for crypto)
  - v1.2 produces kurtosis: -0.03 (near-Gaussian!)
  - The buffer mechanism smooths out tail events — it creates mean-reversion
    that dampens exactly the extreme moves we need to match
- "The core innovation of the model — the directional buffer — is also its
  statistical weakness. The mechanism that makes it structurally interesting
  for microstructure modeling actively hurts moment matching."
- This is a genuine insight, not an excuse

---

### Slide 7: Temporal Validation
- "A single window could be unrepresentative. Here are N non-overlapping windows."
- Present win rates, average ranks, pooled significance
- This is the definitive test

---

### Slide 8: Market Comparison
- "Even if the model matches moments poorly, does it price options reasonably?"
- Show RMSE against Deribit digital option prices
- Caveat: moment matching ≠ pricing accuracy, so this is a partially independent test

---

### Slide 9: What's Genuinely Novel
- Despite the negative performance result, the structural ideas are new:
  1. Multi-timescale directional buffers (no precedent in crypto SDE literature)
  2. Stochastic vol target (extension of Heston's framework)
  3. Buffer-modulated drift (path-dependent mean-reversion)
  4. Quantified jump discovery (formal result on resolution-dependence)
- "The ideas may find their application in pricing/hedging rather than distribution matching."

---

### Slide 10: Limitations
- Read through honestly. Don't rush.
- Key message: "We know where this model falls short. That's the point of rigorous testing."

---

### Slide 11: Proposition
- Brief theoretical result
- The buffer creates a friction-like mechanism — can prove it reduces drift
- But this drift reduction is also what kills tail properties

---

### Slide 12: The Verdict
- Auto-populated based on results
- Be prepared for: "The model does not reliably outperform simpler alternatives"
- Deliver with confidence, not apology

---

### Slide 12b: What We Learned
- Reframe: "A negative result done rigorously is more valuable than a false positive"
- The methodological contribution is solid regardless of v1.2's performance
- The structural tension (microstructure fidelity vs statistical fit) is a genuine finding
- "The honest pitch: we built something novel, tested it properly, and learned from it."

---

### Slide 13: Next Steps & Questions
- Future directions: test at lower frequencies, decouple buffer from vol, test on hedging
- Open questions for the committee
- "Thank you. Questions?"

---

## Anticipated Questions and Responses

**Q: With 11 parameters, isn't v1.2 just overfitting?**
A: "Actually, the opposite. Despite having 11 parameters, it performs WORSE than
1-4 parameter models. If anything, the extra degrees of freedom are poorly utilized
by the buffer mechanism. The issue isn't overfitting — it's a structural mismatch
between what buffers do (smooth returns) and what we need (heavy tails)."

**Q: Why not just add a heavy-tailed noise term?**
A: "That would address the kurtosis problem but defeat the purpose. We could add
t-distributed noise to any model. The interesting question is whether the buffer
mechanism itself generates the right distributional features, and the answer is
currently no — at least not for minute-scale moment matching."

**Q: Have you tested at lower frequencies?**
A: "Not yet. At 15-min or 1-hour resolution, buffers have more time to accumulate
directional pressure, and jump mechanisms become active again. This is the most
promising direction for future work."

**Q: What about transaction costs and practical use?**
A: "We explicitly list this as a limitation. The model is currently a theoretical
framework. Any production use would require transaction cost modeling, execution
simulation, and risk management overlays."

**Q: Is the buffer mechanism actually observed in market data?**
A: "The directional tension concept is motivated by order flow imbalance research
(Cont & Kakushadze 2020, Cartea & Jaimungal 2016). We don't directly estimate
buffers from order flow — that's future work and a natural connection to the
Hawkes process literature."

**Q: Why present a model that doesn't work?**
A: "Because science isn't about presenting only successes. The buffer mechanism
is a genuinely new structural idea. Showing that it fails at moment matching
but may work for pricing/hedging is an honest contribution that advances
understanding. Most papers only show the good results."
