# Experiments 08A–14

## 08A — Frozen Digital Twin

Establishes the digital-twin parameter baseline used by downstream work.

## 08B — Identification and Validation

Validates the identified motor model and establishes the basis for subsequent controller evaluation.

## 09A — Baseline Closed-Loop Control

Provides the baseline control condition against which the constraint-aware strategy is compared.

## 09B-v3 — Constraint-Aware Control

Introduces reference governance/constraint handling while retaining a control structure suitable for systematic robustness testing.

## 10 — Uncertainty Robustness

100 independent Monte-Carlo realizations with independent ±10% uncertainty across six motor parameters.

Seed: `20260912`.

No retuning or re-identification.

## 11 — Statistical Analysis

Reanalysis of Experiment 10 using bootstrap confidence intervals, Wilcoxon tests, Holm correction and rank-biserial effect sizes.

No new plant realizations.

## 12 — Interaction Analysis

Second-order interaction analysis of the Experiment 10 uncertainty sample.

No new plant realizations.

## 13 — Targeted Interaction Confirmation

1,000 new independent Monte-Carlo realizations.

Seed: `20260913`.

Targeted interactions:

- R×Kt
- R×b
- Kt×b
- Kt×Ke
- Kt×J

No retuning or re-identification.

## 14 — Synthesis and Reproducibility Audit

Audits the evidence chain, protocol metadata and output artifacts.

No new plant simulations.

Final result:

```text
24 PASS
0 WARN
0 FAIL
```
