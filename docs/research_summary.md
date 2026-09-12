# Research Summary

## Working title

**Constraint-Aware Digital-Twin Control of a DC Motor Under Parametric Uncertainty**

## Abstract

This project develops and evaluates a reproducible digital-twin control workflow for a DC motor. The methodology progresses from dynamic modelling and system identification to baseline closed-loop control, constraint-aware reference governance, Monte-Carlo uncertainty analysis, statistical inference, parameter-interaction analysis, independent confirmation, and reproducibility auditing.

The robustness campaign evaluates independent ±10% uncertainty in six motor parameters while keeping the identified digital twin and controller gains fixed. Experiment 10 evaluates 100 Monte-Carlo realizations, while Experiment 13 evaluates 1,000 new realizations using an independent seed and targeted interaction analysis. Experiments 11 and 12 provide statistical and interaction-level reanalysis of the original robustness sample.

The results show that the constraint-aware strategy achieves positive median improvements across the principal robustness metrics and wins in the majority of uncertain plant realizations. Aggregate improvements remain statistically supported after multiple-comparison correction in the Experiment 10 analysis. Independent confirmation in Experiment 13 reproduces the broad robustness pattern and confirms several targeted parameter interactions.

The findings are bounded by the tested model and ±10% uncertainty domain. The work is simulation-based and does not yet constitute hardware validation.

## Main contribution

The principal contribution is not a single controller gain. It is the **integrated and reproducible engineering workflow** connecting:

```text
Digital Twin
    ↓
Identification
    ↓
Control
    ↓
Constraint Governance
    ↓
Uncertainty Quantification
    ↓
Statistical Validation
    ↓
Interaction Analysis
    ↓
Independent Confirmation
    ↓
Reproducibility Audit
```

## Current evidence

Two robustness campaigns provide 1,100 unique Monte-Carlo plant realizations.

The final reproducibility audit reports:

```text
24 PASS
0 WARN
0 FAIL
```

## Boundary of claims

The evidence supports claims about performance under the specified simulation and uncertainty protocol. It does not establish universal superiority, hardware performance, or behavior outside the tested uncertainty regime.

## Future validation

The next scientific validation step should be hardware-in-the-loop or physical motor testing, followed by investigation of measurement noise, actuator nonlinearities, online identification and real-time implementation.
