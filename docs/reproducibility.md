# Reproducibility Protocol

## Frozen assumptions

The robustness evidence uses a frozen identified digital twin and frozen controller gains.

### Identified twin

```text
R  = 2.011540
L  = 0.481997
Kt = 0.105805
Ke = 0.098059
J  = 0.021550
b  = 0.010715
```

### Controller

```text
K_AUG = [4.97875849, 6.45871589, -11.78057976]
```

### Operating conditions

```text
Reference = 30 rad/s
Voltage = 0–12 V
Load torque = 0.05 N·m
Uncertainty = independent uniform ±10%
```

## Seeds

```text
Experiment 10 = 20260912
Experiment 13 = 20260913
```

Experiment 13 deliberately uses a new seed and a new realization set.

## Statistical procedures

Experiment 11:

- 10,000 bootstrap resamples;
- 95% confidence intervals;
- paired Wilcoxon testing;
- Holm step-down correction;
- rank-biserial effect size.

Experiment 12:

- 5,000 bootstrap resamples;
- 5,000 permutation resamples.

Experiment 13:

- 3,000 bootstrap resamples;
- 3,000 permutation resamples.

## Missing data

Unavailable settling-time values are treated as missing rather than being converted to zero.

## Audit

Experiment 14 generates:

- evidence matrix;
- reproducibility audit;
- cross-experiment summary;
- project manifest;
- synthesis report.

Final audit:

```text
PASS = 24
WARN = 0
FAIL = 0
```

The project should retain the generated CSV/TXT evidence files with the corresponding scripts.
