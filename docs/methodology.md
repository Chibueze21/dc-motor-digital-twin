# Methodology

## Research design

Project One follows a staged computational research design. Each experiment has a defined role and downstream assumptions are frozen where required.

The key methodological principle is separation of concerns:

- model identification is separated from controller evaluation;
- baseline and proposed control strategies are compared under the same plant conditions;
- robustness analysis uses frozen controller parameters;
- statistical analysis is performed after the simulation campaign;
- interaction analysis is separated from aggregate performance analysis;
- independent confirmation uses a new random realization set;
- reproducibility is audited after the evidence chain is complete.

## Parameter uncertainty

For robustness studies, each uncertain motor parameter is sampled independently from a uniform ±10% interval around its nominal value.

No parameter is re-identified after sampling.

## Controller freezing

The controller used by the robustness campaigns is frozen before uncertainty evaluation. This prevents the robustness result from becoming a collection of separately tuned controllers.

## Performance metrics

The study evaluates tracking, transient, actuator, current, and digital-twin fidelity metrics, including:

- RMS tracking error;
- integral absolute error;
- integral squared error;
- overshoot;
- settling time;
- saturation duration;
- RMS voltage;
- RMS current;
- twin-to-true speed RMSE.

## Statistical analysis

Aggregate improvements are evaluated using paired non-parametric inference and multiple-comparison correction. Bootstrap procedures provide uncertainty intervals for descriptive statistics.

Interaction analysis uses standardized parameter deviations and pairwise interaction terms. Targeted interactions are subsequently tested on an independent larger Monte-Carlo campaign.

## Reproducibility

All robustness experiments specify seeds, uncertainty ranges, frozen parameters, simulation configuration, and output artifacts. Experiment 14 produces a machine-readable audit and SHA-256 project manifest.

## Scientific scope

The conclusions are conditional on the tested model, controller, numerical integration procedure, and ±10% uncertainty domain. They should not be generalized to physical hardware or untested uncertainty regimes without further validation.
