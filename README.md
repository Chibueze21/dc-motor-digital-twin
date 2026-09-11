# DC Motor Digital Twin & Control Lab

A computational engineering study of **DC motor modeling, simulation, feedback control, system identification, digital-twin validation, and constraint-aware control transfer**.

The project develops a physics-based DC motor model and progressively builds a digital-twin workflow in which experimentally identified parameters are used to synthesize controllers and evaluate their transfer from the identified model to the true plant.

## Objectives

* Develop a mathematical model of a DC motor.
* Implement numerical simulation of motor dynamics.
* Validate the physics-based model against expected motor behavior.
* Design and evaluate PI/PID feedback control.
* Investigate state-space and integral-augmented state-feedback control.
* Perform parameter/system identification from simulated measurements.
* Validate an identified motor model using an independent excitation trajectory.
* Establish a digital-twin representation of the motor.
* Design controllers using the identified digital twin.
* Evaluate controller transfer from the digital twin to the true plant.
* Investigate actuator-constraint-aware control using reference governance.
* Establish a foundation for future machine-learning augmentation.

## Technology

* Python
* NumPy
* SciPy
* Matplotlib
* Git
* C/C++
* CMake

## Research Progression

The project is organized as a progressive engineering workflow:

```text
Mathematical Model
       ↓
Numerical Simulation
       ↓
Physics-Based Validation
       ↓
Feedback Control
       ↓
State-Space Control
       ↓
System Identification
       ↓
Independent Digital-Twin Validation
       ↓
Digital-Twin Closed-Loop Control
       ↓
Constraint-Aware Control Transfer
```

## Experiments

### 01–03 — Motor Modeling and Validation

The initial experiments establish the mathematical DC motor model, numerical simulation framework, and physics-based validation procedures.

### 05–06 — PI/PID Control

Feedback control experiments investigate speed regulation using PI and PID controllers, including actuator limits and practical controller behavior.

### 07 — State-Space Control

State-feedback control is developed using the motor's controllable state-space representation.

An integral-augmented formulation is then introduced to reject constant load disturbances and eliminate steady-state tracking error.

For the nominal motor under a 30 rad/s reference and 0.05 N·m load torque, the integral-augmented controller achieved:

* Final speed: **30.0000 rad/s**
* Steady-state error: **0 rad/s**
* Settling time: **2.554 s**
* Overshoot: **0%**

### 08 — System Identification and Digital-Twin Validation

Motor parameters are identified from simulated noisy measurements using nonlinear least-squares estimation.

The identified model achieved:

* Mean parameter error: **4.471%**
* Current RMSE: **0.01284 A**
* Speed RMSE: **0.05637 rad/s**
* Current R²: **0.999863**
* Speed R²: **0.999949**

An independent validation trajectory was subsequently used without re-fitting the parameters.

The independent validation produced:

* Current RMSE: **0.02235 A**
* Speed RMSE: **0.09797 rad/s**
* Current R²: **0.999782**
* Speed R²: **0.999794**

This established the identified model as a usable digital-twin representation for subsequent control experiments.

### 09A — Digital-Twin Closed-Loop Control Transfer

The integral-augmented state-feedback controller was synthesized **exclusively from the identified digital-twin model** and then applied without retuning to both:

1. the identified digital twin, and
2. the true motor model.

The experiment evaluates whether controller behavior transfers from the identified model to the underlying plant despite parameter mismatch.

For the true plant:

* Final speed: **30.0000 rad/s**
* Peak speed: **35.8333 rad/s**
* Overshoot: **19.44%**
* Settling time: **6.066 s**
* Maximum voltage: **12 V**

The experiment demonstrated successful closed-loop tracking and successful controller transfer, with measurable but bounded model-to-plant deviation.

### 09B — Constraint-Aware Digital-Twin Control

The 09B experiment investigates whether the same controller can achieve improved transient performance when actuator constraints are explicitly accounted for through **reference governance**, without retuning the state-feedback gains.

The controller gains remain frozen from the 09A design.

Three development stages were retained:

* **09B-v1** — initial reference-governed implementation.
* **09B-v2** — expanded performance and transfer metrics.
* **09B-v3** — final phase-matched evaluation methodology.

The authoritative result is **09B-v3**.

The reference governor limits the commanded reference rate to:

```text
18 rad/s²
```

The governor reaches the 30 rad/s target after:

```text
1.666667 s
```

To ensure an apples-to-apples comparison, 09A and 09B-v3 are evaluated over the identical post-governor interval:

```text
t ≥ 1.666667 s
```

#### True-Plant Results

| Metric             |          09A |       09B-v3 | Improvement |
| ------------------ | -----------: | -----------: | ----------: |
| RMS tracking error | 2.8159 rad/s | 2.3892 rad/s |  **15.15%** |
| IAE                |      15.3745 |       7.9549 |  **48.26%** |
| ISE                |      66.0462 |      47.4334 |  **28.18%** |
| Overshoot          |       19.44% |        6.83% |  **64.88%** |
| Settling time      |      6.066 s |      4.112 s |  **32.21%** |
| Saturation time    |      3.300 s |      1.598 s |  **51.58%** |
| RMS voltage        |    10.6809 V |    10.3384 V |   **3.21%** |

The constraint-aware controller therefore improved both transient performance and actuator utilization over the phase-matched evaluation window.

#### Digital-Twin Transfer

The proposed trajectory also improved the agreement between the identified digital twin and the true plant.

For speed:

* 09A transfer RMSE: **0.2659 rad/s**
* 09B-v3 transfer RMSE: **0.1601 rad/s**
* Improvement: **39.80%**

This indicates that the constraint-aware trajectory produced a closer digital-twin-to-plant correspondence under the evaluated conditions.

## Current Status

**Research prototype — control and digital-twin workflow established.**

The project has progressed beyond basic motor simulation to:

* physics-based modeling
* feedback control
* state-space control
* system identification
* independent digital-twin validation
* digital-twin-based controller synthesis
* true-plant controller transfer
* actuator-constraint-aware control
* phase-matched comparative evaluation

Future work will investigate more advanced digital-twin capabilities, including state estimation, online parameter adaptation, uncertainty quantification, and machine-learning augmentation.

## Repository Structure

```text
DC-Motor-Digital-Twin/
├── src/
│   ├── motor.py
│   └── controllers.py
│
├── simulations/
│   ├── 01_*.py
│   ├── 02_*.py
│   ├── ...
│   ├── 08_*.py
│   ├── 09A_*.py
│   └── 09B_constraint_aware_control_v3.py
│
├── results/
│   └── generated experiment outputs
│
├── notebooks/
│
├── tests/
│
├── .gitignore
└── README.md
```

## Research Direction

The long-term objective is to evolve this project from a conventional motor-control simulation into a reusable **engineering digital-twin framework for robotic systems**.

Planned extensions include:

* online system identification
* uncertainty-aware simulation
* state estimation
* adaptive control
* model-predictive control
* reinforcement-learning augmentation
* hardware-in-the-loop validation
* robotic actuator integration
* deployment-oriented simulation and testing
