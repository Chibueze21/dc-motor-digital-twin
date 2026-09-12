"""
Experiment 12 — Uncertainty Landscape & Interaction Analysis

Purpose
-------
Extend Experiment 11 by examining pairwise interactions among the six
uncertain DC-motor parameters. This is a post-processing experiment:
Experiment 10 is NOT rerun and no controller gains are changed.

Input
-----
results/10_uncertainty_robustness_summary.csv

Outputs
-------
results/12_interaction_statistics.csv
results/12_interaction_matrix.csv
results/12_interaction_summary.txt
results/12_interaction_heatmap_rms.png
results/12_interaction_heatmap_overshoot.png
results/12_interaction_heatmap_settling.png
results/12_interaction_heatmap_twin_true.png
results/12_interaction_effects_rms.png
results/12_interaction_effects_overshoot.png
results/12_interaction_effects_settling.png
results/12_interaction_effects_twin_true.png

Method
------
For each selected outcome, the model uses standardized parameter deviations:

    z_i = (p_i - nominal_i) / (0.10 * nominal_i)

and fits a first-order interaction model:

    y = beta0 + sum(beta_i z_i) + sum(beta_ij z_i z_j)

where y is the Experiment-10 improvement percentage.

The analysis reports:
- standardized main-effect coefficients
- pairwise interaction coefficients
- bootstrap 95% confidence intervals
- permutation p-values for interaction terms
- Holm correction within each outcome
- model R^2 and adjusted R^2

Important:
A significant interaction term indicates an association within the sampled
uncertainty region. It does NOT establish a causal physical law or guarantee
behavior outside the ±10% uncertainty domain.
"""

from pathlib import Path
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
INPUT_CSV = RESULTS_DIR / "10_uncertainty_robustness_summary.csv"

SEED = 20260912
BOOTSTRAP_RESAMPLES = 5000
PERMUTATION_RESAMPLES = 5000
ALPHA = 0.05

TRUE_NOMINAL = {
    "R": 2.0,
    "L": 0.5,
    "Kt": 0.1,
    "Ke": 0.1,
    "J": 0.02,
    "b": 0.01,
}

OUTCOMES = [
    {
        "key": "rms",
        "label": "RMS error",
        "column": "rms_improvement_pct",
    },
    {
        "key": "overshoot",
        "label": "Overshoot",
        "column": "overshoot_improvement_pct",
    },
    {
        "key": "settling",
        "label": "Settling time",
        "column": "settling_improvement_pct",
    },
    {
        "key": "twin_true",
        "label": "Twin→true speed RMSE",
        "column": "twin_true_speed_rmse_improvement_pct",
    },
]


def require_input():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Experiment 10 CSV was not found: {INPUT_CSV}"
        )


def load_data():
    require_input()

    df = pd.read_csv(INPUT_CSV)
    df = df.replace(
        ["<br>", "<BR>", "", " ", "nan", "NaN", "None"],
        np.nan,
    )

    required = set(TRUE_NOMINAL.keys())
    required.update(o["column"] for o in OUTCOMES)

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Experiment 10 CSV is missing required columns:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    for col in df.columns:
        if col != "sample":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if len(df) != 100:
        warnings.warn(
            f"Expected 100 Monte-Carlo rows, found {len(df)}.",
            RuntimeWarning,
        )

    return df


def standardized_parameters(df):
    """
    Standardize parameter deviations relative to the ±10% uncertainty range.

    z = 0 at nominal.
    z = -1 at -10%.
    z = +1 at +10%.
    """
    z = pd.DataFrame(index=df.index)

    for parameter, nominal in TRUE_NOMINAL.items():
        z[parameter] = (
            (df[parameter].to_numpy(dtype=float) - nominal)
            / (0.10 * nominal)
        )

    return z


def design_matrix(z):
    names = ["Intercept"]
    columns = [np.ones(len(z))]

    for parameter in TRUE_NOMINAL:
        names.append(parameter)
        columns.append(z[parameter].to_numpy(dtype=float))

    for a, b in combinations(TRUE_NOMINAL.keys(), 2):
        names.append(f"{a}×{b}")
        columns.append(
            z[a].to_numpy(dtype=float) * z[b].to_numpy(dtype=float)
        )

    X = np.column_stack(columns)
    return X, names


def fit_least_squares(X, y):
    beta, _, rank, singular_values = np.linalg.lstsq(
        X, y, rcond=None
    )

    fitted = X @ beta
    residuals = y - fitted

    sse = float(np.sum(residuals ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))

    r2 = np.nan if sst == 0 else 1.0 - sse / sst

    n, k = X.shape
    if n > k and np.isfinite(r2):
        adjusted_r2 = 1.0 - (
            (1.0 - r2) * (n - 1) / (n - k)
        )
    else:
        adjusted_r2 = np.nan

    return beta, fitted, residuals, rank, singular_values, r2, adjusted_r2


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)

    finite = np.where(np.isfinite(p_values))[0]
    if len(finite) == 0:
        return adjusted

    order = finite[np.argsort(p_values[finite])]
    m = len(order)
    running_max = 0.0

    for rank, idx in enumerate(order):
        value = (m - rank) * p_values[idx]
        running_max = max(running_max, value)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def bootstrap_coefficients(X, y, rng):
    n = len(y)
    estimates = np.empty((BOOTSTRAP_RESAMPLES, X.shape[1]))

    for i in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, n, size=n)
        beta, _, _, _, _, _, _ = fit_least_squares(
            X[indices], y[indices]
        )
        estimates[i] = beta

    low = np.percentile(estimates, 100 * ALPHA / 2, axis=0)
    high = np.percentile(
        estimates,
        100 * (1 - ALPHA / 2),
        axis=0,
    )

    return low, high


def permutation_interaction_pvalues(X, y, names, beta_observed, rng):
    """
    Permutation test for each interaction coefficient.

    The response is permuted while the complete parameter design remains
    fixed. This tests whether the observed interaction coefficient is unusual
    under exchangeability of the outcome with respect to the design.
    """
    interaction_indices = [
        i for i, name in enumerate(names)
        if "×" in name
    ]

    observed = np.abs(beta_observed[interaction_indices])
    counts = np.zeros(len(interaction_indices), dtype=int)

    for _ in range(PERMUTATION_RESAMPLES):
        y_perm = rng.permutation(y)
        beta_perm, _, _, _, _, _, _ = fit_least_squares(
            X, y_perm
        )
        counts += (
            np.abs(beta_perm[interaction_indices]) >= observed
        )

    p_values = (counts + 1) / (PERMUTATION_RESAMPLES + 1)

    return interaction_indices, p_values


def analyze_outcome(df, z, outcome, rng):
    column = outcome["column"]

    y_all = df[column].to_numpy(dtype=float)
    mask = np.isfinite(y_all)

    y = y_all[mask]
    z_valid = z.loc[mask].reset_index(drop=True)

    X, names = design_matrix(z_valid)

    (
        beta,
        fitted,
        residuals,
        rank,
        singular_values,
        r2,
        adjusted_r2,
    ) = fit_least_squares(X, y)

    boot_low, boot_high = bootstrap_coefficients(
        X, y, rng
    )

    (
        interaction_indices,
        interaction_p,
    ) = permutation_interaction_pvalues(
        X, y, names, beta, rng
    )

    adjusted_p = holm_adjust(interaction_p)

    interaction_p_by_index = {
        idx: p for idx, p in zip(
            interaction_indices, interaction_p
        )
    }
    interaction_adj_by_index = {
        idx: p for idx, p in zip(
            interaction_indices, adjusted_p
        )
    }

    rows = []

    for i, name in enumerate(names):
        is_interaction = "×" in name

        rows.append(
            {
                "outcome": outcome["label"],
                "outcome_key": outcome["key"],
                "term": name,
                "term_type": (
                    "interaction"
                    if is_interaction
                    else (
                        "main_effect"
                        if name != "Intercept"
                        else "intercept"
                    )
                ),
                "n": len(y),
                "coefficient": beta[i],
                "bootstrap_ci95_low": boot_low[i],
                "bootstrap_ci95_high": boot_high[i],
                "permutation_p_raw": (
                    interaction_p_by_index.get(i, np.nan)
                ),
                "permutation_p_holm": (
                    interaction_adj_by_index.get(i, np.nan)
                ),
                "significant_holm": (
                    bool(interaction_adj_by_index[i] < ALPHA)
                    if i in interaction_adj_by_index
                    else False
                ),
                "model_rank": rank,
                "r2": r2,
                "adjusted_r2": adjusted_r2,
            }
        )

    return pd.DataFrame(rows)


def interaction_matrix(statistics, outcome_key):
    sub = statistics[
        (statistics["outcome_key"] == outcome_key)
        & (statistics["term_type"] == "interaction")
    ]

    matrix = pd.DataFrame(
        np.nan,
        index=list(TRUE_NOMINAL.keys()),
        columns=list(TRUE_NOMINAL.keys()),
    )

    for _, row in sub.iterrows():
        a, b = row["term"].split("×")
        value = row["coefficient"]
        matrix.loc[a, b] = value
        matrix.loc[b, a] = value

    # Avoid np.fill_diagonal on a potentially read-only DataFrame view.
    # Explicit diagonal assignment is robust across NumPy/Pandas versions.
    for parameter in matrix.index:
        matrix.loc[parameter, parameter] = 0.0

    return matrix


def plot_heatmap(statistics, outcome, filename):
    matrix = interaction_matrix(
        statistics,
        outcome["key"],
    )

    fig, ax = plt.subplots(figsize=(8, 6.5))
    image = ax.imshow(matrix.values, aspect="auto")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    ax.set_title(
        f"Experiment 12 — Pairwise Interaction Coefficients\n"
        f"{outcome['label']}"
    )

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = matrix.iloc[i, j]
            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                )

    fig.colorbar(image, ax=ax, label="Interaction coefficient")
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_top_interactions(statistics, outcome, filename):
    sub = statistics[
        (statistics["outcome_key"] == outcome["key"])
        & (statistics["term_type"] == "interaction")
    ].copy()

    sub["abs_coefficient"] = sub["coefficient"].abs()
    sub = sub.sort_values(
        "abs_coefficient",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(9, 6.5))

    ax.barh(
        sub["term"],
        sub["coefficient"],
    )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Standardized interaction coefficient"
    )
    ax.set_ylabel("Parameter pair")
    ax.set_title(
        f"Experiment 12 — Pairwise Interaction Effects\n"
        f"{outcome['label']}"
    )
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_matrix_files(statistics):
    for outcome in OUTCOMES:
        matrix = interaction_matrix(
            statistics,
            outcome["key"],
        )

        safe = outcome["key"]
        matrix.to_csv(
            RESULTS_DIR / f"12_interaction_matrix_{safe}.csv"
        )


def write_report(statistics, df):
    path = RESULTS_DIR / "12_interaction_summary.txt"

    with path.open("w", encoding="utf-8") as f:
        f.write("EXPERIMENT 12 — UNCERTAINTY LANDSCAPE & INTERACTION ANALYSIS\n")
        f.write("=" * 76 + "\n\n")

        f.write("Purpose\n")
        f.write("-------\n")
        f.write(
            "Post-processing analysis of Experiment 10 Monte-Carlo data. "
            "No Experiment 10 simulations were rerun and no controller gains "
            "were retuned.\n\n"
        )

        f.write("Dataset and method\n")
        f.write("------------------\n")
        f.write(f"Input: {INPUT_CSV}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write("Parameter uncertainty domain: ±10%\n")
        f.write(f"Bootstrap resamples: {BOOTSTRAP_RESAMPLES:,}\n")
        f.write(
            f"Permutation resamples: {PERMUTATION_RESAMPLES:,}\n"
        )
        f.write(f"Random seed: {SEED}\n")
        f.write(
            "Interaction model: standardized main effects + all pairwise "
            "parameter products\n"
        )
        f.write(
            "Permutation p-values are corrected with Holm step-down "
            "within each outcome.\n\n"
        )

        for outcome in OUTCOMES:
            f.write("\n")
            f.write(outcome["label"] + "\n")
            f.write("-" * len(outcome["label"]) + "\n")

            sub = statistics[
                statistics["outcome_key"] == outcome["key"]
            ]

            model_row = sub.iloc[0]
            f.write(
                f"n = {int(model_row['n'])}, "
                f"R² = {model_row['r2']:.4f}, "
                f"adjusted R² = {model_row['adjusted_r2']:.4f}\n"
            )

            interactions = sub[
                sub["term_type"] == "interaction"
            ].copy()

            interactions["abs_coefficient"] = (
                interactions["coefficient"].abs()
            )
            interactions = interactions.sort_values(
                "abs_coefficient",
                ascending=False,
            )

            f.write("\nPairwise interactions by absolute effect:\n")

            for _, row in interactions.iterrows():
                marker = (
                    " *"
                    if bool(row["significant_holm"])
                    else ""
                )

                f.write(
                    f"  {row['term']}: "
                    f"beta={row['coefficient']:.4f}, "
                    f"CI95=[{row['bootstrap_ci95_low']:.4f}, "
                    f"{row['bootstrap_ci95_high']:.4f}], "
                    f"pHolm={row['permutation_p_holm']:.6g}"
                    f"{marker}\n"
                )

            significant = interactions[
                interactions["significant_holm"]
            ]

            if len(significant):
                f.write(
                    "\nHolm-significant interactions: "
                    + ", ".join(significant["term"].tolist())
                    + "\n"
                )
            else:
                f.write(
                    "\nHolm-significant interactions: none\n"
                )

        f.write("\n\nInterpretation guardrails\n")
        f.write("-------------------------\n")
        f.write(
            "A positive interaction coefficient means that the joint "
            "parameter deviations are associated with a larger improvement "
            "than would be represented by the additive main-effect terms "
            "alone. A negative coefficient indicates the opposite within "
            "the sampled domain.\n"
        )
        f.write(
            "These are statistical associations inside the ±10% uncertainty "
            "region. They do not establish causality, nonlinear behavior "
            "outside the sampled region, or hardware-level validation.\n"
        )
        f.write(
            "The interaction model is deliberately used as an exploratory "
            "second-order landscape approximation rather than as a complete "
            "physical motor model.\n"
        )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("EXPERIMENT 12 — UNCERTAINTY LANDSCAPE & INTERACTION ANALYSIS")
    print("=" * 76)
    print("Input: results/10_uncertainty_robustness_summary.csv")
    print("No Experiment 10 rerun.")
    print("No controller gain retuning.")
    print(f"Bootstrap resamples: {BOOTSTRAP_RESAMPLES:,}")
    print(f"Permutation resamples: {PERMUTATION_RESAMPLES:,}")
    print(f"Random seed: {SEED}")
    print()

    df = load_data()
    z = standardized_parameters(df)
    rng = np.random.default_rng(SEED)

    all_results = []

    for outcome in OUTCOMES:
        print(f"Analyzing: {outcome['label']}")
        result = analyze_outcome(
            df,
            z,
            outcome,
            rng,
        )
        all_results.append(result)

    statistics = pd.concat(
        all_results,
        ignore_index=True,
    )

    statistics.to_csv(
        RESULTS_DIR / "12_interaction_statistics.csv",
        index=False,
    )

    write_matrix_files(statistics)
    write_report(statistics, df)

    plot_specs = [
        (
            "rms",
            "12_interaction_heatmap_rms.png",
            "12_interaction_effects_rms.png",
        ),
        (
            "overshoot",
            "12_interaction_heatmap_overshoot.png",
            "12_interaction_effects_overshoot.png",
        ),
        (
            "settling",
            "12_interaction_heatmap_settling.png",
            "12_interaction_effects_settling.png",
        ),
        (
            "twin_true",
            "12_interaction_heatmap_twin_true.png",
            "12_interaction_effects_twin_true.png",
        ),
    ]

    outcome_by_key = {
        x["key"]: x for x in OUTCOMES
    }

    for key, heatmap, effects in plot_specs:
        outcome = outcome_by_key[key]
        plot_heatmap(
            statistics,
            outcome,
            heatmap,
        )
        plot_top_interactions(
            statistics,
            outcome,
            effects,
        )

    print()
    print("TOP INTERACTIONS")
    print("-" * 76)

    for outcome in OUTCOMES:
        sub = statistics[
            (statistics["outcome_key"] == outcome["key"])
            & (statistics["term_type"] == "interaction")
        ].copy()

        sub["abs_coefficient"] = (
            sub["coefficient"].abs()
        )
        sub = sub.sort_values(
            "abs_coefficient",
            ascending=False,
        )

        print(f"\n{outcome['label']}")

        for _, row in sub.head(5).iterrows():
            marker = (
                " *"
                if bool(row["significant_holm"])
                else ""
            )

            print(
                f"  {row['term']:>7s}: "
                f"beta={row['coefficient']: .4f}, "
                f"CI95=[{row['bootstrap_ci95_low']: .4f}, "
                f"{row['bootstrap_ci95_high']: .4f}], "
                f"pHolm={row['permutation_p_holm']:.3g}"
                f"{marker}"
            )

    print()
    print("OUTPUTS")
    print("-" * 76)

    outputs = [
        RESULTS_DIR / "12_interaction_statistics.csv",
        RESULTS_DIR / "12_interaction_summary.txt",
        RESULTS_DIR / "12_interaction_matrix_rms.csv",
        RESULTS_DIR / "12_interaction_matrix_overshoot.csv",
        RESULTS_DIR / "12_interaction_matrix_settling.csv",
        RESULTS_DIR / "12_interaction_matrix_twin_true.csv",
        RESULTS_DIR / "12_interaction_heatmap_rms.png",
        RESULTS_DIR / "12_interaction_heatmap_overshoot.png",
        RESULTS_DIR / "12_interaction_heatmap_settling.png",
        RESULTS_DIR / "12_interaction_heatmap_twin_true.png",
        RESULTS_DIR / "12_interaction_effects_rms.png",
        RESULTS_DIR / "12_interaction_effects_overshoot.png",
        RESULTS_DIR / "12_interaction_effects_settling.png",
        RESULTS_DIR / "12_interaction_effects_twin_true.png",
    ]

    for output in outputs:
        print(output)

    print()
    print("EXPERIMENT 12 COMPLETE")


if __name__ == "__main__":
    main()
