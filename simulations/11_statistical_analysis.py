"""
Experiment 11 — Statistical Analysis & Publication Figures

Purpose
-------
Analyze the existing Experiment 10 Monte-Carlo results without rerunning or
modifying Experiment 10.

Input
-----
results/10_uncertainty_robustness_summary.csv

Outputs
-------
results/11_statistical_summary.csv
results/11_statistical_summary.txt
results/11_pairwise_statistics.csv
results/11_improvement_distributions.png
results/11_improvement_boxplots.png
results/11_improvement_ecdf.png
results/11_negative_tail_rms_ise.png
results/11_parameter_sensitivity.csv
results/11_parameter_sensitivity.png

Methodological notes
--------------------
- 100 Experiment-10 Monte-Carlo realizations are expected.
- Missing settling-time values remain NaN; they are never converted to zero.
- Improvement is already defined in Experiment 10 as 100*(09A-09B)/09A.
- Paired Wilcoxon signed-rank tests compare the raw 09A and 09B metrics.
- Holm correction is applied across the nine metric-level tests.
- Bootstrap confidence intervals are percentile intervals.
- Parameter sensitivity uses Spearman correlation between relative parameter
  deviation and improvement, with Holm correction across six parameters for
  each metric.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr, rankdata


RESULTS_DIR = Path("results")
INPUT_CSV = RESULTS_DIR / "10_uncertainty_robustness_summary.csv"

SEED = 20260912
BOOTSTRAP_RESAMPLES = 10_000
ALPHA = 0.05

TRUE_NOMINAL = {
    "R": 2.0,
    "L": 0.5,
    "Kt": 0.1,
    "Ke": 0.1,
    "J": 0.02,
    "b": 0.01,
}

METRICS = [
    {
        "key": "rms",
        "label": "RMS error",
        "improvement": "rms_improvement_pct",
        "a": "09A_rms",
        "b": "09B_rms",
    },
    {
        "key": "iae",
        "label": "IAE",
        "improvement": "iae_improvement_pct",
        "a": "09A_iae",
        "b": "09B_iae",
    },
    {
        "key": "ise",
        "label": "ISE",
        "improvement": "ise_improvement_pct",
        "a": "09A_ise",
        "b": "09B_ise",
    },
    {
        "key": "overshoot",
        "label": "Overshoot",
        "improvement": "overshoot_improvement_pct",
        "a": "09A_overshoot_pct",
        "b": "09B_overshoot_pct",
    },
    {
        "key": "settling",
        "label": "Settling time",
        "improvement": "settling_improvement_pct",
        "a": "09A_settling_s",
        "b": "09B_settling_s",
    },
    {
        "key": "saturation",
        "label": "Saturation duration",
        "improvement": "saturation_improvement_pct",
        "a": "09A_saturation_s",
        "b": "09B_saturation_s",
    },
    {
        "key": "rms_voltage",
        "label": "RMS voltage",
        "improvement": "rms_voltage_improvement_pct",
        "a": "09A_rms_voltage",
        "b": "09B_rms_voltage",
    },
    {
        "key": "rms_current",
        "label": "RMS current",
        "improvement": "rms_current_improvement_pct",
        "a": "09A_rms_current",
        "b": "09B_rms_current",
    },
    {
        "key": "twin_true_speed_rmse",
        "label": "Twin→true speed RMSE",
        "improvement": "twin_true_speed_rmse_improvement_pct",
        "a": "09A_twin_true_speed_rmse",
        "b": "09B_twin_true_speed_rmse",
    },
]


def require_input():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Experiment 10 CSV was not found: {INPUT_CSV}\n"
            "Run Experiment 10 first, or place its CSV in results/."
        )


def load_data():
    require_input()

    df = pd.read_csv(INPUT_CSV)

    # Preserve missing values such as <br> from copied Markdown tables.
    missing_tokens = ["<br>", "<BR>", "", " ", "nan", "NaN", "None"]
    df = df.replace(missing_tokens, np.nan)

    required = {"sample"}
    for metric in METRICS:
        required.update(
            [metric["improvement"], metric["a"], metric["b"]]
        )
    required.update(TRUE_NOMINAL.keys())

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Experiment 10 CSV is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
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


def bootstrap_ci(values, statistic, rng, resamples=BOOTSTRAP_RESAMPLES):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        value = float(statistic(values))
        return value, value

    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    samples = values[indices]
    stats = np.apply_along_axis(statistic, 1, samples)

    lower = np.percentile(stats, 100 * ALPHA / 2)
    upper = np.percentile(stats, 100 * (1 - ALPHA / 2))
    return float(lower), float(upper)


def win_tie_loss(improvements):
    values = np.asarray(improvements, dtype=float)
    values = values[np.isfinite(values)]

    wins = int(np.sum(values > 0))
    ties = int(np.sum(values == 0))
    losses = int(np.sum(values < 0))

    return wins, ties, losses


def matched_pairs_rank_biserial(a, b):
    """
    Matched-pairs rank-biserial correlation.

    Positive values indicate that 09B is lower/better than 09A for metrics
    where lower values are desirable.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mask = np.isfinite(a) & np.isfinite(b)
    differences = a[mask] - b[mask]
    differences = differences[differences != 0]

    if len(differences) == 0:
        return np.nan

    ranks = rankdata(np.abs(differences), method="average")
    positive_rank_sum = np.sum(ranks[differences > 0])
    negative_rank_sum = np.sum(ranks[differences < 0])
    total_rank_sum = positive_rank_sum + negative_rank_sum

    if total_rank_sum == 0:
        return np.nan

    return float((positive_rank_sum - negative_rank_sum) / total_rank_sum)


def holm_adjust(p_values):
    """Holm step-down adjustment, preserving original order."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)

    finite_idx = np.where(np.isfinite(p_values))[0]
    if len(finite_idx) == 0:
        return adjusted

    order = finite_idx[np.argsort(p_values[finite_idx])]
    running_max = 0.0

    m = len(order)
    for rank, idx in enumerate(order):
        adjusted_value = (m - rank) * p_values[idx]
        running_max = max(running_max, adjusted_value)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def ecdf(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.array([]), np.array([])

    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def analyze_metrics(df):
    rng = np.random.default_rng(SEED)

    rows = []
    wilcoxon_p = []

    for metric in METRICS:
        imp = df[metric["improvement"]].to_numpy(dtype=float)
        a = df[metric["a"]].to_numpy(dtype=float)
        b = df[metric["b"]].to_numpy(dtype=float)

        imp_finite = imp[np.isfinite(imp)]
        a_finite = a[np.isfinite(a)]
        b_finite = b[np.isfinite(b)]

        mean_ci = bootstrap_ci(imp_finite, np.mean, rng)
        median_ci = bootstrap_ci(imp_finite, np.median, rng)

        pair_mask = np.isfinite(a) & np.isfinite(b)
        paired_a = a[pair_mask]
        paired_b = b[pair_mask]
        differences = paired_a - paired_b
        nonzero = differences[differences != 0]

        if len(nonzero) >= 1:
            try:
                test = wilcoxon(
                    paired_a,
                    paired_b,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                p_value = float(test.pvalue)
                statistic = float(test.statistic)
            except ValueError:
                p_value = np.nan
                statistic = np.nan
        else:
            p_value = np.nan
            statistic = np.nan

        wins, ties, losses = win_tie_loss(imp)

        rows.append(
            {
                "metric": metric["label"],
                "key": metric["key"],
                "n_improvement": len(imp_finite),
                "n_paired": len(paired_a),
                "mean_improvement_pct": np.mean(imp_finite)
                if len(imp_finite)
                else np.nan,
                "median_improvement_pct": np.median(imp_finite)
                if len(imp_finite)
                else np.nan,
                "std_improvement_pct": np.std(imp_finite, ddof=1)
                if len(imp_finite) > 1
                else np.nan,
                "min_improvement_pct": np.min(imp_finite)
                if len(imp_finite)
                else np.nan,
                "max_improvement_pct": np.max(imp_finite)
                if len(imp_finite)
                else np.nan,
                "p05_improvement_pct": np.percentile(imp_finite, 5)
                if len(imp_finite)
                else np.nan,
                "p95_improvement_pct": np.percentile(imp_finite, 95)
                if len(imp_finite)
                else np.nan,
                "mean_bootstrap_ci95_low": mean_ci[0],
                "mean_bootstrap_ci95_high": mean_ci[1],
                "median_bootstrap_ci95_low": median_ci[0],
                "median_bootstrap_ci95_high": median_ci[1],
                "wins_09B": wins,
                "ties": ties,
                "losses_09B": losses,
                "win_rate_pct": 100 * wins / len(imp_finite)
                if len(imp_finite)
                else np.nan,
                "negative_tail_present": bool(np.any(imp_finite < 0))
                if len(imp_finite)
                else False,
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_raw": p_value,
                "rank_biserial": matched_pairs_rank_biserial(
                    paired_a, paired_b
                ),
            }
        )

        wilcoxon_p.append(p_value)

    result = pd.DataFrame(rows)
    result["wilcoxon_p_holm"] = holm_adjust(wilcoxon_p)
    result["wilcoxon_significant_holm"] = (
        result["wilcoxon_p_holm"] < ALPHA
    )

    return result


def parameter_sensitivity(df):
    rows = []

    for metric in METRICS:
        improvement = df[metric["improvement"]].to_numpy(dtype=float)

        p_values = []
        local_rows = []

        for parameter, nominal in TRUE_NOMINAL.items():
            deviation_col = f"_{parameter}_relative_deviation"
            deviation = (df[parameter].to_numpy(dtype=float) - nominal) / nominal

            mask = np.isfinite(improvement) & np.isfinite(deviation)

            if np.sum(mask) >= 3:
                rho, p_value = spearmanr(
                    deviation[mask], improvement[mask]
                )
                rho = float(rho)
                p_value = float(p_value)
            else:
                rho = np.nan
                p_value = np.nan

            local_rows.append(
                {
                    "metric": metric["label"],
                    "key": metric["key"],
                    "parameter": parameter,
                    "spearman_rho": rho,
                    "p_raw": p_value,
                    "n": int(np.sum(mask)),
                }
            )
            p_values.append(p_value)

        adjusted = holm_adjust(p_values)

        for row, p_adj in zip(local_rows, adjusted):
            row["p_holm_within_metric"] = p_adj
            row["significant_holm"] = bool(
                np.isfinite(p_adj) and p_adj < ALPHA
            )
            rows.append(row)

    return pd.DataFrame(rows)


def plot_improvement_distributions(df, summary):
    labels = [m["label"] for m in METRICS]
    values = [
        df[m["improvement"]].dropna().to_numpy(dtype=float)
        for m in METRICS
    ]

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_ylabel("09B-v3 improvement over 09A (%)")
    ax.set_title("Experiment 11 — Monte-Carlo Improvement Distributions")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "11_improvement_distributions.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_improvement_boxplots(df):
    primary_keys = [
        "rms",
        "iae",
        "ise",
        "overshoot",
        "settling",
        "saturation",
    ]
    selected = [m for m in METRICS if m["key"] in primary_keys]

    labels = [m["label"] for m in selected]
    values = [
        df[m["improvement"]].dropna().to_numpy(dtype=float)
        for m in selected
    ]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.boxplot(values, tick_labels=labels, showmeans=True)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_ylabel("Improvement (%)")
    ax.set_title(
        "Experiment 11 — Primary Closed-Loop Robustness Metrics"
    )
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "11_improvement_boxplots.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_improvement_ecdf(df):
    primary_keys = [
        "rms",
        "iae",
        "ise",
        "overshoot",
        "settling",
        "saturation",
    ]
    selected = [m for m in METRICS if m["key"] in primary_keys]

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for metric in selected:
        x, y = ecdf(df[metric["improvement"]].to_numpy(dtype=float))
        if len(x):
            ax.step(x, y, where="post", label=metric["label"])

    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("09B-v3 improvement over 09A (%)")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title("Experiment 11 — Empirical CDF of Improvement")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "11_improvement_ecdf.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_negative_tails(df):
    metrics = [
        ("rms_improvement_pct", "RMS error improvement (%)"),
        ("ise_improvement_pct", "ISE improvement (%)"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for column, label in metrics:
        values = df[column].dropna().to_numpy(dtype=float)
        ax.hist(values, bins=18, alpha=0.45, label=label)

    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Improvement (%)")
    ax.set_ylabel("Count")
    ax.set_title(
        "Experiment 11 — Negative Improvement Tails for RMS and ISE"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "11_negative_tail_rms_ise.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_parameter_sensitivity(sensitivity):
    subset = sensitivity[sensitivity["key"] == "rms"].copy()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(subset["parameter"], subset["spearman_rho"])
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("True-plant parameter")
    ax.set_ylabel("Spearman ρ")
    ax.set_title(
        "Experiment 11 — Parameter Sensitivity of RMS Improvement"
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "11_parameter_sensitivity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_text_report(summary, sensitivity, df):
    output = RESULTS_DIR / "11_statistical_summary.txt"

    with output.open("w", encoding="utf-8") as f:
        f.write("EXPERIMENT 11 — STATISTICAL ANALYSIS\n")
        f.write("=" * 72 + "\n\n")
        f.write("Purpose\n")
        f.write("-------\n")
        f.write(
            "Statistical analysis of the existing Experiment 10 Monte-Carlo "
            "results. No Experiment 10 simulations were rerun.\n\n"
        )

        f.write("Dataset\n")
        f.write("-------\n")
        f.write(f"Input: {INPUT_CSV}\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Bootstrap resamples: {BOOTSTRAP_RESAMPLES:,}\n")
        f.write(f"Random seed: {SEED}\n")
        f.write("Confidence level: 95%\n")
        f.write("Wilcoxon correction: Holm step-down\n\n")

        f.write("Metric summary\n")
        f.write("--------------\n")

        for _, row in summary.iterrows():
            f.write(f"\n{row['metric']}\n")
            f.write(
                f"  n improvement = {int(row['n_improvement'])}\n"
            )
            f.write(f"  n paired = {int(row['n_paired'])}\n")
            f.write(
                f"  mean improvement = "
                f"{row['mean_improvement_pct']:.3f}%\n"
            )
            f.write(
                f"  median improvement = "
                f"{row['median_improvement_pct']:.3f}%\n"
            )
            f.write(
                f"  std = {row['std_improvement_pct']:.3f}%\n"
            )
            f.write(
                f"  5th–95th percentile = "
                f"{row['p05_improvement_pct']:.3f}% to "
                f"{row['p95_improvement_pct']:.3f}%\n"
            )
            f.write(
                f"  mean bootstrap 95% CI = "
                f"{row['mean_bootstrap_ci95_low']:.3f}% to "
                f"{row['mean_bootstrap_ci95_high']:.3f}%\n"
            )
            f.write(
                f"  median bootstrap 95% CI = "
                f"{row['median_bootstrap_ci95_low']:.3f}% to "
                f"{row['median_bootstrap_ci95_high']:.3f}%\n"
            )
            f.write(
                f"  09B wins/ties/losses = "
                f"{int(row['wins_09B'])}/"
                f"{int(row['ties'])}/"
                f"{int(row['losses_09B'])}\n"
            )
            f.write(
                f"  win rate = {row['win_rate_pct']:.2f}%\n"
            )
            f.write(
                f"  negative improvement tail = "
                f"{bool(row['negative_tail_present'])}\n"
            )
            f.write(
                f"  Wilcoxon p (raw) = "
                f"{row['wilcoxon_p_raw']:.6g}\n"
            )
            f.write(
                f"  Wilcoxon p (Holm) = "
                f"{row['wilcoxon_p_holm']:.6g}\n"
            )
            f.write(
                f"  Holm-significant = "
                f"{bool(row['wilcoxon_significant_holm'])}\n"
            )
            f.write(
                f"  matched-pairs rank-biserial = "
                f"{row['rank_biserial']:.4f}\n"
            )

        f.write("\n\nInterpretation\n")
        f.write("--------------\n")
        f.write(
            "Positive improvement means that 09B-v3 produced a lower "
            "metric than frozen 09A under the Experiment 10 definition.\n"
        )
        f.write(
            "The analysis therefore evaluates robustness of the previously "
            "observed controller comparison under the sampled uncertainty "
            "realizations; it is not hardware validation.\n"
        )

        strong = summary[
            summary["win_rate_pct"] >= 95.0
        ]["metric"].tolist()

        if strong:
            f.write(
                "Metrics with at least 95% 09B win rate: "
                + ", ".join(strong)
                + ".\n"
            )

        negative = summary[
            summary["negative_tail_present"]
        ]["metric"].tolist()

        if negative:
            f.write(
                "Negative improvement tails were observed for: "
                + ", ".join(negative)
                + ". These tails should be retained in reporting and "
                "should not be described as universal superiority.\n"
            )

        significant = summary[
            summary["wilcoxon_significant_holm"]
        ]["metric"].tolist()

        f.write(
            "Metrics remaining significant after Holm correction: "
            + (", ".join(significant) if significant else "none")
            + ".\n"
        )

        f.write("\nParameter sensitivity\n")
        f.write("---------------------\n")
        for key in ["rms", "iae", "ise", "overshoot"]:
            sub = sensitivity[sensitivity["key"] == key]
            f.write(f"\n{key}:\n")
            for _, row in sub.iterrows():
                f.write(
                    f"  {row['parameter']}: rho="
                    f"{row['spearman_rho']:.4f}, "
                    f"Holm p={row['p_holm_within_metric']:.6g}\n"
                )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EXPERIMENT 11 — STATISTICAL ANALYSIS")
    print("=" * 72)
    print("Input: results/10_uncertainty_robustness_summary.csv")
    print("No Experiment 10 rerun.")
    print(f"Bootstrap resamples: {BOOTSTRAP_RESAMPLES:,}")
    print(f"Random seed: {SEED}")
    print()

    df = load_data()

    summary = analyze_metrics(df)
    sensitivity = parameter_sensitivity(df)

    summary_csv = RESULTS_DIR / "11_statistical_summary.csv"
    pairwise_csv = RESULTS_DIR / "11_pairwise_statistics.csv"
    sensitivity_csv = RESULTS_DIR / "11_parameter_sensitivity.csv"

    summary.to_csv(summary_csv, index=False)

    pairwise_columns = [
        "metric",
        "key",
        "n_paired",
        "wilcoxon_statistic",
        "wilcoxon_p_raw",
        "wilcoxon_p_holm",
        "wilcoxon_significant_holm",
        "rank_biserial",
    ]
    summary[pairwise_columns].to_csv(pairwise_csv, index=False)
    sensitivity.to_csv(sensitivity_csv, index=False)

    plot_improvement_distributions(df, summary)
    plot_improvement_boxplots(df)
    plot_improvement_ecdf(df)
    plot_negative_tails(df)
    plot_parameter_sensitivity(sensitivity)
    write_text_report(summary, sensitivity, df)

    print("KEY RESULTS")
    print("-" * 72)

    display_columns = [
        "metric",
        "mean_improvement_pct",
        "median_improvement_pct",
        "win_rate_pct",
        "wilcoxon_p_holm",
        "rank_biserial",
    ]

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 180,
        "display.float_format", "{:.3f}".format,
    ):
        print(summary[display_columns].to_string(index=False))

    print()
    print("OUTPUTS")
    print("-" * 72)
    for path in [
        summary_csv,
        RESULTS_DIR / "11_statistical_summary.txt",
        pairwise_csv,
        sensitivity_csv,
        RESULTS_DIR / "11_improvement_distributions.png",
        RESULTS_DIR / "11_improvement_boxplots.png",
        RESULTS_DIR / "11_improvement_ecdf.png",
        RESULTS_DIR / "11_negative_tail_rms_ise.png",
        RESULTS_DIR / "11_parameter_sensitivity.png",
    ]:
        print(path)

    print()
    print("EXPERIMENT 11 COMPLETE")


if __name__ == "__main__":
    main()
