"""
FreshRAG Experiment Analysis (Phase 4).

Statistical analysis of scored eval instances. Supports multiple experiments:
  4.1 — Mechanism Isolation: two-way ANOVA (scenario × timestamp_condition)

Usage:
    python experiment_analysis.py --experiment 4.1 --scored data/benchmark/scored_210.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_METRICS = ("FA", "HR", "AU", "CRS", "POS")


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_scored(path: Path) -> pd.DataFrame:
    """Load scored JSONL into a DataFrame with flattened score columns."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            flat = {
                "instance_id": rec["instance_id"],
                "qa_id": rec["qa_id"],
                "scenario": rec["scenario"],
                "timestamp_condition": rec.get("timestamp_condition", "actual"),
                "domain": rec["domain"],
                "predicted_mechanism": rec["predicted_mechanism"],
                "change_type": rec["change_type"],
            }
            for m in _METRICS:
                flat[m] = rec["scores"][m]
            records.append(flat)

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} scored instances from {path}")
    print(f"  Scenarios: {sorted(df['scenario'].unique())}")
    print(f"  Timestamp conditions: {sorted(df['timestamp_condition'].unique())}")
    print(f"  Domains: {sorted(df['domain'].unique())}")
    print(f"  Mechanisms: {sorted(df['predicted_mechanism'].unique())}")
    return df


# ---------------------------------------------------------------------------
# 2. Two-way ANOVA (Type II SS)
# ---------------------------------------------------------------------------

def _ss_between(groups: list[np.ndarray], grand_mean: float) -> float:
    """Sum of squares between groups."""
    return sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)


def two_way_anova(
    df: pd.DataFrame,
    metric: str,
    factor_a: str,
    factor_b: str,
) -> dict:
    """
    Compute Type II two-way ANOVA for a metric with two categorical factors.

    Returns dict with main effects, interaction, F-stats, p-values, eta-squared.
    """
    y = df[metric].values
    grand_mean = y.mean()
    ss_total = np.sum((y - grand_mean) ** 2)
    n_total = len(y)

    # Main effect A
    groups_a = [df.loc[df[factor_a] == lvl, metric].values
                for lvl in sorted(df[factor_a].unique())]
    ss_a = _ss_between(groups_a, grand_mean)
    df_a = len(groups_a) - 1

    # Main effect B
    groups_b = [df.loc[df[factor_b] == lvl, metric].values
                for lvl in sorted(df[factor_b].unique())]
    ss_b = _ss_between(groups_b, grand_mean)
    df_b = len(groups_b) - 1

    # Cell means for interaction
    cells = {}
    for a_lvl in sorted(df[factor_a].unique()):
        for b_lvl in sorted(df[factor_b].unique()):
            mask = (df[factor_a] == a_lvl) & (df[factor_b] == b_lvl)
            cells[(a_lvl, b_lvl)] = df.loc[mask, metric].values

    # SS for the full model (all cells)
    ss_cells = sum(
        len(g) * (g.mean() - grand_mean) ** 2
        for g in cells.values() if len(g) > 0
    )
    ss_ab = ss_cells - ss_a - ss_b
    df_ab = df_a * df_b

    # Residual
    ss_resid = ss_total - ss_cells
    df_resid = n_total - len(cells)

    if df_resid <= 0 or ss_resid <= 0:
        ms_resid = 1e-10
    else:
        ms_resid = ss_resid / df_resid

    # F-statistics
    def _f_and_p(ss, df_effect):
        if df_effect == 0:
            return 0.0, 1.0
        ms = ss / df_effect
        f_stat = ms / ms_resid
        p_val = 1.0 - stats.f.cdf(f_stat, df_effect, df_resid)
        return float(f_stat), float(p_val)

    f_a, p_a = _f_and_p(ss_a, df_a)
    f_b, p_b = _f_and_p(ss_b, df_b)
    f_ab, p_ab = _f_and_p(ss_ab, df_ab)

    # Effect sizes (eta-squared)
    eta2_a = ss_a / ss_total if ss_total > 0 else 0
    eta2_b = ss_b / ss_total if ss_total > 0 else 0
    eta2_ab = ss_ab / ss_total if ss_total > 0 else 0

    return {
        "metric": metric,
        "factor_a": factor_a,
        "factor_b": factor_b,
        "main_effect_a": {
            "SS": round(float(ss_a), 4), "df": df_a,
            "F": round(f_a, 4), "p": round(p_a, 6),
            "eta_squared": round(float(eta2_a), 4),
        },
        "main_effect_b": {
            "SS": round(float(ss_b), 4), "df": df_b,
            "F": round(f_b, 4), "p": round(p_b, 6),
            "eta_squared": round(float(eta2_b), 4),
        },
        "interaction": {
            "SS": round(float(ss_ab), 4), "df": df_ab,
            "F": round(f_ab, 4), "p": round(p_ab, 6),
            "eta_squared": round(float(eta2_ab), 4),
        },
        "residual": {
            "SS": round(float(ss_resid), 4), "df": df_resid,
        },
    }


# ---------------------------------------------------------------------------
# 3. Post-hoc pairwise comparisons (Bonferroni-corrected t-tests)
# ---------------------------------------------------------------------------

def pairwise_comparisons(
    df: pd.DataFrame,
    metric: str,
    factor: str,
) -> list[dict]:
    """Bonferroni-corrected pairwise t-tests for all levels of a factor."""
    levels = sorted(df[factor].unique())
    n_comparisons = len(levels) * (len(levels) - 1) // 2
    results = []

    for a, b in combinations(levels, 2):
        vals_a = df.loc[df[factor] == a, metric].values
        vals_b = df.loc[df[factor] == b, metric].values
        t_stat, p_raw = stats.ttest_ind(vals_a, vals_b, equal_var=False)
        p_adj = min(p_raw * n_comparisons, 1.0)
        diff = float(vals_a.mean() - vals_b.mean())
        results.append({
            "pair": f"{a} vs {b}",
            "mean_diff": round(diff, 4),
            "t_stat": round(float(t_stat), 4),
            "p_raw": round(float(p_raw), 6),
            "p_bonferroni": round(float(p_adj), 6),
            "significant": bool(p_adj < 0.05),
        })

    return results


# ---------------------------------------------------------------------------
# 4. Cell means table
# ---------------------------------------------------------------------------

def cell_means_table(
    df: pd.DataFrame,
    metric: str,
    factor_a: str,
    factor_b: str,
) -> dict:
    """Compute mean ± std for each cell in the factor_a × factor_b grid."""
    table = {}
    for a_lvl in sorted(df[factor_a].unique()):
        row = {}
        for b_lvl in sorted(df[factor_b].unique()):
            mask = (df[factor_a] == a_lvl) & (df[factor_b] == b_lvl)
            vals = df.loc[mask, metric]
            row[b_lvl] = {
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std()), 4),
                "n": int(len(vals)),
            }
        table[a_lvl] = row
    return table


# ---------------------------------------------------------------------------
# 5. Experiment 4.1 — Mechanism Isolation
# ---------------------------------------------------------------------------

def experiment_4_1(df: pd.DataFrame) -> dict:
    """
    Full factorial analysis: scenario (S1-S5) × timestamp_condition (actual/none/misleading).
    Two-way ANOVA + post-hoc + cell means for each metric.
    Also stratified by predicted_mechanism.
    """
    results: dict = {"experiment": "4.1", "description": "Mechanism Isolation"}

    # Overall ANOVA for each metric
    anova_results = {}
    posthoc_results = {}
    cell_means = {}

    for metric in _METRICS:
        anova_results[metric] = two_way_anova(
            df, metric, "scenario", "timestamp_condition"
        )
        posthoc_results[metric] = {
            "by_scenario": pairwise_comparisons(df, metric, "scenario"),
            "by_timestamp_condition": pairwise_comparisons(
                df, metric, "timestamp_condition"
            ),
        }
        cell_means[metric] = cell_means_table(
            df, metric, "scenario", "timestamp_condition"
        )

    results["anova"] = anova_results
    results["posthoc"] = posthoc_results
    results["cell_means"] = cell_means

    # Stratified by mechanism
    mechanism_results = {}
    for mech in sorted(df["predicted_mechanism"].unique()):
        mech_df = df[df["predicted_mechanism"] == mech]
        if len(mech_df) < 30:
            continue
        mech_anova = {}
        for metric in _METRICS:
            mech_anova[metric] = two_way_anova(
                mech_df, metric, "scenario", "timestamp_condition"
            )
        mechanism_results[mech] = {
            "n": len(mech_df),
            "anova": mech_anova,
        }
    results["by_mechanism"] = mechanism_results

    return results


# ---------------------------------------------------------------------------
# 6. Pretty printing
# ---------------------------------------------------------------------------

def _sig_marker(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def print_experiment_4_1(results: dict) -> None:
    """Print formatted results for Experiment 4.1."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4.1: MECHANISM ISOLATION")
    print("Full factorial ANOVA: scenario (S1-S5) × timestamp_condition")
    print("=" * 80)

    # ANOVA summary table
    print(f"\n{'Metric':<6s}  {'Effect':<25s}  {'F':>8s}  {'p':>10s}  {'η²':>6s}  {'Sig':>4s}")
    print("-" * 70)
    for metric in _METRICS:
        anova = results["anova"][metric]
        for label, key in [
            ("scenario", "main_effect_a"),
            ("timestamp_condition", "main_effect_b"),
            ("scenario × ts_cond", "interaction"),
        ]:
            eff = anova[key]
            sig = _sig_marker(eff["p"])
            print(f"{metric:<6s}  {label:<25s}  {eff['F']:>8.2f}  {eff['p']:>10.6f}  "
                  f"{eff['eta_squared']:>6.4f}  {sig:>4s}")
        print()

    # Cell means for FA (the primary metric)
    print("\n--- Cell Means: FA (Factual Accuracy) ---")
    cm = results["cell_means"]["FA"]
    scenarios = sorted(cm.keys())
    ts_conds = sorted(next(iter(cm.values())).keys())
    print(f"{'':>5s}  " + "  ".join(f"{t:>16s}" for t in ts_conds))
    for scenario in scenarios:
        row = cm[scenario]
        cells = "  ".join(
            f"{row[t]['mean']:>6.3f} ±{row[t]['std']:>5.3f} (n={row[t]['n']})"
            for t in ts_conds
        )
        print(f"{scenario:>5s}  {cells}")

    # Cell means for POS
    print("\n--- Cell Means: POS (Preference for Outdated) ---")
    cm = results["cell_means"]["POS"]
    print(f"{'':>5s}  " + "  ".join(f"{t:>16s}" for t in ts_conds))
    for scenario in scenarios:
        row = cm[scenario]
        cells = "  ".join(
            f"{row[t]['mean']:>6.3f} ±{row[t]['std']:>5.3f} (n={row[t]['n']})"
            for t in ts_conds
        )
        print(f"{scenario:>5s}  {cells}")

    # Significant post-hoc comparisons for FA
    print("\n--- Significant Pairwise Comparisons (FA, Bonferroni-corrected) ---")
    for factor_label, factor_key in [
        ("By scenario", "by_scenario"),
        ("By timestamp condition", "by_timestamp_condition"),
    ]:
        print(f"\n  {factor_label}:")
        comparisons = results["posthoc"]["FA"][factor_key]
        sig_comps = [c for c in comparisons if c["significant"]]
        if not sig_comps:
            print("    (no significant differences)")
        for c in sig_comps:
            print(f"    {c['pair']:>25s}  diff={c['mean_diff']:>+.4f}  "
                  f"p={c['p_bonferroni']:.6f} {_sig_marker(c['p_bonferroni'])}")

    # Mechanism breakdown
    if results.get("by_mechanism"):
        print("\n--- ANOVA by Mechanism (FA only, scenario effect) ---")
        print(f"{'Mechanism':<10s}  {'n':>5s}  {'F':>8s}  {'p':>10s}  {'η²':>6s}  {'Sig':>4s}")
        print("-" * 50)
        for mech, mech_data in sorted(results["by_mechanism"].items()):
            eff = mech_data["anova"]["FA"]["main_effect_a"]
            sig = _sig_marker(eff["p"])
            print(f"{mech:<10s}  {mech_data['n']:>5d}  {eff['F']:>8.2f}  "
                  f"{eff['p']:>10.6f}  {eff['eta_squared']:>6.4f}  {sig:>4s}")


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FreshRAG experiment analysis — statistical tests on scored instances",
    )
    parser.add_argument(
        "--experiment", "-e",
        required=True,
        choices=["4.1"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--scored", "-s",
        required=True,
        help="Path to scored JSONL file",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON path (default: data/benchmark/experiment_{id}.json)",
    )

    args = parser.parse_args()

    scored_path = Path(args.scored)
    if not scored_path.exists():
        sys.exit(f"Error: scored file not found: {scored_path}")

    df = load_scored(scored_path)

    if args.experiment == "4.1":
        results = experiment_4_1(df)
        print_experiment_4_1(results)
    else:
        sys.exit(f"Unknown experiment: {args.experiment}")

    # Save results
    output_path = (
        Path(args.output) if args.output
        else scored_path.parent / f"experiment_{args.experiment.replace('.', '_')}.json"
    )
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
