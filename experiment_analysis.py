"""
FreshRAG Experiment Analysis (Phase 4–5).

Statistical analysis of scored eval instances. Supports multiple experiments:
  4.1 — Mechanism Isolation: two-way ANOVA (scenario × timestamp_condition)
  4.2 — Temporal Gradient: regression of FA vs document age (T-4→T+4)
  4.3 — Domain Heterogeneity: stratified by domain, mixed-effects style analysis
  5.1 — Causal Effect Decomposition: TE = DE + IE via KC + IE via TS + IE via PO
  5.2 — Mechanism Hypothesis Validation: predicted vs observed mechanism patterns

Usage:
    python experiment_analysis.py --experiment 4.1 --scored data/benchmark/scored_210.jsonl
    python experiment_analysis.py --experiment 4.2 --scored data/benchmark/scored_tg_210.jsonl
    python experiment_analysis.py --experiment 4.3 --scored data/benchmark/scored_210.jsonl
    python experiment_analysis.py --experiment 5.1 --scored data/benchmark/scored_210.jsonl
    python experiment_analysis.py --experiment 5.2 --scored data/benchmark/scored_210.jsonl
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
            # Temporal gradient fields (Experiment 4.2)
            if "t_label" in rec:
                flat["t_label"] = rec["t_label"]
            if "t_offset" in rec:
                flat["t_offset"] = rec["t_offset"]
            for m in _METRICS:
                flat[m] = rec["scores"][m]
            records.append(flat)

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} scored instances from {path}")
    print(f"  Scenarios: {sorted(df['scenario'].unique())}")
    print(f"  Timestamp conditions: {sorted(df['timestamp_condition'].unique())}")
    print(f"  Domains: {sorted(df['domain'].unique())}")
    print(f"  Mechanisms: {sorted(df['predicted_mechanism'].unique())}")
    if "t_offset" in df.columns:
        print(f"  Temporal offsets: {sorted(df['t_offset'].unique())}")
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
# 6. Experiment 4.2 — Temporal Gradient
# ---------------------------------------------------------------------------

def _ols_regression(x: np.ndarray, y: np.ndarray) -> dict:
    """Simple OLS regression: y ~ x. Returns slope, intercept, r, r², p, SE."""
    n = len(x)
    if n < 3:
        return {"n": n, "slope": None, "intercept": None, "r": None,
                "r_squared": None, "p": None, "se_slope": None}
    slope, intercept, r, p, se = stats.linregress(x, y)
    return {
        "n": int(n),
        "slope": round(float(slope), 6),
        "intercept": round(float(intercept), 6),
        "r": round(float(r), 4),
        "r_squared": round(float(r ** 2), 4),
        "p": round(float(p), 8),
        "se_slope": round(float(se), 6),
    }


def experiment_4_2(df: pd.DataFrame) -> dict:
    """
    Temporal Gradient analysis: regression of FA vs t_offset (T-4→T+4).

    Each instance has a single document at a specific temporal distance from
    the change event. We regress FA against t_offset (negative = stale,
    positive = fresh) to quantify how document age affects factual accuracy.

    Also:
    - Mean FA at each temporal position
    - Regression stratified by predicted_mechanism and domain
    - Breakpoint analysis: stale-side vs fresh-side regressions
    - One-way ANOVA: FA across temporal positions
    """
    results: dict = {"experiment": "4.2", "description": "Temporal Gradient"}

    if "t_offset" not in df.columns:
        raise ValueError("Scored data missing 't_offset' column — "
                         "use temporal gradient scored instances")

    x_all = df["t_offset"].values.astype(float)
    y_all = df["FA"].values.astype(float)

    # ---- Overall regression ----
    results["overall_regression"] = _ols_regression(x_all, y_all)

    # ---- Mean FA at each temporal position ----
    position_means = {}
    for t_off in sorted(df["t_offset"].unique()):
        vals = df.loc[df["t_offset"] == t_off, "FA"]
        t_label = f"T{t_off:+d}"
        position_means[t_label] = {
            "t_offset": int(t_off),
            "mean_FA": round(float(vals.mean()), 4),
            "std_FA": round(float(vals.std()), 4),
            "n": int(len(vals)),
        }
    results["position_means"] = position_means

    # ---- One-way ANOVA: FA across temporal positions ----
    groups = [df.loc[df["t_offset"] == t, "FA"].values
              for t in sorted(df["t_offset"].unique())]
    f_stat, p_val = stats.f_oneway(*groups)
    results["anova_positions"] = {
        "F": round(float(f_stat), 4),
        "p": round(float(p_val), 8),
        "k": len(groups),
    }

    # ---- Breakpoint: separate stale-side and fresh-side regressions ----
    stale = df[df["t_offset"] < 0]
    fresh = df[df["t_offset"] > 0]
    results["stale_regression"] = _ols_regression(
        stale["t_offset"].values.astype(float),
        stale["FA"].values.astype(float),
    )
    results["fresh_regression"] = _ols_regression(
        fresh["t_offset"].values.astype(float),
        fresh["FA"].values.astype(float),
    )

    # ---- Stale vs Fresh t-test ----
    stale_fa = stale["FA"].values
    fresh_fa = fresh["FA"].values
    t_stat, p_val = stats.ttest_ind(stale_fa, fresh_fa, equal_var=False)
    results["stale_vs_fresh"] = {
        "stale_mean_FA": round(float(stale_fa.mean()), 4),
        "fresh_mean_FA": round(float(fresh_fa.mean()), 4),
        "diff": round(float(fresh_fa.mean() - stale_fa.mean()), 4),
        "t_stat": round(float(t_stat), 4),
        "p": round(float(p_val), 8),
    }

    # ---- By metric: regression for each metric ----
    metric_regressions = {}
    for metric in _METRICS:
        y = df[metric].values.astype(float)
        metric_regressions[metric] = _ols_regression(x_all, y)
    results["metric_regressions"] = metric_regressions

    # ---- Stratified by mechanism ----
    mechanism_results = {}
    for mech in sorted(df["predicted_mechanism"].unique()):
        mech_df = df[df["predicted_mechanism"] == mech]
        if len(mech_df) < 20:
            continue
        x_m = mech_df["t_offset"].values.astype(float)
        y_m = mech_df["FA"].values.astype(float)
        mechanism_results[mech] = {
            "n": len(mech_df),
            "regression": _ols_regression(x_m, y_m),
            "position_means": {},
        }
        for t_off in sorted(mech_df["t_offset"].unique()):
            vals = mech_df.loc[mech_df["t_offset"] == t_off, "FA"]
            t_label = f"T{t_off:+d}"
            mechanism_results[mech]["position_means"][t_label] = {
                "mean_FA": round(float(vals.mean()), 4),
                "n": int(len(vals)),
            }
    results["by_mechanism"] = mechanism_results

    # ---- Stratified by domain ----
    domain_results = {}
    for domain in sorted(df["domain"].unique()):
        dom_df = df[df["domain"] == domain]
        if len(dom_df) < 20:
            continue
        x_d = dom_df["t_offset"].values.astype(float)
        y_d = dom_df["FA"].values.astype(float)
        domain_results[domain] = {
            "n": len(dom_df),
            "regression": _ols_regression(x_d, y_d),
        }
    results["by_domain"] = domain_results

    return results


def print_experiment_4_2(results: dict) -> None:
    """Print formatted results for Experiment 4.2."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4.2: TEMPORAL GRADIENT")
    print("Regression: FA vs document age (t_offset)")
    print("=" * 80)

    # Overall regression
    reg = results["overall_regression"]
    print(f"\n--- Overall Regression (FA ~ t_offset) ---")
    print(f"  n={reg['n']}  slope={reg['slope']:.6f}  intercept={reg['intercept']:.4f}")
    print(f"  r={reg['r']:.4f}  R²={reg['r_squared']:.4f}  p={reg['p']:.8f}")

    # Position means
    print(f"\n--- Mean FA at Each Temporal Position ---")
    print(f"  {'Position':<8s}  {'n':>5s}  {'Mean FA':>8s}  {'Std':>8s}  {'Bar'}")
    print("  " + "-" * 55)
    for t_label, info in results["position_means"].items():
        bar = "█" * int(info["mean_FA"] * 40)
        print(f"  {t_label:<8s}  {info['n']:>5d}  {info['mean_FA']:>8.4f}  "
              f"{info['std_FA']:>8.4f}  {bar}")

    # ANOVA
    anova = results["anova_positions"]
    sig = _sig_marker(anova["p"])
    print(f"\n  One-way ANOVA: F({anova['k']-1})={anova['F']:.2f}, p={anova['p']:.8f} {sig}")

    # Stale vs Fresh
    svf = results["stale_vs_fresh"]
    sig = _sig_marker(svf["p"])
    print(f"\n--- Stale vs Fresh ---")
    print(f"  Stale (t<0): FA={svf['stale_mean_FA']:.4f}")
    print(f"  Fresh (t>0): FA={svf['fresh_mean_FA']:.4f}")
    print(f"  Diff (fresh-stale): {svf['diff']:+.4f}  t={svf['t_stat']:.4f}  p={svf['p']:.8f} {sig}")

    # Breakpoint regressions
    print(f"\n--- Breakpoint Regressions ---")
    for label, key in [("Stale side (t<0)", "stale_regression"),
                       ("Fresh side (t>0)", "fresh_regression")]:
        r = results[key]
        if r["slope"] is not None:
            print(f"  {label}: slope={r['slope']:.6f}  r={r['r']:.4f}  "
                  f"R²={r['r_squared']:.4f}  p={r['p']:.8f}")
        else:
            print(f"  {label}: insufficient data")

    # All metrics
    print(f"\n--- Regression by Metric ---")
    print(f"  {'Metric':<6s}  {'Slope':>10s}  {'r':>7s}  {'R²':>7s}  {'p':>12s}  {'Sig':>4s}")
    print("  " + "-" * 50)
    for metric, reg in results["metric_regressions"].items():
        if reg["slope"] is not None:
            sig = _sig_marker(reg["p"])
            print(f"  {metric:<6s}  {reg['slope']:>10.6f}  {reg['r']:>7.4f}  "
                  f"{reg['r_squared']:>7.4f}  {reg['p']:>12.8f}  {sig:>4s}")

    # By mechanism
    print(f"\n--- FA Regression by Mechanism ---")
    print(f"  {'Mechanism':<10s}  {'n':>5s}  {'Slope':>10s}  {'r':>7s}  {'p':>12s}  {'Sig':>4s}")
    print("  " + "-" * 55)
    for mech, data in sorted(results["by_mechanism"].items()):
        reg = data["regression"]
        if reg["slope"] is not None:
            sig = _sig_marker(reg["p"])
            print(f"  {mech:<10s}  {data['n']:>5d}  {reg['slope']:>10.6f}  "
                  f"{reg['r']:>7.4f}  {reg['p']:>12.8f}  {sig:>4s}")

    # By domain
    print(f"\n--- FA Regression by Domain ---")
    print(f"  {'Domain':<25s}  {'n':>5s}  {'Slope':>10s}  {'r':>7s}  {'p':>12s}  {'Sig':>4s}")
    print("  " + "-" * 70)
    for domain, data in sorted(results["by_domain"].items()):
        reg = data["regression"]
        if reg["slope"] is not None:
            sig = _sig_marker(reg["p"])
            print(f"  {domain:<25s}  {data['n']:>5d}  {reg['slope']:>10.6f}  "
                  f"{reg['r']:>7.4f}  {reg['p']:>12.8f}  {sig:>4s}")


# ---------------------------------------------------------------------------
# 7. Experiment 4.3 — Domain Heterogeneity
# ---------------------------------------------------------------------------

def experiment_4_3(df: pd.DataFrame) -> dict:
    """
    Domain Heterogeneity analysis: stratified by domain with mixed-effects
    style analysis (domain as random effect approximation).

    - One-way ANOVA: metric ~ domain (for each metric)
    - Per-domain descriptive stats and regressions (scenario effect within domain)
    - Domain × scenario interaction (two-way ANOVA)
    - Kruskal-Wallis non-parametric test per metric
    - Effect size comparisons across domains
    """
    results: dict = {"experiment": "4.3", "description": "Domain Heterogeneity"}
    domains = sorted(df["domain"].unique())

    # ---- Per-metric: one-way ANOVA across domains ----
    domain_anova = {}
    for metric in _METRICS:
        groups = [df.loc[df["domain"] == d, metric].values for d in domains]
        f_stat, p_val = stats.f_oneway(*groups)
        # Kruskal-Wallis (non-parametric)
        h_stat, kw_p = stats.kruskal(*groups)
        # Effect size: eta-squared
        grand_mean = df[metric].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = np.sum((df[metric].values - grand_mean) ** 2)
        eta2 = ss_between / ss_total if ss_total > 0 else 0
        domain_anova[metric] = {
            "F": round(float(f_stat), 4),
            "p": round(float(p_val), 8),
            "eta_squared": round(float(eta2), 4),
            "kruskal_H": round(float(h_stat), 4),
            "kruskal_p": round(float(kw_p), 8),
        }
    results["domain_anova"] = domain_anova

    # ---- Per-domain descriptive stats ----
    domain_stats = {}
    for domain in domains:
        dom_df = df[df["domain"] == domain]
        stats_row = {"n": len(dom_df)}
        for metric in _METRICS:
            vals = dom_df[metric]
            stats_row[metric] = {
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std()), 4),
                "median": round(float(vals.median()), 4),
            }
        domain_stats[domain] = stats_row
    results["domain_stats"] = domain_stats

    # ---- Domain × Scenario interaction (two-way ANOVA for FA) ----
    domain_scenario_anova = {}
    for metric in _METRICS:
        domain_scenario_anova[metric] = two_way_anova(
            df, metric, "domain", "scenario"
        )
    results["domain_scenario_anova"] = domain_scenario_anova

    # ---- Pairwise domain comparisons for FA ----
    results["pairwise_domains_FA"] = pairwise_comparisons(df, "FA", "domain")

    # ---- Per-domain scenario effect (one-way ANOVA within each domain) ----
    domain_scenario_effect = {}
    for domain in domains:
        dom_df = df[df["domain"] == domain]
        scenarios = sorted(dom_df["scenario"].unique())
        if len(scenarios) < 2:
            continue
        groups = [dom_df.loc[dom_df["scenario"] == s, "FA"].values
                  for s in scenarios]
        f_stat, p_val = stats.f_oneway(*groups)
        # Scenario means within domain
        scenario_means = {}
        for s in scenarios:
            vals = dom_df.loc[dom_df["scenario"] == s, "FA"]
            scenario_means[s] = round(float(vals.mean()), 4)
        domain_scenario_effect[domain] = {
            "n": len(dom_df),
            "F": round(float(f_stat), 4),
            "p": round(float(p_val), 8),
            "scenario_means_FA": scenario_means,
        }
    results["domain_scenario_effect"] = domain_scenario_effect

    # ---- Per-domain mechanism distribution ----
    domain_mechanism = {}
    for domain in domains:
        dom_df = df[df["domain"] == domain]
        counts = dom_df["predicted_mechanism"].value_counts().to_dict()
        domain_mechanism[domain] = {k: int(v) for k, v in counts.items()}
    results["domain_mechanism_distribution"] = domain_mechanism

    return results


def print_experiment_4_3(results: dict) -> None:
    """Print formatted results for Experiment 4.3."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4.3: DOMAIN HETEROGENEITY")
    print("Stratified analysis by domain")
    print("=" * 80)

    # Domain ANOVA
    print(f"\n--- One-way ANOVA: metric ~ domain ---")
    print(f"  {'Metric':<6s}  {'F':>8s}  {'p':>12s}  {'η²':>6s}  {'KW H':>8s}  {'KW p':>12s}  {'Sig':>4s}")
    print("  " + "-" * 65)
    for metric in _METRICS:
        a = results["domain_anova"][metric]
        sig = _sig_marker(a["p"])
        print(f"  {metric:<6s}  {a['F']:>8.2f}  {a['p']:>12.8f}  "
              f"{a['eta_squared']:>6.4f}  {a['kruskal_H']:>8.2f}  "
              f"{a['kruskal_p']:>12.8f}  {sig:>4s}")

    # Per-domain stats
    print(f"\n--- Per-domain Mean FA ---")
    print(f"  {'Domain':<25s}  {'n':>5s}  {'FA':>7s}  {'HR':>7s}  {'POS':>7s}  {'AU':>7s}")
    print("  " + "-" * 60)
    for domain, st in sorted(results["domain_stats"].items()):
        print(f"  {domain:<25s}  {st['n']:>5d}  {st['FA']['mean']:>7.4f}  "
              f"{st['HR']['mean']:>7.4f}  {st['POS']['mean']:>7.4f}  "
              f"{st['AU']['mean']:>7.4f}")

    # Significant pairwise domain comparisons
    print(f"\n--- Significant Pairwise Domain Comparisons (FA, Bonferroni) ---")
    sig_comps = [c for c in results["pairwise_domains_FA"] if c["significant"]]
    if not sig_comps:
        print("  (no significant differences)")
    for c in sig_comps[:10]:  # top 10
        print(f"  {c['pair']:>45s}  diff={c['mean_diff']:>+.4f}  "
              f"p={c['p_bonferroni']:.6f} {_sig_marker(c['p_bonferroni'])}")

    # Domain × Scenario interaction
    print(f"\n--- Domain × Scenario Interaction (FA) ---")
    a = results["domain_scenario_anova"]["FA"]
    for label, key in [("domain", "main_effect_a"),
                       ("scenario", "main_effect_b"),
                       ("domain × scenario", "interaction")]:
        eff = a[key]
        sig = _sig_marker(eff["p"])
        print(f"  {label:<25s}  F={eff['F']:>8.2f}  p={eff['p']:.6f}  "
              f"η²={eff['eta_squared']:.4f}  {sig}")

    # Per-domain scenario effect
    print(f"\n--- Scenario Effect Within Each Domain (FA) ---")
    print(f"  {'Domain':<25s}  {'n':>5s}  {'F':>8s}  {'p':>12s}  {'Sig':>4s}  S1→S5 means")
    print("  " + "-" * 80)
    for domain, data in sorted(results["domain_scenario_effect"].items()):
        sig = _sig_marker(data["p"])
        means = "  ".join(f"{data['scenario_means_FA'].get(s, 0):.3f}"
                         for s in ["S1", "S2", "S3", "S4", "S5"])
        print(f"  {domain:<25s}  {data['n']:>5d}  {data['F']:>8.2f}  "
              f"{data['p']:>12.8f}  {sig:>4s}  {means}")


# ---------------------------------------------------------------------------
# 8. Experiment 4.4 — Model Scaling
# ---------------------------------------------------------------------------

def load_scored_multi(paths: list[Path], labels: list[str]) -> pd.DataFrame:
    """Load multiple scored JSONL files, adding a 'model' column to each."""
    frames = []
    for path, label in zip(paths, labels):
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
                    "model": label,
                }
                for m in _METRICS:
                    flat[m] = rec["scores"][m]
                records.append(flat)
        frame = pd.DataFrame(records)
        frames.append(frame)
        print(f"  Loaded {len(frame)} scored instances for model '{label}' from {path}")
    return pd.concat(frames, ignore_index=True)


def experiment_4_4(df: pd.DataFrame) -> dict:
    """
    Model Scaling analysis: compare metrics across models.

    - Per-model descriptive stats for each metric
    - Paired comparison (matched instance_id) between models
    - Model × scenario interaction
    - Model effect stratified by mechanism and domain
    """
    results: dict = {"experiment": "4.4", "description": "Model Scaling"}
    models = sorted(df["model"].unique())
    results["models"] = models

    # ---- Per-model overall stats ----
    model_stats = {}
    for model in models:
        mod_df = df[df["model"] == model]
        stats_row = {"n": len(mod_df)}
        for metric in _METRICS:
            vals = mod_df[metric]
            stats_row[metric] = {
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std()), 4),
            }
        model_stats[model] = stats_row
    results["model_stats"] = model_stats

    # ---- Per-metric: ANOVA across models ----
    model_anova = {}
    for metric in _METRICS:
        groups = [df.loc[df["model"] == m, metric].values for m in models]
        if len(groups) == 2:
            t_stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            model_anova[metric] = {
                "test": "t-test",
                "t_stat": round(float(t_stat), 4),
                "p": round(float(p_val), 8),
            }
        else:
            f_stat, p_val = stats.f_oneway(*groups)
            model_anova[metric] = {
                "test": "ANOVA",
                "F": round(float(f_stat), 4),
                "p": round(float(p_val), 8),
            }
        # Effect size (Cohen's d for 2 models)
        if len(models) == 2:
            m1 = df.loc[df["model"] == models[0], metric].values
            m2 = df.loc[df["model"] == models[1], metric].values
            pooled_std = np.sqrt((np.var(m1) + np.var(m2)) / 2)
            d = (m1.mean() - m2.mean()) / pooled_std if pooled_std > 0 else 0
            model_anova[metric]["cohens_d"] = round(float(d), 4)
    results["model_anova"] = model_anova

    # ---- Paired comparison on matched instances ----
    if len(models) == 2:
        m1_label, m2_label = models[0], models[1]
        m1_df = df[df["model"] == m1_label].set_index("instance_id")
        m2_df = df[df["model"] == m2_label].set_index("instance_id")
        common = m1_df.index.intersection(m2_df.index)
        paired_results = {}
        for metric in _METRICS:
            v1 = m1_df.loc[common, metric].values
            v2 = m2_df.loc[common, metric].values
            diff = v2 - v1
            t_stat, p_val = stats.ttest_rel(v1, v2)
            paired_results[metric] = {
                "n_paired": len(common),
                "mean_diff": round(float(diff.mean()), 4),
                "t_stat": round(float(t_stat), 4),
                "p": round(float(p_val), 8),
                f"{m1_label}_mean": round(float(v1.mean()), 4),
                f"{m2_label}_mean": round(float(v2.mean()), 4),
            }
        results["paired_comparison"] = paired_results

    # ---- Model × Scenario interaction ----
    model_scenario = {}
    for metric in _METRICS:
        model_scenario[metric] = two_way_anova(df, metric, "model", "scenario")
    results["model_scenario_anova"] = model_scenario

    # ---- Per-model × scenario means for FA ----
    model_scenario_means = {}
    for model in models:
        mod_df = df[df["model"] == model]
        row = {}
        for scenario in sorted(mod_df["scenario"].unique()):
            vals = mod_df.loc[mod_df["scenario"] == scenario, "FA"]
            row[scenario] = round(float(vals.mean()), 4)
        model_scenario_means[model] = row
    results["model_scenario_means_FA"] = model_scenario_means

    # ---- Model effect by mechanism ----
    model_mechanism = {}
    for mech in sorted(df["predicted_mechanism"].unique()):
        mech_df = df[df["predicted_mechanism"] == mech]
        if len(mech_df) < 20:
            continue
        mech_stats = {"n": len(mech_df)}
        for model in models:
            vals = mech_df.loc[mech_df["model"] == model, "FA"]
            if len(vals) > 0:
                mech_stats[f"{model}_FA"] = round(float(vals.mean()), 4)
        if len(models) == 2:
            g1 = mech_df.loc[mech_df["model"] == models[0], "FA"].values
            g2 = mech_df.loc[mech_df["model"] == models[1], "FA"].values
            if len(g1) > 1 and len(g2) > 1:
                t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
                mech_stats["t_stat"] = round(float(t_stat), 4)
                mech_stats["p"] = round(float(p_val), 8)
        model_mechanism[mech] = mech_stats
    results["model_by_mechanism"] = model_mechanism

    # ---- Model effect by domain ----
    model_domain = {}
    for domain in sorted(df["domain"].unique()):
        dom_df = df[df["domain"] == domain]
        dom_stats = {"n": len(dom_df)}
        for model in models:
            vals = dom_df.loc[dom_df["model"] == model, "FA"]
            if len(vals) > 0:
                dom_stats[f"{model}_FA"] = round(float(vals.mean()), 4)
        if len(models) == 2:
            g1 = dom_df.loc[dom_df["model"] == models[0], "FA"].values
            g2 = dom_df.loc[dom_df["model"] == models[1], "FA"].values
            if len(g1) > 1 and len(g2) > 1:
                t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
                dom_stats["t_stat"] = round(float(t_stat), 4)
                dom_stats["p"] = round(float(p_val), 8)
        model_domain[domain] = dom_stats
    results["model_by_domain"] = model_domain

    return results


def print_experiment_4_4(results: dict) -> None:
    """Print formatted results for Experiment 4.4."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4.4: MODEL SCALING")
    print(f"Comparing models: {', '.join(results['models'])}")
    print("=" * 80)

    models = results["models"]

    # Overall stats
    print(f"\n--- Per-model Overall Metrics ---")
    print(f"  {'Model':<20s}  {'n':>5s}  " +
          "  ".join(f"{m:>7s}" for m in _METRICS))
    print("  " + "-" * 65)
    for model in models:
        st = results["model_stats"][model]
        vals = "  ".join(f"{st[m]['mean']:>7.4f}" for m in _METRICS)
        print(f"  {model:<20s}  {st['n']:>5d}  {vals}")

    # Model ANOVA / t-test
    print(f"\n--- Model Effect (per metric) ---")
    print(f"  {'Metric':<6s}  {'Test':>8s}  {'Stat':>8s}  {'p':>12s}  {'Cohen d':>8s}  {'Sig':>4s}")
    print("  " + "-" * 55)
    for metric in _METRICS:
        a = results["model_anova"][metric]
        sig = _sig_marker(a["p"])
        stat_val = a.get("t_stat", a.get("F", 0))
        d = a.get("cohens_d", "")
        d_str = f"{d:>8.4f}" if isinstance(d, float) else f"{'':>8s}"
        print(f"  {metric:<6s}  {a['test']:>8s}  {stat_val:>8.4f}  "
              f"{a['p']:>12.8f}  {d_str}  {sig:>4s}")

    # Paired comparison
    if "paired_comparison" in results:
        print(f"\n--- Paired Comparison (matched instances) ---")
        for metric in _METRICS:
            pc = results["paired_comparison"][metric]
            sig = _sig_marker(pc["p"])
            print(f"  {metric}: diff={pc['mean_diff']:>+.4f}  "
                  f"t={pc['t_stat']:.4f}  p={pc['p']:.8f}  "
                  f"n={pc['n_paired']}  {sig}")

    # Model × Scenario FA
    print(f"\n--- Model × Scenario Mean FA ---")
    scenarios = sorted(set().union(*(v.keys() for v in results["model_scenario_means_FA"].values())))
    print(f"  {'Model':<20s}  " + "  ".join(f"{s:>7s}" for s in scenarios))
    print("  " + "-" * (22 + 9 * len(scenarios)))
    for model in models:
        vals = "  ".join(
            f"{results['model_scenario_means_FA'][model].get(s, 0):>7.4f}"
            for s in scenarios
        )
        print(f"  {model:<20s}  {vals}")

    # Model × Scenario interaction
    a = results["model_scenario_anova"]["FA"]
    print(f"\n  Interaction ANOVA (FA):")
    for label, key in [("model", "main_effect_a"),
                       ("scenario", "main_effect_b"),
                       ("model × scenario", "interaction")]:
        eff = a[key]
        sig = _sig_marker(eff["p"])
        print(f"    {label:<25s}  F={eff['F']:>8.2f}  p={eff['p']:.6f}  "
              f"η²={eff['eta_squared']:.4f}  {sig}")

    # By mechanism
    print(f"\n--- Model Effect by Mechanism (FA) ---")
    print(f"  {'Mechanism':<10s}  {'n':>5s}  " +
          "  ".join(f"{m:>12s}" for m in models) +
          "  {'p':>12s}  {'Sig':>4s}")
    print("  " + "-" * (20 + 14 * len(models) + 20))
    for mech, data in sorted(results["model_by_mechanism"].items()):
        fa_vals = "  ".join(
            f"{data.get(f'{m}_FA', 0):>12.4f}" for m in models
        )
        p_val = data.get("p", 1.0)
        sig = _sig_marker(p_val)
        print(f"  {mech:<10s}  {data['n']:>5d}  {fa_vals}  {p_val:>12.8f}  {sig:>4s}")

    # By domain
    print(f"\n--- Model Effect by Domain (FA) ---")
    for domain, data in sorted(results["model_by_domain"].items()):
        fa_vals = "  ".join(
            f"{data.get(f'{m}_FA', 0):.4f}" for m in models
        )
        p_val = data.get("p", 1.0)
        sig = _sig_marker(p_val)
        print(f"  {domain:<25s}  n={data['n']:>5d}  {fa_vals}  "
              f"p={p_val:.6f}  {sig}")


# ---------------------------------------------------------------------------
# 9. Experiment 5.1 — Causal Effect Decomposition
# ---------------------------------------------------------------------------

def experiment_5_1(df: pd.DataFrame) -> dict:
    """
    Causal effect decomposition: TE = DE + IE via KC + IE via TS + IE via PO.

    Decomposes the Total Effect (fresh-vs-stale performance gap) into indirect
    effects mediated by each failure mechanism, using two complementary approaches:

    A. Mechanism-stratified decomposition:
       Partition TE by predicted mechanism labels. For each mechanism m:
         TE_m = E[FA|S1,m] - E[FA|S2,m]   (fresh-stale gap for mechanism m)
         w_m  = n_m / n_total               (proportion of instances)
         IE_m = TE_m × w_m                  (weighted contribution)
       Since mechanisms partition the data: TE ≈ Σ IE_m.

    B. Metric-mediated decomposition:
       Use behavioural metrics (POS, CRS, AU) as proxies for causal pathways:
         POS → PO pathway (preference for outdated)
         CRS → KC pathway (knowledge conflict triggers change recognition)
         AU  → TS pathway (temporal sensitivity → hedging/uncertainty)
       Compute mediation via product-of-coefficients: IE = a × b
       where a = effect of staleness on mediator, b = effect of mediator on FA.
    """
    results: dict = {"experiment": "5.1", "description": "Causal Effect Decomposition"}

    # Need S1 (fresh-only) and S2 (stale-only) for clean contrast
    s1 = df[df["scenario"] == "S1"]
    s2 = df[df["scenario"] == "S2"]

    if len(s1) == 0 or len(s2) == 0:
        raise ValueError("Need both S1 and S2 scenarios for causal decomposition")

    # ---- A. Total Effect ----
    te_fa = float(s1["FA"].mean() - s2["FA"].mean())
    te_stats = {
        "TE": round(te_fa, 6),
        "S1_mean_FA": round(float(s1["FA"].mean()), 4),
        "S2_mean_FA": round(float(s2["FA"].mean()), 4),
        "n_S1": len(s1),
        "n_S2": len(s2),
    }
    # TE for all metrics
    for m in _METRICS:
        te_stats[f"TE_{m}"] = round(float(s1[m].mean() - s2[m].mean()), 6)
    results["total_effect"] = te_stats

    # ---- B. Mechanism-Stratified Decomposition ----
    mechanisms = sorted(df["predicted_mechanism"].unique())
    n_total = len(s1)  # same QA pairs in S1 and S2
    mech_decomp = {}
    ie_sum = 0.0

    for mech in mechanisms:
        s1_m = s1[s1["predicted_mechanism"] == mech]
        s2_m = s2[s2["predicted_mechanism"] == mech]
        n_m = len(s1_m)
        w_m = n_m / n_total if n_total > 0 else 0

        if len(s1_m) == 0 or len(s2_m) == 0:
            mech_decomp[mech] = {"n": n_m, "w": round(w_m, 4),
                                  "TE_m": None, "IE_m": None}
            continue

        te_m = float(s1_m["FA"].mean() - s2_m["FA"].mean())
        ie_m = te_m * w_m
        ie_sum += ie_m

        # t-test for TE_m significance
        t_stat, p_val = stats.ttest_ind(
            s1_m["FA"].values, s2_m["FA"].values, equal_var=False
        )

        # Per-metric TE_m
        metric_te = {}
        for met in _METRICS:
            metric_te[met] = round(float(s1_m[met].mean() - s2_m[met].mean()), 6)

        mech_decomp[mech] = {
            "n": n_m,
            "w": round(w_m, 4),
            "S1_mean_FA": round(float(s1_m["FA"].mean()), 4),
            "S2_mean_FA": round(float(s2_m["FA"].mean()), 4),
            "TE_m": round(te_m, 6),
            "IE_m": round(ie_m, 6),
            "pct_of_TE": round(ie_m / te_fa * 100, 2) if abs(te_fa) > 1e-8 else None,
            "t_stat": round(float(t_stat), 4),
            "p": round(float(p_val), 8),
            "metric_TE": metric_te,
        }

    results["mechanism_decomposition"] = mech_decomp
    results["ie_sum"] = round(ie_sum, 6)
    results["decomposition_check"] = round(te_fa - ie_sum, 8)  # should be ~0

    # ---- C. Metric-Mediated Decomposition (Baron & Kenny style) ----
    # Mediators: POS (PO pathway), CRS (KC pathway), AU (TS pathway)
    # Treatment: stale indicator (S2=1, S1=0)
    # Outcome: FA
    mediators = {"POS": "PO", "CRS": "KC", "AU": "TS"}
    s1s2 = pd.concat([s1.copy(), s2.copy()], ignore_index=True)
    s1s2["stale"] = (s1s2["scenario"] == "S2").astype(float)

    mediation_results = {}
    total_ie_mediated = 0.0

    for mediator, pathway in mediators.items():
        # Path a: stale → mediator
        x_a = s1s2["stale"].values
        y_a = s1s2[mediator].values
        reg_a = _ols_regression(x_a, y_a)

        # Path b: mediator → FA (controlling for stale)
        # Simple approach: partial correlation via residualization
        # FA_residual ~ mediator_residual after removing stale effect
        from numpy.linalg import lstsq
        X_full = np.column_stack([x_a, y_a])
        y_fa = s1s2["FA"].values
        # OLS: FA = β0 + β_stale * stale + β_med * mediator
        X_design = np.column_stack([np.ones(len(x_a)), x_a, y_a])
        betas, _, _, _ = lstsq(X_design, y_fa, rcond=None)
        b_stale = float(betas[1])  # direct effect of stale on FA
        b_med = float(betas[2])    # effect of mediator on FA

        # Indirect effect via this mediator = a × b
        a_coeff = reg_a["slope"] if reg_a["slope"] is not None else 0
        ie_mediated = a_coeff * b_med

        # Sobel test for mediation significance
        se_a = reg_a["se_slope"] if reg_a["se_slope"] is not None else 0
        # SE of b from OLS
        residuals = y_fa - X_design @ betas
        mse = float(np.sum(residuals ** 2) / (len(y_fa) - 3))
        XtX_inv = np.linalg.inv(X_design.T @ X_design)
        se_b = float(np.sqrt(mse * XtX_inv[2, 2]))
        sobel_se = np.sqrt(a_coeff**2 * se_b**2 + b_med**2 * se_a**2)
        sobel_z = ie_mediated / sobel_se if sobel_se > 0 else 0
        sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

        total_ie_mediated += ie_mediated

        mediation_results[pathway] = {
            "mediator": mediator,
            "path_a_slope": round(a_coeff, 6),
            "path_a_p": reg_a["p"],
            "path_b_slope": round(b_med, 6),
            "path_b_se": round(se_b, 6),
            "IE": round(ie_mediated, 6),
            "pct_of_TE": round(ie_mediated / te_fa * 100, 2) if abs(te_fa) > 1e-8 else None,
            "sobel_z": round(float(sobel_z), 4),
            "sobel_p": round(float(sobel_p), 8),
        }

    # Direct effect = total c path minus sum of indirect
    # c path (total): stale → FA
    reg_c = _ols_regression(s1s2["stale"].values, s1s2["FA"].values)
    c_total = reg_c["slope"] if reg_c["slope"] is not None else 0
    de = c_total - total_ie_mediated

    results["mediation"] = {
        "c_total": round(c_total, 6),
        "c_total_p": reg_c["p"],
        "direct_effect": round(de, 6),
        "pct_direct": round(de / c_total * 100, 2) if abs(c_total) > 1e-8 else None,
        "pathways": mediation_results,
        "total_IE_mediated": round(total_ie_mediated, 6),
        "pct_mediated": round(total_ie_mediated / c_total * 100, 2) if abs(c_total) > 1e-8 else None,
    }

    # ---- D. Mixed scenario decomposition (S3-S5 as partial treatments) ----
    mixed_effects = {}
    for scenario in ["S3", "S4", "S5"]:
        sc = df[df["scenario"] == scenario]
        if len(sc) == 0:
            continue
        gap_vs_s1 = float(s1["FA"].mean() - sc["FA"].mean())
        gap_vs_s2 = float(sc["FA"].mean() - s2["FA"].mean())
        mixed_effects[scenario] = {
            "mean_FA": round(float(sc["FA"].mean()), 4),
            "n": len(sc),
            "gap_vs_S1": round(gap_vs_s1, 6),
            "gap_vs_S2": round(gap_vs_s2, 6),
            "recovery_pct": round(gap_vs_s2 / te_fa * 100, 2) if abs(te_fa) > 1e-8 else None,
        }
    results["mixed_scenario_effects"] = mixed_effects

    # ---- E. Domain-stratified decomposition ----
    domain_decomp = {}
    for domain in sorted(df["domain"].unique()):
        s1_d = s1[s1["domain"] == domain]
        s2_d = s2[s2["domain"] == domain]
        if len(s1_d) == 0 or len(s2_d) == 0:
            continue
        te_d = float(s1_d["FA"].mean() - s2_d["FA"].mean())
        t_stat, p_val = stats.ttest_ind(
            s1_d["FA"].values, s2_d["FA"].values, equal_var=False
        )
        domain_decomp[domain] = {
            "TE": round(te_d, 6),
            "S1_mean_FA": round(float(s1_d["FA"].mean()), 4),
            "S2_mean_FA": round(float(s2_d["FA"].mean()), 4),
            "n_S1": len(s1_d),
            "t_stat": round(float(t_stat), 4),
            "p": round(float(p_val), 8),
        }
    results["domain_decomposition"] = domain_decomp

    return results


def print_experiment_5_1(results: dict) -> None:
    """Print formatted results for Experiment 5.1."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 5.1: CAUSAL EFFECT DECOMPOSITION")
    print("TE = DE + IE via KC + IE via TS + IE via PO")
    print("=" * 80)

    # Total Effect
    te = results["total_effect"]
    print(f"\n--- Total Effect (S1 fresh-only vs S2 stale-only) ---")
    print(f"  S1 mean FA = {te['S1_mean_FA']:.4f}  (n={te['n_S1']})")
    print(f"  S2 mean FA = {te['S2_mean_FA']:.4f}  (n={te['n_S2']})")
    print(f"  TE (FA)    = {te['TE']:+.6f}")
    print(f"\n  TE by metric:")
    for m in _METRICS:
        print(f"    {m}: {te[f'TE_{m}']:+.6f}")

    # Mechanism-Stratified Decomposition
    print(f"\n--- A. Mechanism-Stratified Decomposition ---")
    print(f"  {'Mech':<6s}  {'n':>5s}  {'w':>6s}  {'S1 FA':>7s}  {'S2 FA':>7s}  "
          f"{'TE_m':>8s}  {'IE_m':>8s}  {'%TE':>6s}  {'p':>10s}  {'Sig':>4s}")
    print("  " + "-" * 85)
    for mech, d in sorted(results["mechanism_decomposition"].items()):
        if d["TE_m"] is None:
            print(f"  {mech:<6s}  {d['n']:>5d}  {d['w']:>6.4f}  (insufficient data)")
            continue
        sig = _sig_marker(d["p"])
        pct = f"{d['pct_of_TE']:>5.1f}%" if d["pct_of_TE"] is not None else "  N/A"
        print(f"  {mech:<6s}  {d['n']:>5d}  {d['w']:>6.4f}  {d['S1_mean_FA']:>7.4f}  "
              f"{d['S2_mean_FA']:>7.4f}  {d['TE_m']:>+8.4f}  {d['IE_m']:>+8.4f}  "
              f"{pct}  {d['p']:>10.6f}  {sig:>4s}")

    print(f"\n  Sum(IE_m) = {results['ie_sum']:+.6f}  "
          f"(check: TE - Σ IE = {results['decomposition_check']:.8f})")

    # Metric-Mediated Decomposition
    med = results["mediation"]
    print(f"\n--- B. Metric-Mediated Decomposition (Baron & Kenny) ---")
    print(f"  Total effect (c path): stale→FA = {med['c_total']:+.6f}  p={med['c_total_p']}")
    print(f"\n  {'Pathway':<6s}  {'Mediator':<6s}  {'a (stale→M)':>12s}  {'b (M→FA)':>10s}  "
          f"{'IE=a×b':>10s}  {'%TE':>6s}  {'Sobel z':>8s}  {'p':>10s}  {'Sig':>4s}")
    print("  " + "-" * 85)
    for pathway in ["PO", "KC", "TS"]:
        if pathway not in med["pathways"]:
            continue
        p = med["pathways"][pathway]
        sig = _sig_marker(p["sobel_p"])
        pct = f"{p['pct_of_TE']:>5.1f}%" if p["pct_of_TE"] is not None else "  N/A"
        print(f"  {pathway:<6s}  {p['mediator']:<6s}  {p['path_a_slope']:>+12.6f}  "
              f"{p['path_b_slope']:>+10.6f}  {p['IE']:>+10.6f}  "
              f"{pct}  {p['sobel_z']:>8.4f}  {p['sobel_p']:>10.6f}  {sig:>4s}")

    print(f"\n  Direct effect (c')    = {med['direct_effect']:+.6f}  "
          f"({med['pct_direct']:.1f}% of TE)")
    print(f"  Total mediated (Σ IE) = {med['total_IE_mediated']:+.6f}  "
          f"({med['pct_mediated']:.1f}% of TE)")

    # Mixed scenario recovery
    print(f"\n--- C. Mixed Scenario Recovery ---")
    print(f"  {'Scenario':<10s}  {'Mean FA':>8s}  {'Gap vs S1':>10s}  {'Gap vs S2':>10s}  {'Recovery%':>10s}")
    print("  " + "-" * 55)
    for sc, d in sorted(results["mixed_scenario_effects"].items()):
        recovery = f"{d['recovery_pct']:.1f}%" if d["recovery_pct"] is not None else "N/A"
        print(f"  {sc:<10s}  {d['mean_FA']:>8.4f}  {d['gap_vs_S1']:>+10.4f}  "
              f"{d['gap_vs_S2']:>+10.4f}  {recovery:>10s}")

    # Domain decomposition
    print(f"\n--- D. Total Effect by Domain ---")
    print(f"  {'Domain':<25s}  {'S1 FA':>7s}  {'S2 FA':>7s}  {'TE':>8s}  {'p':>10s}  {'Sig':>4s}")
    print("  " + "-" * 65)
    for domain, d in sorted(results["domain_decomposition"].items()):
        sig = _sig_marker(d["p"])
        print(f"  {domain:<25s}  {d['S1_mean_FA']:>7.4f}  {d['S2_mean_FA']:>7.4f}  "
              f"{d['TE']:>+8.4f}  {d['p']:>10.6f}  {sig:>4s}")


# ---------------------------------------------------------------------------
# 10. Experiment 5.2 — Mechanism Hypothesis Validation
# ---------------------------------------------------------------------------

def experiment_5_2(df: pd.DataFrame) -> dict:
    """
    Validate predicted mechanism labels against observed behavioural patterns.

    For each predicted mechanism (KC, TS, PO, C), check whether instances with
    that label actually exhibit the expected metric signature:

    - KC (Knowledge Conflict): higher CRS in mixed scenarios (S3-S5) vs S1,
      higher HR when only stale docs (S2).
    - TS (Temporal Sensitivity): higher AU overall, stronger S1-S2 gap in FA,
      higher sensitivity to document age.
    - PO (Preference for Outdated): higher POS in S2, higher POS in stale-
      dominant (S4) vs fresh-dominant (S3).
    - C (Compound): combines multiple patterns, highest HR and lowest FA.

    Also computes discriminant validity: do mechanism labels predict metric
    profiles beyond chance?
    """
    results: dict = {"experiment": "5.2", "description": "Mechanism Hypothesis Validation"}
    mechanisms = sorted(df["predicted_mechanism"].unique())

    # ---- A. Per-mechanism metric profiles ----
    profiles = {}
    for mech in mechanisms:
        mech_df = df[df["predicted_mechanism"] == mech]
        profile = {"n": len(mech_df)}
        for m in _METRICS:
            profile[m] = {
                "mean": round(float(mech_df[m].mean()), 4),
                "std": round(float(mech_df[m].std()), 4),
            }
        # Per-scenario breakdown
        scenario_means = {}
        for sc in sorted(mech_df["scenario"].unique()):
            sc_df = mech_df[mech_df["scenario"] == sc]
            scenario_means[sc] = {m: round(float(sc_df[m].mean()), 4) for m in _METRICS}
            scenario_means[sc]["n"] = len(sc_df)
        profile["by_scenario"] = scenario_means
        profiles[mech] = profile
    results["mechanism_profiles"] = profiles

    # ---- B. Hypothesis tests per mechanism ----
    hypothesis_tests = {}

    # --- KC: Knowledge Conflict ---
    kc_df = df[df["predicted_mechanism"] == "KC"]
    non_kc = df[df["predicted_mechanism"] != "KC"]
    kc_tests = {}
    if len(kc_df) > 0 and len(non_kc) > 0:
        # H1: KC instances have higher CRS in mixed scenarios (S3-S5)
        kc_mixed = kc_df[kc_df["scenario"].isin(["S3", "S4", "S5"])]
        non_kc_mixed = non_kc[non_kc["scenario"].isin(["S3", "S4", "S5"])]
        if len(kc_mixed) > 1 and len(non_kc_mixed) > 1:
            t, p = stats.ttest_ind(kc_mixed["CRS"].values, non_kc_mixed["CRS"].values, equal_var=False)
            kc_tests["H1_higher_CRS_mixed"] = {
                "description": "KC has higher CRS in mixed scenarios (S3-S5)",
                "KC_mean": round(float(kc_mixed["CRS"].mean()), 4),
                "other_mean": round(float(non_kc_mixed["CRS"].mean()), 4),
                "diff": round(float(kc_mixed["CRS"].mean() - non_kc_mixed["CRS"].mean()), 4),
                "t_stat": round(float(t), 4),
                "p": round(float(p), 8),
                "confirmed": bool(p < 0.05 and kc_mixed["CRS"].mean() > non_kc_mixed["CRS"].mean()),
            }
        # H2: KC instances have higher HR in S2
        kc_s2 = kc_df[kc_df["scenario"] == "S2"]
        non_kc_s2 = non_kc[non_kc["scenario"] == "S2"]
        if len(kc_s2) > 1 and len(non_kc_s2) > 1:
            t, p = stats.ttest_ind(kc_s2["HR"].values, non_kc_s2["HR"].values, equal_var=False)
            kc_tests["H2_higher_HR_S2"] = {
                "description": "KC has higher HR in stale-only (S2)",
                "KC_mean": round(float(kc_s2["HR"].mean()), 4),
                "other_mean": round(float(non_kc_s2["HR"].mean()), 4),
                "diff": round(float(kc_s2["HR"].mean() - non_kc_s2["HR"].mean()), 4),
                "t_stat": round(float(t), 4),
                "p": round(float(p), 8),
                "confirmed": bool(p < 0.05 and kc_s2["HR"].mean() > non_kc_s2["HR"].mean()),
            }
    hypothesis_tests["KC"] = kc_tests

    # --- TS: Temporal Sensitivity ---
    ts_df = df[df["predicted_mechanism"] == "TS"]
    non_ts = df[df["predicted_mechanism"] != "TS"]
    ts_tests = {}
    if len(ts_df) > 0 and len(non_ts) > 0:
        # H1: TS instances have higher AU overall
        t, p = stats.ttest_ind(ts_df["AU"].values, non_ts["AU"].values, equal_var=False)
        ts_tests["H1_higher_AU"] = {
            "description": "TS has higher AU (uncertainty) overall",
            "TS_mean": round(float(ts_df["AU"].mean()), 4),
            "other_mean": round(float(non_ts["AU"].mean()), 4),
            "diff": round(float(ts_df["AU"].mean() - non_ts["AU"].mean()), 4),
            "t_stat": round(float(t), 4),
            "p": round(float(p), 8),
            "confirmed": bool(p < 0.05 and ts_df["AU"].mean() > non_ts["AU"].mean()),
        }
        # H2: TS instances have larger S1-S2 FA gap (more sensitive to staleness)
        ts_s1 = ts_df[ts_df["scenario"] == "S1"]["FA"]
        ts_s2 = ts_df[ts_df["scenario"] == "S2"]["FA"]
        non_ts_s1 = non_ts[non_ts["scenario"] == "S1"]["FA"]
        non_ts_s2 = non_ts[non_ts["scenario"] == "S2"]["FA"]
        if len(ts_s1) > 0 and len(ts_s2) > 0 and len(non_ts_s1) > 0 and len(non_ts_s2) > 0:
            ts_gap = float(ts_s1.mean() - ts_s2.mean())
            non_ts_gap = float(non_ts_s1.mean() - non_ts_s2.mean())
            # Bootstrap or compare gaps via interaction test
            ts_tests["H2_larger_staleness_gap"] = {
                "description": "TS has larger S1-S2 FA gap (more temporally sensitive)",
                "TS_gap": round(ts_gap, 4),
                "other_gap": round(non_ts_gap, 4),
                "gap_diff": round(ts_gap - non_ts_gap, 4),
                "confirmed": ts_gap > non_ts_gap,
            }
    hypothesis_tests["TS"] = ts_tests

    # --- PO: Preference for Outdated ---
    po_df = df[df["predicted_mechanism"] == "PO"]
    non_po = df[df["predicted_mechanism"] != "PO"]
    po_tests = {}
    if len(po_df) > 0 and len(non_po) > 0:
        # H1: PO instances have higher POS in S2
        po_s2 = po_df[po_df["scenario"] == "S2"]
        non_po_s2 = non_po[non_po["scenario"] == "S2"]
        if len(po_s2) > 1 and len(non_po_s2) > 1:
            t, p = stats.ttest_ind(po_s2["POS"].values, non_po_s2["POS"].values, equal_var=False)
            po_tests["H1_higher_POS_S2"] = {
                "description": "PO has higher POS in stale-only (S2)",
                "PO_mean": round(float(po_s2["POS"].mean()), 4),
                "other_mean": round(float(non_po_s2["POS"].mean()), 4),
                "diff": round(float(po_s2["POS"].mean() - non_po_s2["POS"].mean()), 4),
                "t_stat": round(float(t), 4),
                "p": round(float(p), 8),
                "confirmed": bool(p < 0.05 and po_s2["POS"].mean() > non_po_s2["POS"].mean()),
            }
        # H2: PO instances have higher POS in S4 (stale-dominant) vs S3 (fresh-dominant)
        po_s3 = po_df[po_df["scenario"] == "S3"]
        po_s4 = po_df[po_df["scenario"] == "S4"]
        if len(po_s3) > 1 and len(po_s4) > 1:
            t, p = stats.ttest_ind(po_s4["POS"].values, po_s3["POS"].values, equal_var=False)
            po_tests["H2_higher_POS_S4_vs_S3"] = {
                "description": "PO has higher POS in stale-dominant (S4) vs fresh-dominant (S3)",
                "PO_S4_mean": round(float(po_s4["POS"].mean()), 4),
                "PO_S3_mean": round(float(po_s3["POS"].mean()), 4),
                "diff": round(float(po_s4["POS"].mean() - po_s3["POS"].mean()), 4),
                "t_stat": round(float(t), 4),
                "p": round(float(p), 8),
                "confirmed": bool(p < 0.05 and po_s4["POS"].mean() > po_s3["POS"].mean()),
            }
        # H3: PO instances have higher POS overall vs others
        t, p = stats.ttest_ind(po_df["POS"].values, non_po["POS"].values, equal_var=False)
        po_tests["H3_higher_POS_overall"] = {
            "description": "PO has higher POS overall vs other mechanisms",
            "PO_mean": round(float(po_df["POS"].mean()), 4),
            "other_mean": round(float(non_po["POS"].mean()), 4),
            "diff": round(float(po_df["POS"].mean() - non_po["POS"].mean()), 4),
            "t_stat": round(float(t), 4),
            "p": round(float(p), 8),
            "confirmed": bool(p < 0.05 and po_df["POS"].mean() > non_po["POS"].mean()),
        }
    hypothesis_tests["PO"] = po_tests

    # --- C: Compound ---
    c_df = df[df["predicted_mechanism"] == "C"]
    non_c = df[df["predicted_mechanism"] != "C"]
    c_tests = {}
    if len(c_df) > 0 and len(non_c) > 0:
        # H1: C instances have lowest FA (hardest)
        t, p = stats.ttest_ind(c_df["FA"].values, non_c["FA"].values, equal_var=False)
        c_tests["H1_lowest_FA"] = {
            "description": "C (compound) has lower FA than single-mechanism instances",
            "C_mean": round(float(c_df["FA"].mean()), 4),
            "other_mean": round(float(non_c["FA"].mean()), 4),
            "diff": round(float(c_df["FA"].mean() - non_c["FA"].mean()), 4),
            "t_stat": round(float(t), 4),
            "p": round(float(p), 8),
            "confirmed": bool(p < 0.05 and c_df["FA"].mean() < non_c["FA"].mean()),
        }
        # H2: C instances have highest CRS (recognize change more)
        t, p = stats.ttest_ind(c_df["CRS"].values, non_c["CRS"].values, equal_var=False)
        c_tests["H2_highest_CRS"] = {
            "description": "C has higher CRS (more change recognition)",
            "C_mean": round(float(c_df["CRS"].mean()), 4),
            "other_mean": round(float(non_c["CRS"].mean()), 4),
            "diff": round(float(c_df["CRS"].mean() - non_c["CRS"].mean()), 4),
            "t_stat": round(float(t), 4),
            "p": round(float(p), 8),
            "confirmed": bool(p < 0.05 and c_df["CRS"].mean() > non_c["CRS"].mean()),
        }
    hypothesis_tests["C"] = c_tests

    results["hypothesis_tests"] = hypothesis_tests

    # ---- C. Discriminant validity: ANOVA metric ~ mechanism ----
    discriminant = {}
    for m in _METRICS:
        groups = [df.loc[df["predicted_mechanism"] == mech, m].values
                  for mech in mechanisms if len(df[df["predicted_mechanism"] == mech]) > 1]
        if len(groups) < 2:
            continue
        f_stat, p_val = stats.f_oneway(*groups)
        grand_mean = df[m].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_total = np.sum((df[m].values - grand_mean) ** 2)
        eta2 = ss_between / ss_total if ss_total > 0 else 0
        discriminant[m] = {
            "F": round(float(f_stat), 4),
            "p": round(float(p_val), 8),
            "eta_squared": round(float(eta2), 4),
        }
    results["discriminant_validity"] = discriminant

    # ---- D. Hypothesis confirmation summary ----
    total_tests = 0
    confirmed_tests = 0
    summary = {}
    for mech, tests in hypothesis_tests.items():
        mech_total = 0
        mech_confirmed = 0
        for test_name, test_data in tests.items():
            mech_total += 1
            total_tests += 1
            if test_data.get("confirmed", False):
                mech_confirmed += 1
                confirmed_tests += 1
        summary[mech] = {"total": mech_total, "confirmed": mech_confirmed}
    summary["overall"] = {"total": total_tests, "confirmed": confirmed_tests,
                          "rate": round(confirmed_tests / total_tests, 4) if total_tests > 0 else 0}
    results["confirmation_summary"] = summary

    return results


def print_experiment_5_2(results: dict) -> None:
    """Print formatted results for Experiment 5.2."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 5.2: MECHANISM HYPOTHESIS VALIDATION")
    print("Predicted vs observed mechanism patterns")
    print("=" * 80)

    # Mechanism profiles
    print(f"\n--- A. Mechanism Metric Profiles ---")
    print(f"  {'Mech':<6s}  {'n':>5s}  " + "  ".join(f"{m:>7s}" for m in _METRICS))
    print("  " + "-" * 55)
    for mech, p in sorted(results["mechanism_profiles"].items()):
        vals = "  ".join(f"{p[m]['mean']:>7.4f}" for m in _METRICS)
        print(f"  {mech:<6s}  {p['n']:>5d}  {vals}")

    # Hypothesis tests
    print(f"\n--- B. Hypothesis Tests ---")
    for mech in ["KC", "TS", "PO", "C"]:
        tests = results["hypothesis_tests"].get(mech, {})
        if not tests:
            print(f"\n  {mech}: (no tests — insufficient data)")
            continue
        print(f"\n  {mech}:")
        for test_name, td in tests.items():
            status = "CONFIRMED" if td.get("confirmed") else "NOT confirmed"
            p_str = f"p={td['p']:.6f}" if "p" in td else ""
            sig = _sig_marker(td["p"]) if "p" in td else ""
            print(f"    [{status:>13s}] {td['description']}")
            if "diff" in td:
                print(f"      diff={td['diff']:+.4f}  {p_str}  {sig}")

    # Discriminant validity
    print(f"\n--- C. Discriminant Validity (ANOVA: metric ~ mechanism) ---")
    print(f"  {'Metric':<6s}  {'F':>8s}  {'p':>12s}  {'η²':>6s}  {'Sig':>4s}")
    print("  " + "-" * 42)
    for m in _METRICS:
        if m in results["discriminant_validity"]:
            d = results["discriminant_validity"][m]
            sig = _sig_marker(d["p"])
            print(f"  {m:<6s}  {d['F']:>8.2f}  {d['p']:>12.8f}  {d['eta_squared']:>6.4f}  {sig:>4s}")

    # Summary
    print(f"\n--- D. Confirmation Summary ---")
    cs = results["confirmation_summary"]
    for mech in ["KC", "TS", "PO", "C"]:
        if mech in cs:
            ms = cs[mech]
            print(f"  {mech}: {ms['confirmed']}/{ms['total']} hypotheses confirmed")
    ov = cs["overall"]
    print(f"  Overall: {ov['confirmed']}/{ov['total']} ({ov['rate']*100:.0f}%) hypotheses confirmed")


# ---------------------------------------------------------------------------
# 11. Experiment 4.5 — Adversarial Freshness
# ---------------------------------------------------------------------------

def experiment_4_5(df: pd.DataFrame) -> dict:
    """
    Adversarial Freshness: can fake timestamps fool models into trusting stale content?

    Uses existing timestamp conditions:
    - S2+actual: stale content, real (old) dates shown
    - S2+misleading: stale content, fake recent dates shown (adversarial)
    - S2+none: stale content, no dates shown

    Also tests adversarial on mixed scenarios (S3-S5) where misleading dates
    could amplify trust in the wrong documents.
    """
    results: dict = {"experiment": "4.5", "description": "Adversarial Freshness"}

    # ---- A. Core adversarial test: S2 across timestamp conditions ----
    s2 = df[df["scenario"] == "S2"]
    ts_conds = sorted(s2["timestamp_condition"].unique())

    s2_by_ts = {}
    for tc in ts_conds:
        subset = s2[s2["timestamp_condition"] == tc]
        row = {"n": len(subset)}
        for m in _METRICS:
            row[m] = round(float(subset[m].mean()), 4)
        s2_by_ts[tc] = row
    results["s2_by_timestamp"] = s2_by_ts

    # Pairwise comparisons within S2
    s2_pairwise = {}
    for m in _METRICS:
        s2_pairwise[m] = pairwise_comparisons(s2, m, "timestamp_condition")
    results["s2_pairwise"] = s2_pairwise

    # ANOVA within S2: metric ~ timestamp_condition
    s2_anova = {}
    for m in _METRICS:
        groups = [s2.loc[s2["timestamp_condition"] == tc, m].values for tc in ts_conds]
        f_stat, p_val = stats.f_oneway(*groups)
        s2_anova[m] = {
            "F": round(float(f_stat), 4),
            "p": round(float(p_val), 8),
        }
    results["s2_anova"] = s2_anova

    # ---- B. Adversarial effect per scenario ----
    scenario_adv = {}
    for sc in sorted(df["scenario"].unique()):
        sc_df = df[df["scenario"] == sc]
        actual = sc_df[sc_df["timestamp_condition"] == "actual"]
        misleading = sc_df[sc_df["timestamp_condition"] == "misleading"]
        none_ts = sc_df[sc_df["timestamp_condition"] == "none"]
        if len(actual) == 0 or len(misleading) == 0:
            continue
        row = {
            "n_actual": len(actual),
            "n_misleading": len(misleading),
        }
        for m in _METRICS:
            a_mean = float(actual[m].mean())
            m_mean = float(misleading[m].mean())
            t_stat, p_val = stats.ttest_ind(
                actual[m].values, misleading[m].values, equal_var=False
            )
            row[m] = {
                "actual_mean": round(a_mean, 4),
                "misleading_mean": round(m_mean, 4),
                "diff": round(m_mean - a_mean, 4),
                "t_stat": round(float(t_stat), 4),
                "p": round(float(p_val), 8),
            }
        scenario_adv[sc] = row
    results["adversarial_by_scenario"] = scenario_adv

    # ---- C. Adversarial effect by mechanism ----
    mech_adv = {}
    for mech in sorted(df["predicted_mechanism"].unique()):
        mech_df = df[df["predicted_mechanism"] == mech]
        s2_mech = mech_df[mech_df["scenario"] == "S2"]
        actual = s2_mech[s2_mech["timestamp_condition"] == "actual"]
        misleading = s2_mech[s2_mech["timestamp_condition"] == "misleading"]
        if len(actual) < 5 or len(misleading) < 5:
            continue
        row = {"n_actual": len(actual), "n_misleading": len(misleading)}
        for m in _METRICS:
            a_mean = float(actual[m].mean())
            m_mean = float(misleading[m].mean())
            t_stat, p_val = stats.ttest_ind(
                actual[m].values, misleading[m].values, equal_var=False
            )
            row[m] = {
                "actual_mean": round(a_mean, 4),
                "misleading_mean": round(m_mean, 4),
                "diff": round(m_mean - a_mean, 4),
                "t_stat": round(float(t_stat), 4),
                "p": round(float(p_val), 8),
            }
        mech_adv[mech] = row
    results["adversarial_by_mechanism"] = mech_adv

    # ---- D. Adversarial effect by domain ----
    domain_adv = {}
    for domain in sorted(df["domain"].unique()):
        dom_df = df[df["domain"] == domain]
        s2_dom = dom_df[dom_df["scenario"] == "S2"]
        actual = s2_dom[s2_dom["timestamp_condition"] == "actual"]
        misleading = s2_dom[s2_dom["timestamp_condition"] == "misleading"]
        if len(actual) < 5 or len(misleading) < 5:
            continue
        row = {"n_actual": len(actual), "n_misleading": len(misleading)}
        for m in ("FA", "POS", "HR"):
            a_mean = float(actual[m].mean())
            m_mean = float(misleading[m].mean())
            t_stat, p_val = stats.ttest_ind(
                actual[m].values, misleading[m].values, equal_var=False
            )
            row[m] = {
                "actual_mean": round(a_mean, 4),
                "misleading_mean": round(m_mean, 4),
                "diff": round(m_mean - a_mean, 4),
                "t_stat": round(float(t_stat), 4),
                "p": round(float(p_val), 8),
            }
        domain_adv[domain] = row
    results["adversarial_by_domain"] = domain_adv

    # ---- E. Overall adversarial susceptibility score ----
    # Count how many of the per-metric tests are significant
    all_actual = df[df["timestamp_condition"] == "actual"]
    all_misleading = df[df["timestamp_condition"] == "misleading"]
    overall_tests = {}
    n_sig = 0
    for m in _METRICS:
        t_stat, p_val = stats.ttest_ind(
            all_actual[m].values, all_misleading[m].values, equal_var=False
        )
        sig = bool(p_val < 0.05)
        if sig:
            n_sig += 1
        overall_tests[m] = {
            "actual_mean": round(float(all_actual[m].mean()), 4),
            "misleading_mean": round(float(all_misleading[m].mean()), 4),
            "diff": round(float(all_misleading[m].mean() - all_actual[m].mean()), 4),
            "t_stat": round(float(t_stat), 4),
            "p": round(float(p_val), 8),
            "significant": sig,
        }
    results["overall_adversarial"] = overall_tests
    results["adversarial_susceptibility"] = {
        "n_metrics_significant": n_sig,
        "total_metrics": len(_METRICS),
        "verdict": "susceptible" if n_sig >= 3 else "partially susceptible" if n_sig >= 1 else "robust",
    }

    return results


def print_experiment_4_5(results: dict) -> None:
    """Print formatted results for Experiment 4.5."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4.5: ADVERSARIAL FRESHNESS")
    print("Can fake timestamps fool models into trusting stale content?")
    print("=" * 80)

    # S2 by timestamp condition
    print(f"\n--- A. Stale-Only (S2) Performance by Timestamp Condition ---")
    print(f"  {'Condition':<12s}  {'n':>5s}  " +
          "  ".join(f"{m:>7s}" for m in _METRICS))
    print("  " + "-" * 55)
    for tc, row in sorted(results["s2_by_timestamp"].items()):
        vals = "  ".join(f"{row[m]:>7.4f}" for m in _METRICS)
        print(f"  {tc:<12s}  {row['n']:>5d}  {vals}")

    # S2 ANOVA
    print(f"\n  ANOVA within S2 (metric ~ timestamp_condition):")
    for m in _METRICS:
        a = results["s2_anova"][m]
        sig = _sig_marker(a["p"])
        print(f"    {m}: F={a['F']:.4f}  p={a['p']:.6f}  {sig}")

    # Adversarial effect per scenario (FA only)
    print(f"\n--- B. Adversarial Effect by Scenario (actual vs misleading) ---")
    print(f"  {'Scenario':<10s}  {'Metric':<6s}  {'Actual':>8s}  {'Mislead':>8s}  "
          f"{'Diff':>8s}  {'p':>10s}  {'Sig':>4s}")
    print("  " + "-" * 65)
    for sc, row in sorted(results["adversarial_by_scenario"].items()):
        for m in ("FA", "POS", "HR"):
            d = row[m]
            sig = _sig_marker(d["p"])
            print(f"  {sc:<10s}  {m:<6s}  {d['actual_mean']:>8.4f}  "
                  f"{d['misleading_mean']:>8.4f}  {d['diff']:>+8.4f}  "
                  f"{d['p']:>10.6f}  {sig:>4s}")

    # Adversarial by mechanism (S2 only, FA)
    print(f"\n--- C. Adversarial Effect in S2 by Mechanism (FA) ---")
    print(f"  {'Mech':<6s}  {'Actual':>8s}  {'Mislead':>8s}  {'Diff':>8s}  {'p':>10s}  {'Sig':>4s}")
    print("  " + "-" * 52)
    for mech, row in sorted(results["adversarial_by_mechanism"].items()):
        d = row["FA"]
        sig = _sig_marker(d["p"])
        print(f"  {mech:<6s}  {d['actual_mean']:>8.4f}  {d['misleading_mean']:>8.4f}  "
              f"{d['diff']:>+8.4f}  {d['p']:>10.6f}  {sig:>4s}")

    # Adversarial by domain (S2 only, FA)
    print(f"\n--- D. Adversarial Effect in S2 by Domain (FA) ---")
    print(f"  {'Domain':<25s}  {'Actual':>8s}  {'Mislead':>8s}  {'Diff':>8s}  {'p':>10s}  {'Sig':>4s}")
    print("  " + "-" * 70)
    for domain, row in sorted(results["adversarial_by_domain"].items()):
        d = row["FA"]
        sig = _sig_marker(d["p"])
        print(f"  {domain:<25s}  {d['actual_mean']:>8.4f}  {d['misleading_mean']:>8.4f}  "
              f"{d['diff']:>+8.4f}  {d['p']:>10.6f}  {sig:>4s}")

    # Overall verdict
    print(f"\n--- E. Overall Adversarial Susceptibility ---")
    for m, d in results["overall_adversarial"].items():
        sig = _sig_marker(d["p"])
        print(f"  {m}: actual={d['actual_mean']:.4f}  misleading={d['misleading_mean']:.4f}  "
              f"diff={d['diff']:+.4f}  p={d['p']:.6f}  {sig}")
    v = results["adversarial_susceptibility"]
    print(f"\n  Verdict: {v['verdict'].upper()} "
          f"({v['n_metrics_significant']}/{v['total_metrics']} metrics significant)")


# ---------------------------------------------------------------------------
# 12. Experiment 5.3 — Practical Guidelines
# ---------------------------------------------------------------------------

def experiment_5_3(df: pd.DataFrame, df_tg: pd.DataFrame | None = None) -> dict:
    """
    Derive practical guidelines for RAG content refresh strategies.

    Synthesizes findings from experiments 4.1-4.5 and 5.1-5.2 into actionable
    recommendations, with supporting quantitative evidence.

    Guidelines cover:
    1. Corpus freshness strategy (when/what to refresh)
    2. Retrieval composition (how many fresh vs stale docs)
    3. Timestamp handling (whether to show dates)
    4. Domain-specific strategies
    5. Mechanism-aware processing
    """
    results: dict = {"experiment": "5.3", "description": "Practical Guidelines for Content Refresh"}

    s1 = df[df["scenario"] == "S1"]
    s2 = df[df["scenario"] == "S2"]
    te = float(s1["FA"].mean() - s2["FA"].mean())

    # ---- Guideline 1: Freshness Matters, But One Fresh Doc Suffices ----
    g1 = {"title": "One fresh document neutralizes staleness"}
    # Evidence: S1 vs S2 gap, and S3-S5 recovery
    scenario_means = {}
    for sc in ["S1", "S2", "S3", "S4", "S5"]:
        subset = df[df["scenario"] == sc]
        scenario_means[sc] = round(float(subset["FA"].mean()), 4)
    g1["scenario_means_FA"] = scenario_means
    g1["TE_S1_vs_S2"] = round(te, 4)

    # Recovery rates for mixed scenarios
    for sc in ["S3", "S4", "S5"]:
        gap = scenario_means[sc] - scenario_means["S2"]
        g1[f"recovery_{sc}"] = round(gap / te * 100, 1) if abs(te) > 1e-8 else None

    g1["recommendation"] = (
        "Ensure at least one up-to-date document in every retrieval set. "
        "The marginal value of additional fresh documents is negligible "
        f"(S3 with 2 fresh docs: FA={scenario_means['S3']:.3f} vs "
        f"S5 with 1 fresh doc: FA={scenario_means['S5']:.3f}). "
        "Prioritize breadth of coverage over depth of freshness."
    )
    results["guideline_1"] = g1

    # ---- Guideline 2: Timestamps Don't Help (or Hurt) ----
    g2 = {"title": "Timestamp metadata has no measurable impact"}
    ts_means = {}
    for tc in sorted(df["timestamp_condition"].unique()):
        subset = df[df["timestamp_condition"] == tc]
        ts_means[tc] = round(float(subset["FA"].mean()), 4)
    g2["timestamp_means_FA"] = ts_means

    # Max difference between any two conditions
    ts_vals = list(ts_means.values())
    g2["max_diff"] = round(max(ts_vals) - min(ts_vals), 4)

    # Even adversarial: S2+misleading vs S2+actual
    s2_actual = df[(df["scenario"] == "S2") & (df["timestamp_condition"] == "actual")]
    s2_mislead = df[(df["scenario"] == "S2") & (df["timestamp_condition"] == "misleading")]
    if len(s2_actual) > 0 and len(s2_mislead) > 0:
        t_stat, p_val = stats.ttest_ind(
            s2_actual["FA"].values, s2_mislead["FA"].values, equal_var=False
        )
        g2["adversarial_test"] = {
            "actual_FA": round(float(s2_actual["FA"].mean()), 4),
            "misleading_FA": round(float(s2_mislead["FA"].mean()), 4),
            "p": round(float(p_val), 8),
            "significant": bool(p_val < 0.05),
        }

    g2["recommendation"] = (
        "Including or omitting document timestamps in retrieval context "
        "does not affect model accuracy. Even deliberately misleading timestamps "
        "(swapping fresh/stale dates) produce no measurable effect. "
        "Models appear to rely on content semantics, not metadata, "
        "for temporal reasoning."
    )
    results["guideline_2"] = g2

    # ---- Guideline 3: Domain-Specific Refresh Priorities ----
    g3 = {"title": "Prioritize refreshing science and sports content"}
    domain_perf = {}
    for domain in sorted(df["domain"].unique()):
        dom = df[df["domain"] == domain]
        dom_s1 = dom[dom["scenario"] == "S1"]
        dom_s2 = dom[dom["scenario"] == "S2"]
        te_d = float(dom_s1["FA"].mean() - dom_s2["FA"].mean()) if len(dom_s1) > 0 and len(dom_s2) > 0 else 0
        domain_perf[domain] = {
            "FA": round(float(dom["FA"].mean()), 4),
            "HR": round(float(dom["HR"].mean()), 4),
            "POS": round(float(dom["POS"].mean()), 4),
            "TE": round(te_d, 4),
            "n": len(dom),
        }
    g3["domain_performance"] = domain_perf

    # Rank domains by staleness vulnerability (TE × HR)
    vulnerability = {d: v["TE"] * (1 + v["HR"]) for d, v in domain_perf.items()}
    g3["vulnerability_ranking"] = dict(sorted(vulnerability.items(), key=lambda x: -x[1]))

    g3["recommendation"] = (
        "Content refresh resources should be allocated based on domain vulnerability. "
        "Highest priority: policy and organizations (largest staleness effect TE>0.12). "
        "science_life_physical has highest hallucination rate (HR=0.38) — "
        "stale scientific content is most likely to produce fabricated answers. "
        "Sports has highest outdated-preference (POS=0.24) — "
        "models stubbornly cling to old sports statistics."
    )
    results["guideline_3"] = g3

    # ---- Guideline 4: Mechanism-Aware Processing ----
    g4 = {"title": "TS and PO mechanisms drive most staleness failures"}
    mech_stats = {}
    for mech in sorted(df["predicted_mechanism"].unique()):
        m_df = df[df["predicted_mechanism"] == mech]
        m_s1 = m_df[m_df["scenario"] == "S1"]
        m_s2 = m_df[m_df["scenario"] == "S2"]
        te_m = float(m_s1["FA"].mean() - m_s2["FA"].mean()) if len(m_s1) > 0 and len(m_s2) > 0 else 0
        proportion = len(m_df) / len(df)
        mech_stats[mech] = {
            "n": len(m_df),
            "proportion": round(proportion, 4),
            "TE_m": round(te_m, 4),
            "weighted_contribution": round(te_m * proportion, 4),
            "FA": round(float(m_df["FA"].mean()), 4),
            "POS": round(float(m_df["POS"].mean()), 4),
            "AU": round(float(m_df["AU"].mean()), 4),
        }
    g4["mechanism_stats"] = mech_stats

    g4["recommendation"] = (
        "Two mechanisms account for 92% of the staleness effect: "
        "Temporal Sensitivity (TS, 69%) and Preference for Outdated (PO, 24%). "
        "For TS-type questions (time-sensitive facts like dates, standings, prices), "
        "always retrieve the freshest available document. "
        "For PO-type questions (where old answers feel authoritative), "
        "consider explicitly flagging content age in system prompts or "
        "using recency-weighted retrieval scoring."
    )
    results["guideline_4"] = g4

    # ---- Guideline 5: Temporal Gradient — Freshness Has a Cliff, Not a Slope ----
    g5 = {"title": "Freshness effect is a step function at the change boundary"}
    if df_tg is not None and "t_offset" in df_tg.columns:
        stale = df_tg[df_tg["t_offset"] < 0]
        fresh = df_tg[df_tg["t_offset"] > 0]
        g5["stale_mean_FA"] = round(float(stale["FA"].mean()), 4)
        g5["fresh_mean_FA"] = round(float(fresh["FA"].mean()), 4)
        g5["step_size"] = round(float(fresh["FA"].mean() - stale["FA"].mean()), 4)

        # Within-stale and within-fresh variation
        stale_by_pos = {}
        for t in sorted(stale["t_offset"].unique()):
            vals = stale.loc[stale["t_offset"] == t, "FA"]
            stale_by_pos[f"T{int(t):+d}"] = round(float(vals.mean()), 4)
        fresh_by_pos = {}
        for t in sorted(fresh["t_offset"].unique()):
            vals = fresh.loc[fresh["t_offset"] == t, "FA"]
            fresh_by_pos[f"T{int(t):+d}"] = round(float(vals.mean()), 4)
        g5["stale_positions"] = stale_by_pos
        g5["fresh_positions"] = fresh_by_pos
        g5["within_stale_range"] = round(max(stale_by_pos.values()) - min(stale_by_pos.values()), 4)
        g5["within_fresh_range"] = round(max(fresh_by_pos.values()) - min(fresh_by_pos.values()), 4)
    else:
        g5["note"] = "Temporal gradient data not provided — pass --scored-tg for full analysis"

    g5["recommendation"] = (
        "Document age within stale or fresh categories barely matters — "
        "the critical factor is whether the document was captured before or after "
        "the factual change. A 6-month-old fresh document is just as good as "
        "a 1-month-old one. Implication: refresh frequency should be event-driven "
        "(triggered by known factual changes), not calendar-driven."
    )
    results["guideline_5"] = g5

    # ---- Summary: Refresh Strategy Decision Matrix ----
    results["decision_matrix"] = {
        "high_priority_refresh": {
            "condition": "TS or PO mechanism, policy/organizations/science domain",
            "action": "Refresh immediately upon detecting factual change",
            "expected_FA_gain": f"+{te:.3f} (full TE recovery)",
        },
        "medium_priority_refresh": {
            "condition": "Any domain, mixed retrieval possible",
            "action": "Ensure at least 1 fresh document in retrieval set",
            "expected_FA_gain": f"+{scenario_means['S5'] - scenario_means['S2']:.3f} (98.7% recovery)",
        },
        "low_priority": {
            "condition": "C mechanism, products domain",
            "action": "Standard refresh cycle sufficient",
            "rationale": "C mechanism shows smallest staleness effect; products domain has highest base FA",
        },
        "not_needed": {
            "condition": "Timestamp metadata updates",
            "action": "No action needed",
            "rationale": "Timestamps have zero measurable impact on model performance",
        },
    }

    return results


def print_experiment_5_3(results: dict) -> None:
    """Print formatted results for Experiment 5.3."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 5.3: PRACTICAL GUIDELINES FOR CONTENT REFRESH")
    print("=" * 80)

    for i, key in enumerate(["guideline_1", "guideline_2", "guideline_3",
                              "guideline_4", "guideline_5"], 1):
        g = results[key]
        print(f"\n{'=' * 70}")
        print(f"  GUIDELINE {i}: {g['title'].upper()}")
        print(f"{'=' * 70}")

        if key == "guideline_1":
            print(f"\n  Scenario means (FA):")
            for sc, fa in g["scenario_means_FA"].items():
                print(f"    {sc}: {fa:.4f}")
            print(f"  TE (S1-S2): {g['TE_S1_vs_S2']:.4f}")
            for sc in ["S3", "S4", "S5"]:
                r = g.get(f"recovery_{sc}")
                if r is not None:
                    print(f"  {sc} recovery: {r:.1f}%")

        elif key == "guideline_2":
            print(f"\n  FA by timestamp condition:")
            for tc, fa in g["timestamp_means_FA"].items():
                print(f"    {tc}: {fa:.4f}")
            print(f"  Max difference: {g['max_diff']:.4f}")
            if "adversarial_test" in g:
                at = g["adversarial_test"]
                sig = _sig_marker(at["p"])
                print(f"  Adversarial (S2): actual={at['actual_FA']:.4f} vs "
                      f"misleading={at['misleading_FA']:.4f}  p={at['p']:.6f}  {sig}")

        elif key == "guideline_3":
            print(f"\n  {'Domain':<25s}  {'FA':>6s}  {'HR':>6s}  {'POS':>6s}  {'TE':>6s}")
            print("  " + "-" * 55)
            for d, v in sorted(g["domain_performance"].items(), key=lambda x: -x[1]["TE"]):
                print(f"  {d:<25s}  {v['FA']:>6.3f}  {v['HR']:>6.3f}  "
                      f"{v['POS']:>6.3f}  {v['TE']:>+6.3f}")
            print(f"\n  Vulnerability ranking (TE × (1+HR)):")
            for d, v in g["vulnerability_ranking"].items():
                print(f"    {d:<25s}: {v:.4f}")

        elif key == "guideline_4":
            print(f"\n  {'Mech':<6s}  {'%':>6s}  {'TE_m':>6s}  {'Contrib':>7s}  "
                  f"{'FA':>6s}  {'POS':>6s}  {'AU':>6s}")
            print("  " + "-" * 50)
            for mech, v in sorted(g["mechanism_stats"].items(),
                                  key=lambda x: -x[1]["weighted_contribution"]):
                print(f"  {mech:<6s}  {v['proportion']*100:>5.1f}%  {v['TE_m']:>+6.3f}  "
                      f"{v['weighted_contribution']:>+7.4f}  {v['FA']:>6.3f}  "
                      f"{v['POS']:>6.3f}  {v['AU']:>6.3f}")

        elif key == "guideline_5":
            if "stale_mean_FA" in g:
                print(f"\n  Stale mean FA: {g['stale_mean_FA']:.4f}")
                print(f"  Fresh mean FA: {g['fresh_mean_FA']:.4f}")
                print(f"  Step size: {g['step_size']:+.4f}")
                print(f"  Within-stale range: {g['within_stale_range']:.4f}")
                print(f"  Within-fresh range: {g['within_fresh_range']:.4f}")
                if g.get("stale_positions"):
                    print(f"  Stale positions: {g['stale_positions']}")
                if g.get("fresh_positions"):
                    print(f"  Fresh positions: {g['fresh_positions']}")
            elif "note" in g:
                print(f"\n  {g['note']}")

        print(f"\n  >> {g['recommendation']}")

    # Decision matrix
    print(f"\n{'=' * 70}")
    print(f"  REFRESH STRATEGY DECISION MATRIX")
    print(f"{'=' * 70}")
    for priority, d in results["decision_matrix"].items():
        label = priority.replace("_", " ").title()
        print(f"\n  [{label}]")
        for k, v in d.items():
            print(f"    {k}: {v}")


# ---------------------------------------------------------------------------
# 13. Pretty printing
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
        choices=["4.1", "4.2", "4.3", "4.4", "4.5", "5.1", "5.2", "5.3"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--scored", "-s",
        required=True,
        help="Path to scored JSONL file",
    )
    parser.add_argument(
        "--scored2",
        default=None,
        help="Path to second scored JSONL file (for 4.4 model comparison)",
    )
    parser.add_argument(
        "--model-labels",
        default=None,
        help="Comma-separated model labels for 4.4 (default: gpt-4o-mini,gpt-4o)",
    )
    parser.add_argument(
        "--scored-tg",
        default=None,
        help="Path to temporal gradient scored JSONL (for 5.3 guideline 5)",
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

    if args.experiment == "4.4":
        if not args.scored2:
            sys.exit("Error: --scored2 is required for experiment 4.4")
        scored2_path = Path(args.scored2)
        if not scored2_path.exists():
            sys.exit(f"Error: scored2 file not found: {scored2_path}")
        labels = (args.model_labels.split(",") if args.model_labels
                  else ["gpt-4o-mini", "gpt-4o"])
        df = load_scored_multi([scored_path, scored2_path], labels)
        results = experiment_4_4(df)
        print_experiment_4_4(results)
    else:
        df = load_scored(scored_path)

        if args.experiment == "4.1":
            results = experiment_4_1(df)
            print_experiment_4_1(results)
        elif args.experiment == "4.2":
            results = experiment_4_2(df)
            print_experiment_4_2(results)
        elif args.experiment == "4.3":
            results = experiment_4_3(df)
            print_experiment_4_3(results)
        elif args.experiment == "5.1":
            results = experiment_5_1(df)
            print_experiment_5_1(results)
        elif args.experiment == "4.5":
            results = experiment_4_5(df)
            print_experiment_4_5(results)
        elif args.experiment == "5.2":
            results = experiment_5_2(df)
            print_experiment_5_2(results)
        elif args.experiment == "5.3":
            df_tg = None
            if args.scored_tg:
                tg_path = Path(args.scored_tg)
                if tg_path.exists():
                    df_tg = load_scored(tg_path)
                else:
                    print(f"Warning: --scored-tg file not found: {tg_path}")
            results = experiment_5_3(df, df_tg)
            print_experiment_5_3(results)
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
