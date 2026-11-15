"""Generate the Q1 evidence report and figures from saved validated results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"R": "#64748B", "I": "#C17B2B", "C": "#087F8C"}


def table(frame: pd.DataFrame, digits: int = 2) -> str:
    """Format a small Markdown table without an extra dependency."""
    def cell(value):
        if isinstance(value, (bool, np.bool_)):
            return "Yes" if value else "No"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value).replace("|", "/")
    lines = ["| " + " | ".join(frame.columns) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def save_figure(fig, folder: Path, name: str):
    """PNG for slides; SVG with editable text for later figure editing."""
    fig.savefig(folder / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(folder / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_report(root: Path):
    results = root / "results"
    report_dir = results / "report"
    figures = report_dir / "figures"
    required = ["base/comparison.csv", "base/settlement.csv", "base/C/config.json",
                "base/inputs.csv", "validation.json", "sensitivity/validation.json",
                "sensitivity/one_dimensional.csv", "sensitivity/break_even.csv",
                "alternate_billing/comparison.csv"]
    missing = [name for name in required if not (results / name).exists()]
    if missing:
        raise FileNotFoundError("Run base, validate and sensitivity first. Missing: " + ", ".join(missing))
    validation = json.loads((results / "validation.json").read_text(encoding="utf-8"))
    sensitivity = json.loads((results / "sensitivity/validation.json").read_text(encoding="utf-8"))
    if not validation["all_passed"] or not sensitivity["all_passed"]:
        raise ValueError("Refusing to publish materials from failed validation results")
    config = json.loads((results / "base/C/config.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(results / "base/comparison.csv").set_index("model")
    members = pd.read_csv(results / "base/settlement.csv")
    r, i, c = (summary.loc[mode] for mode in ["R", "I", "C"])
    count = len(members)
    worse_r = int((members.benefit_vs_R_DKK < -1e-6).sum())
    worse_i = int((members.benefit_vs_I_DKK < -1e-6).sum())
    saving = float(r.total_cost_DKK - c.total_cost_DKK)
    saving_pct = float(100 * saving / r.total_cost_DKK)
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})

    # Member benefits against the no-contract option.
    fig, ax = plt.subplots(figsize=(10.5, 6.7))
    values = members.benefit_vs_R_DKK.to_numpy()
    y = np.arange(count)
    ax.barh(y, values, color=np.where(values >= -1e-6, COLORS["C"], "#BC4B51"), height=.65)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_yticks(y, members.member)
    ax.invert_yaxis()
    margin = max(1., float(np.max(np.abs(values))) * .25)
    ax.set_xlim(min(0., float(values.min())) - margin, max(0., float(values.max())) + margin)
    ax.set_xlabel("Member benefit vs R (DKK/day): R cost minus C settlement cost")
    ax.set_title("Lower total cost does not guarantee individual benefits", loc="left", pad=38, weight="bold")
    ax.text(0, 1.025, f"Base case: total cost falls {saving_pct:.1f}%; {worse_r}/{count} members are worse off",
            transform=ax.transAxes, color="#475569")
    for position, value in zip(y, values):
        ax.text(value + (.3 if value >= 0 else -.3), position, f"{value:+.2f}",
                va="center", ha="left" if value >= 0 else "right", fontsize=9)
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    fig.text(.13, .015, "C1-C4: increasing discomfort cost | PV: solar | B: battery | PVB: both\n"
             "Base/d1; cyclic battery; below-cap billing; selected optimal dual solution.", fontsize=9, color="#475569")
    fig.subplots_adjust(left=.14, right=.96, bottom=.17, top=.82)
    save_figure(fig, figures, "01_member_benefits")
    members[["member", "archetype", "cost_R_DKK", "cost_I_DKK", "total_member_cost_DKK",
             "benefit_vs_R_DKK", "benefit_vs_I_DKK"]].to_csv(report_dir / "chart_data.csv", index=False)

    # Import profiles and sensitivity evidence.
    hourly = {mode: pd.read_csv(results / f"base/{mode}/hourly.csv") for mode in COLORS}
    inputs = pd.read_csv(results / "base/inputs.csv")
    fig, ax = plt.subplots(figsize=(10.5, 4.5), layout="constrained")
    for mode, frame in hourly.items():
        ax.step(frame.hour, frame.import_kWh, where="mid", color=COLORS[mode], label=mode, linewidth=2)
    ax.step(inputs.hour, inputs.cap_kW, where="mid", color="#172638", linestyle="--", label="Community cap")
    ax.set(xlabel="Hour (0-23)", ylabel="Gross import (kW; one-hour intervals)",
           title="Base case: community coordination reduces peak imports", xticks=range(0, 24, 2))
    ax.legend(ncol=4, frameon=False)
    ax.grid(alpha=.15)
    save_figure(fig, figures, "02_import_profiles")
    sweep = pd.read_csv(results / "sensitivity/one_dimensional.csv")
    fig, ax = plt.subplots(figsize=(10.5, 4.5), layout="constrained")
    for mode in COLORS:
        frame = sweep[(sweep.dimension == "beta") & (sweep.model == mode)].sort_values("value")
        ax.plot(frame.value, frame.total_cost_DKK, color=COLORS[mode], label=mode,
                marker="o" if mode != "R" else None, linestyle="--" if mode == "R" else "-")
    ax.axvline(config["beta"], color="#94A3B8", linestyle=":", label="Base discount")
    ax.set(xlabel="Grid-tariff discount ratio", ylabel="Total cost (DKK/day)",
           title="Contract value depends on the tariff discount")
    ax.legend(frameon=False, ncol=4)
    ax.grid(alpha=.15)
    save_figure(fig, figures, "03_discount_sensitivity")

    facts = {"scope": "base/d1, cyclic SOC, below_cap billing, selected optimal dual solution",
             "members": count, "R_cost_DKK": float(r.total_cost_DKK),
             "I_cost_DKK": float(i.total_cost_DKK), "C_cost_DKK": float(c.total_cost_DKK),
             "saving_vs_R_DKK": saving, "saving_vs_R_pct": saving_pct,
             "R_peak_kW": float(r.peak_import_kW), "C_peak_kW": float(c.peak_import_kW),
             "worse_off_vs_R": worse_r, "worse_off_vs_I": worse_i,
             "manager_surplus_DKK": float(c.manager_surplus_DKK),
             "base_validation_checks": len(validation["tests"]),
             "sensitivity_configurations": sensitivity["contract_configurations"]}
    (report_dir / "facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    headline = (f"C lowers total cost from {r.total_cost_DKK:.2f} to {c.total_cost_DKK:.2f} DKK/day "
                f"({saving_pct:.2f}%), but {worse_r}/{count} members are worse off than R.")
    comparison = summary.reset_index()[["model", "total_cost_DKK", "peak_import_kW",
                                        "contract_excess_kWh", "manager_surplus_DKK"]]
    thresholds = pd.read_csv(results / "sensitivity/break_even.csv")
    billing = pd.read_csv(results / "alternate_billing/comparison.csv")
    report = f"""# Q1 evidence report

Generated from saved results. Edit the model/configuration and rerun experiments to change the evidence; do not hand-edit numerical results.

## Main result

{headline}
Against I, {worse_i}/{count} members are worse off. The manager surplus is {c.manager_surplus_DKK:.2f} DKK/day.
These are base-case findings under the selected optimal dual solution, not universal claims.

{table(comparison)}

R has no contract penalty: its zero *contract* exceedance must not be read as zero feeder overload. Gross imports for R/I sum individual imports; C uses the common meter. This is the assignment metric, not a full network-security assessment.

## Mechanism and settlement

Pooling permits internal exchange and shared use of import capacity. Total cost includes external energy, grid fees, penalties, PV costs and discomfort. Internal payments are transfers, not extra community costs.
Each member pays sum(lambda[t] * net[i,t]) and also bears PV and discomfort costs. Benefit is the independently optimized R/I cost minus the C settlement cost. Budget balance requires manager surplus = 0, not merely a nonnegative surplus.

The manager surplus equals sum(mu[t] * cap[t]) under this formulation. Member costs sum to community cost plus manager surplus. No redistribution is implemented.

## Model assumptions

- Perfect information, direct control, 24 one-hour intervals, base member mix.
- Cyclic battery SOC; continuous devices; no internal power-flow model or battery degradation cost.
- Soft import cap. In below_cap billing only non-excess imports pay the discounted import tariff; all imported energy still pays the energy price.
- The alternative all_metered interpretation is computed separately; course clarification is still needed.
- No explicit charge/discharge or meter import/export exclusivity binaries. Solved cases are checked for simultaneous flows; this is not a guarantee for arbitrary new data.

See [formulation](docs/formulation.md) for variables, constraints, signs and pricing.

## Validation

Base and auxiliary checks passed: {len(validation['tests'])}. Distinct sensitivity contracts checked: {sensitivity['contract_configurations']}.
Checks cover balances, bounds, SOC, objective reconstruction, LP stationarity/complementarity/strong duality, independent member price responses, cap-value finite differences and input hashes.
Alternative optimal bases can change schedules and prices. Matching one hourly schedule is not required for optimality.
The minimum possible individual-contract exceedance is {validation['minimum_possible_contract_excess_kWh']['I']:.6f} kWh/day; the community minimum is {validation['minimum_possible_contract_excess_kWh']['C']:.6f} kWh/day.
For an additional independent formulation and explicit-dual check run `python audit_q1_course.py`; its fresh results are in results/course_audit/.

## Sensitivity evidence

Thresholds refer to below_cap billing, with other inputs held at their baseline values. A bracket is an interval, not an exact point. No crossing in a tested interval does not establish a global impossibility.

{table(thresholds, 6)}

## Alternative billing

{table(billing[['billing', 'model', 'total_cost_DKK', 'penalty_DKK']])}

## Figures and exported data

- [Member benefits](results/report/figures/01_member_benefits.png)
- [Import profiles](results/report/figures/02_import_profiles.png)
- [Discount sensitivity](results/report/figures/03_discount_sensitivity.png)
- [Chart data](results/report/chart_data.csv)
- [Summary facts](results/report/facts.json)

"""
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    print("Q1 report and figures generated in results/report/.", flush=True)
