"""Run the Q1 contract sweeps and generate their comparison figure."""
from dataclasses import replace
from pathlib import Path
import json

import matplotlib
import numpy as np
import pandas as pd

from src.data_loader import load_community
from .model import CommunityModel, Q1Config

matplotlib.use("Agg")
from matplotlib import pyplot as plt


RESULTS = Path(__file__).resolve().parent / "results"



def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def solve_modes(config=Q1Config(), modes=("R", "I", "C")):
    """Load original data anew; solve named modes with one reusable builder."""
    model = CommunityModel(load_community(), config)
    try:
        return {mode: model.solve(mode) for mode in modes}
    finally:
        model.close()


def settlement_table(results):
    """Positive benefit means the member is better off under C's dual settlement."""
    table = results["C"].members.copy()
    for mode in ["R", "I"]:
        baseline = results[mode].members.set_index("member").total_member_cost_DKK
        table[f"cost_{mode}_DKK"] = table.member.map(baseline)
        table[f"benefit_vs_{mode}_DKK"] = table[f"cost_{mode}_DKK"] - table.total_member_cost_DKK
        table[f"IR_vs_{mode}"] = table[f"benefit_vs_{mode}_DKK"] >= -1e-6
    return table


def sensitivity_experiment():
    """Isolate discount/cap/penalty effects, then examine discount-cap interaction."""
    folder = RESULTS / "sensitivity"
    folder.mkdir(parents=True, exist_ok=True)
    r = solve_modes(modes=("R",))["R"]
    base = Q1Config()
    specs = {
        "beta": [0, .1, .2, .3, .4, .5, .6, .7, .75, .8, .9, .95],
        "cap_scale": [.5, .65, .8, .9, 1, 1.1, 1.2, 1.35, 1.5, 1.75, 2, 2.5, 3, 3.5, 5, 8],
        "alpha": [.25, .5, 1, 2, 3, 5, 10, 20, 50],
    }
    rows, member_rows, check_rows = [], [], []
    cache = {}

    def evaluate(cfg):
        if cfg not in cache:
            cache[cfg] = solve_modes(cfg, modes=("I", "C"))
        solved = cache[cfg]
        if solved["C"].summary["total_cost_DKK"] > solved["I"].summary["total_cost_DKK"]+1e-6:
            raise AssertionError("Pooling bound violated")
        return solved

    for dimension, values in specs.items():
        for value in values:
            cfg = replace(base, **{dimension: value})
            solved = evaluate(cfg)
            current = {"R": r, **solved}
            settle = settlement_table(current)
            for mode, result in current.items():
                row = dict(dimension=dimension, value=value, **result.summary)
                # R dispatch is unchanged but exceedance must use the varied cap.
                if mode == "R":
                    cap = np.array(r.config["cap_kW"])*cfg.cap_scale
                    excess = np.maximum(r.hourly.import_kWh.to_numpy()-cap, 0)
                    row.update(feeder_excess_kWh=float(excess.sum()), feeder_excess_hours=int((excess>1e-6).sum()),
                               max_feeder_excess_kW=float(excess.max()))
                row["benefit_vs_R_DKK"] = r.summary["total_cost_DKK"]-row["total_cost_DKK"]
                rows.append(row)
                if mode != "R":
                    check_rows.append(dict(dimension=dimension, value=value, model=mode, **result.checks))
            member_rows.extend(settle.assign(dimension=dimension, value=value).to_dict("records"))
        print(f"Sensitivity {dimension}: {len(values)} values solved", flush=True)
    pd.DataFrame(rows).to_csv(folder / "one_dimensional.csv", index=False)
    pd.DataFrame(member_rows).to_csv(folder / "members.csv", index=False)
    pd.DataFrame(check_rows).to_csv(folder / "checks.csv", index=False)
    grid = []
    for beta in [0, .25, .5, .75, .95]:
        for scale in [.5, .65, .8, 1, 1.2, 1.5, 2, 2.5, 3.5, 5, 8]:
            solved = evaluate(replace(base, beta=beta, cap_scale=scale))
            for mode, result in solved.items():
                grid.append(dict(beta=beta, cap_scale=scale, **result.summary,
                                 benefit_vs_R_DKK=r.summary["total_cost_DKK"]-result.summary["total_cost_DKK"]))
        print(f"Discount-cap grid: beta={beta} solved", flush=True)
    pd.DataFrame(grid).to_csv(folder / "discount_cap_grid.csv", index=False)
    # Refine break-even points inside the tested interval; never infer them from
    # coarse chart intersections. Objectives are monotone in discount/cap here.
    thresholds = []
    for dimension, lo0, hi0 in [("beta", 0., .95), ("cap_scale", .5, 3.5)]:
        for mode in ["I", "C"]:
            def benefit(value):
                return r.summary["total_cost_DKK"]-evaluate(replace(base, **{dimension: value}))[mode].summary["total_cost_DKK"]
            lo, hi = lo0, hi0
            flo, fhi = benefit(lo), benefit(hi)
            if flo >= 0:
                thresholds.append(dict(dimension=dimension, model=mode, status="beneficial_at_lower_bound", lower=lo, upper=lo))
            elif fhi < 0:
                thresholds.append(dict(dimension=dimension, model=mode, status="not_beneficial_in_interval", lower=lo, upper=hi))
            else:
                for _ in range(13):
                    mid = (lo+hi)/2
                    if benefit(mid) >= 0: hi = mid
                    else: lo = mid
                thresholds.append(dict(dimension=dimension, model=mode, status="bracketed", lower=lo, upper=hi))
    pd.DataFrame(thresholds).to_csv(folder / "break_even.csv", index=False)
    # Residual checks apply to all solved configurations, including grid/refinement.
    max_checks = {}
    for cfg, solved in cache.items():
        for mode, result in solved.items():
            for key, value in result.checks.items():
                max_checks[key] = max(max_checks.get(key, 0), value)
    if any(v > 1e-5 for v in max_checks.values()):
        raise AssertionError(f"Sensitivity validation failed: {max_checks}")
    write_json(folder / "validation.json", {"contract_configurations": len(cache), "all_passed": True, "max_checks": max_checks})
    print(f"Sensitivity complete: {len(cache)} distinct contracts validated", flush=True)


def plot_sensitivity():
    folder = RESULTS / "sensitivity"
    data = pd.read_csv(folder / "one_dimensional.csv")
    columns = [("beta", "Tariff discount", 0.75), ("cap_scale", "Import cap multiplier", 1.0),
               ("alpha", "Exceedance penalty (DKK/kWh)", 10.0)]
    metrics = [("total_cost_DKK", "Total cost (DKK/day)"),
               ("feeder_excess_kWh", "Feeder exceedance (kWh/day)")]
    styles = {"R": ("#566475", "--"), "I": ("#b04c24", "-"), "C": ("#08766e", "-")}
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex="col", sharey="row", layout="constrained")
    for j, (dimension, title, base) in enumerate(columns):
        axes[0, j].set_title(title)
        for i, (metric, label) in enumerate(metrics):
            ax = axes[i, j]
            for mode, (color, style) in styles.items():
                selected = data[(data["dimension"] == dimension) & (data["model"] == mode)].sort_values("value")
                ax.plot(selected["value"], selected[metric], label=mode, color=color, linestyle=style,
                        marker="o" if mode != "R" else None, markersize=3, linewidth=1.7)
            ax.axvline(base, color="#9da4ad", linewidth=1, linestyle=":")
            ax.grid(axis="y", alpha=0.22)
            if j == 0:
                ax.set_ylabel(label)
        axes[1, j].set_xlabel(title)
    axes[0, 0].legend(title="Model", loc="upper right")
    fig.suptitle("Q1 sensitivity on day d1 · dotted lines mark the base contract", fontsize=14, weight="bold")
    fig.savefig(folder / "contract_sensitivity.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    sensitivity_experiment()
    plot_sensitivity()
