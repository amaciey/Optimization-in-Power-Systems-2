"""Reproduce Q1: python run_q1.py all (or base / validate / sensitivity / report)."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import platform
import sys

import gurobipy as gp
import numpy as np
import pandas as pd

from src.data_loader import load_community
from src.q1_model import CommunityModel, Q1Config

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


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


def base_experiment():
    """Base dispatch, cost decomposition, settlement, and alternate tariff reading."""
    results = solve_modes()
    for mode, result in results.items():
        result.save(RESULTS / "base" / mode)
    summary = pd.DataFrame([r.summary for r in results.values()])
    summary["benefit_vs_R_DKK"] = results["R"].summary["total_cost_DKK"]-summary.total_cost_DKK
    summary["peak_reduction_vs_R_pct"] = 100*(1-summary.peak_import_kW/results["R"].summary["peak_import_kW"])
    summary.to_csv(RESULTS / "base" / "comparison.csv", index=False)
    settlement_table(results).to_csv(RESULTS / "base" / "settlement.csv", index=False)
    com = load_community()
    pd.DataFrame({"hour": range(24), "price_DKK_per_kWh": com.energy_price,
                  "reference_load_per_member_kWh": com.members[0].reference_load,
                  "available_PV_per_system_kWh": next(m.pv_available for m in com.members if m.has_pv),
                  "cap_kW": com.contract.capacity_limit_kW}).to_csv(RESULTS / "base" / "inputs.csv", index=False)
    alternate = solve_modes(Q1Config(billing="all_metered"), modes=("I", "C"))

    for mode, result in alternate.items():
        result.save(RESULTS / "alternate_billing" / mode)
    pd.DataFrame([dict(billing="below_cap", **results[m].summary) for m in ["I", "C"]]
                 +[dict(billing="all_metered", **alternate[m].summary) for m in ["I", "C"]]
                 ).to_csv(RESULTS / "alternate_billing" / "comparison.csv", index=False)
    versions = {"python": sys.version, "platform": platform.platform(), "gurobi": gp.gurobi.version(),
                "numpy": np.__version__, "pandas": pd.__version__, "solver_method": "dual simplex",
                "threads": 1, "feasibility_tolerance": 1e-8, "optimality_tolerance": 1e-8,
                "randomness": "Q1 is deterministic; solver seed=0; no sampled scenarios",
                "source_share_sum": sum(com.contract.share.values()),
                "contract": results["C"].config}
    write_json(RESULTS / "environment.json", versions)
    print(summary[["model", "total_cost_DKK", "peak_import_kW", "feeder_excess_kWh", "manager_surplus_DKK"]].to_string(index=False), flush=True)
    return results


def validate_experiment(results=None):
    """Economic/physical checks, LP certificates, limiting cases and dual pricing."""
    results = results or solve_modes()
    report = {"tests": {}, "notes": []}

    def test(name, error, tolerance=1e-6):
        error = float(error)
        report["tests"][name] = {"error": error, "tolerance": tolerance, "passed": error <= tolerance}

    for mode, result in results.items():
        for key, value in result.checks.items():
            test(f"{mode}:{key}", value)
    com = load_community()
    ref = com.contract.reference_model_R
    for source_key, key, tolerance in [("cost_DKK", "total_cost_DKK", .005),
                                       ("daily_import_kWh", "import_kWh", .05),
                                       ("peak_import_kW", "peak_import_kW", .005)]:
        test("published_reference:"+key, abs(results["R"].summary[key]-ref[source_key]), tolerance)
    report["reference_exceedance"] = {"supplied_kWh": ref["exceedance_kWh"],
                                      "computed_kWh": results["R"].summary["feeder_excess_kWh"]}
    test("pooling_cost_C_le_I", max(0, results["C"].summary["total_cost_DKK"]-results["I"].summary["total_cost_DKK"]))
    test("normalized_cap_shares", abs(sum(results["C"].config["normalized_shares"].values())-1))
    # Removing the contract with separate metering must recover R, not pooled C.
    limit = solve_modes(Q1Config(beta=0, cap_scale=100), modes=("I", "C"))
    test("separate_meter_no_contract_limit", abs(limit["I"].summary["total_cost_DKK"]-results["R"].summary["total_cost_DKK"]))
    test("pooled_no_contract_no_more_costly", max(0, limit["C"].summary["total_cost_DKK"]-results["R"].summary["total_cost_DKK"]))
    # Each centralized member's achieved private cost plus payment must be a
    # minimum at the dual prices. Equality of schedules is not required with ties.
    builder = CommunityModel(com)
    report["minimum_possible_contract_excess_kWh"] = {m: builder.minimum_contract_excess(m) for m in ["I", "C"]}
    test("community_zero_excess_physically_feasible", report["minimum_possible_contract_excess_kWh"]["C"])
    test("individual_base_attains_minimum_excess", abs(results["I"].summary["contract_excess_kWh"]
         -report["minimum_possible_contract_excess_kWh"]["I"]))
    lam = results["C"].hourly.internal_price_DKK_per_kWh.to_numpy()
    response_rows = []
    for row in results["C"].members.itertuples():
        optimum = builder.best_response_value(com.member(row.member), lam)
        gap = row.total_member_cost_DKK-optimum
        response_rows.append({"member": row.member, "central_settlement_cost_DKK": row.total_member_cost_DKK,
                              "best_response_cost_DKK": optimum, "gap_DKK": gap})
        test("dual_price_best_response:"+row.member, abs(gap))
    builder.close()
    pd.DataFrame(response_rows).to_csv(RESULTS / "base" / "price_response_validation.csv", index=False)
    # Shadow prices are subgradients. At a kink use both one-sided differences.
    step = 1e-4
    cap0 = com.contract.capacity_limit_kW[0]
    minus = solve_modes(Q1Config(cap_scale=1-step/cap0), modes=("C",))["C"].summary["total_cost_DKK"]
    plus = solve_modes(Q1Config(cap_scale=1+step/cap0), modes=("C",))["C"].summary["total_cost_DKK"]
    f0 = results["C"].summary["total_cost_DKK"]
    left, right = (f0-minus)/step, (plus-f0)/step
    subgradient = -results["C"].hourly.cap_shadow_DKK_per_kWh.sum()
    test("cap_shadow_finite_difference", max(0, left-subgradient, subgradient-right), 1e-5)
    report["cap_derivative"] = dict(left=left, shadow_subgradient=subgradient, right=right)
    # Examine an alternative optimal basis, without asserting uniqueness.
    class PrimalSimplexModel(CommunityModel):
        def _new_model(self, name):
            model = super()._new_model(name)
            model.Params.Method = 0
            return model
    other = PrimalSimplexModel(com)
    altR, altC = other.solve("R"), other.solve("C")
    other.close()
    test("reference_alternative_basis_objective", abs(altR.summary["total_cost_DKK"]-results["R"].summary["total_cost_DKK"]))
    test("community_alternative_basis_objective", abs(altC.summary["total_cost_DKK"]-results["C"].summary["total_cost_DKK"]))
    report["alternative_basis"] = {"R_exceedance_kWh": altR.summary["feeder_excess_kWh"],
                                   "R_peak_kW": altR.summary["peak_import_kW"],
                                   "C_manager_surplus_DKK": altC.summary["manager_surplus_DKK"],
                                   "C_max_price_change": float(np.max(np.abs(altC.hourly.internal_price_DKK_per_kWh-lam)))}
    # An equal mixture of two optimal LP solutions is also optimal. Recompute
    # its physical R cost rather than assuming that rounded table values match.
    a = results["R"].dispatch.set_index(["member", "hour"])
    b = altR.dispatch.set_index(["member", "hour"])
    mix = (a[["load_kWh", "pv_kWh", "charge_kWh", "discharge_kWh", "soc_kWh"]]
           + b[["load_kWh", "pv_kWh", "charge_kWh", "discharge_kWh", "soc_kWh"]])/2
    mix["net_kWh"] = mix.load_kWh+mix.charge_kWh-mix.pv_kWh-mix.discharge_kWh
    mix["import_kWh"] = mix.net_kWh.clip(lower=0)
    mix["export_kWh"] = (-mix.net_kWh).clip(lower=0)
    mix_cost = 0.
    for member in com.members:
        z = mix.loc[member.consumer_id]
        mix_cost += float(np.dot(member.energy_price, z.net_kWh)
                          + member.import_tariff*z.import_kWh.sum()+member.export_tariff*z.export_kWh.sum()
                          + member.pv_marginal_cost*z.pv_kWh.sum()
                          + member.linear_disutility*np.abs(z.load_kWh.to_numpy()-member.reference_load).sum())
    mix_import = mix.groupby("hour").import_kWh.sum().to_numpy()
    mix_excess = float(np.maximum(mix_import-com.contract.capacity_limit_kW, 0).sum())
    test("reference_optimal_mixture_cost", abs(mix_cost-results["R"].summary["total_cost_DKK"]))
    report["optimal_mixture"] = {"cost_DKK": mix_cost, "exceedance_kWh": mix_excess,
                                 "difference_from_published_exceedance_kWh": mix_excess-ref["exceedance_kWh"],
                                 "matches_published_rounding": abs(mix_excess-ref["exceedance_kWh"]) <= .05,
                                 "explanation": "Equal mixture of two computed optimal R schedules; not a claim about the instructor's actual schedule."}
    mix.reset_index().to_csv(RESULTS / "base" / "reference_optimal_mixture.csv", index=False)
    altR.save(RESULTS / "validation_alternative_basis" / "R")
    altC.save(RESULTS / "validation_alternative_basis" / "C")
    # Source data must remain byte-identical.
    manifest = json.loads((ROOT / "source_data_sha256.json").read_text(encoding="utf-8"))
    for filename, expected in manifest.items():
        test("original_data:"+filename, int(hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() != expected), 0)
    if abs(ref["exceedance_kWh"]-results["R"].summary["feeder_excess_kWh"]) > .05:
        report["notes"].append("R may have non-unique optimal schedules. Alternative-basis and optimal-mixture diagnostics are recorded. Published exceedance is descriptive, not a mandatory optimality test; do not force a selected optimal schedule or its arbitrary mixture to match it.")
    report["all_passed"] = all(t["passed"] for t in report["tests"].values())
    write_json(RESULTS / "validation.json", report)
    if not report["all_passed"]:
        raise AssertionError({k: v for k, v in report["tests"].items() if not v["passed"]})
    print(f"Validation: {len(report['tests'])} checks passed; non-unique reference dispatch audited.", flush=True)
    return report


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


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["all", "base", "validate", "sensitivity", "report"], nargs="?", default="all")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    results = None
    if args.experiment in {"all", "base"}:
        results = base_experiment()
    if args.experiment in {"all", "validate"}:
        if not (RESULTS / "base").exists(): results = base_experiment()
        validate_experiment(results)
    if args.experiment in {"all", "sensitivity"}:
        sensitivity_experiment()
    if args.experiment in {"all", "report"}:
        from src.q1_reporting import generate_report
        generate_report(ROOT)


if __name__ == "__main__":
    main()
