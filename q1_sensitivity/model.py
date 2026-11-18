"""Q1 linear dispatch models, in kWh per one-hour interval and DKK.

One builder implements R (separate meters, no contract), I (separate meters,
individual contracts), and C (one community meter). Independent blocks of R/I
are solved separately, exactly, to respect the size-limited Gurobi license.
No integer variables or artificial regularization are added to the dispatch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from pathlib import Path
import json

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from src.data_loader import CommunityData, InputData


@dataclass(frozen=True)
class Q1Config:
    """Contract overrides; None uses the supplied data. SOC is cyclic by default."""
    beta: float | None = None
    alpha: float | None = None
    cap_scale: float = 1.0
    billing: str = "below_cap"
    terminal: str = "cyclic"
    solver_output: bool = False


@dataclass
class Q1Result:
    """Solved physical dispatch, economic ledger, prices and numerical checks."""
    mode: str
    summary: dict
    members: pd.DataFrame
    hourly: pd.DataFrame
    dispatch: pd.DataFrame
    duals: pd.DataFrame
    checks: dict
    config: dict

    def save(self, folder: Path) -> None:
        """Save every table and machine-readable summary, without rounding."""
        folder.mkdir(parents=True, exist_ok=True)
        for name in ["members", "hourly", "dispatch", "duals"]:
            getattr(self, name).to_csv(folder / f"{name}.csv", index=False)
        for name in ["summary", "checks", "config"]:
            (folder / f"{name}.json").write_text(
                json.dumps(getattr(self, name), indent=2, ensure_ascii=False), encoding="utf-8")


def _values(v, n):
    return np.array([v[t].X if isinstance(v[t], gp.Var) else float(v[t]) for t in range(n)])


def _lp_certificate(model: gp.Model) -> dict:
    """Independently reconstruct LP feasibility, stationarity and strong duality.

For min c'x with row dual Pi, reduced costs satisfy RC=c-A'Pi. Bound
contributions RC*LB (RC>=0) or RC*UB (RC<0) complete the dual objective.
"""
    variables = model.getVars()
    x = np.array([v.X for v in variables])
    rc = np.array([v.RC for v in variables])
    residual = np.array([v.Obj for v in variables]) - rc
    dual_obj = float(model.ObjCon)
    primal, dual_sign, comp = 0.0, 0.0, 0.0
    for row in model.getConstrs():
        expr = model.getRow(row)
        lhs = 0.0
        for k in range(expr.size()):
            j, a = expr.getVar(k).index, expr.getCoeff(k)
            lhs += a * x[j]
            residual[j] -= a * row.Pi
        slack = row.RHS - lhs
        violation = abs(slack) if row.Sense == "=" else max(0., -slack if row.Sense == "<" else slack)
        primal = max(primal, violation)
        if row.Sense != "=":
            dual_sign = max(dual_sign, max(0., row.Pi if row.Sense == "<" else -row.Pi))
            comp = max(comp, abs(row.Pi * slack))
        dual_obj += row.RHS * row.Pi
    for j, v in enumerate(variables):
        primal = max(primal, max(0., v.LB - x[j], x[j] - v.UB))
        if abs(rc[j]) > 1e-10:
            bound = v.LB if rc[j] >= 0 else v.UB
            if abs(bound) >= GRB.INFINITY:
                dual_sign = max(dual_sign, abs(rc[j]))
            else:
                dual_obj += rc[j] * bound
                comp = max(comp, abs(rc[j] * (x[j] - bound)))
    return {
        "primal_max_violation": float(primal),
        "dual_sign_max_violation": float(dual_sign),
        "stationarity_max_residual": float(np.max(np.abs(residual))),
        "complementarity_max_residual": float(comp),
        "strong_duality_absolute_gap_DKK": float(abs(model.ObjVal - dual_obj)),
    }


class CommunityModel:
    """Reusable dispatch builder. All time intervals have duration one hour."""

    def __init__(self, community: CommunityData, config: Q1Config | None = None):
        self.com = community
        self.cfg = config or Q1Config()
        self.beta = community.contract.tariff_discount_ratio if self.cfg.beta is None else self.cfg.beta
        self.alpha = community.contract.penalty_DKK_per_kWh if self.cfg.alpha is None else self.cfg.alpha
        self.cap = np.array(community.contract.capacity_limit_kW) * self.cfg.cap_scale
        total = sum(community.contract.share.values())
        # Supplied six-decimal shares have a small sum-rounding error.
        self.shares = {k: v / total for k, v in community.contract.share.items()}
        if not (0 <= self.beta <= 1 and self.alpha >= 0 and self.cfg.cap_scale > 0):
            raise ValueError("Require 0<=beta<=1, alpha>=0 and cap_scale>0")
        if self.cfg.billing not in {"below_cap", "all_metered"}:
            raise ValueError("billing must be below_cap or all_metered")
        if self.cfg.terminal not in {"cyclic", "free"}:
            raise ValueError("terminal must be cyclic or free")
        if self.cfg.billing == "below_cap" and self.alpha < (1-self.beta)*community.members[0].import_tariff:
            raise ValueError("Convex below-cap tariff formulation requires alpha >= discounted import tariff")
        if community.contract.export_capacity_limit_kW is not None:
            raise ValueError("This Q1 implementation uses the supplied contract without an export cap")
        for m in community.members:
            if m.quadratic_disutility not in (None, 0) or m.consumption_utility not in (None, 0):
                raise ValueError("This builder implements the supplied linear-disutility load")
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", int(self.cfg.solver_output))
        self.env.start()

    def close(self):
        """Release the Gurobi environment after experiments."""
        self.env.dispose()

    def _new_model(self, name):
        model = gp.Model(name, env=self.env)
        model.Params.Method = 1  # reproducible dual simplex basis, no objective perturbation
        model.Params.Threads = 1
        model.Params.Seed = 0
        model.Params.FeasibilityTol = 1e-8
        model.Params.OptimalityTol = 1e-8
        return model

    def _add_member(self, model: gp.Model, member: InputData):
        """Add load, optional PV/battery and return net demand expressions."""
        n, cid = member.n_hours, member.consumer_id
        h = range(n)
        load = model.addVars(h, lb={t: float(member.load_lower_kWh[t]) for t in h},
                             ub={t: float(member.load_upper_kWh[t]) for t in h}, name=f"load_{cid}")
        dev = model.addVars(h, lb=0, name=f"absdev_{cid}")
        model.addConstrs((dev[t] >= load[t]-member.reference_load[t] for t in h), name=f"dev_plus_{cid}")
        model.addConstrs((dev[t] >= member.reference_load[t]-load[t] for t in h), name=f"dev_minus_{cid}")
        daily = model.addConstr(gp.quicksum(load.values()) >= member.min_daily_energy_kWh, name=f"daily_{cid}")
        zero = {t: 0.0 for t in h}
        pv = model.addVars(h, lb=0, ub={t: float(member.pv_available[t]) for t in h}, name=f"pv_{cid}") if member.has_pv else zero
        ch, dis, soc = zero, zero, zero
        if member.has_battery:
            ch = model.addVars(h, ub=member.battery_max_charge_kW, name=f"charge_{cid}")
            dis = model.addVars(h, ub=member.battery_max_discharge_kW, name=f"discharge_{cid}")
            soc = model.addVars(h, ub=member.battery_capacity_kWh, name=f"soc_{cid}")
            for t in h:
                prev = member.battery_initial_soc_kWh if t == 0 else soc[t-1]
                model.addConstr(soc[t] == prev + member.battery_charging_efficiency*ch[t]
                                - dis[t]/member.battery_discharging_efficiency, name=f"soc_balance_{cid}_{t}")
            if self.cfg.terminal == "cyclic":
                model.addConstr(soc[n-1] == member.battery_initial_soc_kWh, name=f"terminal_{cid}")
        net = {t: load[t]+ch[t]-pv[t]-dis[t] for t in h}
        private = member.linear_disutility*gp.quicksum(dev.values()) + member.pv_marginal_cost*gp.quicksum(pv.values())
        return dict(load=load, deviation=dev, pv=pv, charge=ch, discharge=dis, soc=soc, net=net, private=private, daily=daily)

    def _solve_block(self, members: list[InputData], mode: str, cap: np.ndarray):
        model = self._new_model(f"Q1_{mode}")
        n = members[0].n_hours
        h = range(n)
        variables = {m.consumer_id: self._add_member(model, m) for m in members}
        imp = model.addVars(h, name="grid_import")
        exp = model.addVars(h, name="grid_export")
        # Gurobi Pi on (import-export-net=0) is the positive internal price.
        balances = {t: model.addConstr(imp[t]-exp[t] == gp.quicksum(variables[m.consumer_id]["net"][t] for m in members),
                                      name=f"meter_balance_{t}") for t in h}
        a = members[0].import_tariff * (1 if mode == "R" else 1-self.beta)
        b = members[0].export_tariff * (1 if mode == "R" else 1-self.beta)
        penalty_coefficient = self.alpha - (a if self.cfg.billing == "below_cap" else 0)
        excess, caps = {t: 0. for t in h}, {}
        if mode != "R":
            excess = model.addVars(h, name="excess")
            caps = {t: model.addConstr(imp[t]-excess[t] <= float(cap[t]), name=f"cap_{t}") for t in h}
            model.addConstrs((excess[t] <= imp[t] for t in h), name="excess_within_import")
        price = members[0].energy_price
        meter_cost = gp.quicksum(price[t]*(imp[t]-exp[t])+a*imp[t]+b*exp[t]
                                +(penalty_coefficient*excess[t] if mode != "R" else 0) for t in h)
        model.setObjective(meter_cost + gp.quicksum(v["private"] for v in variables.values()), GRB.MINIMIZE)
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"{mode}: expected optimal LP, status={model.Status}")
        checks = _lp_certificate(model)
        ui, ue = _values(imp, n), _values(exp, n)
        actual_excess = np.maximum(ui-cap, 0) if mode != "R" else np.zeros(n)
        billed_import = ui-actual_excess if mode != "R" and self.cfg.billing == "below_cap" else ui
        energy = price*(ui-ue)
        tariffs = a*billed_import+b*ue
        penalties = self.alpha*actual_excess
        lam = np.array([balances[t].Pi for t in h])
        mu = np.array([-caps[t].Pi for t in h]) if caps else np.zeros(n)
        hourly = pd.DataFrame({"hour": list(h), "import_kWh": ui, "export_kWh": ue,
                               "cap_kW": cap, "contract_excess_kWh": actual_excess,
                               "energy_DKK": energy, "tariff_DKK": tariffs, "penalty_DKK": penalties,
                               "internal_price_DKK_per_kWh": lam, "cap_shadow_DKK_per_kWh": mu})
        rows, ledger = [], []
        for member in members:
            v = variables[member.consumer_id]
            z = {key: _values(v[key], n) for key in ["load", "deviation", "pv", "charge", "discharge", "soc"]}
            net = z["load"]+z["charge"]-z["pv"]-z["discharge"]
            discomfort = member.linear_disutility*np.abs(z["load"]-member.reference_load)
            pv_cost = member.pv_marginal_cost*z["pv"]
            # R/I: external bill. C: dual-price transfer; physical PV cost/disutility remain private.
            payment = lam*net if mode == "C" else energy+tariffs+penalties
            for t in h:
                rows.append({"member": member.consumer_id, "slope_class": member.slope_class,
                             "archetype": member.archetype, "hour": t, "reference_kWh": member.reference_load[t],
                             **{key+"_kWh": value[t] for key, value in z.items()}, "net_kWh": net[t],
                             "discomfort_DKK": discomfort[t], "pv_cost_DKK": pv_cost[t], "payment_DKK": payment[t]})
            ledger.append({"member": member.consumer_id, "slope_class": member.slope_class,
                           "archetype": member.archetype, "has_pv": member.has_pv, "has_battery": member.has_battery,
                           "load_kWh": z["load"].sum(), "net_kWh": net.sum(), "pv_kWh": z["pv"].sum(),
                           "shifted_out_kWh": np.maximum(member.reference_load-z["load"], 0).sum(),
                           "absolute_deviation_kWh": np.abs(z["load"]-member.reference_load).sum(),
                           "discomfort_DKK": discomfort.sum(), "pv_cost_DKK": pv_cost.sum(),
                           "payment_DKK": payment.sum(), "total_member_cost_DKK": (payment+discomfort+pv_cost).sum(),
                           "daily_minimum_shadow_DKK_per_kWh": v["daily"].Pi})
        dual_rows = [{"constraint": c.ConstrName, "sense": c.Sense, "rhs": c.RHS,
                      "slack": c.Slack, "dual": c.Pi} for c in model.getConstrs()]
        dual_rows += [{"constraint": "bound:"+v.VarName, "sense": "bounds", "rhs": None,
                       "slack": None, "dual": v.RC} for v in model.getVars()]
        stats = {"objective": model.ObjVal, "solver_seconds": model.Runtime,
                 "variables": model.NumVars, "constraints": model.NumConstrs}
        model.dispose()
        return hourly, ledger, rows, dual_rows, checks, stats

    def solve(self, mode: str) -> Q1Result:
        """Solve R/I/C and compute comparable welfare costs and feeder metrics."""
        if mode not in {"R", "I", "C"}:
            raise ValueError("mode must be R, I or C")
        start = perf_counter()
        groups = [self.com.members] if mode == "C" else [[m] for m in self.com.members]
        blocks = [self._solve_block(group, mode, self.cap if mode == "C" else self.cap*self.shares[group[0].consumer_id])
                  for group in groups]
        ledger = pd.DataFrame([row for block in blocks for row in block[1]])
        dispatch = pd.DataFrame([row for block in blocks for row in block[2]])
        duals = pd.DataFrame([dict(block=j, **row) for j, block in enumerate(blocks) for row in block[3]])
        hourly = blocks[0][0].copy()
        if mode != "C":
            cols = ["import_kWh", "export_kWh", "contract_excess_kWh", "energy_DKK", "tariff_DKK", "penalty_DKK"]
            hourly[cols] = sum((block[0][cols] for block in blocks))
            hourly["internal_price_DKK_per_kWh"] = np.nan
            hourly["cap_shadow_DKK_per_kWh"] = np.nan
        hourly["cap_kW"] = self.cap
        hourly["feeder_excess_kWh"] = np.maximum(hourly.import_kWh-self.cap, 0)
        hourly["net_grid_kWh"] = hourly.import_kWh-hourly.export_kWh
        net_member_positive = dispatch.assign(positive=dispatch.net_kWh.clip(lower=0)).groupby("hour").positive.sum()
        hourly["internal_exchange_kWh"] = np.maximum(net_member_positive.to_numpy()-hourly.import_kWh.to_numpy(), 0) if mode == "C" else 0.
        grid = float(hourly[["energy_DKK", "tariff_DKK", "penalty_DKK"]].to_numpy().sum())
        private = float(ledger[["discomfort_DKK", "pv_cost_DKK"]].to_numpy().sum())
        summary = {
            "model": mode, "total_cost_DKK": grid+private, "grid_cost_DKK": grid,
            "energy_DKK": float(hourly.energy_DKK.sum()), "tariff_DKK": float(hourly.tariff_DKK.sum()),
            "penalty_DKK": float(hourly.penalty_DKK.sum()), "discomfort_DKK": float(ledger.discomfort_DKK.sum()),
            "pv_cost_DKK": float(ledger.pv_cost_DKK.sum()), "import_kWh": float(hourly.import_kWh.sum()),
            "export_kWh": float(hourly.export_kWh.sum()), "peak_import_kW": float(hourly.import_kWh.max()),
            "feeder_excess_kWh": float(hourly.feeder_excess_kWh.sum()),
            "feeder_excess_hours": int((hourly.feeder_excess_kWh > 1e-6).sum()),
            "max_feeder_excess_kW": float(hourly.feeder_excess_kWh.max()),
            "contract_excess_kWh": float(hourly.contract_excess_kWh.sum()),
            "load_kWh": float(ledger.load_kWh.sum()), "pv_kWh": float(ledger.pv_kWh.sum()),
            "shifted_out_kWh": float(ledger.shifted_out_kWh.sum()),
            "internal_exchange_kWh": float(hourly.internal_exchange_kWh.sum()),
            "manager_surplus_DKK": float(ledger.payment_DKK.sum()-grid) if mode == "C" else 0.,
            "sum_member_cost_DKK": float(ledger.total_member_cost_DKK.sum()),
            "cap_rent_DKK": float(np.dot(hourly.cap_shadow_DKK_per_kWh, self.cap)) if mode == "C" else 0.,
            "solver_seconds": sum(block[5]["solver_seconds"] for block in blocks),
            "wall_seconds": perf_counter()-start,
            "largest_LP_variables": max(block[5]["variables"] for block in blocks),
            "largest_LP_constraints": max(block[5]["constraints"] for block in blocks),
            "number_of_LPs": len(blocks),
        }
        checks = {key: max(block[4][key] for block in blocks) for key in blocks[0][4]}
        checks["objective_reconstruction_error_DKK"] = abs(summary["total_cost_DKK"]-sum(block[5]["objective"] for block in blocks))
        checks["aggregate_balance_max_error_kWh"] = float(np.max(np.abs(dispatch.groupby("hour").net_kWh.sum().to_numpy()-hourly.net_grid_kWh.to_numpy())))
        checks["max_simultaneous_battery_charge_discharge_kWh"] = float(np.minimum(dispatch.charge_kWh, dispatch.discharge_kWh).max())
        checks["max_simultaneous_meter_import_export_kWh"] = float(max(np.minimum(block[0].import_kWh, block[0].export_kWh).max() for block in blocks))
        checks["max_daily_energy_excess_kWh"] = float(max(abs(row.load_kWh-self.com.member(row.member).min_daily_energy_kWh) for row in ledger.itertuples()))
        if mode == "C":
            checks["settlement_cap_rent_identity_error_DKK"] = abs(summary["manager_surplus_DKK"]-summary["cap_rent_DKK"])
        config = dict(asdict(self.cfg), beta=self.beta, alpha=self.alpha, day=self.com.day,
                      cap_kW=self.cap.tolist(), normalized_shares=self.shares,
                      supplied_share_sum=sum(self.com.contract.share.values()))
        return Q1Result(mode, summary, ledger, hourly, dispatch, duals, checks, config)

    def best_response_value(self, member: InputData, price: np.ndarray) -> float:
        """Validate decentralization: independent minimum at a fixed internal price."""
        model = self._new_model("member_price_response")
        v = self._add_member(model, member)
        model.setObjective(v["private"]+gp.quicksum(float(price[t])*v["net"][t] for t in range(member.n_hours)))
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError("Member price response not optimal")
        value = model.ObjVal
        model.dispose()
        return value

    def minimum_contract_excess(self, mode: str) -> float:
        """Auxiliary feasibility LP: minimum possible billed exceedance in kWh.

        Monetary costs are deliberately excluded, to distinguish a physical
        inability to meet caps from a cost-optimal choice to exceed them.
        """
        if mode not in {"I", "C"}:
            raise ValueError("Minimum violation is defined for contract models I/C")
        groups = [self.com.members] if mode == "C" else [[m] for m in self.com.members]
        total = 0.
        for group in groups:
            model = self._new_model("minimum_contract_excess")
            values = [self._add_member(model, member) for member in group]
            cap = self.cap if mode == "C" else self.cap*self.shares[group[0].consumer_id]
            e = model.addVars(range(24), name="excess")
            for t in range(24):
                model.addConstr(e[t] >= gp.quicksum(v["net"][t] for v in values)-cap[t])
            model.setObjective(gp.quicksum(e.values()))
            model.optimize()
            if model.Status != GRB.OPTIMAL:
                raise RuntimeError("Minimum violation LP not optimal")
            total += model.ObjVal
            model.dispose()
        return total
