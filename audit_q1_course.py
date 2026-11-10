"""Audit Q1 against Weeks 3/4/5/7 methods: python audit_q1_course.py.

The independent primal uses separate below-cap/above-cap import variables.
The explicit dual is independently optimized for two single-member LPs.
Analytical KKT checks use the named device equations, not the model matrix.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from src.data_loader import load_community
from src.q1_model import CommunityModel, Q1Config

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/course_audit"


def _err(values):
    return float(np.max(np.abs(np.asarray(values, dtype=float))))


def physical_checks(com, result):
    """Re-evaluate original physical requirements directly from saved schedules."""
    errors = {}
    for m in com.members:
        d = result.dispatch[result.dispatch.member == m.consumer_id].sort_values("hour")
        l, g, ch, dis, soc = [d[k+"_kWh"].to_numpy() for k in ["load", "pv", "charge", "discharge", "soc"]]
        errors[m.consumer_id+":load_bounds"] = max(0., float(np.max(m.load_lower_kWh-l)), float(np.max(l-m.load_upper_kWh)))
        errors[m.consumer_id+":daily_minimum"] = max(0., float(m.min_daily_energy_kWh-l.sum()))
        errors[m.consumer_id+":abs_deviation"] = _err(d.deviation_kWh-np.abs(l-m.reference_load))
        errors[m.consumer_id+":PV_bounds"] = max(0., float(np.max(-g)), float(np.max(g-m.pv_available)))
        errors[m.consumer_id+":member_balance"] = _err(d.net_kWh-l-ch+g+dis)
        if m.has_battery:
            prev = np.r_[m.battery_initial_soc_kWh, soc[:-1]]
            errors[m.consumer_id+":SOC_dynamics"] = _err(soc-prev-m.battery_charging_efficiency*ch+dis/m.battery_discharging_efficiency)
            errors[m.consumer_id+":cyclic_SOC"] = abs(float(soc[-1]-m.battery_initial_soc_kWh))
            errors[m.consumer_id+":battery_bounds"] = max(0., float(np.max(-ch)), float(np.max(-dis)),
                float(np.max(-soc)), float(np.max(ch-m.battery_max_charge_kW)),
                float(np.max(dis-m.battery_max_discharge_kW)), float(np.max(soc-m.battery_capacity_kWh)))
        else:
            errors[m.consumer_id+":absent_battery"] = max(_err(ch), _err(dis), _err(soc))
    h = result.hourly
    net = result.dispatch.groupby("hour").net_kWh.sum().to_numpy()
    errors["meter_balance"] = _err(h.import_kWh-h.export_kWh-net)
    if result.mode == "C":
        errors["community_excess_definition"] = _err(h.contract_excess_kWh-np.maximum(h.import_kWh-h.cap_kW, 0))
        errors["settlement_identity"] = abs(result.summary["sum_member_cost_DKK"]-result.summary["total_cost_DKK"]-result.summary["manager_surplus_DKK"])
    # No assumption that the minimum daily energy inequality must always bind.
    return errors


def analytical_kkt(com, result):
    """Check all variable stationarity equations using the lecture sign convention.

Pi for the coded SOC equality is minus the lecture's stored-energy value psi.
The cyclic terminal equality contributes an extra final-hour multiplier.
"""
    errors = {}
    for j, member in enumerate(com.members):
        block = 0 if result.mode == "C" else j
        duals = result.duals[result.duals.block == block].set_index("constraint").dual
        cid = member.consumer_id
        get = lambda name: float(duals.loc[name])
        lam = np.array([get(f"meter_balance_{t}") for t in range(24)])
        daily = get(f"daily_{cid}")
        plus = np.array([get(f"dev_plus_{cid}[{t}]") for t in range(24)])
        minus = np.array([get(f"dev_minus_{cid}[{t}]") for t in range(24)])
        rc = lambda stem: np.array([get(f"bound:{stem}_{cid}[{t}]") for t in range(24)])
        errors[cid+":load_stationarity"] = _err(rc("load")-(lam-daily+plus-minus))
        errors[cid+":deviation_stationarity"] = _err(rc("absdev")-(member.linear_disutility-plus-minus))
        if member.has_pv:
            errors[cid+":PV_stationarity"] = _err(rc("pv")-(member.pv_marginal_cost-lam))
        if member.has_battery:
            psi = -np.array([get(f"soc_balance_{cid}_{t}") for t in range(24)])
            errors[cid+":charge_stationarity"] = _err(rc("charge")-(lam-member.battery_charging_efficiency*psi))
            errors[cid+":discharge_stationarity"] = _err(rc("discharge")-(-lam+psi/member.battery_discharging_efficiency))
            terminal = get(f"terminal_{cid}")
            expected = psi-np.r_[psi[1:], terminal]
            errors[cid+":SOC_stationarity"] = _err(rc("soc")-expected)
            # Slides' psi_t=psi_(t+1) implication applies only when SOC is interior.
            d = result.dispatch[result.dispatch.member==cid].sort_values("hour")
            interior = (d.soc_kWh.to_numpy()[:-1]>1e-6)&(d.soc_kWh.to_numpy()[:-1]<member.battery_capacity_kWh-1e-6)
            errors[cid+":interior_SOC_value"] = _err((psi[:-1]-psi[1:])[interior]) if interior.any() else 0.
        a = member.import_tariff*(1 if result.mode=="R" else 1-result.config["beta"])
        b = member.export_tariff*(1 if result.mode=="R" else 1-result.config["beta"])
        p = member.energy_price
        mu = np.array([-get(f"cap_{t}") for t in range(24)]) if result.mode!="R" else np.zeros(24)
        nu = np.array([-get(f"excess_within_import[{t}]") for t in range(24)]) if result.mode!="R" else np.zeros(24)
        grid_rc = lambda name: np.array([get(f"bound:{name}[{t}]") for t in range(24)])
        errors[cid+":grid_import_stationarity"] = _err(grid_rc("grid_import")-(p+a-lam+mu-nu))
        errors[cid+":grid_export_stationarity"] = _err(grid_rc("grid_export")-(-p+b+lam))
        if result.mode!="R":
            coefficient = result.config["alpha"]-(a if result.config["billing"]=="below_cap" else 0)
            errors[cid+":excess_stationarity"] = _err(grid_rc("excess")-(coefficient-mu+nu))
    return errors


def explicit_dual(primal, env):
    """Construct and solve the bound-inclusive dual; do not read primal Pi/RC.

For row Ax<=b use pi<=0; Ax>=b uses pi>=0; equality pi is free.
L=c'x+pi'(b-Ax)+eta'(LB-x)+zeta'(x-UB).
Dual: max b'pi+LB'eta-UB'zeta, A'pi+eta-zeta=c.
"""
    dual = gp.Model("independent_explicit_dual", env=env)
    dual.Params.Threads = 1
    dual.Params.FeasibilityTol = 1e-8
    dual.Params.OptimalityTol = 1e-8
    x = primal.getVars()
    equations = [gp.LinExpr() for _ in x]
    obj = gp.LinExpr(primal.ObjCon)
    for row in primal.getConstrs():
        lower, upper = (0, GRB.INFINITY) if row.Sense==">" else (-GRB.INFINITY, 0) if row.Sense=="<" else (-GRB.INFINITY, GRB.INFINITY)
        pi = dual.addVar(lb=lower, ub=upper, name="row_"+row.ConstrName)
        obj += row.RHS*pi
        expr = primal.getRow(row)
        for k in range(expr.size()): equations[expr.getVar(k).index] += expr.getCoeff(k)*pi
    for v in x:
        if v.LB > -GRB.INFINITY:
            eta = dual.addVar(name="lower_"+v.VarName)
            obj += v.LB*eta
            equations[v.index] += eta
        if v.UB < GRB.INFINITY:
            zeta = dual.addVar(name="upper_"+v.VarName)
            obj -= v.UB*zeta
            equations[v.index] -= zeta
    for v in x: dual.addConstr(equations[v.index] == v.Obj)
    dual.setObjective(obj, GRB.MAXIMIZE)
    dual.optimize()
    if dual.Status != GRB.OPTIMAL: raise RuntimeError("Explicit dual failed")
    gap = abs(primal.ObjVal-dual.ObjVal)
    row = dict(primal_DKK=primal.ObjVal, dual_DKK=dual.ObjVal, absolute_gap_DKK=gap,
               dual_variables=dual.NumVars, dual_constraints=dual.NumConstrs)
    dual.dispose()
    return row


def independent_primal(com, mode, env, billing="below_cap", terminal="cyclic", solve_dual=False):
    """Fresh implementation, with q variables and two import price blocks.

No original model-building helper is called. For C this uses 1992 variables,
so it remains within the current license limit. 'minimum' is the lecture's
terminal-SOC inequality, with E_end set to the initial SOC.
"""
    groups = [com.members] if mode=="C" else [[m] for m in com.members]
    cost = 0.
    dual_rows = []
    for group in groups:
        model = gp.Model("lecture_reference_"+mode, env=env)
        model.Params.Threads = 1
        model.Params.FeasibilityTol = 1e-8
        model.Params.OptimalityTol = 1e-8
        hours = range(24)
        private = gp.LinExpr()
        q = {}
        for m in group:
            cid=m.consumer_id
            l = model.addVars(hours, lb={t:float(m.load_lower_kWh[t]) for t in hours}, ub={t:float(m.load_upper_kWh[t]) for t in hours}, name="l_"+cid)
            d = model.addVars(hours, name="d_"+cid)
            q[cid] = model.addVars(hours, lb=-GRB.INFINITY, name="q_"+cid)
            model.addConstr(gp.quicksum(l.values()) >= m.min_daily_energy_kWh)
            for t in hours:
                model.addConstr(l[t]-m.reference_load[t] <= d[t])
                model.addConstr(m.reference_load[t]-l[t] <= d[t])
            g = model.addVars(hours, ub={t:float(m.pv_available[t]) for t in hours}, name="g_"+cid) if m.has_pv else {t:0 for t in hours}
            ch, dis = {t:0 for t in hours}, {t:0 for t in hours}
            if m.has_battery:
                ch=model.addVars(hours,ub=m.battery_max_charge_kW,name="ch_"+cid)
                dis=model.addVars(hours,ub=m.battery_max_discharge_kW,name="dis_"+cid)
                soc=model.addVars(hours,ub=m.battery_capacity_kWh,name="soc_"+cid)
                for t in hours:
                    previous = m.battery_initial_soc_kWh if t==0 else soc[t-1]
                    model.addConstr(soc[t]-previous == m.battery_charging_efficiency*ch[t]-dis[t]/m.battery_discharging_efficiency)
                if terminal=="cyclic": model.addConstr(soc[23] == m.battery_initial_soc_kWh)
                elif terminal=="minimum": model.addConstr(soc[23] >= m.battery_initial_soc_kWh)
                else: raise ValueError(terminal)
            for t in hours: model.addConstr(q[cid][t]+g[t]+dis[t] == l[t]+ch[t])
            private += m.linear_disutility*gp.quicksum(d.values())+m.pv_marginal_cost*gp.quicksum(g.values())
        first=group[0]
        a=first.import_tariff*(1 if mode=="R" else 1-com.contract.tariff_discount_ratio)
        b=first.export_tariff*(1 if mode=="R" else 1-com.contract.tariff_discount_ratio)
        cap=com.contract.capacity_limit_kW.copy()
        if mode=="I": cap *= com.contract.share[first.consumer_id]/sum(com.contract.share.values())
        normal=model.addVars(hours,ub=GRB.INFINITY if mode=="R" else {t:float(cap[t]) for t in hours},name="normal_import")
        excess={t:0 for t in hours} if mode=="R" else model.addVars(hours,name="above_cap_import")
        exp=model.addVars(hours,name="export")
        obj=private
        for t in hours:
            model.addConstr(normal[t]+excess[t]-exp[t] == gp.quicksum(q[m.consumer_id][t] for m in group))
            obj += (first.energy_price[t]+a)*normal[t]+(-first.energy_price[t]+b)*exp[t]
            if mode!="R":
                above_price=first.energy_price[t]+com.contract.penalty_DKK_per_kWh+(a if billing=="all_metered" else 0)
                obj += above_price*excess[t]
        model.setObjective(obj)
        model.optimize()
        if model.Status!=GRB.OPTIMAL: raise RuntimeError("Independent lecture primal failed")
        cost += model.ObjVal
        if solve_dual and first.consumer_id=="C1_PVB":
            dual_rows.append(dict(model=mode,member=first.consumer_id,**explicit_dual(model,env)))
        model.dispose()
    return cost,dual_rows


def main():
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True,exist_ok=True)
    com=load_community()
    builder=CommunityModel(com)
    results={mode:builder.solve(mode) for mode in ["R","I","C"]}
    builder.close()
    tests=[]
    def check(name,error,tolerance=1e-6):
        error=float(error)
        tests.append(dict(test=name,error=error,tolerance=tolerance,passed=np.isfinite(error) and error<=tolerance))
    for mode,result in results.items():
        for name,error in physical_checks(com,result).items():check(mode+":physical:"+name,error)
        for name,error in analytical_kkt(com,result).items():check(mode+":KKT:"+name,error)
    env=gp.Env(empty=True);env.setParam("OutputFlag",0);env.start()
    comparisons,dual_rows=[],[]
    for mode in ["R","I","C"]:
        independent,duals=independent_primal(com,mode,env,solve_dual=mode!="C")
        check(mode+":independent_primal",abs(independent-results[mode].summary["total_cost_DKK"]))
        minimum,_=independent_primal(com,mode,env,terminal="minimum")
        comparisons.append(dict(model=mode,original_cyclic_DKK=results[mode].summary["total_cost_DKK"],
                                 independent_cyclic_DKK=independent,lecture_minimum_terminal_DKK=minimum))
        # A relaxed minimum terminal condition must not be more expensive than equality.
        check(mode+":minimum_terminal_relaxation",max(0,minimum-independent))
        dual_rows+=duals
    for row in dual_rows:check(row["model"]+":explicit_dual_gap",row["absolute_gap_DKK"])
    env.dispose()
    c=results["C"]
    check("budget_identity",abs(c.summary["sum_member_cost_DKK"]-c.summary["total_cost_DKK"]-c.summary["manager_surplus_DKK"]))
    check("capacity_rent_identity",abs(c.summary["manager_surplus_DKK"]-c.summary["cap_rent_DKK"]))
    test_frame=pd.DataFrame(tests)
    test_frame.to_csv(OUT/"checks.csv",index=False)
    pd.DataFrame(comparisons).to_csv(OUT/"independent_primals.csv",index=False)
    pd.DataFrame(dual_rows).to_csv(OUT/"explicit_duals.csv",index=False)
    summary=dict(passed=bool(test_frame.passed.all()),checks=len(tests),max_error=float(test_frame.error.max()),
                 failed=test_frame.loc[~test_frame.passed].to_dict("records"),
                 baseline={m:r.summary["total_cost_DKK"] for m,r in results.items()},
                 conclusions={"core_LP_formulation":"consistent with lecture method on supplied case",
                              "terminal_SOC":"cyclic equality is a chosen stricter condition, not the slides' minimum inequality",
                              "billing":"not resolved by these lectures; both assignment-text interpretations remain documented",
                              "reference_exceedance":"non-unique dispatch metric must not be forced to an arbitrary published optimal schedule"})
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(pd.DataFrame(dual_rows).to_string(index=False))
    print(f"Course audit: {len(tests)} checks; passed={summary['passed']}; max error={summary['max_error']:.3e}")
    if not summary["passed"]:raise AssertionError(summary["failed"])


if __name__=="__main__":main()
