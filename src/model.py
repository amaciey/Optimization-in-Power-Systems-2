from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import CommunityData, ContractData, InputData


@dataclass
class Results:
    """Primal and dual solution of one model run."""

    question: str
    status: str
    objective: float
    hourly: pd.DataFrame                   # one row per hour: variables, prices, hourly duals
    duals: dict[str, float] = field(default_factory=dict)   # duals of non-hourly constraints
    meta: dict = field(default_factory=dict)                 # anything else worth keeping (scenario name, ...)

    def save(self, folder: Path | str, tag: str = "") -> None:
        """Write ``hourly`` to CSV and the scalar values to a small text file."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.question}{'_' + tag if tag else ''}"
        self.hourly.to_csv(folder / f"{stem}_hourly.csv", index_label="hour")
        with open(folder / f"{stem}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"status    : {self.status}\nobjective : {self.objective:.4f} DKK\n")
            for k, v in self.duals.items():
                f.write(f"dual[{k}] : {v:.4f}\n")

            for k, v in self.meta.get("economics", {}).items():
                f.write(f"{k:<17}: {v:.4f} DKK\n")
                
    def __str__(self) -> str:
        cols = [c for c in self.hourly.columns if not c.startswith("dual_")]
        return (
            f"status: {self.status} | objective: {self.objective:.2f} DKK\n"
            f"daily totals (kWh): " + ", ".join(f"{c}={self.hourly[c].sum():.1f}" for c in cols if c in ("import", "export", "load", "pv"))
            + (f"\nduals: {self.duals}" if self.duals else "")
        )


def _dual(c) -> float:
    """Dual value of a linear (``Pi``) or quadratic (``QCPi``, requires QCPDual=1) constraint."""
    return c.QCPi if isinstance(c, gp.QConstr) else c.Pi


class FlexibleConsumerModel:
    """TODO: Add docstring"""

    def __init__(self, data: InputData, name: str = "flexible_consumer", verbose: bool = False):
        self.data = data
        self.T = range(data.n_hours)
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.m.Params.QCPDual = 1          # only relevant if you add a quadratic constraint (none is needed in Assignment 1)
        self.var: dict[str, gp.tupledict | gp.Var] = {}   # decision variables by name
        self.con: dict[str, gp.tupledict | gp.Constr] = {}  # constraints by name (duals read from here)

    # ------------------------------------------------------------------ 2. build
    def build(self) -> "FlexibleConsumerModel":
    
        d, m, T = self.data, self.m, self.T

        # --- Decision variables --------------------------------------------------------
        self.var["load"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="load")
        self.var["pv"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="pv")
        self.var["import"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="import")
        self.var["export"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="export")

        L, PV, P_imp, P_exp  = self.var["load"], self.var["pv"], self.var["import"], self.var["export"]

        # Effective import and export tariffs (DKK/kWh)
        p_imp = d.energy_price + d.import_tariff
        p_exp = d.energy_price - d.export_tariff
        u_L = d.consumption_utility if d.consumption_utility is not None else 0.0
        c_pv = d.pv_marginal_cost


        # --- Objective ---------------------------------------------------------------
        m.setObjective(gp.quicksum(p_imp[t]*P_imp[t] - p_exp[t]*P_exp[t] + c_pv*PV[t] - u_L*L[t] for t in T), GRB.MINIMIZE)

        # --- Constraints -------------------------------------------------------------
        # 1. Hourly power balance: Load + Export = PV + Import
        self.con["balance"] = m.addConstrs(
            (L[t] + P_exp[t] - PV[t] - P_imp[t] == 0 for t in T), name="balance")

        # 2. Flexible load operational limits, including hourly deviation bounds when provided
        load_min = d.load_lower_kWh if d.load_lower_kWh is not None else np.full(len(T), d.load_min_kWh)
        load_max = d.load_upper_kWh if d.load_upper_kWh is not None else np.full(len(T), d.load_max_kWh)
        self.con["load_min"] = m.addConstrs(
            (load_min[t] - L[t] <= 0 for t in T), name="load_min")
        
        self.con["load_max"] = m.addConstrs(
            (L[t] - load_max[t] <= 0 for t in T), name="load_max")

        if d.min_daily_energy_kWh is not None:
            self.con["min_daily_energy"] = m.addConstr(
                gp.quicksum(L[t] for t in T) >= d.min_daily_energy_kWh,
                name="min_daily_energy",
            )

        # 3. PV operational limits: 0 <= PV_t <= PV_available
        self.con["pv_min"] = m.addConstrs(
            (-PV[t] <= 0 for t in T), name="pv_min")
        
        self.con["pv_max"] = m.addConstrs(
            (PV[t] - d.pv_available[t] <= 0 for t in T), name="pv_max")

        # 4. Non-negativity of grid exchanges
        self.con["import_min"] = m.addConstrs(
            (-P_imp[t] <= 0 for t in T), name="import_min")
        
        self.con["export_min"] = m.addConstrs((-P_exp[t] <= 0 for t in T), name="export_min")

        m.update()
        return self

    # ------------------------------------------------------------------ 3. solve
    def solve(self) -> Results:
        """Optimise and return primal values, objective and dual values."""
        m = self.m
        m.update()
        if m.NumConstrs == 0 and m.NumQConstrs == 0:
            raise NotImplementedError(
                "The model has no constraints: complete FlexibleConsumerModel.build() in src/model.py first."
            )
        m.optimize()
        status = _status_name(m.Status)
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimisation ended with status {status}. Check the model (m.computeIIS() helps for infeasibility).")
        return self._extract_results(status)

    # --------------------------------------------------------------- extraction
    def _extract_results(self, status: str) -> Results:
        d, T = self.data, list(self.T)
        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = d.energy_price
        hourly["pv_available"] = d.pv_available
        if d.reference_load is not None:
            hourly["reference_load"] = d.reference_load

        # Primal values: every hourly variable family in self.var becomes a column
        for name, v in self.var.items():
            if isinstance(v, gp.tupledict):
                hourly[name] = [v[t].X for t in T]
        scalars = {name: v.X for name, v in self.var.items() if isinstance(v, gp.Var)}

        # Dual values: every constraint family in self.con becomes a 'dual_<name>' column or scalar
        duals: dict[str, float] = {}
        for name, c in self.con.items():
            try:
                if isinstance(c, gp.tupledict):
                    hourly[f"dual_{name}"] = [_dual(c[t]) for t in T]
                else:
                    duals[name] = _dual(c)
            except (AttributeError, gp.GurobiError):
                # No duals available (e.g. model with integer variables)
                pass

        # Implementation of missing parts in question 1e
        p_imp = d.energy_price + d.import_tariff   
        p_exp = d.energy_price - d.export_tariff          
        economics = {
            "procurement_cost": float((p_imp * hourly["import"] - p_exp * hourly["export"]).sum()),
            "pv_cost": float((d.pv_marginal_cost * hourly["pv"]).sum()),
        }
        if d.consumption_utility is not None:
            economics["utility"] = float((d.consumption_utility * hourly["load"]).sum())
            economics["net_utility"] = economics["utility"] - economics["procurement_cost"] - economics["pv_cost"]

        return Results(
            question=d.question,
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            duals=duals,
            meta={"scalar_variables": scalars, "economics": economics},
        )

class DisutilityConsumer(FlexibleConsumerModel):

    def __init__(
        self,
        data: InputData,
        name: str = "disutility_consumer",
        verbose: bool = False,
        contract_type: str = "R",
        contract: ContractData = None
    ):
        super().__init__(data=data, name=name, verbose=verbose)
        self.contract_type = contract_type
        self.contract = contract
    
    def build(self) -> "DisutilityConsumer":
        """Declare decision variables, objective and constraints."""
        super().build()
        d, m, T, con = self.data, self.m, self.T, self.contract

        # Auxiliary variables for linear absolute value decomposition: |L_t - ref_t| = delta_pos + delta_neg
        self.var["delta_pos"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="delta_pos")
        self.var["delta_neg"] = m.addVars(T, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="delta_neg")

        L = self.var["load"]
        PV = self.var["pv"]
        P_imp = self.var["import"]
        P_exp = self.var["export"]
        delta_pos = self.var["delta_pos"]
        delta_neg = self.var["delta_neg"]

        # Effective import and export tariffs
        p_imp = d.energy_price + d.import_tariff
        p_exp = d.energy_price - d.export_tariff
        c_pv = d.pv_marginal_cost
        c_L = d.linear_disutility if d.linear_disutility is not None else 0.0
        ref_load = d.reference_load if d.reference_load is not None else np.zeros(len(T))

        # --- Objective ---------------------------------------------------------------
        # Minimize total cost: procurement costs + PV costs + linear disutility penalty
        match self.contract_type:
            case "R":
                m.setObjective(
                    gp.quicksum(p_imp[t]*P_imp[t] - p_exp[t]*P_exp[t] + c_pv*PV[t] + c_L*(delta_pos[t] + delta_neg[t]) for t in T), GRB.MINIMIZE)
            case "I":
                betaDSO = con.tariff_discount_ratio
                alphaDSO = con.penalty_DKK_per_kWh
                
                P_iDSO = con.individual_capacity_limit_kW[d.consumer_id]
                
                # Apply the tariff discount (1 - beta) to the grid tariffs
                p_imp_I = d.energy_price + (1 - betaDSO) * d.import_tariff
                p_exp_I = d.energy_price - (1 - betaDSO) * d.export_tariff
                
                # Auxiliary variable for imports exceeding the individual cap
                self.var["exceedance"] = m.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="exceedance")
                exceedance = self.var["exceedance"]
                
                # Link exceedance variable to the import and cap
                self.con["exceedance_def"] = m.addConstrs(
                    (exceedance[t] >= P_imp[t] - P_iDSO[t] for t in T), name="exceedance_def"
                )
                
                m.setObjective(
                    gp.quicksum(
                        p_imp_I[t] * P_imp[t] - p_exp_I[t] * P_exp[t] 
                        + c_pv * PV[t] 
                        + c_L * (delta_pos[t] + delta_neg[t]) 
                        + alphaDSO * exceedance[t] 
                        for t in T
                    ), 
                    GRB.MINIMIZE
                )

        # --- Constraints -------------------------------------------------------------
        # Linear deviation link: L_t - reference_load_t = delta_pos_t - delta_neg_t
        self.con["deviation_def"] = m.addConstrs(
            (L[t] - ref_load[t] - (delta_pos[t] - delta_neg[t]) == 0 for t in T), name="deviation_def")

        # Non-negativity of auxiliary deviation variables
        self.con["delta_pos_min"] = m.addConstrs(
            (-delta_pos[t] <= 0 for t in T), name="delta_pos_min")
        
        self.con["delta_neg_min"] = m.addConstrs(
            (-delta_neg[t] <= 0 for t in T), name="delta_neg_min")

        m.update()
        return self

    def _extract_results(self, status: str) -> Results:
        # 1. Get the base results from the parent class
        results = super()._extract_results(status)
        
        # 2. If it's an Individual contract, recalculate the economics correctly
        if self.contract_type == "I" and self.contract:
            hourly = results.hourly
            betaDSO = self.contract.tariff_discount_ratio
            alphaDSO = self.contract.penalty_DKK_per_kWh
            
            # Recalculate discounted tariffs
            p_imp_I = self.data.energy_price + (1 - betaDSO) * self.data.import_tariff
            p_exp_I = self.data.energy_price - (1 - betaDSO) * self.data.export_tariff
            
            # Calculate correct procurement and new penalty costs
            proc_cost = float((p_imp_I * hourly["import"] - p_exp_I * hourly["export"]).sum())
            pen_cost = float((alphaDSO * hourly["exceedance"]).sum())
            
            # Update the economics dictionary
            results.meta["economics"]["procurement_cost"] = proc_cost
            results.meta["economics"]["penalty_cost"] = pen_cost
            
            # Fix net utility to subtract the correct costs
            if "utility" in results.meta["economics"]:
                results.meta["economics"]["net_utility"] = (
                    results.meta["economics"]["utility"] 
                    - proc_cost 
                    - results.meta["economics"]["pv_cost"] 
                    - pen_cost
                )
                
        return results


class BatteryDisutilityConsumer(DisutilityConsumer):
    """Linear-disutility consumer model with an explicitly configured battery."""

    def _battery_parameters(self) -> tuple[float, float, float, float, float, float, float]:
        d = self.data
        values = {
            "battery_capacity_kWh": d.battery_capacity_kWh,
            "battery_max_charge_kW": d.battery_max_charge_kW,
            "battery_max_discharge_kW": d.battery_max_discharge_kW,
            "battery_charging_efficiency": d.battery_charging_efficiency,
            "battery_discharging_efficiency": d.battery_discharging_efficiency,
            "battery_initial_soc_kWh": d.battery_initial_soc_kWh,
        }
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValueError(
                f"Incomplete battery data for {d.consumer_id!r}; missing: {', '.join(missing)}"
            )

        capacity = values["battery_capacity_kWh"]
        max_charge = values["battery_max_charge_kW"]
        max_discharge = values["battery_max_discharge_kW"]
        charging_efficiency = values["battery_charging_efficiency"]
        discharging_efficiency = values["battery_discharging_efficiency"]
        initial_soc = values["battery_initial_soc_kWh"]
        assert capacity is not None
        assert max_charge is not None
        assert max_discharge is not None
        assert charging_efficiency is not None
        assert discharging_efficiency is not None
        assert initial_soc is not None

        if capacity < 0 or max_charge < 0 or max_discharge < 0:
            raise ValueError(f"Battery capacities and power limits must be non-negative for {d.consumer_id!r}.")
        if not 0 < charging_efficiency <= 1 or not 0 < discharging_efficiency <= 1:
            raise ValueError(f"Battery efficiencies must be in (0, 1] for {d.consumer_id!r}.")
        if not 0 <= initial_soc <= capacity:
            raise ValueError(f"Initial battery state of charge must be within capacity for {d.consumer_id!r}.")
        if d.battery_final_soc_kWh is not None and not 0 <= d.battery_final_soc_kWh <= capacity:
            raise ValueError(f"Final battery state of charge must be within capacity for {d.consumer_id!r}.")

        final_soc = d.battery_final_soc_kWh if d.battery_final_soc_kWh is not None else initial_soc
        return (
            capacity,
            max_charge,
            max_discharge,
            charging_efficiency,
            discharging_efficiency,
            initial_soc,
            final_soc,
        )

    def build(self) -> "BatteryDisutilityConsumer":
        super().build()
        d, m, T = self.data, self.m, self.T
        E_cap, P_ch_max, P_dis_max, eta_ch, eta_dis, E_0, E_final = self._battery_parameters()
        load = self.var["load"]
        pv = self.var["pv"]
        imports = self.var["import"]
        exports = self.var["export"]

        self.var["charge"] = m.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="charge")
        self.var["discharge"] = m.addVars(T, lb=0.0, vtype=GRB.CONTINUOUS, name="discharge")
        self.var["soc"] = m.addVars(T, lb=0.0, ub=E_cap, vtype=GRB.CONTINUOUS, name="soc")
        charge = self.var["charge"]
        discharge = self.var["discharge"]
        soc = self.var["soc"]

        m.remove(list(self.con["balance"].values()))
        self.con["balance"] = m.addConstrs(
            (load[t] + exports[t] + charge[t] - pv[t] - imports[t] - discharge[t] == 0 for t in T),
            name="balance",
        )
        self.con["soc_dynamics"] = m.addConstrs(
            (
                soc[t]
                - (E_0 if t == 0 else soc[t - 1])
                - eta_ch * charge[t]
                + discharge[t] / eta_dis
                == 0
                for t in T
            ),
            name="soc_dynamics",
        )
        self.con["charge_max"] = m.addConstrs(
            (charge[t] <= P_ch_max for t in T), name="charge_max"
        )
        self.con["discharge_max"] = m.addConstrs(
            (discharge[t] <= P_dis_max for t in T), name="discharge_max"
        )
        self.con["soc_end"] = m.addConstr(soc[T[-1]] == E_final, name="soc_end")

        m.update()
        return self


class Community:
    """Community model for independent R/I meters or one joint C / DP meter."""

    def __init__(
        self,
        data: CommunityData,
        contract_type: str = "R",
        verbose: bool = False,
        budget_balance: bool = False,
        individual_rationality: bool = False,
        ir_benchmark: str = "I",
        allow_infeasibility_slack: bool = False,
        slack_penalty: float = 1e4,
        price_discrimination: bool = False,
        use_sos1: bool = False,
    ):
        self.data = data
        self.contract_type = contract_type.upper()
        self.verbose = verbose
        self.budget_balance = budget_balance
        self.individual_rationality = individual_rationality
        self.ir_benchmark = ir_benchmark.upper()
        self.allow_infeasibility_slack = allow_infeasibility_slack
        self.slack_penalty = slack_penalty
        self.price_discrimination = price_discrimination
        self.use_sos1 = use_sos1

        self.consumers: dict[str, DisutilityConsumer | BatteryDisutilityConsumer] = {}
        self.m: gp.Model | None = None
        self.var = {}
        self.con = {}

        # Using a single joint Gurobi model for both "C" and "DP"
        if self.contract_type in ("C", "DP"):
            self.m = gp.Model(f"{data.community}_{self.contract_type}")
            self.m.Params.OutputFlag = int(verbose)
            if self.contract_type == "C":
                self.m.Params.Method = 1
                self.m.Params.Threads = 1
            return

        # Independent member models for "R" and "I"
        for member in data.members:
            if member.consumer_id in self.consumers:
                raise ValueError(f"Duplicate community member id: {member.consumer_id!r}")
            consumer_class = BatteryDisutilityConsumer if member.has_battery else DisutilityConsumer
            self.consumers[member.consumer_id] = consumer_class(
                data=member,
                name=f"{data.community}_{member.consumer_id}",
                verbose=verbose,
                contract_type=contract_type,
                contract=data.contract,
            )

    def _compute_benchmark_costs(self) -> dict[str, float]:
        """Solve standalone benchmark (R or I) to obtain C_i^{bench} for Individual Rationality."""
        bench_type = self.ir_benchmark if self.ir_benchmark in ("R", "I") else "I"
        bench_comm = Community(data=self.data, contract_type=bench_type, verbose=False)
        bench_comm.build()
        bench_res = bench_comm.solve()
        return {cid: float(res.objective) for cid, res in bench_res.items()}

    def _add_complementarity(
        self,
        primal_slack: gp.LinExpr | gp.Var,
        dual_var: gp.Var,
        M_primal: float,
        M_dual: float,
        name: str,
    ) -> gp.Var | None:
        """Enforce 0 <= primal_slack _|_ dual_var >= 0 via big-M binary or SOS1."""
        m = self.m
        assert m is not None
        if self.use_sos1:
            s = m.addVar(lb=0.0, name=f"slack_{name}")
            m.addConstr(s == primal_slack, name=f"slack_def_{name}")
            m.addSOS(GRB.SOS_TYPE1, [s, dual_var])
            return None
        u = m.addVar(vtype=GRB.BINARY, name=f"u_{name}")
        m.addConstr(primal_slack <= M_primal * (1 - u), name=f"comp_p_{name}")
        m.addConstr(dual_var <= M_dual * u, name=f"comp_d_{name}")
        return u

    def build(self) -> "Community":
        """Build independent member models (R, I) or the joint community model (C, DP)."""
        if self.contract_type not in ("C", "DP"):
            for consumer in self.consumers.values():
                consumer.build()
            return self

        if self.m is None:
            raise RuntimeError(f"Community {self.contract_type} model was not initialized.")
        d, m, T = self.data, self.m, range(self.data.members[0].n_hours)
        contract, first = d.contract, d.members[0]

        # Contract parameters
        beta = contract.tariff_discount_ratio
        alpha = contract.penalty_DKK_per_kWh
        discounted_import = (1 - beta) * first.import_tariff
        discounted_export = (1 - beta) * first.export_tariff

        if contract.export_capacity_limit_kW is not None:
            raise ValueError(f"Formulation {self.contract_type} does not support an export capacity limit.")
        if alpha < discounted_import:
            raise ValueError(f"{self.contract_type} requires alpha to be at least the discounted import tariff.")

        # --- 1. Shared Primal Variables & Feasibility Constraints (Common to C and DP) ---
        members = {}
        member_constrs = {}
        private_cost = gp.LinExpr()

        for member in d.members:
            if member.quadratic_disutility not in (None, 0) or member.consumption_utility not in (None, 0):
                raise ValueError(f"{self.contract_type} supports the supplied linear-disutility load only.")
            cid = member.consumer_id

            values = {
                "load": m.addVars(
                    T,
                    lb={t: float(member.load_lower_kWh[t]) for t in T},
                    ub={t: float(member.load_upper_kWh[t]) for t in T},
                    name=f"load_{cid}",
                ),
                "pv": m.addVars(
                    T,
                    lb=0.0,
                    ub={t: float(member.pv_available[t]) for t in T},
                    name=f"pv_{cid}",
                ),
                "delta_pos": m.addVars(T, lb=0.0, name=f"delta_pos_{cid}"),
                "delta_neg": m.addVars(T, lb=0.0, name=f"delta_neg_{cid}"),
            }

            c_dev = m.addConstrs(
                (
                    values["load"][t] - float(member.reference_load[t])
                    == values["delta_pos"][t] - values["delta_neg"][t]
                    for t in T
                ),
                name=f"deviation_{cid}",
            )
            c_energy = m.addConstr(
                gp.quicksum(values["load"][t] for t in T) >= float(member.min_daily_energy_kWh),
                name=f"min_daily_energy_{cid}",
            )
            member_constrs[cid] = {"deviation": c_dev, "min_daily_energy": c_energy}

            if member.has_battery:
                values["charge"] = m.addVars(T, lb=0.0, ub=float(member.battery_max_charge_kW), name=f"charge_{cid}")
                values["discharge"] = m.addVars(
                    T, lb=0.0, ub=float(member.battery_max_discharge_kW), name=f"discharge_{cid}"
                )
                values["soc"] = m.addVars(T, lb=0.0, ub=float(member.battery_capacity_kWh), name=f"soc_{cid}")
                soc_con = {}
                for t in T:
                    previous = float(member.battery_initial_soc_kWh) if t == 0 else values["soc"][t - 1]
                    soc_con[t] = m.addConstr(
                        values["soc"][t]
                        == previous
                        + float(member.battery_charging_efficiency) * values["charge"][t]
                        - values["discharge"][t] / float(member.battery_discharging_efficiency),
                        name=f"soc_balance_{cid}_{t}",
                    )
                c_soc_end = m.addConstr(
                    values["soc"][member.n_hours - 1] == float(member.battery_initial_soc_kWh),
                    name=f"soc_terminal_{cid}",
                )
                member_constrs[cid]["soc_balance"] = soc_con
                member_constrs[cid]["soc_terminal"] = c_soc_end

            members[cid] = values
            private_cost += float(member.pv_marginal_cost) * gp.quicksum(values["pv"][t] for t in T)
            private_cost += float(member.linear_disutility) * gp.quicksum(
                values["delta_pos"][t] + values["delta_neg"][t] for t in T
            )

        imp = m.addVars(T, lb=0.0, name="community_import")
        exp = m.addVars(T, lb=0.0, name="community_export")
        excess = m.addVars(T, lb=0.0, name="community_excess")
        self.var = {"members": members, "import": imp, "export": exp, "excess": excess}
        self.con["members"] = member_constrs

        def net(member, t):
            v = members[member.consumer_id]
            expr = v["load"][t] - v["pv"][t]
            if member.has_battery:
                expr += v["charge"][t] - v["discharge"][t]
            return expr

        self.con["balance"] = m.addConstrs(
            (imp[t] - exp[t] == gp.quicksum(net(member, t) for member in d.members) for t in T),
            name="community_balance",
        )
        self.con["capacity"] = m.addConstrs(
            (imp[t] - excess[t] <= float(contract.capacity_limit_kW[t]) for t in T),
            name="community_capacity",
        )
        self.con["excess_import"] = m.addConstrs(
            (excess[t] <= imp[t] for t in T),
            name="community_excess_import",
        )

        meter_cost = gp.quicksum(
            float(first.energy_price[t]) * (imp[t] - exp[t])
            + discounted_import * imp[t]
            + discounted_export * exp[t]
            + (alpha - discounted_import) * excess[t]
            for t in T
        )

        # If formulation is "C", set objective and return immediately
        if self.contract_type == "C":
            m.setObjective(meter_cost + private_cost, GRB.MINIMIZE)
            m.update()
            return self

        # --- 2. Additional Upper/Lower-Level KKT & Settlement Structure for "DP" ---
        # Safe big-M constants for dual and primal bounds
        max_price_bound = float(np.max(first.energy_price) + alpha + 5.0)
        M_dual = 2.0 * max_price_bound

        # Internal price variable x_t (uniform across members, or discriminated per member if enabled)
        x_uniform = m.addVars(T, lb=-max_price_bound, ub=max_price_bound, name="internal_price")
        self.var["internal_price"] = x_uniform
        if self.price_discrimination:
            x_member = {
                member.consumer_id: m.addVars(
                    T, lb=-max_price_bound, ub=max_price_bound, name=f"internal_price_{member.consumer_id}"
                )
                for member in d.members
            }
            self.var["member_internal_price"] = x_member
        else:
            x_member = {member.consumer_id: x_uniform for member in d.members}

        dual_objs = {}
        member_priv_costs = {}
        duals_dict = {}

        for member in d.members:
            cid = member.consumer_id
            v = members[cid]
            x_i = x_member[cid]
            c_L = float(member.linear_disutility)
            c_pv = float(member.pv_marginal_cost)
            L_min = member.load_lower_kWh
            L_max = member.load_upper_kWh
            L_ref = member.reference_load
            E_min = float(member.min_daily_energy_kWh)
            d_max = float(member.max_hourly_deviation_kWh or member.load_max_kWh)

            # Follower dual variables
            mu_delta = m.addVars(T, lb=-c_L, ub=c_L, name=f"mu_delta_{cid}")
            mu_minus = m.addVars(T, lb=0.0, ub=M_dual, name=f"mu_minus_{cid}")
            mu_plus = m.addVars(T, lb=0.0, ub=M_dual, name=f"mu_plus_{cid}")
            mu_E = m.addVar(lb=0.0, ub=M_dual, name=f"mu_E_{cid}")
            nu_minus = m.addVars(T, lb=0.0, ub=M_dual, name=f"nu_minus_{cid}")
            nu_plus = m.addVars(T, lb=0.0, ub=M_dual, name=f"nu_plus_{cid}")
            omega_plus = m.addVars(T, lb=0.0, ub=2.0 * c_L, name=f"omega_plus_{cid}")
            omega_minus = m.addVars(T, lb=0.0, ub=2.0 * c_L, name=f"omega_minus_{cid}")

            # Stationarity conditions
            m.addConstrs(
                (x_i[t] + mu_delta[t] - mu_minus[t] + mu_plus[t] - mu_E == 0.0 for t in T),
                name=f"stat_load_{cid}",
            )
            m.addConstrs(
                (c_pv - x_i[t] - nu_minus[t] + nu_plus[t] == 0.0 for t in T),
                name=f"stat_pv_{cid}",
            )
            m.addConstrs((c_L - mu_delta[t] - omega_plus[t] == 0.0 for t in T), name=f"stat_dpos_{cid}")
            m.addConstrs((c_L + mu_delta[t] - omega_minus[t] == 0.0 for t in T), name=f"stat_dneg_{cid}")

            # Complementarity conditions for load, daily energy, PV, and deviations
            M_E = float(np.sum(L_max) - E_min) + 1e-3
            self._add_complementarity(
                gp.quicksum(v["load"][t] for t in T) - E_min, mu_E, M_E, M_dual, f"Emin_{cid}"
            )
            for t in T:
                span_L = float(L_max[t] - L_min[t]) + 1e-4
                self._add_complementarity(
                    v["load"][t] - float(L_min[t]), mu_minus[t], span_L, M_dual, f"Lmin_{cid}_{t}"
                )
                self._add_complementarity(
                    float(L_max[t]) - v["load"][t], mu_plus[t], span_L, M_dual, f"Lmax_{cid}_{t}"
                )
                self._add_complementarity(
                    v["delta_pos"][t], omega_plus[t], d_max, 2.0 * c_L, f"dpos_{cid}_{t}"
                )
                self._add_complementarity(
                    v["delta_neg"][t], omega_minus[t], d_max, 2.0 * c_L, f"dneg_{cid}_{t}"
                )
                pv_avail = float(member.pv_available[t])
                if pv_avail > 0:
                    self._add_complementarity(v["pv"][t], nu_minus[t], pv_avail, M_dual, f"pvmin_{cid}_{t}")
                    self._add_complementarity(
                        pv_avail - v["pv"][t], nu_plus[t], pv_avail, M_dual, f"pvmax_{cid}_{t}"
                    )
                else:
                    m.addConstr(nu_plus[t] == 0.0, name=f"nu_plus_zero_{cid}_{t}")

            # Strong-duality objective expression g_i(Phi_i)
            g_i = (
                gp.quicksum(
                    -float(L_ref[t]) * mu_delta[t]
                    + float(L_min[t]) * mu_minus[t]
                    - float(L_max[t]) * mu_plus[t]
                    - float(member.pv_available[t]) * nu_plus[t]
                    for t in T
                )
                + E_min * mu_E
            )

            # Battery KKT stationarity, complementarity, and dual objective terms
            if member.has_battery:
                E_cap = float(member.battery_capacity_kWh)
                P_ch = float(member.battery_max_charge_kW)
                P_dis = float(member.battery_max_discharge_kW)
                eta_ch = float(member.battery_charging_efficiency)
                eta_dis = float(member.battery_discharging_efficiency)
                E_0 = float(member.battery_initial_soc_kWh)

                xi = m.addVars(T, lb=-M_dual, ub=M_dual, name=f"xi_{cid}")
                xi_end = m.addVar(lb=-M_dual, ub=M_dual, name=f"xi_end_{cid}")
                phi_ch_min = m.addVars(T, lb=0.0, ub=M_dual, name=f"phi_ch_min_{cid}")
                phi_ch_max = m.addVars(T, lb=0.0, ub=M_dual, name=f"phi_ch_max_{cid}")
                phi_dis_min = m.addVars(T, lb=0.0, ub=M_dual, name=f"phi_dis_min_{cid}")
                phi_dis_max = m.addVars(T, lb=0.0, ub=M_dual, name=f"phi_dis_max_{cid}")
                psi_min = m.addVars(T, lb=0.0, ub=M_dual, name=f"psi_min_{cid}")
                psi_max = m.addVars(T, lb=0.0, ub=M_dual, name=f"psi_max_{cid}")

                m.addConstrs(
                    (x_i[t] - eta_ch * xi[t] - phi_ch_min[t] + phi_ch_max[t] == 0.0 for t in T),
                    name=f"stat_ch_{cid}",
                )
                m.addConstrs(
                    (-x_i[t] + (1.0 / eta_dis) * xi[t] - phi_dis_min[t] + phi_dis_max[t] == 0.0 for t in T),
                    name=f"stat_dis_{cid}",
                )
                for t in T:
                    if t < member.n_hours - 1:
                        m.addConstr(
                            xi[t] - xi[t + 1] - psi_min[t] + psi_max[t] == 0.0,
                            name=f"stat_soc_{cid}_{t}",
                        )
                    else:
                        m.addConstr(
                            xi[t] + xi_end - psi_min[t] + psi_max[t] == 0.0,
                            name=f"stat_soc_{cid}_{t}",
                        )
                    self._add_complementarity(v["charge"][t], phi_ch_min[t], P_ch, M_dual, f"chmin_{cid}_{t}")
                    self._add_complementarity(
                        P_ch - v["charge"][t], phi_ch_max[t], P_ch, M_dual, f"chmax_{cid}_{t}"
                    )
                    self._add_complementarity(
                        v["discharge"][t], phi_dis_min[t], P_dis, M_dual, f"dismin_{cid}_{t}"
                    )
                    self._add_complementarity(
                        P_dis - v["discharge"][t], phi_dis_max[t], P_dis, M_dual, f"dismax_{cid}_{t}"
                    )
                    self._add_complementarity(v["soc"][t], psi_min[t], E_cap, M_dual, f"socmin_{cid}_{t}")
                    self._add_complementarity(
                        E_cap - v["soc"][t], psi_max[t], E_cap, M_dual, f"socmax_{cid}_{t}"
                    )

                g_i -= E_0 * (xi[0] + xi_end) + gp.quicksum(
                    P_ch * phi_ch_max[t] + P_dis * phi_dis_max[t] + E_cap * psi_max[t] for t in T
                )

            dual_objs[cid] = g_i
            member_priv_costs[cid] = c_pv * gp.quicksum(v["pv"][t] for t in T) + c_L * gp.quicksum(
                v["delta_pos"][t] + v["delta_neg"][t] for t in T
            )
            duals_dict[cid] = {"mu_E": mu_E, "mu_delta": mu_delta, "mu_minus": mu_minus, "mu_plus": mu_plus}

        self.var["follower_duals"] = duals_dict
        self.dual_objs = dual_objs
        self.member_priv_costs = member_priv_costs

        # --- 3. Parameterized Settlement Constraints (Budget Balance & Individual Rationality) ---
        slack_penalty_expr = gp.LinExpr()

        if self.budget_balance:
            total_member_payments = gp.quicksum(
                dual_objs[m_i.consumer_id] - member_priv_costs[m_i.consumer_id] for m_i in d.members
            )
            if self.allow_infeasibility_slack:
                sigma_bb = m.addVar(lb=0.0, name="slack_BB")
                self.var["slack_BB"] = sigma_bb
                self.con["budget_balance"] = m.addConstr(
                    meter_cost - total_member_payments <= sigma_bb, name="budget_balance"
                )
                slack_penalty_expr += self.slack_penalty * sigma_bb
            else:
                self.con["budget_balance"] = m.addConstr(
                    total_member_payments == meter_cost, name="budget_balance"
                )

        if self.individual_rationality:
            bench_costs = self._compute_benchmark_costs()
            self.benchmark_costs = bench_costs
            if self.allow_infeasibility_slack:
                sigma_ir = m.addVars(d.member_ids, lb=0.0, name="slack_IR")
                self.var["slack_IR"] = sigma_ir
                self.con["individual_rationality"] = m.addConstrs(
                    (dual_objs[cid] - bench_costs[cid] <= sigma_ir[cid] for cid in d.member_ids),
                    name="individual_rationality",
                )
                slack_penalty_expr += self.slack_penalty * gp.quicksum(sigma_ir[cid] for cid in d.member_ids)
            else:
                self.con["individual_rationality"] = m.addConstrs(
                    (dual_objs[cid] <= bench_costs[cid] for cid in d.member_ids),
                    name="individual_rationality",
                )

        m.setObjective(meter_cost + private_cost + slack_penalty_expr, GRB.MINIMIZE)
        m.update()
        return self

    def solve(self):
        """Solve the selected formulation and return member or community results."""
        if self.contract_type not in ("C", "DP"):
            return {member_id: consumer.solve() for member_id, consumer in self.consumers.items()}
        if self.m is None or not self.var:
            raise RuntimeError(f"Community {self.contract_type} model must be built before solving.")

        self.m.optimize()
        status = _status_name(self.m.Status)

        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Community formulation {self.contract_type} ended with status {status}.")

        d, T = self.data, list(range(self.data.members[0].n_hours))
        contract, first = d.contract, d.members[0]

        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["energy_price"] = first.energy_price
        hourly["community_import"] = [self.var["import"][t].X for t in T]
        hourly["community_export"] = [self.var["export"][t].X for t in T]
        hourly["community_excess"] = [self.var["excess"][t].X for t in T]
        hourly["capacity"] = contract.capacity_limit_kW

        if self.contract_type == "C":
            hourly["dual_internal_price"] = [self.con["balance"][t].Pi for t in T]
            hourly["dual_capacity"] = [-self.con["capacity"][t].Pi for t in T]
        else:
            hourly["dual_internal_price"] = [self.var["internal_price"][t].X for t in T]
            hourly["dual_capacity"] = np.zeros(len(T))
            if self.price_discrimination:
                for member in d.members:
                    cid = member.consumer_id
                    hourly[f"{cid}_internal_price"] = [
                        self.var["member_internal_price"][cid][t].X for t in T
                    ]

        for member in d.members:
            for name, variables in self.var["members"][member.consumer_id].items():
                hourly[f"{member.consumer_id}_{name}"] = [variables[t].X for t in T]

        # Alias columns expected by plotting.py
        hourly["price"] = first.energy_price
        hourly["import"] = hourly["community_import"]
        hourly["export"] = hourly["community_export"]
        hourly["load"] = sum(hourly[f"{m.consumer_id}_load"] for m in d.members)
        hourly["pv"] = sum(hourly[f"{m.consumer_id}_pv"] for m in d.members)
        hourly["pv_available"] = sum(m.pv_available for m in d.members)
        hourly["reference_load"] = sum(m.reference_load for m in d.members)
        hourly["dual_balance"] = -hourly["dual_internal_price"]

        # Calculate community economics
        energy = first.energy_price * (hourly["community_import"] - hourly["community_export"])
        tariff = (
            (1 - contract.tariff_discount_ratio) * first.import_tariff * hourly["community_import"]
            + (1 - contract.tariff_discount_ratio) * first.export_tariff * hourly["community_export"]
        )
        penalty = (
            contract.penalty_DKK_per_kWh - (1 - contract.tariff_discount_ratio) * first.import_tariff
        ) * hourly["community_excess"]
        dso_cost = float((energy + tariff + penalty).sum())

        # Per-member settlement and welfare summary
        settlement = {}
        total_collected = 0.0
        for member in d.members:
            cid = member.consumer_id
            net_q = (
                hourly[f"{cid}_load"]
                - hourly[f"{cid}_pv"]
                + (hourly[f"{cid}_charge"] - hourly[f"{cid}_discharge"] if member.has_battery else 0.0)
            )
            p_vec = (
                hourly[f"{cid}_internal_price"]
                if (self.contract_type == "DP" and self.price_discrimination)
                else hourly["dual_internal_price"]
            )
            payment = float((p_vec * net_q).sum())
            priv = float(
                (
                    member.pv_marginal_cost * hourly[f"{cid}_pv"]
                    + member.linear_disutility * (hourly[f"{cid}_delta_pos"] + hourly[f"{cid}_delta_neg"])
                ).sum()
            )
            settlement[cid] = {
                "payment_DKK": payment,
                "private_cost_DKK": priv,
                "total_cost_DKK": payment + priv,
            }
            if hasattr(self, "benchmark_costs") and cid in self.benchmark_costs:
                settlement[cid]["benchmark_cost_DKK"] = self.benchmark_costs[cid]
                settlement[cid]["ir_gain_DKK"] = self.benchmark_costs[cid] - (payment + priv)
            total_collected += payment

        economics = {
            "energy_cost": float(energy.sum()),
            "tariff_cost": float(tariff.sum()),
            "excess_penalty_cost": float(penalty.sum()),
            "dso_cost": dso_cost,
            "collected_from_members": total_collected,
            "manager_budget_surplus": total_collected - dso_cost,
        }
        if "slack_BB" in self.var:
            economics["slack_BB_DKK"] = float(self.var["slack_BB"].X)
        if "slack_IR" in self.var:
            economics["total_slack_IR_DKK"] = float(
                sum(self.var["slack_IR"][cid].X for cid in d.member_ids)
            )

        true_social_cost = dso_cost + sum(s["private_cost_DKK"] for s in settlement.values())

        return Results(
            question=self.contract_type,
            status=status,
            objective=true_social_cost,
            hourly=hourly,
            duals={
                "capacity_rent": float(np.dot(hourly["dual_capacity"], contract.capacity_limit_kW))
            },
            meta={"economics": economics, "settlement": settlement},
        )


class CommunityOptimizationModel:
    """Build and solve a community formulation from shared community data.

    R/I use independent member meters; C and DP use one joint community meter.
    """

    _SUPPORTED_FORMULATIONS = {"R", "I", "C", "DP"}

    def __init__(
        self,
        community: CommunityData,
        formulation: str,
        verbose: bool = False,
        **kwargs,
    ):
        self.community = community
        self.formulation = formulation.upper()
        self.verbose = verbose
        self.kwargs = kwargs
        self.model: Community | None = None
        self.results: dict[str, Results] | Results | None = None
        self._validate_community()

    def _validate_community(self) -> None:
        """Validate dimensions shared by every member model."""
        if not self.community.members:
            raise ValueError("CommunityOptimizationModel requires at least one member.")

        n_hours = self.community.members[0].n_hours
        if n_hours == 0:
            raise ValueError("Community members must contain at least one time period.")
        if any(member.n_hours != n_hours for member in self.community.members):
            raise ValueError("All community members must use the same time horizon.")

        if self.formulation not in self._SUPPORTED_FORMULATIONS:
            raise NotImplementedError(
                f"Formulation {self.formulation!r} is not implemented; "
                f"available formulations: {sorted(self._SUPPORTED_FORMULATIONS)}."
            )

    def build_model(self) -> "CommunityOptimizationModel":
        """Construct fresh member models for the selected formulation."""
        self.model = Community(
            data=self.community,
            contract_type=self.formulation,
            verbose=self.verbose,
            **self.kwargs,
        )
        self.model.build()
        self.results = None
        return self

    def solve(self) -> dict[str, Results] | Results:
        """Solve the selected formulation."""
        if self.model is None:
            self.build_model()

        if self.model is None:
            raise RuntimeError("CommunityOptimizationModel failed to build its member models.")
        self.results = self.model.solve()
        return self.results

    def extract_results(self) -> dict[str, Results] | Results:
        """Return solved member results without re-solving the models."""
        if self.results is None:
            raise RuntimeError("CommunityOptimizationModel has not been solved yet.")
        return self.results

_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC", GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    return _STATUS.get(code, f"STATUS_{code}")