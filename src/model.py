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
    """Community model for independent R/I meters or one joint C meter."""

    def __init__(
        self,
        data: CommunityData,
        contract_type: str = "R",
        verbose: bool = False,
    ):
        self.data = data
        self.contract_type = contract_type.upper()
        self.verbose = verbose
        self.consumers: dict[str, DisutilityConsumer | BatteryDisutilityConsumer] = {}
        self.m: gp.Model | None = None
        self.var = {}
        self.con = {}

        # using different gurobi model for the community "C" case compared to the "R" and "I" cases
        if self.contract_type == "C":
            self.m = gp.Model(f"{data.community}_C")
            self.m.Params.OutputFlag = int(verbose)
            self.m.Params.Method = 1
            self.m.Params.Threads = 1
            return

        # Adding members to the self.consumers dict according to the json data passed in to the class
        for member in data.members:
            if member.consumer_id in self.consumers: 
                raise ValueError(f"Duplicate community member id: {member.consumer_id!r}")
            consumer_class = BatteryDisutilityConsumer if member.has_battery else DisutilityConsumer
            self.consumers[member.consumer_id] = consumer_class(
                data=member,
                name=f"{data.community}_{member.consumer_id}",
                verbose=verbose,
                contract_type=contract_type,
                contract=data.contract
            )

    def build(self) -> "Community":
        """Build independent member models or the joint community model."""
        if self.contract_type != "C":
            for consumer in self.consumers.values():
                consumer.build() # builds an individual model for each consumer if the contract type is not "C"
            return self

        if self.m is None:
            raise RuntimeError("Community C model was not initialized.")
        d, m, T = self.data, self.m, range(self.data.members[0].n_hours)
        contract, first = d.contract, d.members[0]
        
        # Defining contract terms as variables
        beta = contract.tariff_discount_ratio
        alpha = contract.penalty_DKK_per_kWh
        discounted_import = (1 - beta) * first.import_tariff
        discounted_export = (1 - beta) * first.export_tariff
        
        # Errors out if malformed contract values for "C" case
        if contract.export_capacity_limit_kW is not None:
            raise ValueError("Community formulation C does not support an export capacity limit.")
        if alpha < discounted_import:
            raise ValueError("C requires alpha to be at least the discounted import tariff.")

        members = {}
        private_cost = gp.LinExpr()
        for member in d.members:
            if member.quadratic_disutility not in (None, 0) or member.consumption_utility not in (None, 0):
                raise ValueError("C supports the supplied linear-disutility load only.")
            cid = member.consumer_id
            
            # Adds variables to gurobi model
            values = {
                "load": m.addVars(T, lb={t: float(member.load_lower_kWh[t]) for t in T},
                                  ub={t: float(member.load_upper_kWh[t]) for t in T}, name=f"load_{cid}"),
                "pv": m.addVars(T, lb=0, ub={t: float(member.pv_available[t]) for t in T}, name=f"pv_{cid}"),
                "delta_pos": m.addVars(T, lb=0, name=f"delta_pos_{cid}"),
                "delta_neg": m.addVars(T, lb=0, name=f"delta_neg_{cid}"),
            }

            # Adds constraints to gurobi model
            m.addConstrs((values["load"][t] - member.reference_load[t] ==
                          values["delta_pos"][t] - values["delta_neg"][t] for t in T), name=f"deviation_{cid}")
            m.addConstr(gp.quicksum(values["load"][t] for t in T) >= member.min_daily_energy_kWh,
                        name=f"min_daily_energy_{cid}")

            # Adds battery-specific variables and constraints if the consumer has a battery
            if member.has_battery:
                values["charge"] = m.addVars(T, lb=0, ub=member.battery_max_charge_kW, name=f"charge_{cid}")
                values["discharge"] = m.addVars(T, lb=0, ub=member.battery_max_discharge_kW, name=f"discharge_{cid}")
                values["soc"] = m.addVars(T, lb=0, ub=member.battery_capacity_kWh, name=f"soc_{cid}")
                for t in T:
                    previous = member.battery_initial_soc_kWh if t == 0 else values["soc"][t - 1]
                    m.addConstr(values["soc"][t] == previous
                                + member.battery_charging_efficiency * values["charge"][t]
                                - values["discharge"][t] / member.battery_discharging_efficiency,
                                name=f"soc_balance_{cid}_{t}")
                m.addConstr(values["soc"][member.n_hours - 1] == member.battery_initial_soc_kWh,
                            name=f"soc_terminal_{cid}")

            # Adds each member's variables and constraints to the overall dict for the community
            members[cid] = values
            private_cost += member.pv_marginal_cost * gp.quicksum(values["pv"][t] for t in T)
            private_cost += member.linear_disutility * gp.quicksum(
                values["delta_pos"][t] + values["delta_neg"][t] for t in T)

        # Defines community-level variables for import, export, and excess
        imp = m.addVars(T, lb=0, name="community_import")
        exp = m.addVars(T, lb=0, name="community_export")
        excess = m.addVars(T, lb=0, name="community_excess")
        self.var = {"members": members, "import": imp, "export": exp, "excess": excess}

        # calculating net import for each member in each hour
        def net(member, t):
            values = members[member.consumer_id]
            expression = values["load"][t] - values["pv"][t]
            if member.has_battery:
                expression += values["charge"][t] - values["discharge"][t]
            return expression

        # Setting-up overall community balance, capacity, and excess import constraints
        self.con["balance"] = m.addConstrs(
            (imp[t] - exp[t] == gp.quicksum(net(member, t) for member in d.members) for t in T),
            name="community_balance")
        self.con["capacity"] = m.addConstrs(
            (imp[t] - excess[t] <= float(contract.capacity_limit_kW[t]) for t in T),
            name="community_capacity")
        self.con["excess_import"] = m.addConstrs(
            (excess[t] <= imp[t] for t in T), name="community_excess_import")
        meter_cost = gp.quicksum( # calculates the total cost for the community
            first.energy_price[t] * (imp[t] - exp[t])
            + discounted_import * imp[t] + discounted_export * exp[t]
            + (alpha - discounted_import) * excess[t] for t in T)
        m.setObjective(meter_cost + private_cost, GRB.MINIMIZE)
        m.update()
        return self

    def solve(self):
        """Solve the selected formulation and return member or community results."""
        if self.contract_type != "C":
            return {member_id: consumer.solve() for member_id, consumer in self.consumers.items()}
        if self.m is None or not self.var:
            raise RuntimeError("Community C model must be built before solving.")
        
        # Optimizing full community model for the "C" formulation
        self.m.optimize()
        status = _status_name(self.m.Status)
        
        # Logging status of solver
        if self.m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Community formulation C ended with status {status}.")
        
        d, T = self.data, list(range(self.data.members[0].n_hours))
        contract, first = d.contract, d.members[0]
        
        # Creating hourly dataframe for community dispatch variables
        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["energy_price"] = first.energy_price
        hourly["community_import"] = [self.var["import"][t].X for t in T]
        hourly["community_export"] = [self.var["export"][t].X for t in T]
        hourly["community_excess"] = [self.var["excess"][t].X for t in T]
        hourly["capacity"] = contract.capacity_limit_kW
        hourly["dual_internal_price"] = [self.con["balance"][t].Pi for t in T]
        hourly["dual_capacity"] = [-self.con["capacity"][t].Pi for t in T]
        
        # Adding individual member variables to hourly df
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
        
        # Calculating total costs using hourly df values
        energy = first.energy_price * (hourly["community_import"] - hourly["community_export"])
        tariff = ((1 - contract.tariff_discount_ratio) * first.import_tariff * hourly["community_import"]
                  + (1 - contract.tariff_discount_ratio) * first.export_tariff * hourly["community_export"])
        penalty = (contract.penalty_DKK_per_kWh - (1 - contract.tariff_discount_ratio) * first.import_tariff) * hourly["community_excess"]
        
        return Results(
            question="C", status=status, objective=self.m.ObjVal, hourly=hourly,
            duals={"capacity_rent": float(np.dot(hourly["dual_capacity"], contract.capacity_limit_kW))},
            meta={"economics": {"energy_cost": float(energy.sum()), "tariff_cost": float(tariff.sum()),
                                "excess_penalty_cost": float(penalty.sum())}},
        )


class CommunityOptimizationModel:
    """Build and solve a community formulation from shared community data.

    R/I use independent member meters; C uses one joint community meter.
    """

    _SUPPORTED_FORMULATIONS = {"R", "I", "C"}

    def __init__(
        self,
        community: CommunityData,
        formulation: str,
        verbose: bool = False,
    ):
        self.community = community
        self.formulation = formulation.upper()
        self.verbose = verbose
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
        contract_type = self.formulation
        self.model = Community(
            data=self.community,
            contract_type=contract_type,
            verbose=self.verbose,
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