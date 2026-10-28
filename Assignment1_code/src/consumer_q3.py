"""Question 3.(a) - minimum daily energy requirement (QP). Solution text, Section 3.1, Eqs. (3.1)-(3.2).

    min  sum_t  C(p_im_t, p_ex_t, p_PV_t) + D_Q(l_t)                                 (3.1)
    s.t. core constraints (1.2)-(1.5)
         E_min - sum_t l_t <= 0                                  (mu)                    (3.2)
"""
import gurobipy as gp
import pandas as pd
from gurobipy import GRB


class MinEnergyConsumer:

    def __init__(self, data):
        self.data = data
        self.T = range(1, data.n_hours + 1)                  # hours t = 1, ..., 24
        self.m = gp.Model("Q3a_min_energy_consumer")
        self.m.Params.OutputFlag = 0
        self.build()

    def build(self):
        d, m, T = self.data, self.m, self.T

        # Input parameters (hourly ones as dicts indexed by t = 1..24)
        pi_DA = {t: d.energy_price[t - 1] for t in T}
        pi_im = {t: pi_DA[t] + d.import_tariff for t in T}
        pi_ex = {t: pi_DA[t] - d.export_tariff for t in T}
        PV_bar = {t: d.pv_available[t - 1] for t in T}
        l_ref = {t: d.reference_load[t - 1] for t in T}
        c_PV, c_Q = d.pv_marginal_cost, d.quadratic_disutility
        L_min, L_max = d.load_min_kWh, d.load_max_kWh
        E_min = d.min_daily_energy_kWh                           # minimum daily consumption, kWh

        # Decision variables (kWh in hour t): all unbounded, every bound is an explicit constraint
        self.p_im = m.addVars(T, lb=-GRB.INFINITY, name="p_im")
        self.p_ex = m.addVars(T, lb=-GRB.INFINITY, name="p_ex")
        self.p_PV = m.addVars(T, lb=-GRB.INFINITY, name="p_PV")
        self.l = m.addVars(T, lb=-GRB.INFINITY, name="l")
        p_im, p_ex, p_PV, l = self.p_im, self.p_ex, self.p_PV, self.l

        # Objective (3.1): procurement cost C plus quadratic disutility D_Q
        self.C = gp.quicksum(pi_im[t] * p_im[t] - pi_ex[t] * p_ex[t] + c_PV * p_PV[t] for t in T)
        self.D = gp.quicksum(c_Q * (l[t] - l_ref[t]) * (l[t] - l_ref[t]) for t in T)
        m.setObjective(self.C + self.D, GRB.MINIMIZE)

        # Constraints, written exactly as h(x) = 0 or g(x) <= 0 in the solution text
        c = self.con = {}
        c["bal"] = m.addConstrs((l[t] + p_ex[t] - p_im[t] - p_PV[t] == 0 for t in T), name="bal")  # (1.2)
        c["PV_up"] = m.addConstrs((p_PV[t] - PV_bar[t] <= 0 for t in T), name="PV_up")            # (1.3)
        c["PV_lo"] = m.addConstrs((-p_PV[t] <= 0 for t in T), name="PV_lo")                       # (1.3)
        c["l_up"] = m.addConstrs((l[t] - L_max <= 0 for t in T), name="l_up")                     # (1.4)
        c["l_lo"] = m.addConstrs((L_min - l[t] <= 0 for t in T), name="l_lo")                     # (1.4)
        c["im_lo"] = m.addConstrs((-p_im[t] <= 0 for t in T), name="im_lo")                       # (1.5)
        c["ex_lo"] = m.addConstrs((-p_ex[t] <= 0 for t in T), name="ex_lo")                       # (1.5)
        c["E_min"] = m.addConstr(E_min - gp.quicksum(l[t] for t in T) <= 0, name="E_min")          # (3.2)

    def solve(self):
        """Solve; return (summary dict, hourly DataFrame). KKT multiplier = -Pi (see consumer_q1.py)."""
        d, m, T, c = self.data, self.m, self.T, self.con
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"{m.ModelName}: Gurobi status {m.Status}")
        X = lambda var: [var[t].X for t in T]
        Y = lambda con: [-con[t].Pi for t in T]

        hourly = pd.DataFrame({
            "pi_DA": d.energy_price, "PV_bar": d.pv_available, "l_ref": d.reference_load,
            "l": X(self.l), "p_PV": X(self.p_PV), "p_im": X(self.p_im), "p_ex": X(self.p_ex),
            "lambda": Y(c["bal"]), "rho_up": Y(c["PV_up"]), "rho_lo": Y(c["PV_lo"]),
            "omega_up": Y(c["l_up"]), "omega_lo": Y(c["l_lo"]), "alpha": Y(c["im_lo"]), "beta": Y(c["ex_lo"]),
        }, index=pd.Index(T, name="t"))

        summary = {
            "objective (3.1) = C + D_Q [DKK]": m.ObjVal,
            "procurement cost C [DKK]": self.C.getValue(),
            "disutility D_Q [DKK]": self.D.getValue(),
            "mu, multiplier of (3.2) [DKK/kWh]": -c["E_min"].Pi,
        }
        return summary, hourly
