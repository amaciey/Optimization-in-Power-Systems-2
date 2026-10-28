"""Question 2.(b) - linear disutility (LP). Solution text, Section 2.1, Eqs. (2.1)-(2.3).

    min  sum_t  C(p_im_t, p_ex_t, p_PV_t) + D_L(d_pos_t, d_neg_t)                    (2.1)
         D_L = c_L (d_pos_t + d_neg_t)
    s.t. core constraints (1.2)-(1.5)
         l_t - l_ref_t - d_pos_t + d_neg_t = 0                  (theta_t)               (2.2)
         -d_pos_t <= 0,   -d_neg_t <= 0                         (psi_pos_t, psi_neg_t)  (2.3)
"""
import gurobipy as gp
import pandas as pd
from gurobipy import GRB


class LinearDisutilityConsumer:

    def __init__(self, data):
        self.data = data
        self.T = range(1, data.n_hours + 1)                  # hours t = 1, ..., 24
        self.m = gp.Model("Q2b_linear_disutility_consumer")
        self.m.Params.OutputFlag = 0
        self.build()

    def build(self):
        d, m, T = self.data, self.m, self.T

        # Input parameters (hourly ones as dicts indexed by t = 1..24)
        pi_DA = {t: d.energy_price[t - 1] for t in T}
        pi_im = {t: pi_DA[t] + d.import_tariff for t in T}
        pi_ex = {t: pi_DA[t] - d.export_tariff for t in T}
        PV_bar = {t: d.pv_available[t - 1] for t in T}
        l_ref = {t: d.reference_load[t - 1] for t in T}        # reference consumption, kWh
        c_PV, c_L = d.pv_marginal_cost, d.linear_disutility    # DKK/kWh
        L_min, L_max = d.load_min_kWh, d.load_max_kWh

        # Decision variables (kWh in hour t): all unbounded, every bound is an explicit constraint
        self.p_im = m.addVars(T, lb=-GRB.INFINITY, name="p_im")
        self.p_ex = m.addVars(T, lb=-GRB.INFINITY, name="p_ex")
        self.p_PV = m.addVars(T, lb=-GRB.INFINITY, name="p_PV")
        self.l = m.addVars(T, lb=-GRB.INFINITY, name="l")
        self.d_pos = m.addVars(T, lb=-GRB.INFINITY, name="d_pos")   # consumption above the reference
        self.d_neg = m.addVars(T, lb=-GRB.INFINITY, name="d_neg")   # consumption below the reference
        p_im, p_ex, p_PV, l, d_pos, d_neg = self.p_im, self.p_ex, self.p_PV, self.l, self.d_pos, self.d_neg

        # Objective (2.1): procurement cost C plus linear disutility D_L
        self.C = gp.quicksum(pi_im[t] * p_im[t] - pi_ex[t] * p_ex[t] + c_PV * p_PV[t] for t in T)
        self.D = gp.quicksum(c_L * (d_pos[t] + d_neg[t]) for t in T)
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
        c["split"] = m.addConstrs(
            (l[t] - l_ref[t] - d_pos[t] + d_neg[t] == 0 for t in T), name="split")                # (2.2)
        c["dpos_lo"] = m.addConstrs((-d_pos[t] <= 0 for t in T), name="dpos_lo")                  # (2.3)
        c["dneg_lo"] = m.addConstrs((-d_neg[t] <= 0 for t in T), name="dneg_lo")                  # (2.3)

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
            "l": X(self.l), "d_pos": X(self.d_pos), "d_neg": X(self.d_neg),
            "p_PV": X(self.p_PV), "p_im": X(self.p_im), "p_ex": X(self.p_ex),
            "lambda": Y(c["bal"]), "rho_up": Y(c["PV_up"]), "rho_lo": Y(c["PV_lo"]),
            "omega_up": Y(c["l_up"]), "omega_lo": Y(c["l_lo"]), "alpha": Y(c["im_lo"]), "beta": Y(c["ex_lo"]),
            "theta": Y(c["split"]), "psi_pos": Y(c["dpos_lo"]), "psi_neg": Y(c["dneg_lo"]),
        }, index=pd.Index(T, name="t"))

        summary = {
            "objective (2.1) = C + D_L [DKK]": m.ObjVal,
            "procurement cost C [DKK]": self.C.getValue(),
            "disutility D_L [DKK]": self.D.getValue(),
        }
        return summary, hourly
