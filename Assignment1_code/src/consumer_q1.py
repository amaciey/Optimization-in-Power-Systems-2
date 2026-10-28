"""Question 1 - price-elastic consumer (LP). Solution text, Section 1, Eqs. (1.1)-(1.5).

    min  sum_t  C(p_im_t, p_ex_t, p_PV_t) - U(l_t)                                   (1.1)
         C = pi_im_t p_im_t - pi_ex_t p_ex_t + c_PV p_PV_t,   U = u_L l_t
    s.t. l_t + p_ex_t - p_im_t - p_PV_t = 0                  (lambda_t)                (1.2)
         p_PV_t - PV_bar_t <= 0,   -p_PV_t <= 0              (rho_up_t, rho_lo_t)      (1.3)
         l_t - L_max <= 0,         L_min - l_t <= 0          (omega_up_t, omega_lo_t)  (1.4)
         -p_im_t <= 0,             -p_ex_t <= 0              (alpha_t, beta_t)         (1.5)

The same class solves Case A (Q1_caseA) and Case B (Q1_caseB): only c_PV differs in the data.
"""
import gurobipy as gp
import pandas as pd
from gurobipy import GRB


class PriceElasticConsumer:

    def __init__(self, data):
        self.data = data
        self.T = range(1, data.n_hours + 1)                  # hours t = 1, ..., 24
        self.m = gp.Model("Q1_price_elastic_consumer")
        self.m.Params.OutputFlag = 0
        self.build()

    def build(self):
        d, m, T = self.data, self.m, self.T

        # Input parameters (hourly ones as dicts indexed by t = 1..24)
        pi_DA = {t: d.energy_price[t - 1] for t in T}          # electricity price, DKK/kWh
        pi_im = {t: pi_DA[t] + d.import_tariff for t in T}     # effective import price, DKK/kWh
        pi_ex = {t: pi_DA[t] - d.export_tariff for t in T}     # effective export price, DKK/kWh
        PV_bar = {t: d.pv_available[t - 1] for t in T}         # available PV production, kWh
        c_PV, u_L = d.pv_marginal_cost, d.consumption_utility  # DKK/kWh
        L_min, L_max = d.load_min_kWh, d.load_max_kWh          # kWh

        # Decision variables (kWh in hour t): all unbounded, every bound is an explicit constraint
        self.p_im = m.addVars(T, lb=-GRB.INFINITY, name="p_im")   # import from the grid
        self.p_ex = m.addVars(T, lb=-GRB.INFINITY, name="p_ex")   # export to the grid
        self.p_PV = m.addVars(T, lb=-GRB.INFINITY, name="p_PV")   # PV production used
        self.l = m.addVars(T, lb=-GRB.INFINITY, name="l")         # load consumption
        p_im, p_ex, p_PV, l = self.p_im, self.p_ex, self.p_PV, self.l

        # Objective (1.1): procurement cost C minus utility U
        self.C = gp.quicksum(pi_im[t] * p_im[t] - pi_ex[t] * p_ex[t] + c_PV * p_PV[t] for t in T)
        self.U = gp.quicksum(u_L * l[t] for t in T)
        m.setObjective(self.C - self.U, GRB.MINIMIZE)

        # Constraints, written exactly as h(x) = 0 or g(x) <= 0 in the solution text
        c = self.con = {}
        c["bal"] = m.addConstrs((l[t] + p_ex[t] - p_im[t] - p_PV[t] == 0 for t in T), name="bal")  # (1.2)
        c["PV_up"] = m.addConstrs((p_PV[t] - PV_bar[t] <= 0 for t in T), name="PV_up")            # (1.3)
        c["PV_lo"] = m.addConstrs((-p_PV[t] <= 0 for t in T), name="PV_lo")                       # (1.3)
        c["l_up"] = m.addConstrs((l[t] - L_max <= 0 for t in T), name="l_up")                     # (1.4)
        c["l_lo"] = m.addConstrs((L_min - l[t] <= 0 for t in T), name="l_lo")                     # (1.4)
        c["im_lo"] = m.addConstrs((-p_im[t] <= 0 for t in T), name="im_lo")                       # (1.5)
        c["ex_lo"] = m.addConstrs((-p_ex[t] <= 0 for t in T), name="ex_lo")                       # (1.5)

    def solve(self):
        """Solve; return (summary dict, hourly DataFrame of primal values and KKT multipliers).

        Gurobi's Pi is d(objective)/d(right-hand side). With every constraint written as h(x) = 0 or
        g(x) <= 0, as in the solution text, the KKT multiplier of the text is always -Pi.
        """
        d, m, T, c = self.data, self.m, self.T, self.con
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"{m.ModelName}: Gurobi status {m.Status}")
        X = lambda var: [var[t].X for t in T]          # primal values
        Y = lambda con: [-con[t].Pi for t in T]        # KKT multipliers

        hourly = pd.DataFrame({
            "pi_DA": d.energy_price, "PV_bar": d.pv_available,
            "l": X(self.l), "p_PV": X(self.p_PV), "p_im": X(self.p_im), "p_ex": X(self.p_ex),
            "lambda": Y(c["bal"]), "rho_up": Y(c["PV_up"]), "rho_lo": Y(c["PV_lo"]),
            "omega_up": Y(c["l_up"]), "omega_lo": Y(c["l_lo"]), "alpha": Y(c["im_lo"]), "beta": Y(c["ex_lo"]),
        }, index=pd.Index(T, name="t"))

        summary = {
            "objective (1.1) = C - U = -surplus [DKK]": m.ObjVal,
            "procurement cost C [DKK]": self.C.getValue(),
            "utility U [DKK]": self.U.getValue(),
        }
        return summary, hourly
