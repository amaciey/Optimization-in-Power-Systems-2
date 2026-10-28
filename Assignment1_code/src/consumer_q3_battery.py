"""Question 3.(g) - minimum daily energy requirement with a battery (QP). Solution text, Section 3.2, Eqs. (3.5)-(3.11).

    min  sum_t  C(p_im_t, p_ex_t, p_PV_t) + D_Q(l_t)                                             (3.5)
    s.t. core constraints (1.3)-(1.5) and E_min requirement (3.2)
         l_t + p_ex_t + p_ch_t - p_im_t - p_PV_t - p_dis_t = 0       (lambda_t)                    (3.6)
         s_1 - S_init - eta_ch p_ch_1 + p_dis_1 / eta_dis = 0         (sigma_1)                     (3.7)
         s_t - s_{t-1} - eta_ch p_ch_t + p_dis_t / eta_dis = 0, t > 1 (sigma_t)                     (3.8)
         p_ch_t - P_ch_bar <= 0, p_dis_t - P_dis_bar <= 0, s_t - S_bar <= 0
                                                                      (nu_up, upsilon_up, kappa_up) (3.9)
         -p_ch_t <= 0, -p_dis_t <= 0, -s_t <= 0                       (nu_lo, upsilon_lo, kappa_lo) (3.10)
         S_init - s_24 <= 0                                           (phi)                         (3.11)

No binary variable prevents simultaneous charging and discharging: with eta_ch * eta_dis < 1 and positive
effective export prices it is never optimal (solution text, Section 3.2). solve() reports the overlap as a check.
"""
import gurobipy as gp
import pandas as pd
from gurobipy import GRB


class BatteryConsumer:

    def __init__(self, data):
        self.data = data
        self.T = range(1, data.n_hours + 1)                  # hours t = 1, ..., 24
        self.m = gp.Model("Q3g_battery_consumer")
        self.m.Params.OutputFlag = 0
        self.build()

    def build(self):
        d, m, T = self.data, self.m, self.T
        t_end = T[-1]                                        # last hour, 24

        # Input parameters (hourly ones as dicts indexed by t = 1..24)
        pi_DA = {t: d.energy_price[t - 1] for t in T}
        pi_im = {t: pi_DA[t] + d.import_tariff for t in T}
        pi_ex = {t: pi_DA[t] - d.export_tariff for t in T}
        PV_bar = {t: d.pv_available[t - 1] for t in T}
        l_ref = {t: d.reference_load[t - 1] for t in T}
        c_PV, c_Q = d.pv_marginal_cost, d.quadratic_disutility
        L_min, L_max = d.load_min_kWh, d.load_max_kWh
        E_min = d.min_daily_energy_kWh
        S_bar, S_init = d.battery_capacity_kWh, d.battery_initial_soc_kWh                  # kWh
        P_ch_bar, P_dis_bar = d.battery_max_charge_kW, d.battery_max_discharge_kW          # kW = kWh per hour
        eta_ch, eta_dis = d.battery_charging_efficiency, d.battery_discharging_efficiency  # -

        # Decision variables (kWh in hour t): all unbounded, every bound is an explicit constraint
        self.p_im = m.addVars(T, lb=-GRB.INFINITY, name="p_im")
        self.p_ex = m.addVars(T, lb=-GRB.INFINITY, name="p_ex")
        self.p_PV = m.addVars(T, lb=-GRB.INFINITY, name="p_PV")
        self.l = m.addVars(T, lb=-GRB.INFINITY, name="l")
        self.p_ch = m.addVars(T, lb=-GRB.INFINITY, name="p_ch")     # energy drawn to charge the battery
        self.p_dis = m.addVars(T, lb=-GRB.INFINITY, name="p_dis")   # energy delivered by discharging
        self.s = m.addVars(T, lb=-GRB.INFINITY, name="s")           # state of charge at the end of hour t
        p_im, p_ex, p_PV, l = self.p_im, self.p_ex, self.p_PV, self.l
        p_ch, p_dis, s = self.p_ch, self.p_dis, self.s

        # Objective (3.5): procurement cost C plus quadratic disutility D_Q (the battery has no cost)
        self.C = gp.quicksum(pi_im[t] * p_im[t] - pi_ex[t] * p_ex[t] + c_PV * p_PV[t] for t in T)
        self.D = gp.quicksum(c_Q * (l[t] - l_ref[t]) * (l[t] - l_ref[t]) for t in T)
        m.setObjective(self.C + self.D, GRB.MINIMIZE)

        # Constraints, written exactly as h(x) = 0 or g(x) <= 0 in the solution text
        c = self.con = {}
        c["PV_up"] = m.addConstrs((p_PV[t] - PV_bar[t] <= 0 for t in T), name="PV_up")            # (1.3)
        c["PV_lo"] = m.addConstrs((-p_PV[t] <= 0 for t in T), name="PV_lo")                       # (1.3)
        c["l_up"] = m.addConstrs((l[t] - L_max <= 0 for t in T), name="l_up")                     # (1.4)
        c["l_lo"] = m.addConstrs((L_min - l[t] <= 0 for t in T), name="l_lo")                     # (1.4)
        c["im_lo"] = m.addConstrs((-p_im[t] <= 0 for t in T), name="im_lo")                       # (1.5)
        c["ex_lo"] = m.addConstrs((-p_ex[t] <= 0 for t in T), name="ex_lo")                       # (1.5)
        c["E_min"] = m.addConstr(E_min - gp.quicksum(l[t] for t in T) <= 0, name="E_min")          # (3.2)
        c["bal"] = m.addConstrs(
            (l[t] + p_ex[t] + p_ch[t] - p_im[t] - p_PV[t] - p_dis[t] == 0 for t in T), name="bal")  # (3.6)
        c["soc_1"] = m.addConstr(
            s[1] - S_init - eta_ch * p_ch[1] + p_dis[1] / eta_dis == 0, name="soc_1")              # (3.7)
        c["soc"] = m.addConstrs(
            (s[t] - s[t - 1] - eta_ch * p_ch[t] + p_dis[t] / eta_dis == 0 for t in T if t > 1),
            name="soc")                                                                            # (3.8)
        c["ch_up"] = m.addConstrs((p_ch[t] - P_ch_bar <= 0 for t in T), name="ch_up")             # (3.9)
        c["dis_up"] = m.addConstrs((p_dis[t] - P_dis_bar <= 0 for t in T), name="dis_up")         # (3.9)
        c["s_up"] = m.addConstrs((s[t] - S_bar <= 0 for t in T), name="s_up")                     # (3.9)
        c["ch_lo"] = m.addConstrs((-p_ch[t] <= 0 for t in T), name="ch_lo")                       # (3.10)
        c["dis_lo"] = m.addConstrs((-p_dis[t] <= 0 for t in T), name="dis_lo")                    # (3.10)
        c["s_lo"] = m.addConstrs((-s[t] <= 0 for t in T), name="s_lo")                            # (3.10)
        c["s_end"] = m.addConstr(S_init - s[t_end] <= 0, name="s_end")                            # (3.11)

    def solve(self):
        """Solve; return (summary dict, hourly DataFrame). KKT multiplier = -Pi (see consumer_q1.py)."""
        d, m, T, c = self.data, self.m, self.T, self.con
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"{m.ModelName}: Gurobi status {m.Status}")
        X = lambda var: [var[t].X for t in T]
        Y = lambda con: [-con[t].Pi for t in T]
        sigma = [-(c["soc_1"] if t == 1 else c["soc"][t]).Pi for t in T]   # (3.7) for t = 1, (3.8) after

        hourly = pd.DataFrame({
            "pi_DA": d.energy_price, "PV_bar": d.pv_available, "l_ref": d.reference_load,
            "l": X(self.l), "p_PV": X(self.p_PV), "p_im": X(self.p_im), "p_ex": X(self.p_ex),
            "p_ch": X(self.p_ch), "p_dis": X(self.p_dis), "s": X(self.s),
            "lambda": Y(c["bal"]), "rho_up": Y(c["PV_up"]), "rho_lo": Y(c["PV_lo"]),
            "omega_up": Y(c["l_up"]), "omega_lo": Y(c["l_lo"]), "alpha": Y(c["im_lo"]), "beta": Y(c["ex_lo"]),
            "sigma": sigma,
            "nu_up": Y(c["ch_up"]), "nu_lo": Y(c["ch_lo"]),
            "upsilon_up": Y(c["dis_up"]), "upsilon_lo": Y(c["dis_lo"]),
            "kappa_up": Y(c["s_up"]), "kappa_lo": Y(c["s_lo"]),
        }, index=pd.Index(T, name="t"))

        summary = {
            "objective (3.5) = C + D_Q [DKK]": m.ObjVal,
            "procurement cost C [DKK]": self.C.getValue(),
            "disutility D_Q [DKK]": self.D.getValue(),
            "mu, multiplier of (3.2) [DKK/kWh]": -c["E_min"].Pi,
            "phi, multiplier of (3.11) [DKK/kWh]": -c["s_end"].Pi,
            "max_t min(p_ch_t, p_dis_t) [kWh] (overlap check)":
                max(min(self.p_ch[t].X, self.p_dis[t].X) for t in T),
        }
        return summary, hourly
