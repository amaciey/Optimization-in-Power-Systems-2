# 46750 Assignment 1 (2026) - minimal solution code

Base-case models of the flexible consumer of Assignment 1, one class per consumer type, as a clean
starting point for Assignment 2. The code follows the solution text (`Assignment_1_2026_solution.pdf`)
line by line: same notation, same constraints written in the same form, same equation numbers.

## Run

```bash
pip install -r requirements.txt     # gurobipy, numpy, pandas
python main.py                      # solves and prints all six base cases
python main.py Q3 Q3_battery        # only the cases you name
```

The pip version of `gurobipy` ships with a restricted licence that is large enough for these models.

## Files

| File | Content |
|---|---|
| `main.py` | Loads each base case from `data/`, solves it with its model class and prints the primal solution and the KKT multipliers |
| `src/data_loader.py` | `load_question("<case>")` -> `InputData` (unchanged from the Assignment 1 starter code) |
| `src/consumer_q1.py` | `PriceElasticConsumer` - Question 1, LP, Section 1, Eqs. (1.1)-(1.5); used for `Q1_caseA` and `Q1_caseB` |
| `src/consumer_q2_linear.py` | `LinearDisutilityConsumer` - Question 2.(b), LP, Section 2.1, Eqs. (2.1)-(2.3) |
| `src/consumer_q2_quadratic.py` | `QuadraticDisutilityConsumer` - Question 2.(c), QP, Section 2.2, Eq. (2.6) |
| `src/consumer_q3.py` | `MinEnergyConsumer` - Question 3.(a), QP, Section 3.1, Eqs. (3.1)-(3.2) |
| `src/consumer_q3_battery.py` | `BatteryConsumer` - Question 3.(g), QP, Section 3.2, Eqs. (3.5)-(3.11) |
| `data/` | The input data of Assignment 1, as provided (see the starter-code README for every field) |

Each class follows the same three steps: `__init__` stores the data and calls `build()` (parameters,
variables, objective, constraints); `solve()` optimises and returns a `summary` dict (objective, cost
terms, scalar multipliers) and an `hourly` DataFrame indexed by `t = 1, ..., 24` (input data, primal
values, hourly KKT multipliers).

## Notation (as in the solution text)

| Text | Code | Text | Code |
|---|---|---|---|
| p^im_t, p^ex_t | `p_im`, `p_ex` | π^DA_t, π^im_t, π^ex_t | `pi_DA`, `pi_im`, `pi_ex` |
| p^PV_t, ℓ_t | `p_PV`, `l` | PV̄_t, c^PV | `PV_bar`, `c_PV` |
| d^+_t, d^-_t | `d_pos`, `d_neg` | L^min, L^max, u^L | `L_min`, `L_max`, `u_L` |
| p^ch_t, p^dis_t, s_t | `p_ch`, `p_dis`, `s` | ℓ^ref_t, c^L, c^Q, E^min | `l_ref`, `c_L`, `c_Q`, `E_min` |
| C, U, D^L, D^Q | `C`, `U`, `D` | S̄, P̄^ch, P̄^dis, η^ch, η^dis, S^init | `S_bar`, `P_ch_bar`, `P_dis_bar`, `eta_ch`, `eta_dis`, `S_init` |

KKT multipliers: λ `lambda`, ρ̄/ρ̲ `rho_up`/`rho_lo`, ω̄/ω̲ `omega_up`/`omega_lo`, α `alpha`, β `beta`,
θ `theta`, ψ^+/ψ^- `psi_pos`/`psi_neg`, μ `mu`, σ `sigma`, ν̄/ν̲ `nu_up`/`nu_lo`,
ῡ/υ̲ `upsilon_up`/`upsilon_lo`, κ̄/κ̲ `kappa_up`/`kappa_lo`, φ `phi`.

## Conventions

* Objective = net cost in DKK (minimised). In Question 1 it equals minus the surplus.
* Every decision variable is declared unbounded (`lb=-GRB.INFINITY`); every bound, including all
  non-negativity constraints, is an explicit constraint, so that each one has its own multiplier.
* Every constraint is written exactly as in the solution text, as `h(x) == 0` or `g(x) <= 0`, and its
  equation number is given in a comment. Gurobi's `Pi` is d(objective)/d(right-hand side), so with this
  form the KKT multiplier of the text is always `-Pi` (non-negative for every inequality).
* Quadratic disutility: `c_Q * (l_t - l_ref_t)^2` with `c_Q` = `quadratic_disutility_DKK_per_kWh2`.
* Battery: state of charge in hour 1 from `S_init` (3.7) and in later hours (3.8); end of horizon
  `S_init - s_24 <= 0` (3.11).

## Expected output (base cases)

Use these values to check that your own models give the same optimum.

| Case | Objective [DKK] | Load [kWh/day] | Import [kWh] | Export [kWh] | mu [DKK/kWh] |
|---|---:|---:|---:|---:|---:|
| `Q1_caseA` | -2.609 | 35.94 | 12.00 | 3.00 | - |
| `Q1_caseB` | -1.950 | 12.00 | 12.00 | 3.00 | - |
| `Q2_linear` | 44.314 | 23.10 | 0.00 | 3.00 | - |
| `Q2_quadratic` | 39.971 | 22.77 | 4.42 | 3.00 | - |
| `Q3` | 63.311 | 40.00 | 15.47 | 2.41 | 2.369 |
| `Q3_battery` | 60.678 | 40.00 | 17.58 | 3.80 | 2.324 |

Optimal objective values are unique. Hourly values can legitimately differ from those printed here
where the optimum is not unique: some multipliers are not uniquely determined (e.g. `lambda_t` in
Question 1 hours with no import, no export and the load at a bound; `rho_up_t` and `rho_lo_t` in hours
with PV_bar_t = 0, where the two PV bounds (1.3) coincide), and the battery schedule may have alternative
optima. As noted in the solution text, the code does not add l_t >= 0, which is implied by (1.4) since
L_min >= 0.
