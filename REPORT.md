# Q1 evidence report

Generated from saved results. Edit the model/configuration and rerun experiments to change the evidence; do not hand-edit numerical results.

## Main result

C lowers total cost from 398.24 to 372.16 DKK/day (6.55%), but 13/16 members are worse off than R.
Against I, 8/16 members are worse off. The manager surplus is 163.77 DKK/day.
These are base-case findings under the selected optimal dual solution, not universal claims.

| model | total_cost_DKK | peak_import_kW | contract_excess_kWh | manager_surplus_DKK |
| --- | --- | --- | --- | --- |
| R | 398.24 | 27.60 | 0.00 | 0.00 |
| I | 480.13 | 11.08 | 4.29 | 0.00 |
| C | 372.16 | 9.39 | 0.00 | 163.77 |

R has no contract penalty: its zero *contract* exceedance must not be read as zero feeder overload. Gross imports for R/I sum individual imports; C uses the common meter. This is the assignment metric, not a full network-security assessment.

## Mechanism and settlement

Pooling permits internal exchange and shared use of import capacity. Total cost includes external energy, grid fees, penalties, PV costs and discomfort. Internal payments are transfers, not extra community costs.
Each member pays sum(lambda[t] * net[i,t]) and also bears PV and discomfort costs. Benefit is the independently optimized R/I cost minus the C settlement cost. Budget balance requires manager surplus = 0, not merely a nonnegative surplus.

The manager surplus equals sum(mu[t] * cap[t]) under this formulation. Member costs sum to community cost plus manager surplus. No redistribution is implemented.

## Model assumptions

- Perfect information, direct control, 24 one-hour intervals, base member mix.
- Cyclic battery SOC; continuous devices; no internal power-flow model or battery degradation cost.
- Soft import cap. In below_cap billing only non-excess imports pay the discounted import tariff; all imported energy still pays the energy price.
- The alternative all_metered interpretation is computed separately; course clarification is still needed.
- No explicit charge/discharge or meter import/export exclusivity binaries. Solved cases are checked for simultaneous flows; this is not a guarantee for arbitrary new data.

See [formulation](docs/formulation.md) for variables, constraints, signs and pricing.

## Validation

Base and auxiliary checks passed: 74. Distinct sensitivity contracts checked: 124.
Checks cover balances, bounds, SOC, objective reconstruction, LP stationarity/complementarity/strong duality, independent member price responses, cap-value finite differences and input hashes.
Alternative optimal bases can change schedules and prices. Matching one hourly schedule is not required for optimality.
The minimum possible individual-contract exceedance is 4.288092 kWh/day; the community minimum is 0.000000 kWh/day.
For an additional independent formulation and explicit-dual check run `python audit_q1_course.py`; its fresh results are in results/course_audit/.

## Sensitivity evidence

Thresholds refer to below_cap billing, with other inputs held at their baseline values. A bracket is an interval, not an exact point. No crossing in a tested interval does not establish a global impossibility.

| dimension | model | status | lower | upper |
| --- | --- | --- | --- | --- |
| beta | I | not_beneficial_in_interval | 0.000000 | 0.950000 |
| beta | C | bracketed | 0.483698 | 0.483813 |
| cap_scale | I | bracketed | 1.421021 | 1.421387 |
| cap_scale | C | bracketed | 0.875732 | 0.876099 |

## Alternative billing

| billing | model | total_cost_DKK | penalty_DKK |
| --- | --- | --- | --- |
| below_cap | I | 480.13 | 42.88 |
| below_cap | C | 372.16 | 0.00 |
| all_metered | I | 480.67 | 42.88 |
| all_metered | C | 372.16 | 0.00 |

## Figures and exported data

- [Member benefits](results/report/figures/01_member_benefits.png)
- [Import profiles](results/report/figures/02_import_profiles.png)
- [Discount sensitivity](results/report/figures/03_discount_sensitivity.png)
- [Chart data](results/report/chart_data.csv)
- [Summary facts](results/report/facts.json)

