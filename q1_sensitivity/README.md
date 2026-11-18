# Question 1 contract sensitivity

Place this `q1_sensitivity/` directory at the root of the `austin_q1` branch, beside `src/` and `data/`. With the repository's Python dependencies and a working Gurobi installation, run this command from the repository root.

```sh
python -m q1_sensitivity
```

The command uses the existing repository data loader and base-case data. Fresh results are saved to `q1_sensitivity/results/sensitivity/`. The `reference_results/` folder contains a verified run on the Austin branch snapshot with Gurobi 13.0.3.

| File | Contents |
|---|---|
| `one_dimensional.csv` | Cost, peak imports and feeder exceedance for R, I and C in each one-parameter sweep |
| `discount_cap_grid.csv` | Combined discount and cap sweep for I and C |
| `members.csv` | Member costs and benefits against R and I |
| `break_even.csv` | Break-even intervals against R |
| `contract_sensitivity.png` | Cost and feeder exceedance plots |
| `validation.json` | Number of solved configurations and maximum validation residuals |

## Sensitivity analysis for the report

We evaluate the tariff discount, the multiplier on the hourly import cap and the exceedance penalty on day `d1`. The base contract uses a 75% discount, a 9.39 kW hourly cap and a 10 DKK/kWh penalty. Each one-parameter sweep holds the other two settings at their base values. We compare total daily cost with the no-contract reference R and measure feeder exceedance as the sum of hourly gross imports above the common reference cap. Model I allocates the cap using normalized member shares. The base daily costs of R, I and C are 398.24, 480.13 and 372.16 DKK, respectively.

| Parameter varied | Result |
|---|---|
| Tariff discount | C breaks even against R at a discount between 48.37% and 48.38%. I remains more expensive than R over the tested discount range of 0% to 95%. |
| Import cap | C breaks even at approximately 0.876 times the base cap. I needs approximately 1.421 times the base cap. Pooling the import allowance lets C deliver savings at a tighter cap. |
| Exceedance penalty | Raising the penalty from 1 to 5 DKK/kWh reduces C's feeder exceedance from 23.58 to 0.00 kWh/day. At the base penalty of 10 DKK/kWh, I still exceeds the feeder cap by 4.29 kWh/day. Raising the penalty to 50 DKK/kWh leaves this exceedance at 4.29 kWh/day in the selected solution and increases the daily cost from 480.13 to 651.66 DKK. |

The numerical break-even intervals use the `below_cap` billing convention, which applies the discounted import tariff to energy within the cap and the penalty to the excess. Hourly profiles and peak metrics describe the selected optimal schedules. Use the tables and figure from the same run when reporting these metrics.
