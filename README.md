# Q1: start here

This folder contains the code and saved results behind our Assignment 2 Question 1 report. Add it as `assignment2_q1/` at the repository root; keep the team's existing root-level code.

## Read these three files first

1. [REPORT.md](REPORT.md): results and economic interpretation.
2. [docs/formulation.md](docs/formulation.md): mathematical model and assumptions.
3. [src/q1_model.py](src/q1_model.py): implementation of the R, I and C models.

Chinese formulation notes are in [docs/formulation_zh.md](docs/formulation_zh.md).

## Run

From the repository root:

```sh
cd assignment2_q1
python -m pip install -r requirements.txt
python run_q1.py all
```

A suitable Gurobi license is required. `all` runs the base case, validation, sensitivity analysis and report generation. To run a single stage, replace `all` with `base`, `validate`, `sensitivity` or `report`. Report generation uses saved results and does not re-solve the models. Optional additional verification: `python audit_q1_course.py`.

## What the models mean

| Model | Operation |
| --- | --- |
| R | Independent members, no capacity contract |
| I | Independent members, individual capacity contracts |
| C | Centralized community, one meter and shared capacity contract |

Default case: 16 members, day d1, 24 one-hour periods. Battery final SOC equals initial SOC. Individual capacity shares are normalized. Under default `below_cap` billing, all imports pay the energy price; only non-excess imports pay the discounted import network tariff, while excess imports pay the penalty instead. `all_metered` is a separate comparison.

Saved total costs: R = 398.241694, I = 480.133652, C = 372.162434 DKK/day. Positive manager surplus is not strict budget balance; Q1 does not redistribute that surplus.

## Where to find the evidence

| Path | Content |
| --- | --- |
| `results/base/comparison.csv` | R/I/C cost and flexibility comparison |
| `results/base/settlement.csv` | Per-member costs and benefits |
| `results/sensitivity/` | Contract-parameter experiments |
| `results/validation.json` | Saved validation checks |
| `results/course_audit/` | Additional saved mathematical checks |
| `results/report/` | Figures and data used by REPORT.md |
| `data/` | Original supplied data; retained in full to preserve input hashes |

## Connecting this to Q2

Q2's current `src/model.py` is a separate implementation. Align its I billing and share normalization and reproduce the Q1 benchmarks before comparing DP results. Q1's continuous grid interface should not be copied unchanged into a model with budget-balance constraints: Q2 needs exact metered import/export and exceedance handling.

## Provenance

Model, loader, experiment scripts and historical numerical evidence are preserved from the earlier Q1 package. This team edition removes oral-exam and starter materials, simplifies navigation, and directs generated report figures to `results/report/`. The report has been regenerated from saved evidence; the full optimization experiments have not been rerun during packaging. Recorded dependency versions are in `requirements-tested.txt`.

AI assisted code and documentation preparation. Review the assumptions and results as a group and disclose actual AI use and verification in the assignment.
