"""Solve the base case of every consumer model of Assignment 1 and print the solutions.

    python main.py              # all models
    python main.py Q3_battery   # only the cases named on the command line

Each case is loaded from data/ and solved by its own model class (one class per consumer type).
Notation and equation numbers follow the solution text (Assignment_1_2026_solution.pdf).
"""
import sys

import pandas as pd

from src.data_loader import load_question
from src.consumer_q1 import PriceElasticConsumer
from src.consumer_q2_linear import LinearDisutilityConsumer
from src.consumer_q2_quadratic import QuadraticDisutilityConsumer
from src.consumer_q3 import MinEnergyConsumer
from src.consumer_q3_battery import BatteryConsumer

# data case -> model class
CASES = {
    "Q1_caseA": PriceElasticConsumer,             # Question 1 (Section 1), c_PV < u_L
    "Q1_caseB": PriceElasticConsumer,             # Question 1 (Section 1), c_PV > u_L
    "Q2_linear": LinearDisutilityConsumer,        # Question 2.(b) (Section 2.1)
    "Q2_quadratic": QuadraticDisutilityConsumer,  # Question 2.(c) (Section 2.2)
    "Q3": MinEnergyConsumer,                      # Question 3.(a) (Section 3.1)
    "Q3_battery": BatteryConsumer,                # Question 3.(g) (Section 3.2)
}


def print_results(case: str, summary: dict, hourly: pd.DataFrame) -> None:
    """Print the scalar results, the daily energy totals, and the hourly primal and dual values."""
    print("=" * 100)
    print(f"Case {case}")
    print("=" * 100)
    for name, value in summary.items():
        print(f"  {name:<55s} {value:10.3f}")
    energy_cols = [c for c in ("l", "p_PV", "p_im", "p_ex", "p_ch", "p_dis") if c in hourly]
    print("  daily energy [kWh]: " + ", ".join(f"{c} {hourly[c].sum():.2f}" for c in energy_cols))

    table = hourly.round(3) + 0.0                       # + 0.0 removes "-0.000"
    first_dual = list(table.columns).index("lambda")
    fmt = lambda x: f"{x:7.3f}"
    print("\n  Input data and primal solution [DKK/kWh, kWh]")
    print(table.iloc[:, :first_dual].to_string(float_format=fmt))
    print("\n  KKT multipliers [DKK/kWh]")
    print(table.iloc[:, first_dual:].to_string(float_format=fmt))
    print()


def main() -> None:
    cases = sys.argv[1:] or list(CASES)
    for case in cases:
        data = load_question(case)              # base-case input data of this case
        model = CASES[case](data)               # builds the model
        summary, hourly = model.solve()
        print_results(case, summary, hourly)


if __name__ == "__main__":
    main()
