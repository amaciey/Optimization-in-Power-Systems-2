"""Run a community optimization formulation and summarize its results.

    python main.py                                   # base case (16 members), day d1
    python main.py --community my_mix                # your own data/params_community_my_mix.json
    python main.py --day d4                          # another day of the week (and that day's cap)
    python main.py --members                         # also print every member's Assignment 1 summary
    python main.py --formulation I                   # run the individual-contract formulation
    python main.py --demo                            # also run the scenario-generation demonstration (Question 3)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from src.data_loader import load_community, list_communities
from src.model import CommunityOptimizationModel
from src.plotting import plot_duals, plot_inputs, plot_price_ladder, plot_schedule

RESULTS_DIR = Path(__file__).resolve().parent / "results"

FORMULATIONS = ("R", "I", "C", "DP", "SDP")


def run_community_case(
    community,
    formulation: str,
    out: Path,
    show: bool = False,
) -> dict:
    """Build, solve, and persist one formulation for every community member."""
    model = CommunityOptimizationModel(community=community, formulation=formulation)
    model.build_model()
    results = model.solve()

    for member_id, result in results.items():
        member = community.member(member_id)
        member_out = out / member_id
        member_out.mkdir(parents=True, exist_ok=True)
        result.save(member_out, tag=formulation)
        plot_inputs(member, save_to=member_out / "inputs.png")
        plot_schedule(result, member, save_to=member_out / "schedule.png")
        plot_duals(result, member, save_to=member_out / "duals.png")
        plot_price_ladder(result, member, save_to=member_out / "price_ladder.png")
        print(f"{member_id}:\n{result}\n")

    if show:
        plt.show()
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--community", default="base", help=f"base (shipped) or your own params_community_<name>.json; available: {list_communities()}")
    ap.add_argument("--day", default="d1", help="d1 (base case) ... d7")
    ap.add_argument(
        "--formulation",
        choices=FORMULATIONS,
        default="R",
        help="optimization formulation: R, I, C, DP, or SDP",
    )
    ap.add_argument("--members", action="store_true", help="print the Assignment 1 summary of every member")
    ap.add_argument("--demo", action="store_true", help="run the scenario-generation demonstration")
    ap.add_argument("--show", action="store_true", help="display generated plots")
    args = ap.parse_args()

    com = load_community(args.community, day=args.day)
    print(com.summary(), "\n")
    if args.members:
        for m in com.members:
            print(m.summary(), "\n")
    print("Cap profile (kW):", " ".join(f"{x:.1f}" for x in com.contract.capacity_limit_kW))
    ic = com.contract.individual_capacity_limit_kW
    print("Individual caps in model I (kW), hour 0 / hour 11:",
          "  ".join(f"{cid} ({com.member(cid).archetype}) {ic[cid][0]:.2f}/{ic[cid][11]:.2f}" for cid in ("C1", "C1_PVB", "C4", "C4_PVB") if cid in com.member_ids))
    output_dir = RESULTS_DIR / args.community / args.day / args.formulation
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")

    try:
        results = run_community_case(com, args.formulation, output_dir, show=args.show)
        
        # 1. Sum the 'import' column across all members' hourly DataFrames
        total_import = sum(res.hourly["import"] for res in results.values())

        print(res.hourly["import"] for res in results.values())
        
        # 2. Subtract the DSO feeder capacity limit and clip negative values to 0
        limit = com.contract.capacity_limit_kW
        congestion = (total_import - limit).clip(lower=0)
        
        # 3. Print the congestion metrics
        print(f"\n--- Congestion Summary ({args.formulation}) ---")
        print(f"Total import (kWh): {total_import}")
        print(f"Total daily congestion: {congestion.sum():.2f} kWh")
        print(f"Number of congested hours: {(congestion > 0).sum()}")
        print(f"Max hourly congestion: {congestion.max():.2f} kW")


        
    except NotImplementedError as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
