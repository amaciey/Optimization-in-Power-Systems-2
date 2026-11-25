"""Run a community optimization formulation and summarize its results.

    python main.py                                           # base case (16 members), day d1
    python main.py --community my_mix                        # your own data/params_community_my_mix.json
    python main.py --day d4                                  # another day of the week (and that day's cap)
    python main.py --members                                 # also print every member's Assignment 1 summary
    python main.py --formulation I                           # run the individual-contract formulation
    python main.py --formulation DP                          # Q2.2/2.3 base dynamic pricing (cap only)
    python main.py --formulation DP --bb                     # Q2.4 Step 2: DP + Budget Balance
    python main.py --formulation DP --bb --ir --slack        # Q2.4 Step 3: DP + BB + IR (with slack)
    python main.py --formulation DP --bb --ir --disc         # Q2.4 Step 3: DP + BB + IR (price discrimination)
    python main.py --demo                                    # also run the scenario-generation demonstration (Question 3)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_community, list_communities
from src.model import CommunityOptimizationModel
from src.plotting import plot_duals, plot_inputs, plot_price_ladder, plot_schedule

RESULTS_DIR = Path(__file__).resolve().parent / "results"

FORMULATIONS = ("R", "I", "C", "DP", "SDP")


def _dp_variant_tag(args: argparse.Namespace) -> str:
    """Build a descriptive folder/file tag for DP variants so runs do not overwrite each other."""
    parts = ["DP"]
    if args.bb:
        parts.append("BB")
    if args.ir:
        parts.append(f"IR_{args.ir_benchmark}")
    if args.slack:
        parts.append("slack")
    if args.disc:
        parts.append("disc")
    if args.sos1:
        parts.append("sos1")
    return "_".join(parts)


def run_community_case(
    community,
    formulation: str,
    out: Path,
    show: bool = False,
    tag: str | None = None,
    **model_kwargs,
) -> dict:
    """Build, solve, and persist one formulation for every community member."""
    model = CommunityOptimizationModel(
        community=community,
        formulation=formulation,
        **model_kwargs,
    )
    model.build_model()
    results = model.solve()

    if formulation in ("C", "DP"):
        comm_out = out / "community"
        save_tag = tag or formulation
        results.save(comm_out, tag=save_tag)

        # Save per-member settlement table if available
        settlement = results.meta.get("settlement")
        if settlement:
            df_settle = pd.DataFrame.from_dict(settlement, orient="index")
            df_settle.index.name = "consumer_id"
            df_settle.to_csv(comm_out / f"{save_tag}_settlement.csv")

        # Use a PV-owning member for plot_price_ladder so c^PV is shown at 1.41 DKK/kWh
        ref_member = next((m for m in community.members if m.has_pv), community.members[0])
        plot_schedule(results, ref_member, save_to=comm_out / "schedule.png")
        plot_duals(results, ref_member, save_to=comm_out / "duals.png")
        plot_price_ladder(results, ref_member, save_to=comm_out / "price_ladder.png")

        print(f"Community ({save_tag}):\n{results}\n")
        econ = results.meta.get("economics", {})
        if econ:
            print(f"--- Economics ({save_tag}) ---")
            for k, v in econ.items():
                print(f"{k:<24}: {v:.4f} DKK")

        if settlement:
            print(f"\n--- Member Settlement ({save_tag}) ---")
            print(df_settle.round(4).to_string())
            if "ir_gain_DKK" in df_settle.columns:
                blocking = df_settle[df_settle["ir_gain_DKK"] < -1e-4]
                if not blocking.empty:
                    print("\nBlocking members (IR violated):")
                    print(blocking[["total_cost_DKK", "benchmark_cost_DKK", "ir_gain_DKK"]].round(4).to_string())
                else:
                    print("\nAll members satisfy Individual Rationality.")

        if show:
            plt.show()
        return results

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
    # Question 2.4 switches (only used when --formulation DP)
    ap.add_argument("--bb", action="store_true", help="enable Budget Balance constraint (DP)")
    ap.add_argument("--ir", action="store_true", help="enable Individual Rationality constraints (DP)")
    ap.add_argument("--ir-benchmark", choices=("I", "R"), default="I", help="benchmark for IR: I or R (default: I)")
    ap.add_argument("--slack", action="store_true", help="allow penalized slack variables for BB/IR infeasibility (DP)")
    ap.add_argument("--slack-penalty", type=float, default=1e4, help="penalty weight on BB/IR slacks (default: 1e4)")
    ap.add_argument("--disc", action="store_true", help="allow per-member price discrimination x_{i,t} (DP)")
    ap.add_argument("--sos1", action="store_true", help="use Gurobi SOS1 instead of big-M binaries (DP)")

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

    model_kwargs = {}
    run_tag = args.formulation
    if args.formulation == "DP":
        run_tag = _dp_variant_tag(args)
        model_kwargs = dict(
            budget_balance=args.bb,
            individual_rationality=args.ir,
            ir_benchmark=args.ir_benchmark,
            allow_infeasibility_slack=args.slack,
            slack_penalty=args.slack_penalty,
            price_discrimination=args.disc,
            use_sos1=args.sos1,
        )

    output_dir = RESULTS_DIR / args.community / args.day / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        import runpy
        runpy.run_module("src.scenario_generation", run_name="__main__")

    try:
        results = run_community_case(
            com,
            args.formulation,
            output_dir,
            show=args.show,
            tag=run_tag,
            **model_kwargs,
        )

        # 1. Read the shared meter for C and DP, otherwise sum member meters.
        if args.formulation in ("C", "DP"):
            total_import = results.hourly["community_import"]
        else:
            total_import = sum(res.hourly["import"] for res in results.values())

        # 2. Subtract the DSO feeder capacity limit and clip negative values to 0
        limit = com.contract.capacity_limit_kW
        congestion = (total_import - limit).clip(lower=0)

        # 3. Print the congestion metrics
        print(f"\n--- Congestion Summary ({run_tag}) ---")
        print(f"Total import (kWh): {total_import.sum():.2f}")
        print(f"Total daily congestion: {congestion.sum():.2f} kWh")
        print(f"Number of congested hours: {(congestion > 1e-3).sum()}")
        print(f"Max hourly congestion: {congestion.max():.2f} kW")

    except NotImplementedError as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()