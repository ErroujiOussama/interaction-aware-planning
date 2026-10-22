"""Run every (scenario, planner) pair headless and print a metrics table.

    python compare.py                                   # all scenarios x all planners, seed 0
    python compare.py --planners naive orca --seeds 0 1 2 --plots ../results/plots
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")

import metrics  # noqa: E402
import scenario as scenario_mod  # noqa: E402
import viz  # noqa: E402
from planners import PLANNERS  # noqa: E402
from simulation import simulate  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=scenario_mod.available())
    ap.add_argument("--planners", nargs="+", default=sorted(PLANNERS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--model", help="override the kinematic model")
    ap.add_argument("--plots", help="directory for one trajectory .png per run")
    ap.add_argument("--out", help="directory for .npz logs + .json metrics")
    args = ap.parse_args(argv)

    rows = []
    for sc_name in args.scenarios:
        sc = scenario_mod.load(sc_name)
        for planner in args.planners:
            for seed in args.seeds:
                log = simulate(sc, planner, seed, args.model)
                m = metrics.compute(log)
                rows.append((f"{sc.name}/{planner}/s{seed}", m))
                print(metrics.format_table(rows[-1:]).splitlines()[-1], flush=True)
                if args.out:
                    log.save(args.out, m)
                if args.plots:
                    pathlib.Path(args.plots).mkdir(parents=True, exist_ok=True)
                    viz.plot_trajectories(log, pathlib.Path(args.plots) / f"{sc.name}__{planner}__s{seed}.png")
    print()
    print(metrics.format_table(rows))
    return rows


if __name__ == "__main__":
    main()
