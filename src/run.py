"""Run one scenario with one planner.

    python run.py --scenario corridor_2 --planner naive
    python run.py --scenario crossing_4 --planner naive --model unicycle --save out.gif --no-show
    python run.py --scenario circle_swap_8 --planner <name> --param key=value
    python run.py --list
"""

import argparse
import json
import pathlib

import matplotlib

import metrics
import scenario as scenario_mod
from planners import PLANNERS
from robot import MODELS
from simulation import simulate


def parse_params(items):
    params = {}
    for item in items or []:
        key, _, value = item.partition("=")
        try:
            params[key] = json.loads(value)
        except json.JSONDecodeError:
            params[key] = value
    return params


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="corridor_2", help="name in scenarios/ or path to a .yaml")
    ap.add_argument("--planner", default="naive", choices=sorted(PLANNERS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=MODELS, help="override the scenario's kinematic model")
    ap.add_argument("--param", action="append", metavar="KEY=VALUE", help="planner parameter (repeatable)")
    ap.add_argument("--save", help="export the animation (.gif or .mp4)")
    ap.add_argument("--plot", help="save a static trajectory plot (.png)")
    ap.add_argument("--out", help="directory for the .npz log + .json metrics")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    ap.add_argument("--list", action="store_true", help="list scenarios and planners")
    args = ap.parse_args(argv)

    if args.list:
        print("scenarios:", ", ".join(scenario_mod.available()))
        for name, cls in sorted(PLANNERS.items()):
            print(f"planner {name:<10} params: {cls.defaults}")
        return

    if args.no_show:
        matplotlib.use("Agg")
    import viz  # after the backend choice

    log = simulate(args.scenario, args.planner, args.seed, args.model, parse_params(args.param))
    m = metrics.compute(log)
    print(metrics.format_table([(f"{log.scenario}/{log.planner}/s{log.seed}", m)]))
    print(f"\n{m['collisions_robot']} robot-robot collision(s), {m['collisions_wall']} wall collision(s); "
          f"{int(round(m['arrived_rate'] * len(log.radii)))}/{len(log.radii)} robots arrived in {log.t[-1]:.1f} s")

    if args.out:
        print("log saved to", log.save(args.out, m))
    if args.plot:
        pathlib.Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        viz.plot_trajectories(log, args.plot)
    if args.save or not args.no_show:
        viz.animate(log, save=args.save, show=not args.no_show)
    return log, m


if __name__ == "__main__":
    main()
