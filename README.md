## Run (from `src/`)

```bash
python run.py --list                                         # scenarios, planners, parameters
python run.py --scenario doorway_4 --planner prioritized     # animation
python run.py --scenario circle_swap_8 --planner orca --save swap.gif --no-show
python compare.py --planners prioritized orca --seeds 0 1 2  # metrics table
python ../tests/test_sim.py                                  # tests
```

Options: `--model unicycle` (default: single integrator), `--param key=value` (planner parameter), `--seed N`.

## Planners (`src/planners/`)

| Planner | Type | Idea |
|---|---|---|
| `naive` | sanity check | straight to the goal |
| `prioritized` | centralized | robots planned one by one (longest trip first) with a space-time A* on a grid; each plan avoids the reserved motions of the robots planned before; robots follow their plan step by step in time |
| `orca` | decentralized | ORCA (van den Berg 2011, port of RVO2): each neighbor and each nearby wall gives a half-plane of safe velocities; pick the safe velocity closest to the A* direction |

Results, seed 0, single integrator: both reach every goal with 0 collisions on all 6 scenarios.
Unicycle mode is not reliable yet for either planner (grid plans have sharp turns, ORCA assumes holonomic robots).

## Scenarios (`scenarios/*.yaml`)

| Scenario | Robots | What it tests |
|---|---|---|
| `corridor_2` | 2 | head-on swap in a narrow corridor |
| `crossing_4` | 4 | 4-way intersection |
| `circle_swap_4/8/16` | 4, 8, 16 | antipodal swap on a circle, dense conflict |
| `doorway_4` | 4 | two rooms joined by one door |

The YAML format is documented at the top of `src/scenario.py`.

## Code (`src/`)

| File | Role |
|---|---|
| `robot.py`, `world.py` | robot models, walls, collisions, observations |
| `scenario.py` | loads a YAML scenario |
| `simulation.py` | the simulation loop: `simulate(scenario, planner, seed)` |
| `planners/` | `base.py` (interface), `common.py` (A*, point tracking), one file per planner, `template.py` to start a new one |
| `metrics.py`, `logger.py`, `viz.py` | metrics, logs, animation |
| `run.py`, `compare.py` | command line |

A planner implements `reset(ctx)` (once) and `compute_control(obs)` (every step). To add one: copy `template.py`, then register the class in `planners/__init__.py`.
