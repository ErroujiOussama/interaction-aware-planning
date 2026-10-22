"""Load scenarios/*.yaml into an initial World.

Schema (all keys optional except map.bounds and robots|circle):

    name: corridor_2
    dt: 0.1                 # [s]
    t_max: 30.0             # [s] timeout
    goal_tol: 0.1           # [m]
    sensing_range: .inf     # [m] neighbors farther than this are not observed
    model: single_integrator | unicycle
    robot: {radius: 0.25, v_max: 1.0, w_max: 2.0}    # defaults for every robot
    noise: {start: 0.0, goal: 0.0}                   # gaussian std [m], drawn from the run seed
    map:
      bounds: [xmin, ymin, xmax, ymax]               # outer walls
      rects: [[xmin, ymin, xmax, ymax], ...]
      polygons: [[[x, y], ...], ...]
      segments: [[[x1, y1], [x2, y2]], ...]
    robots:                                          # explicit list ...
      - {start: [x, y], goal: [x, y], radius: 0.3}   # per-robot overrides allowed
    circle: {n: 8, radius: 4.0, center: [0, 0]}      # ... or n robots swapping across a circle
"""

import pathlib
from dataclasses import dataclass, field

import numpy as np
import yaml

from robot import Robot, RobotSpec
from world import StaticMap, World

SCENARIO_DIR = pathlib.Path(__file__).resolve().parents[1] / "scenarios"


@dataclass
class Scenario:
    name: str
    static_map: StaticMap
    robots: list  # list of dict(start, goal, spec)
    dt: float = 0.1
    t_max: float = 60.0
    goal_tol: float = 0.1
    sensing_range: float = np.inf
    noise: dict = field(default_factory=dict)
    description: str = ""

    def build_world(self, seed=0, model=None) -> World:
        """Fresh World for one run. `seed` drives start/goal noise; `model` overrides the kinematics."""
        rng = np.random.default_rng(seed)
        s_std, g_std = self.noise.get("start", 0.0), self.noise.get("goal", 0.0)
        robots = []
        for i, r in enumerate(self.robots):
            spec = r["spec"] if model is None else RobotSpec(**{**r["spec"].__dict__, "model": model})
            start = np.asarray(r["start"], float) + rng.normal(0, s_std, 2)
            goal = np.asarray(r["goal"], float) + rng.normal(0, g_std, 2)
            robot = Robot(i, spec, start, goal)
            d = goal - start
            robot.theta = float(np.arctan2(d[1], d[0]))  # start facing the goal
            robots.append(robot)
        return World(robots, self.static_map, self.dt, self.goal_tol, self.sensing_range)


def _rect(xmin, ymin, xmax, ymax):
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]


def load(name_or_path) -> Scenario:
    path = pathlib.Path(name_or_path)
    if not path.exists():
        path = SCENARIO_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml")))
        raise FileNotFoundError(f"scenario {name_or_path!r} not found. Available: {available}")
    cfg = yaml.safe_load(path.read_text())

    m = cfg["map"]
    polygons = [_rect(*r) for r in m.get("rects", [])] + list(m.get("polygons", []))
    static_map = StaticMap(m["bounds"], polygons, m.get("segments", []))

    defaults = {"model": cfg.get("model", "single_integrator"), **cfg.get("robot", {})}
    entries = list(cfg.get("robots", []))
    if "circle" in cfg:
        c = cfg["circle"]
        center = np.asarray(c.get("center", [0.0, 0.0]), float)
        for k in range(c["n"]):
            a = 2 * np.pi * k / c["n"]
            p = center + c["radius"] * np.array([np.cos(a), np.sin(a)])
            entries.append({"start": p.tolist(), "goal": (2 * center - p).tolist()})
    if not entries:
        raise ValueError(f"{path.name}: needs a 'robots' list or a 'circle' block")

    spec_keys = set(RobotSpec.__dataclass_fields__)
    robots = []
    for e in entries:
        spec = RobotSpec(**{**defaults, **{k: v for k, v in e.items() if k in spec_keys}})
        robots.append({"start": e["start"], "goal": e["goal"], "spec": spec})

    return Scenario(
        name=cfg.get("name", path.stem),
        description=cfg.get("description", ""),
        static_map=static_map,
        robots=robots,
        dt=float(cfg.get("dt", 0.1)),
        t_max=float(cfg.get("t_max", 60.0)),
        goal_tol=float(cfg.get("goal_tol", 0.1)),
        sensing_range=float(cfg.get("sensing_range", np.inf)),
        noise=cfg.get("noise", {}) or {},
    )


def available():
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))
