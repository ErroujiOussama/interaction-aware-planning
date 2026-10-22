"""Planner interface. One Planner instance is created per robot for each run.

Lifecycle (driven by simulation.simulate):
    planner = make_planner("orca", **params)
    planner.reset(ctx)                       # once, before t = 0
    u = planner.compute_control(obs)         # every step, until the robot arrives

`compute_control` only gets an Observation (own state + neighbors in sensing
range). Static knowledge (map, own goal) comes through the PlannerContext, like
a map server would on a real robot. `ctx.fleet` and `ctx.shared` exist for
centralized strategies (e.g. prioritized planning): decentralized planners must
not use them.

Output u: (vx, vy) for "single_integrator", (v, omega) for "unicycle"
(ctx.spec.model). The world clips u to the robot's limits.
"""

from dataclasses import dataclass, field

import numpy as np

from robot import RobotSpec
from world import Observation, StaticMap


@dataclass
class PlannerContext:
    robot_id: int
    spec: RobotSpec
    start: np.ndarray
    goal: np.ndarray
    static_map: StaticMap
    dt: float
    fleet: list = field(default_factory=list)   # [{"id", "start", "goal", "spec"}] of every robot (centralized only)
    shared: dict = field(default_factory=dict)  # one dict shared by every planner instance of this run
    t_max: float = 60.0                         # [s] run timeout (planning horizon of offline planners)


class Planner:
    name = "base"
    defaults: dict = {}  # tunable parameters; override with --param key=value

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise TypeError(f"{self.name}: unknown parameter(s) {sorted(unknown)}; known: {sorted(self.defaults)}")
        self.params = {**self.defaults, **params}
        self.ctx: PlannerContext = None
        self.debug = {}  # optional, e.g. {"path": (K, 2) array} is drawn by viz

    def reset(self, ctx: PlannerContext):
        self.ctx = ctx

    def compute_control(self, obs: Observation) -> np.ndarray:
        raise NotImplementedError
