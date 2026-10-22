"""Per-step logs of one run (positions, velocities, controls, solve times) and their .npz/.json export."""

import json
import pathlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RunLog:
    scenario: str
    planner: str
    seed: int
    model: str
    dt: float
    t_max: float
    radii: np.ndarray
    v_max: np.ndarray
    starts: np.ndarray
    goals: np.ndarray
    static_map: object = None
    t: list = field(default_factory=list)            # (T+1,)
    pos: list = field(default_factory=list)          # (T+1, N, 2)
    theta: list = field(default_factory=list)        # (T+1, N)
    vel: list = field(default_factory=list)          # (T+1, N, 2)
    u: list = field(default_factory=list)            # (T, N, 2)   applied (clipped) controls
    solve_time: list = field(default_factory=list)   # (T, N) [s], nan once a robot has arrived
    clearance: list = field(default_factory=list)    # (T+1,) min pairwise surface distance
    collisions: list = field(default_factory=list)   # [(t, i, j)], j = -1 for a wall
    arrival_time: list = field(default_factory=list) # (N,) nan if never arrived
    paths: dict = field(default_factory=dict)        # robot id -> planner debug path (K, 2)

    def record_state(self, world):
        self.t.append(world.t)
        self.pos.append(np.array([r.p for r in world.robots]))
        self.theta.append(np.array([r.theta for r in world.robots]))
        self.vel.append(np.array([r.v for r in world.robots]))
        self.clearance.append(world.pairwise_clearance()[0])

    def record_step(self, applied, solve_times):
        n = len(self.radii)
        self.u.append(np.array([applied[i] for i in range(n)]))
        self.solve_time.append(np.array([solve_times.get(i, np.nan) for i in range(n)]))

    def finalize(self, world):
        self.collisions = list(world.collision_events)
        self.arrival_time = np.array([r.arrival_time if r.arrived else np.nan for r in world.robots])
        for k in ("t", "pos", "theta", "vel", "clearance"):
            setattr(self, k, np.asarray(getattr(self, k)))
        n = len(self.radii)
        self.u = np.asarray(self.u).reshape(-1, n, 2)
        self.solve_time = np.asarray(self.solve_time).reshape(-1, n)
        return self

    def save(self, out_dir, metrics=None):
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{self.scenario}__{self.planner}__s{self.seed}"
        np.savez_compressed(
            out / f"{stem}.npz", t=self.t, pos=self.pos, theta=self.theta, vel=self.vel, u=self.u,
            solve_time=self.solve_time, clearance=self.clearance, radii=self.radii, v_max=self.v_max,
            starts=self.starts, goals=self.goals, arrival_time=self.arrival_time,
            collisions=np.array(self.collisions, dtype=float).reshape(-1, 3),
        )
        meta = {"scenario": self.scenario, "planner": self.planner, "seed": self.seed,
                "model": self.model, "dt": self.dt, "t_max": self.t_max, "metrics": metrics or {}}
        (out / f"{stem}.json").write_text(json.dumps(meta, indent=2, default=float))
        return out / stem
