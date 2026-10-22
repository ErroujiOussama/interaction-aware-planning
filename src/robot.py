"""Robot state and kinematic models.

Two models, chosen per scenario (or overridden on the command line):

- "single_integrator": u = (vx, vy)    p' = u,                 |u| <= v_max
- "unicycle":          u = (v, omega)  p' = v (cos θ, sin θ),  θ' = omega
"""

from dataclasses import dataclass, field

import numpy as np

SINGLE_INTEGRATOR = "single_integrator"
UNICYCLE = "unicycle"
MODELS = (SINGLE_INTEGRATOR, UNICYCLE)


@dataclass
class RobotSpec:
    radius: float = 0.25
    v_max: float = 1.0
    w_max: float = 2.0  # unicycle only [rad/s]
    model: str = SINGLE_INTEGRATOR

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}, expected one of {MODELS}")


@dataclass
class Robot:
    id: int
    spec: RobotSpec
    start: np.ndarray
    goal: np.ndarray
    p: np.ndarray = None   # position (2,)
    theta: float = 0.0     # heading; for single integrator = direction of last motion
    v: np.ndarray = field(default_factory=lambda: np.zeros(2))  # world-frame velocity (2,)
    arrived: bool = False
    arrival_time: float = None

    def __post_init__(self):
        self.start = np.asarray(self.start, dtype=float)
        self.goal = np.asarray(self.goal, dtype=float)
        if self.p is None:
            self.p = self.start.copy()

    def clip_control(self, u):
        u = np.asarray(u, dtype=float).reshape(2)
        if self.spec.model == SINGLE_INTEGRATOR:
            n = np.linalg.norm(u)
            if n > self.spec.v_max:
                u = u * (self.spec.v_max / n)
            return u
        v = np.clip(u[0], -self.spec.v_max, self.spec.v_max)
        w = np.clip(u[1], -self.spec.w_max, self.spec.w_max)
        return np.array([v, w])

    def step(self, u, dt):
        """Apply control u for dt seconds. Returns the clipped control actually applied."""
        u = self.clip_control(u)
        if self.spec.model == SINGLE_INTEGRATOR:
            self.v = u.copy()
            if np.linalg.norm(u) > 1e-6:
                self.theta = float(np.arctan2(u[1], u[0]))
        else:
            v, w = u
            self.theta = float(np.arctan2(np.sin(self.theta + w * dt), np.cos(self.theta + w * dt)))
            self.v = v * np.array([np.cos(self.theta), np.sin(self.theta)])
        self.p = self.p + self.v * dt
        return u
