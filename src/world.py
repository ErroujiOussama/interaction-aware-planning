"""World: robots + static obstacles, stepping, collision checks and observations."""

from dataclasses import dataclass

import numpy as np

from robot import Robot, RobotSpec


# --------------------------------------------------------------------------- static map

class StaticMap:
    """Arena bounds + walls. Every obstacle is stored as line segments.

    bounds:   (xmin, ymin, xmax, ymax); its four edges are walls.
    polygons: list of (K, 2) arrays (closed); kept for drawing.
    """

    def __init__(self, bounds, polygons=(), segments=()):
        self.bounds = tuple(float(b) for b in bounds)
        self.polygons = [np.asarray(poly, dtype=float) for poly in polygons]
        xmin, ymin, xmax, ymax = self.bounds
        box = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        segs = [np.asarray(s, dtype=float).reshape(2, 2) for s in segments]
        for poly in [box, *self.polygons]:
            for k in range(len(poly)):
                segs.append(np.array([poly[k], poly[(k + 1) % len(poly)]]))
        self.segments = np.array(segs)  # (M, 2, 2)

    def distance(self, p):
        """Distance from point p to the closest wall."""
        a, b = self.segments[:, 0], self.segments[:, 1]
        ab = b - a
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0, 1)
        closest = a + t[:, None] * ab
        return float(np.min(np.linalg.norm(p - closest, axis=1)))

    def sample_points(self, step=0.1):
        """Points every `step` metres along all walls (PythonRobotics planners want point obstacles)."""
        pts = []
        for a, b in self.segments:
            n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
            pts.append(a + np.linspace(0, 1, n + 1)[:, None] * (b - a))
        return np.unique(np.round(np.vstack(pts), 6), axis=0)


# --------------------------------------------------------------------------- observations

@dataclass(frozen=True)
class NeighborObs:
    id: int
    p: np.ndarray
    v: np.ndarray
    radius: float


@dataclass(frozen=True)
class Observation:
    """Everything a planner is allowed to see at time t (mirrors a ROS 2 robot's inputs)."""
    t: float
    robot_id: int
    p: np.ndarray
    v: np.ndarray
    theta: float
    goal: np.ndarray
    spec: RobotSpec
    neighbors: tuple  # tuple[NeighborObs], only those within sensing_range


# --------------------------------------------------------------------------- world

class World:
    def __init__(self, robots, static_map, dt=0.1, goal_tol=0.1, sensing_range=np.inf):
        self.robots: list[Robot] = robots
        self.map: StaticMap = static_map
        self.dt = dt
        self.goal_tol = goal_tol
        self.sensing_range = sensing_range
        self.t = 0.0
        self.step_count = 0
        self.collision_events = []  # (t, i, j) ; j = -1 means a wall
        self._in_contact = set()
        self._check_arrivals()

    # ---- observations
    def observe(self, i) -> Observation:
        r = self.robots[i]
        neighbors = tuple(
            NeighborObs(o.id, o.p.copy(), o.v.copy(), o.spec.radius)
            for o in self.robots
            if o.id != i and np.linalg.norm(o.p - r.p) <= self.sensing_range
        )
        return Observation(self.t, i, r.p.copy(), r.v.copy(), r.theta, r.goal.copy(), r.spec, neighbors)

    # ---- dynamics
    def step(self, controls):
        """controls: dict robot_id -> u. Arrived robots stay put. Returns dict of applied controls."""
        applied = {}
        for r in self.robots:
            if r.arrived:
                r.v = np.zeros(2)
                applied[r.id] = np.zeros(2)
            else:
                applied[r.id] = r.step(controls.get(r.id, np.zeros(2)), self.dt)
        self.t += self.dt
        self.step_count += 1
        self._check_collisions()
        self._check_arrivals()
        return applied

    def _check_arrivals(self):
        for r in self.robots:
            if not r.arrived and np.linalg.norm(r.p - r.goal) <= self.goal_tol:
                r.arrived, r.arrival_time = True, self.t
                r.v = np.zeros(2)

    def _check_collisions(self):
        """Robots are not stopped on contact; we log one event each time a pair *enters* contact."""
        contact = set()
        n = len(self.robots)
        for i in range(n):
            ri = self.robots[i]
            for j in range(i + 1, n):
                rj = self.robots[j]
                if np.linalg.norm(ri.p - rj.p) < ri.spec.radius + rj.spec.radius:
                    contact.add((i, j))
            if self.map.distance(ri.p) < ri.spec.radius:
                contact.add((i, -1))
        for pair in sorted(contact - self._in_contact):
            self.collision_events.append((self.t, *pair))
        self._in_contact = contact

    # ---- queries
    def pairwise_clearance(self):
        """Min over pairs of (center distance - r_i - r_j), and min center distance."""
        n = len(self.robots)
        if n < 2:
            return np.inf, np.inf
        P = np.array([r.p for r in self.robots])
        R = np.array([r.spec.radius for r in self.robots])
        D = np.linalg.norm(P[:, None] - P[None], axis=-1)
        iu = np.triu_indices(n, 1)
        return float(np.min(D[iu] - (R[:, None] + R[None])[iu])), float(np.min(D[iu]))

    @property
    def all_arrived(self):
        return all(r.arrived for r in self.robots)
